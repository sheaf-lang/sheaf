// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Code generation - translate CompiledExpr to StableHLO MLIR

use crate::autodiff::grad_simplified;
use crate::compiler::stablehlo::{Register, StableHLOEmitter, StableHLOType};
use crate::core::compiler::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use crate::runtime::{math_ops, nn_ops, tensor_ops};
use std::collections::HashMap;

/// Code generator - converts CompiledExpr to StableHLO
pub struct CodeGenerator {
    emitter: StableHLOEmitter,
    /// Map from variable names to registers and their types
    bindings: HashMap<String, (Register, StableHLOType)>,
    /// Lambdas bound in let forms — stored for inlining, not emitted as SSA.
    lambda_bindings: HashMap<String, CompiledExpr>,
    /// Function registry for user-defined functions
    function_registry: HashMap<String, crate::core::compiler::FunctionDef>,
}

impl CodeGenerator {
    pub fn new() -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
            lambda_bindings: HashMap::new(),
            function_registry: HashMap::new(),
        }
    }

    pub fn with_registry(registry: HashMap<String, crate::core::compiler::FunctionDef>) -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
            lambda_bindings: HashMap::new(),
            function_registry: registry,
        }
    }

    /// Create a CodeGenerator with function parameters bound to %arg0, %arg1, etc.
    pub fn with_function_params(
        registry: HashMap<String, crate::core::compiler::FunctionDef>,
        param_names: &[String],
        param_types: &[StableHLOType],
    ) -> Self {
        let mut bindings = HashMap::new();
        for (i, (name, ty)) in param_names.iter().zip(param_types.iter()).enumerate() {
            bindings.insert(name.clone(), (Register::arg(i), ty.clone()));
        }

        Self {
            emitter: StableHLOEmitter::new(),
            bindings,
            lambda_bindings: HashMap::new(),
            function_registry: registry,
        }
    }

    /// Generate StableHLO for a compiled expression
    pub fn generate(&mut self, expr: &CompiledExpr) -> SheafResult<(Register, StableHLOType)> {
        match expr {
            CompiledExpr::Integer(n) => {
                // Treat integers as floats for now (matches Python behavior)
                let reg = self.emitter.emit_constant_f32(*n as f64);
                Ok((reg, StableHLOType::scalar_f32()))
            }

            CompiledExpr::Float(x) => {
                let reg = self.emitter.emit_constant_f32(*x);
                Ok((reg, StableHLOType::scalar_f32()))
            }

            CompiledExpr::Vector(elements) => {
                match try_flatten_to_constant(elements) {
                    Some((data, shape)) => {
                        let (reg, ty) = self.emitter.emit_nd_tensor_constant(&data, &shape);
                        Ok((reg, ty))
                    }
                    None => Err(SheafError::Compile {
                        message: "Vector contains non-constant expressions; \
                                  cannot emit as tensor constant"
                            .to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    }),
                }
            }

            CompiledExpr::Symbol(name) => {
                // Look up symbol in bindings
                if let Some((reg, ty)) = self.bindings.get(name) {
                    Ok((reg.clone(), ty.clone()))
                } else {
                    Err(SheafError::Compile {
                        message: format!("Undefined symbol in codegen: {}", name),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            }

            CompiledExpr::GetTupleElement { param, indices } => {
                // Resolve a field extracted by with-params via get_tuple_element
                // The param must be in bindings with a Tuple type
                let (param_reg, param_ty) =
                    self.bindings
                        .get(param)
                        .cloned()
                        .ok_or_else(|| SheafError::Compile {
                            message: format!(
                                "GetTupleElement: parameter '{}' not found in bindings",
                                param
                            ),
                            location: crate::core::error::SourceLocation::unknown(),
                        })?;

                // Walk nested tuple type and emit get_tuple_element for each index
                let mut current_reg = param_reg;
                let mut current_ty = param_ty;

                for &idx in indices {
                    let element_ty = match &current_ty {
                        StableHLOType::Tuple(elems) => elems.get(idx).cloned().ok_or_else(|| {
                            SheafError::Compile {
                                message: format!(
                                    "GetTupleElement: index {} out of range for tuple with {} elements",
                                    idx,
                                    elems.len()
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            }
                        })?,
                        other => {
                            return Err(SheafError::Compile {
                                message: format!(
                                    "GetTupleElement: expected tuple type, got {}",
                                    other.to_mlir()
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }
                    };
                    let result_reg = self.emitter.emit_get_tuple_element(
                        &current_reg,
                        &current_ty,
                        idx,
                        &element_ty,
                    );
                    current_reg = result_reg;
                    current_ty = element_ty;
                }

                Ok((current_reg, current_ty))
            }

            CompiledExpr::FunctionCall { name, args } => self.generate_function_call(name, args),

            CompiledExpr::Let { bindings, body } => {
                let mut lambda_names = Vec::new();
                let mut destructured_names: Vec<String> = Vec::new();
                for (name, value_expr) in bindings {
                    if matches!(value_expr, CompiledExpr::Lambda { .. }) {
                        // Store lambda for inlining — no SSA emitted.
                        self.lambda_bindings
                            .insert(name.clone(), value_expr.clone());
                        lambda_names.push(name.clone());
                    } else if name.starts_with('[') && name.ends_with(']') {
                        // Destructuring bind: [a b c] = tuple → get_tuple_element
                        let names: Vec<&str> =
                            name[1..name.len() - 1].split_whitespace().collect();
                        let (tuple_reg, tuple_ty) = self.generate(value_expr)?;
                        let element_types = match &tuple_ty {
                            StableHLOType::Tuple(tys) => tys.clone(),
                            other => {
                                return Err(SheafError::Compile {
                                    message: format!(
                                        "Let destructuring requires a tuple, got: {}",
                                        other.to_mlir()
                                    ),
                                    location: crate::core::error::SourceLocation::unknown(),
                                })
                            }
                        };
                        for (i, n) in names.iter().enumerate() {
                            let elem_reg = self.emitter.emit_get_tuple_element(
                                &tuple_reg,
                                &tuple_ty,
                                i,
                                &element_types[i],
                            );
                            self.bindings
                                .insert(n.to_string(), (elem_reg, element_types[i].clone()));
                            destructured_names.push(n.to_string());
                        }
                    } else {
                        let (reg, ty) = self.generate(value_expr)?;
                        self.bindings.insert(name.clone(), (reg, ty));
                    }
                }
                let result = self.generate(body)?;
                // Clean up (proper scoping)
                for (name, _) in bindings {
                    self.bindings.remove(name);
                }
                for name in &lambda_names {
                    self.lambda_bindings.remove(name);
                }
                for name in &destructured_names {
                    self.bindings.remove(name);
                }
                Ok(result)
            }

            CompiledExpr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let (cond_reg, cond_ty) = self.generate(condition)?;
                let (then_reg, then_ty) = self.generate(then_branch)?;

                if let Some(else_expr) = else_branch {
                    let (else_reg, else_ty) = self.generate(else_expr)?;
                    // Use stablehlo.select: result = select(cond, then, else)
                    let (result_reg, result_ty) = self.emitter.emit_select(
                        &cond_reg, &then_reg, &else_reg, &cond_ty, &then_ty, &else_ty,
                    );
                    Ok((result_reg, result_ty))
                } else {
                    // If without else: just return then_branch
                    // (assumes condition is always true for now)
                    Ok((then_reg, then_ty))
                }
            }

            CompiledExpr::Do(exprs) => {
                // Evaluate all expressions, return the last one
                let mut last_result = None;
                for expr in exprs {
                    last_result = Some(self.generate(expr)?);
                }
                last_result.ok_or_else(|| SheafError::Compile {
                    message: "do requires at least one expression".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }

            CompiledExpr::FunctionRef(name) => Err(SheafError::Compile {
                message: format!("Cannot generate code for bare function reference: {}", name),
                location: crate::core::error::SourceLocation::unknown(),
            }),

            // Lambda and LambdaCall: inline at call site.
            // A bare Lambda without a call is not directly emittable.
            CompiledExpr::Lambda { .. } => Err(SheafError::Compile {
                message: "Cannot emit a lambda without a call site".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),

            CompiledExpr::LambdaCall { callee, args } => {
                let callee = callee.clone();
                let args = args.clone();
                self.inline_lambda_call(&callee, &args)
            }

            CompiledExpr::ValueAndGrad { fn_name, .. } => Err(SheafError::Compile {
                message: format!(
                    "ValueAndGrad '{}' is a module-level form, not an inline expression",
                    fn_name
                ),
                location: crate::core::error::SourceLocation::unknown(),
            }),

            CompiledExpr::InlineValueAndGrad {
                lambda,
                args,
                wrt_indices,
            } => {
                let lambda = lambda.clone();
                let args = args.clone();
                let wrt_indices = wrt_indices.clone();
                self.generate_inline_value_and_grad(&lambda, &args, &wrt_indices)
            }

            _ => Err(SheafError::Compile {
                message: format!("Code generation not yet implemented for: {:?}", expr),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }
    }

    /// Generate code for a function call
    fn generate_function_call(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        // Check if this is a user-defined function in the registry
        // Clone the signature to avoid borrow checker issues
        let signature = self
            .function_registry
            .get(name)
            .and_then(|func_def| func_def.signature.clone());

        if let Some(_signature) = signature {
            // Generate code for each argument
            let mut arg_registers = Vec::new();
            let mut arg_types = Vec::new();

            for arg in args {
                let (reg, ty) = self.generate(arg)?;
                arg_registers.push(reg);
                arg_types.push(ty);
            }

            // Inline user-defined functions when their body is available and
            // the call is monomorphic (arg types are known). This avoids the
            // problem of emitting a func.call to a function compiled with the
            // wrong (scalar) type from inference.
            let func_def = self.function_registry.get(name).cloned();
            if let Some(func_def) = func_def {
                if let Some(body) = &func_def.body_compiled {
                    // Bind arg registers to param names in our bindings map
                    let saved_bindings = self.bindings.clone();
                    for (param, (reg, ty)) in func_def
                        .params
                        .iter()
                        .zip(arg_registers.iter().zip(arg_types.iter()))
                    {
                        self.bindings
                            .insert(param.clone(), (reg.clone(), ty.clone()));
                    }
                    let body = body.clone();
                    let result = self.generate(&body);
                    self.bindings = saved_bindings;
                    return result;
                }
            }

            // Fallback: emit func.call (may have type issues if not monomorphic)
            let sig = self
                .function_registry
                .get(name)
                .and_then(|f| f.signature.clone())
                .unwrap();
            let result_reg =
                self.emitter
                    .emit_func_call(name, &arg_registers, &arg_types, &sig.return_type);
            return Ok((result_reg, sig.return_type.clone()));
        }

        // Binary arithmetic operations
        if matches!(name, "+" | "-" | "*" | "/") && args.len() == 2 {
            let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
            let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
            let (result_reg, result_ty) = math_ops::emit_arithmetic_binop(
                &mut self.emitter,
                name,
                &lhs_reg,
                &rhs_reg,
                &lhs_ty,
                &rhs_ty,
            );
            Ok((result_reg, result_ty))
        }
        // Extended arithmetic: **, //, mod
        else if matches!(name, "**" | "//" | "%" | "mod") && args.len() == 2 {
            let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
            let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
            let (result_reg, result_ty) = math_ops::emit_extended_arithmetic(
                &mut self.emitter,
                name,
                &lhs_reg,
                &rhs_reg,
                &lhs_ty,
                &rhs_ty,
            );
            Ok((result_reg, result_ty))
        }
        // Min/max operations
        else if matches!(name, "min" | "max") && args.len() == 2 {
            let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
            let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
            let (result_reg, result_ty) = math_ops::emit_minmax(
                &mut self.emitter,
                name,
                &lhs_reg,
                &rhs_reg,
                &lhs_ty,
                &rhs_ty,
            );
            Ok((result_reg, result_ty))
        }
        // Comparison operations
        else if matches!(name, "=" | "==" | "!=" | "<" | "<=" | ">" | ">=") && args.len() == 2 {
            let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
            let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
            let (result_reg, result_ty) = math_ops::emit_comparison(
                &mut self.emitter,
                name,
                &lhs_reg,
                &rhs_reg,
                &lhs_ty,
                &rhs_ty,
            );
            Ok((result_reg, result_ty))
        }
        // Matrix multiply
        else if name == "@" && args.len() == 2 {
            let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
            let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
            let (result_reg, result_ty) =
                math_ops::emit_matmul(&mut self.emitter, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty);
            Ok((result_reg, result_ty))
        }
        // Boolean binary operations
        else if matches!(name, "and" | "or") && args.len() == 2 {
            let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
            let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
            let (result_reg, result_ty) = math_ops::emit_boolean_binop(
                &mut self.emitter,
                name,
                &lhs_reg,
                &rhs_reg,
                &lhs_ty,
                &rhs_ty,
            );
            Ok((result_reg, result_ty))
        }
        // Math unary operations: sqrt, exp, log, abs
        else if matches!(name, "sqrt" | "exp" | "log" | "abs") && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg =
                math_ops::emit_math_unary(&mut self.emitter, name, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        }
        // Boolean not
        else if name == "not" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg = math_ops::emit_not(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        }
        // Neural network unary operations: relu, sigmoid, tanh
        else if name == "relu" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg = nn_ops::emit_relu(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        } else if name == "sigmoid" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg = nn_ops::emit_sigmoid(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        } else if name == "tanh" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg = nn_ops::emit_tanh(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        }
        // zeros: (zeros [M N])
        else if name == "zeros" && args.len() == 1 {
            // Extract shape from vector
            if let CompiledExpr::Vector(shape_elems) = &args[0] {
                let shape: Vec<i64> = shape_elems
                    .iter()
                    .map(|e| match e {
                        CompiledExpr::Integer(n) => *n,
                        _ => panic!("Shape element must be integer"),
                    })
                    .collect();
                let (reg, ty) = tensor_ops::emit_zeros(&mut self.emitter, &shape);
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "zeros expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // random-normal: (random-normal key [M N])
        else if name == "random-normal" && args.len() == 2 {
            // Ignore key for now, extract shape
            if let CompiledExpr::Vector(shape_elems) = &args[1] {
                let shape: Vec<i64> = shape_elems
                    .iter()
                    .map(|e| match e {
                        CompiledExpr::Integer(n) => *n,
                        _ => panic!("Shape element must be integer"),
                    })
                    .collect();
                let (reg, ty) = tensor_ops::emit_random_normal(&mut self.emitter, &shape);
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "random-normal expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // ones: (ones [M N])
        else if name == "ones" && args.len() == 1 {
            if let CompiledExpr::Vector(shape_elems) = &args[0] {
                let shape: Vec<i64> = shape_elems
                    .iter()
                    .map(|e| match e {
                        CompiledExpr::Integer(n) => *n,
                        _ => panic!("Shape element must be integer"),
                    })
                    .collect();
                let (reg, ty) = tensor_ops::emit_ones(&mut self.emitter, &shape);
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "ones expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // reshape: (reshape tensor [M N])
        else if name == "reshape" && args.len() == 2 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            if let CompiledExpr::Vector(shape_elems) = &args[1] {
                let new_shape: Vec<i64> = shape_elems
                    .iter()
                    .map(|e| match e {
                        CompiledExpr::Integer(n) => *n,
                        _ => panic!("Shape element must be integer"),
                    })
                    .collect();
                let (reg, ty) = tensor_ops::emit_reshape(
                    &mut self.emitter,
                    &operand_reg,
                    &operand_ty,
                    &new_shape,
                );
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "reshape expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // transpose: (transpose tensor [1 0]) or (transpose tensor) — default perm [1 0]
        else if name == "transpose" && (args.len() == 1 || args.len() == 2) {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let permutation: Vec<i64> = if args.len() == 2 {
                if let CompiledExpr::Vector(perm_elems) = &args[1] {
                    perm_elems
                        .iter()
                        .map(|e| match e {
                            CompiledExpr::Integer(n) => *n,
                            _ => panic!("Permutation element must be integer"),
                        })
                        .collect()
                } else {
                    return Err(SheafError::Compile {
                        message: "transpose expects a vector permutation argument".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            } else {
                // Default: swap last two dims (works for 2D matrices)
                let ndim = operand_ty.shape().len().max(2) as i64;
                let mut perm: Vec<i64> = (0..ndim).collect();
                perm.swap((ndim - 2) as usize, (ndim - 1) as usize);
                perm
            };
            let (reg, ty) = tensor_ops::emit_transpose(
                &mut self.emitter,
                &operand_reg,
                &operand_ty,
                &permutation,
            );
            Ok((reg, ty))
        }
        // arange: (arange N) -> tensor<Nxf32> with [0, 1, 2, ..., N-1]
        else if name == "arange" && args.len() == 1 {
            if let CompiledExpr::Integer(n) = &args[0] {
                let shape = vec![*n];
                let (reg, ty) = tensor_ops::emit_arange(&mut self.emitter, &shape, 0);
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "arange expects an integer argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // concat: (concat [tensor1 tensor2 ...] dim)
        else if name == "concat" && args.len() == 2 {
            if let CompiledExpr::Vector(tensor_exprs) = &args[0] {
                // Generate all tensor operands
                let mut operand_regs = Vec::new();
                let mut operand_types = Vec::new();
                for expr in tensor_exprs {
                    let (reg, ty) = self.generate(expr)?;
                    operand_regs.push(reg);
                    operand_types.push(ty);
                }

                // Get dimension
                if let CompiledExpr::Integer(dim) = &args[1] {
                    let (reg, ty) = tensor_ops::emit_concatenate(
                        &mut self.emitter,
                        &operand_regs,
                        &operand_types,
                        *dim,
                    );
                    Ok((reg, ty))
                } else {
                    Err(SheafError::Compile {
                        message: "concat expects an integer dimension argument".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            } else {
                Err(SheafError::Compile {
                    message: "concat expects a vector of tensors as first argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // where: (where condition x y)
        else if name == "where" && args.len() == 3 {
            let (condition_reg, condition_ty) = self.generate(&args[0])?;
            let (x_reg, x_ty) = self.generate(&args[1])?;
            let (y_reg, y_ty) = self.generate(&args[2])?;
            let (reg, ty) = tensor_ops::emit_where(
                &mut self.emitter,
                &condition_reg,
                &x_reg,
                &y_reg,
                &condition_ty,
                &x_ty,
                &y_ty,
            );
            Ok((reg, ty))
        }
        // swapaxes: (swapaxes x axis1 axis2)
        else if name == "swapaxes" && args.len() == 3 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let axis1 = match &args[1] {
                CompiledExpr::Integer(n) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "swapaxes axis1 must be an integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let axis2 = match &args[2] {
                CompiledExpr::Integer(n) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "swapaxes axis2 must be an integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let (reg, ty) = tensor_ops::emit_swapaxes(
                &mut self.emitter,
                &operand_reg,
                &operand_ty,
                axis1,
                axis2,
            );
            Ok((reg, ty))
        }
        // tril: (tril x)
        else if name == "tril" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let (reg, ty) = tensor_ops::emit_tril(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((reg, ty))
        }
        // sum/mean: (sum x :axis N) or (sum x :axis N :keepdims true)
        else if (name == "sum" || name == "mean") && !args.is_empty() {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;

            // Parse keyword args: :axis N :keepdims bool
            let mut axis: i64 = -1; // default: last axis
            let mut keepdims = false;
            let mut i = 1;
            while i + 1 < args.len() {
                match &args[i] {
                    CompiledExpr::Keyword(k) if k == "axis" => {
                        if let CompiledExpr::Integer(n) = &args[i + 1] {
                            axis = *n;
                        }
                        i += 2;
                    }
                    CompiledExpr::Keyword(k) if k == "keepdims" => {
                        if let CompiledExpr::Boolean(b) = &args[i + 1] {
                            keepdims = *b;
                        }
                        i += 2;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }

            let (reg, ty) = if name == "sum" {
                tensor_ops::emit_sum(&mut self.emitter, &operand_reg, &operand_ty, axis, keepdims)
            } else {
                tensor_ops::emit_mean(&mut self.emitter, &operand_reg, &operand_ty, axis, keepdims)
            };
            Ok((reg, ty))
        } else {
            Err(SheafError::Compile {
                message: format!("Function call not yet supported: {}", name),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    /// Inline a lambda call: generate args, bind params→registers, generate body.
    fn inline_lambda_call(
        &mut self,
        callee: &CompiledExpr,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        // Resolve callee — may be a Lambda directly, or a Symbol bound in lambda_bindings.
        let lambda = match callee {
            CompiledExpr::Lambda { .. } => callee.clone(),
            CompiledExpr::Symbol(name) => {
                self.lambda_bindings
                    .get(name)
                    .cloned()
                    .ok_or_else(|| SheafError::Compile {
                        message: format!("Undefined lambda: {}", name),
                        location: crate::core::error::SourceLocation::unknown(),
                    })?
            }
            other => {
                return Err(SheafError::Compile {
                    message: format!("Expected lambda at call site, got: {:?}", other),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }
        };

        let (params, body) = match lambda {
            CompiledExpr::Lambda { params, body } => (params, *body),
            _ => unreachable!(),
        };

        // Generate argument values.
        let mut arg_regs = Vec::new();
        let mut arg_tys = Vec::new();
        for arg in args {
            let (reg, ty) = self.generate(arg)?;
            arg_regs.push(reg);
            arg_tys.push(ty);
        }

        // Bind param names → (register, type) and generate body.
        let saved = self.bindings.clone();
        for (param, (reg, ty)) in params.iter().zip(arg_regs.iter().zip(arg_tys.iter())) {
            self.bindings
                .insert(param.clone(), (reg.clone(), ty.clone()));
        }
        let result = self.generate(&body);
        self.bindings = saved;
        result
    }

    /// Inline value-and-grad: forward pass + symbolic backward passes → tuple.
    fn generate_inline_value_and_grad(
        &mut self,
        lambda: &CompiledExpr,
        args: &[CompiledExpr],
        wrt_indices: &[usize],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (params, body) = match lambda {
            CompiledExpr::Lambda { params, body } => (params, body),
            _ => {
                return Err(SheafError::Compile {
                    message: "InlineValueAndGrad: expected lambda".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        };

        // Generate argument values
        let mut arg_regs = Vec::new();
        let mut arg_tys = Vec::new();
        for arg in args {
            let (reg, ty) = self.generate(arg)?;
            arg_regs.push(reg);
            arg_tys.push(ty);
        }

        // Bind lambda params → arg registers
        let saved = self.bindings.clone();
        for (param, (reg, ty)) in params.iter().zip(arg_regs.iter().zip(arg_tys.iter())) {
            self.bindings
                .insert(param.clone(), (reg.clone(), ty.clone()));
        }

        // Forward pass
        let (loss_reg, loss_ty) = self.generate(body)?;

        // Backward passes
        let mut grad_regs = Vec::new();
        let mut grad_tys = Vec::new();
        for &idx in wrt_indices {
            let grad_expr = grad_simplified(body, &params[idx]);
            let (grad_reg, grad_ty) = self.generate(&grad_expr)?;
            grad_regs.push(grad_reg);
            grad_tys.push(grad_ty);
        }

        // Restore bindings
        self.bindings = saved;

        // Pack into tuple: (loss, grad0, grad1, ...)
        let mut all_regs = vec![loss_reg];
        all_regs.extend(grad_regs);
        let mut all_tys = vec![loss_ty];
        all_tys.extend(grad_tys);

        Ok(self.emitter.emit_tuple(&all_regs, &all_tys))
    }

    /// Emit a complete function module
    pub fn emit_function(mut self, name: &str, expr: &CompiledExpr) -> SheafResult<String> {
        let (result_reg, result_ty) = self.generate(expr)?;
        self.emitter.emit_return(&result_reg, &result_ty);

        Ok(self.emitter.emit_function_body(name, &result_ty))
    }

    /// Emit a function declaration from a compiled expression
    ///
    /// Generates the body instructions and wraps them in a func.func declaration
    /// with the given parameter types and return type
    pub fn emit_func_declaration(
        mut self,
        name: &str,
        expr: &CompiledExpr,
        param_types: &[StableHLOType],
        return_type: &StableHLOType,
    ) -> SheafResult<String> {
        let (result_reg, result_ty) = self.generate(expr)?;
        self.emitter.emit_return(&result_reg, &result_ty);

        // Clone body to avoid borrow issues
        let body = self.emitter.body.clone();
        Ok(self
            .emitter
            .emit_func_declaration(name, param_types, return_type, &body))
    }

    /// Finalize a multi-output function declaration.
    ///
    /// Emits a `return %r0, %r1, ...` then wraps everything in a `func.func`
    /// with a multi-value return type `-> (t0, t1, ...)`.
    ///
    /// The caller has already called `generate()` for each output and collected
    /// the resulting (Register, StableHLOType) pairs.
    pub fn finish_multi(
        mut self,
        name: &str,
        param_types: &[StableHLOType],
        result_regs: &[crate::compiler::stablehlo::Register],
        result_types: &[StableHLOType],
    ) -> String {
        self.emitter.emit_return_multi(result_regs, result_types);
        let body = self.emitter.body.clone();
        self.emitter
            .emit_func_declaration_multi(name, param_types, result_types, &body)
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Try to flatten a `Vector` of `CompiledExpr` into a constant tensor.
///
/// Returns `Some((flat_data, shape))` if every leaf is a numeric literal
/// (Float or Integer) and all sub-vectors have consistent dimensions.
/// Returns `None` if any element is a non-literal expression (Symbol,
/// FunctionCall, etc.) — the caller should then emit a tuple or report
/// an error.
///
/// Works recursively for arbitrary nesting depth:
///   `[1.0 2.0]`                  → `([1.0, 2.0], [2])`
///   `[[1.0 2.0] [3.0 4.0]]`     → `([1.0, 2.0, 3.0, 4.0], [2, 2])`
///   `[[[1] [2]] [[3] [4]]]`     → `([1.0, 2.0, 3.0, 4.0], [2, 2, 1])`
fn try_flatten_to_constant(elements: &[CompiledExpr]) -> Option<(Vec<f64>, Vec<i64>)> {
    if elements.is_empty() {
        return Some((vec![], vec![0]));
    }

    match &elements[0] {
        CompiledExpr::Float(_) | CompiledExpr::Integer(_) => {
            // Leaf level: all elements must be numeric
            let mut data = Vec::with_capacity(elements.len());
            for e in elements {
                match e {
                    CompiledExpr::Float(x) => data.push(*x),
                    CompiledExpr::Integer(n) => data.push(*n as f64),
                    _ => return None,
                }
            }
            Some((data, vec![elements.len() as i64]))
        }
        CompiledExpr::Vector(_) => {
            // Nested: recurse into each sub-vector, check shapes are uniform
            let mut all_data = Vec::new();
            let mut inner_shape: Option<Vec<i64>> = None;

            for e in elements {
                let sub = match e {
                    CompiledExpr::Vector(sub_elems) => try_flatten_to_constant(sub_elems)?,
                    _ => return None, // mixed Vector / non-Vector
                };
                let (sub_data, sub_shape) = sub;
                match &inner_shape {
                    None => inner_shape = Some(sub_shape),
                    Some(expected) if *expected != sub_shape => return None, // ragged
                    _ => {}
                }
                all_data.extend(sub_data);
            }

            let mut shape = vec![elements.len() as i64];
            shape.extend(inner_shape.unwrap_or_default());
            Some((all_data, shape))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_constant() {
        let mut codegen = CodeGenerator::new();
        let expr = CompiledExpr::Integer(42);
        let result = codegen.generate(&expr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_binop() {
        let mut codegen = CodeGenerator::new();
        let expr = CompiledExpr::FunctionCall {
            name: "+".to_string(),
            args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(2)],
        };
        let result = codegen.generate(&expr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_emit_function() {
        let codegen = CodeGenerator::new();
        let expr = CompiledExpr::FunctionCall {
            name: "+".to_string(),
            args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(2)],
        };
        let mlir = codegen.emit_function("test", &expr);
        assert!(mlir.is_ok());
        let mlir_str = mlir.unwrap();
        assert!(mlir_str.contains("stablehlo.add"));
        assert!(mlir_str.contains("@test"));
    }

    #[test]
    fn test_generate_compare() {
        let mut codegen = CodeGenerator::new();
        let expr = CompiledExpr::FunctionCall {
            name: ">".to_string(),
            args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(2.0)],
        };
        let result = codegen.generate(&expr);
        assert!(result.is_ok());
        let (_, ty) = result.unwrap();
        // Result should be i1 type (boolean results)
        assert!(matches!(ty, StableHLOType::ScalarI1));
    }

    #[test]
    fn test_emit_compare() {
        let codegen = CodeGenerator::new();
        let expr = CompiledExpr::FunctionCall {
            name: "=".to_string(),
            args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(1)],
        };
        let mlir = codegen.emit_function("test_eq", &expr);
        assert!(mlir.is_ok());
        let mlir_str = mlir.unwrap();
        assert!(mlir_str.contains("stablehlo.compare"));
        assert!(mlir_str.contains("comparison_direction = #stablehlo<comparison_direction EQ>"));
    }

    #[test]
    fn test_emit_boolean_and() {
        let codegen = CodeGenerator::new();
        let expr = CompiledExpr::FunctionCall {
            name: "and".to_string(),
            args: vec![
                CompiledExpr::FunctionCall {
                    name: ">".to_string(),
                    args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(2.0)],
                },
                CompiledExpr::FunctionCall {
                    name: "<".to_string(),
                    args: vec![CompiledExpr::Float(1.0), CompiledExpr::Float(3.0)],
                },
            ],
        };
        let mlir = codegen.emit_function("test_and", &expr);
        assert!(mlir.is_ok());
        let mlir_str = mlir.unwrap();
        assert!(mlir_str.contains("stablehlo.compare"));
        assert!(mlir_str.contains("stablehlo.and"));
    }

    #[test]
    fn test_emit_boolean_not() {
        let codegen = CodeGenerator::new();
        let expr = CompiledExpr::FunctionCall {
            name: "not".to_string(),
            args: vec![CompiledExpr::FunctionCall {
                name: ">".to_string(),
                args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(10.0)],
            }],
        };
        let mlir = codegen.emit_function("test_not", &expr);
        assert!(mlir.is_ok());
        let mlir_str = mlir.unwrap();
        assert!(mlir_str.contains("stablehlo.compare"));
        assert!(mlir_str.contains("stablehlo.not"));
    }
}
