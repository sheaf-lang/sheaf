// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Code generation - translate CompiledExpr to StableHLO MLIR

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
    /// Function registry for user-defined functions
    function_registry: HashMap<String, crate::core::compiler::FunctionDef>,
}

impl CodeGenerator {
    pub fn new() -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
            function_registry: HashMap::new(),
        }
    }

    pub fn with_registry(registry: HashMap<String, crate::core::compiler::FunctionDef>) -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
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
                // Check if this is a nested vector representing a 2D tensor
                if let Some(CompiledExpr::Vector(_)) = elements.first() {
                    // This is a 2D tensor like [[1.0, 2.0], [3.0, 4.0]]
                    let mut rows: Vec<Vec<f64>> = Vec::new();

                    for elem in elements {
                        if let CompiledExpr::Vector(row_elems) = elem {
                            let mut row: Vec<f64> = Vec::new();
                            for val in row_elems {
                                match val {
                                    CompiledExpr::Float(x) => row.push(*x),
                                    CompiledExpr::Integer(n) => row.push(*n as f64),
                                    _ => {
                                        return Err(SheafError::Compile {
                                            message: "Tensor elements must be numbers".to_string(),
                                            location: crate::core::error::SourceLocation::unknown(),
                                        });
                                    }
                                }
                            }
                            rows.push(row);
                        } else {
                            return Err(SheafError::Compile {
                                message: "Invalid tensor structure".to_string(),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }
                    }

                    let (reg, ty) = self.emitter.emit_tensor_constant(&rows);
                    Ok((reg, ty))
                } else {
                    // 1D vector - treat as a row vector (1xN tensor)
                    let mut values: Vec<f64> = Vec::new();
                    for elem in elements {
                        match elem {
                            CompiledExpr::Float(x) => values.push(*x),
                            CompiledExpr::Integer(n) => values.push(*n as f64),
                            _ => {
                                return Err(SheafError::Compile {
                                    message:
                                        "Vector elements must be numbers for tensor conversion"
                                            .to_string(),
                                    location: crate::core::error::SourceLocation::unknown(),
                                });
                            }
                        }
                    }
                    let (reg, ty) = self.emitter.emit_tensor_constant(&vec![values]);
                    Ok((reg, ty))
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

            CompiledExpr::FunctionCall { name, args } => self.generate_function_call(name, args),

            CompiledExpr::Let { bindings, body } => {
                // Evaluate bindings and store in bindings map
                for (name, value_expr) in bindings {
                    let (reg, ty) = self.generate(value_expr)?;
                    self.bindings.insert(name.clone(), (reg, ty));
                }
                // Evaluate body with bindings in scope
                let result = self.generate(body)?;
                // Clean up bindings (for proper scoping)
                for (name, _) in bindings {
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

        if let Some(signature) = signature {
            // Generate code for each argument
            let mut arg_registers = Vec::new();
            let mut arg_types = Vec::new();

            for arg in args {
                let (reg, ty) = self.generate(arg)?;
                arg_registers.push(reg);
                arg_types.push(ty);
            }

            // Emit func.call with the signature from the registry
            let result_reg = self.emitter.emit_func_call(
                name,
                &arg_registers,
                &arg_types,
                &signature.return_type,
            );

            return Ok((result_reg, signature.return_type.clone()));
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
        // transpose: (transpose tensor [1 0])
        else if name == "transpose" && args.len() == 2 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            if let CompiledExpr::Vector(perm_elems) = &args[1] {
                let permutation: Vec<i64> = perm_elems
                    .iter()
                    .map(|e| match e {
                        CompiledExpr::Integer(n) => *n,
                        _ => panic!("Permutation element must be integer"),
                    })
                    .collect();
                let (reg, ty) = tensor_ops::emit_transpose(
                    &mut self.emitter,
                    &operand_reg,
                    &operand_ty,
                    &permutation,
                );
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "transpose expects a vector permutation argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        } else {
            Err(SheafError::Compile {
                message: format!("Function call not yet supported: {}", name),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
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
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
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
        // Result should be i64 type (we use i64 for boolean results)
        assert!(matches!(ty, StableHLOType::ScalarI64));
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
        assert!(mlir_str.contains("comparison_direction = EQ"));
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
