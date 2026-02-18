// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Type inference for function signatures

use crate::compiler::stablehlo::StableHLOType;
use crate::core::compiler::{CompiledExpr, CompilerContext};
use crate::core::error::{SheafError, SheafResult, SourceLocation};

/// Function signature (parameter types + return type)
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    pub param_types: Vec<StableHLOType>,
    pub return_type: StableHLOType,
}

/// Infer the signature of a function from its parameters and body
///
/// Strategy:
/// 1. Build a symbol table by traversing the body
/// 2. Infer types for each parameter based on how they're used
/// 3. Infer return type from body
///
/// `known_param_types`: pre-known types for specific params (e.g. from defparams),
/// overrides inference. Maps param name -> StableHLOType.
pub fn infer_function_signature(
    compiler: &CompilerContext,
    params: &[String],
    body_expr: &CompiledExpr,
) -> SheafResult<FunctionSignature> {
    infer_function_signature_with_known(compiler, params, body_expr, &[])
}

/// Like `infer_function_signature` but accepts pre-known param types.
pub fn infer_function_signature_with_known(
    _compiler: &CompilerContext,
    params: &[String],
    body_expr: &CompiledExpr,
    known: &[(String, StableHLOType)],
) -> SheafResult<FunctionSignature> {
    // Build symbol table with inferred types
    let mut symbol_types = std::collections::HashMap::new();

    // Seed with known param types so return type inference can use them
    for (name, ty) in known {
        symbol_types.insert(name.clone(), ty.clone());
    }

    // Also seed GetTupleElement leaf types into symbol_types so
    // infer_type_with_context can resolve field references like W, b
    seed_tuple_element_types(body_expr, &mut symbol_types, known);

    infer_symbol_types(body_expr, &mut symbol_types)?;

    // Infer parameter types from symbol table
    let param_types: Vec<StableHLOType> = params
        .iter()
        .map(|p| {
            symbol_types
                .get(p)
                .cloned()
                .unwrap_or(StableHLOType::scalar_f32())
        })
        .collect();

    // Infer return type from body
    let return_type = infer_type_with_context(body_expr, &symbol_types)?;

    Ok(FunctionSignature {
        param_types,
        return_type,
    })
}

/// Seed symbol_types with the resolved element types of GetTupleElement nodes.
/// This lets return-type inference work for expressions involving typed params.
fn seed_tuple_element_types(
    expr: &CompiledExpr,
    symbol_types: &mut std::collections::HashMap<String, StableHLOType>,
    known: &[(String, StableHLOType)],
) {
    match expr {
        CompiledExpr::GetTupleElement { param, indices } => {
            // Resolve the type of this element from the known param type
            if let Some((_, param_ty)) = known.iter().find(|(n, _)| n == param) {
                if let Some(element_ty) = resolve_tuple_index(param_ty, indices) {
                    // We can't directly name this node, but its type is used in FunctionCall args
                    // Store it in a synthetic key for context
                    let key = format!(
                        "__get_tuple_{}__{}",
                        param,
                        indices
                            .iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join("_")
                    );
                    symbol_types.insert(key, element_ty);
                }
            }
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for arg in args {
                seed_tuple_element_types(arg, symbol_types, known);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                seed_tuple_element_types(v, symbol_types, known);
            }
            seed_tuple_element_types(body, symbol_types, known);
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                seed_tuple_element_types(e, symbol_types, known);
            }
        }
        _ => {}
    }
}

/// Walk a StableHLO tuple type following a sequence of indices.
fn resolve_tuple_index(ty: &StableHLOType, indices: &[usize]) -> Option<StableHLOType> {
    let mut current = ty.clone();
    for &idx in indices {
        match current {
            StableHLOType::Tuple(elems) => {
                current = elems.into_iter().nth(idx)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Infer types for symbols by analyzing how they're used in the expression
fn infer_symbol_types(
    expr: &CompiledExpr,
    symbol_types: &mut std::collections::HashMap<String, StableHLOType>,
) -> SheafResult<()> {
    match expr {
        CompiledExpr::FunctionCall { name, args } => {
            // Infer types from function call context
            if name == "@" && args.len() == 2 {
                // Matrix multiply: (@ lhs rhs)
                // If rhs type is known (e.g. from tuple), use it to constrain lhs
                let rhs_ty = infer_type_with_context(&args[1], symbol_types).ok();
                let lhs_ty = infer_type_with_context(&args[0], symbol_types).ok();

                if let CompiledExpr::Symbol(sym) = &args[0] {
                    let ty = if let Some(rhs) = &rhs_ty {
                        let rhs_shape = rhs.shape();
                        if rhs_shape.len() == 2 {
                            // lhs must have contracting dim = rhs_shape[0], use batch=1
                            StableHLOType::f32_tensor(vec![1, rhs_shape[0]])
                        } else {
                            StableHLOType::f32_tensor(vec![1, 1])
                        }
                    } else {
                        StableHLOType::f32_tensor(vec![1, 1])
                    };
                    symbol_types.entry(sym.clone()).or_insert(ty);
                }
                if let CompiledExpr::Symbol(sym) = &args[1] {
                    let ty = if let Some(lhs) = &lhs_ty {
                        let lhs_shape = lhs.shape();
                        if lhs_shape.len() == 2 {
                            StableHLOType::f32_tensor(vec![lhs_shape[1], 1])
                        } else {
                            StableHLOType::f32_tensor(vec![1, 1])
                        }
                    } else {
                        StableHLOType::f32_tensor(vec![1, 1])
                    };
                    symbol_types.entry(sym.clone()).or_insert(ty);
                }
            }

            // sum/mean: first arg should be at least 2D
            if (name == "sum" || name == "mean") && !args.is_empty() {
                if let CompiledExpr::Symbol(sym) = &args[0] {
                    symbol_types
                        .entry(sym.clone())
                        .or_insert(StableHLOType::f32_tensor(vec![1, 1]));
                }
            }

            // Recurse into arguments
            for arg in args {
                infer_symbol_types(arg, symbol_types)?;
            }
        }

        CompiledExpr::Let { bindings, body } => {
            for (_, value) in bindings {
                infer_symbol_types(value, symbol_types)?;
            }
            infer_symbol_types(body, symbol_types)?;
        }

        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            infer_symbol_types(condition, symbol_types)?;
            infer_symbol_types(then_branch, symbol_types)?;
            if let Some(else_expr) = else_branch {
                infer_symbol_types(else_expr, symbol_types)?;
            }
        }

        CompiledExpr::Do(exprs) => {
            for e in exprs {
                infer_symbol_types(e, symbol_types)?;
            }
        }

        CompiledExpr::Vector(elems) => {
            for e in elems {
                infer_symbol_types(e, symbol_types)?;
            }
        }

        CompiledExpr::Symbol(name) => {
            // Default to scalar if not seen before
            symbol_types
                .entry(name.clone())
                .or_insert(StableHLOType::scalar_f32());
        }

        _ => {}
    }

    Ok(())
}

/// Infer type with a symbol context
fn infer_type_with_context(
    expr: &CompiledExpr,
    symbol_types: &std::collections::HashMap<String, StableHLOType>,
) -> SheafResult<StableHLOType> {
    match expr {
        CompiledExpr::Symbol(name) => Ok(symbol_types
            .get(name)
            .cloned()
            .unwrap_or(StableHLOType::scalar_f32())),

        CompiledExpr::GetTupleElement { param, indices } => {
            // Resolve element type from the param's tuple type in symbol_types
            if let Some(param_ty) = symbol_types.get(param) {
                if let Some(element_ty) = resolve_tuple_index(param_ty, indices) {
                    return Ok(element_ty);
                }
            }
            Ok(StableHLOType::scalar_f32())
        }

        CompiledExpr::FunctionCall { name, args } => {
            // Use context-aware inference for args
            let ctx_infer = |e: &CompiledExpr| infer_type_with_context(e, symbol_types);
            match name.as_str() {
                "+" | "-" | "*" | "/" => {
                    if args.is_empty() {
                        return Ok(StableHLOType::scalar_f32());
                    }
                    ctx_infer(&args[0])
                }
                "@" => {
                    if args.len() != 2 {
                        return Ok(StableHLOType::scalar_f32());
                    }
                    let lhs_ty = ctx_infer(&args[0])?;
                    let rhs_ty = ctx_infer(&args[1])?;
                    let lhs_shape = lhs_ty.shape();
                    let rhs_shape = rhs_ty.shape();
                    if lhs_shape.len() == 2 && rhs_shape.len() == 2 {
                        Ok(StableHLOType::f32_tensor(vec![lhs_shape[0], rhs_shape[1]]))
                    } else {
                        Ok(lhs_ty)
                    }
                }
                // Unary ops (including user-defined like softmax): preserve arg type
                "relu" | "sigmoid" | "tanh" | "sqrt" | "exp" | "log" | "softmax" => {
                    if args.is_empty() {
                        return Ok(StableHLOType::scalar_f32());
                    }
                    ctx_infer(&args[0])
                }
                _ => infer_function_call_type(name, args),
            }
        }

        CompiledExpr::Let { bindings, body } => {
            // Extend symbol_types with let bindings, then infer body type
            let mut extended = symbol_types.clone();
            for (name, val) in bindings {
                let ty = infer_type_with_context(val, &extended)?;
                extended.insert(name.clone(), ty);
            }
            infer_type_with_context(body, &extended)
        }

        CompiledExpr::Do(exprs) => {
            if let Some(last) = exprs.last() {
                infer_type_with_context(last, symbol_types)
            } else {
                Ok(StableHLOType::scalar_f32())
            }
        }

        _ => infer_type(expr),
    }
}

/// Infer the type of a compiled expression
fn infer_type(expr: &CompiledExpr) -> SheafResult<StableHLOType> {
    match expr {
        CompiledExpr::Integer(_) => Ok(StableHLOType::scalar_f32()),
        CompiledExpr::Float(_) => Ok(StableHLOType::scalar_f32()),
        CompiledExpr::Boolean(_) => Ok(StableHLOType::scalar_f32()), // For now

        CompiledExpr::Vector(elements) => {
            if elements.is_empty() {
                return Ok(StableHLOType::scalar_f32());
            }

            // Check if it's a nested vector (matrix)
            if let CompiledExpr::Vector(row) = &elements[0] {
                // 2D tensor
                let rows = elements.len() as i64;
                let cols = row.len() as i64;
                Ok(StableHLOType::f32_tensor(vec![rows, cols]))
            } else {
                // 1D vector - treat as row vector [1xN]
                let len = elements.len() as i64;
                Ok(StableHLOType::f32_tensor(vec![1, len]))
            }
        }

        CompiledExpr::FunctionCall { name, args } => infer_function_call_type(name, args),

        CompiledExpr::Let { body, .. } => infer_type(body),

        CompiledExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            // Infer from then branch (assume both branches have same type)
            let then_ty = infer_type(then_branch)?;
            if let Some(else_expr) = else_branch {
                let else_ty = infer_type(else_expr)?;
                // For Phase 1, just return then_ty
                // Phase 2: check compatibility
                let _ = else_ty;
            }
            Ok(then_ty)
        }

        CompiledExpr::Do(exprs) => {
            if let Some(last) = exprs.last() {
                infer_type(last)
            } else {
                Ok(StableHLOType::scalar_f32())
            }
        }

        CompiledExpr::Symbol(_) => {
            // Symbol should have been resolved, default to scalar
            Ok(StableHLOType::scalar_f32())
        }

        _ => Ok(StableHLOType::scalar_f32()), // Default fallback
    }
}

/// Infer type of a function call based on operation
fn infer_function_call_type(name: &str, args: &[CompiledExpr]) -> SheafResult<StableHLOType> {
    match name {
        // Binary arithmetic ops preserve input type
        "+" | "-" | "*" | "/" => {
            if args.is_empty() {
                return Ok(StableHLOType::scalar_f32());
            }
            infer_type(&args[0])
        }

        // Matrix multiply: [M,K] @ [K,N] -> [M,N]
        "@" => {
            if args.len() != 2 {
                return Ok(StableHLOType::scalar_f32());
            }

            let lhs_ty = infer_type(&args[0])?;
            let rhs_ty = infer_type(&args[1])?;

            let lhs_shape = lhs_ty.shape();
            let rhs_shape = rhs_ty.shape();

            if lhs_shape.len() == 2 && rhs_shape.len() == 2 {
                // [M, K] @ [K, N] -> [M, N]
                Ok(StableHLOType::f32_tensor(vec![lhs_shape[0], rhs_shape[1]]))
            } else {
                // Fallback
                Ok(lhs_ty)
            }
        }

        // Unary ops preserve input type
        "relu" | "sigmoid" | "tanh" | "sqrt" | "exp" | "log" | "softmax" => {
            if args.is_empty() {
                return Ok(StableHLOType::scalar_f32());
            }
            infer_type(&args[0])
        }

        // zeros: (zeros [M N]) -> tensor<MxNxf32>
        "zeros" => {
            if let Some(CompiledExpr::Vector(shape_elems)) = args.first() {
                let shape: Vec<i64> = shape_elems
                    .iter()
                    .filter_map(|e| match e {
                        CompiledExpr::Integer(n) => Some(*n),
                        _ => None,
                    })
                    .collect();
                Ok(StableHLOType::f32_tensor(shape))
            } else {
                Ok(StableHLOType::scalar_f32())
            }
        }

        // random-normal: (random-normal key [M N]) -> tensor<MxNxf32>
        "random-normal" => {
            if args.len() < 2 {
                return Ok(StableHLOType::scalar_f32());
            }
            if let CompiledExpr::Vector(shape_elems) = &args[1] {
                let shape: Vec<i64> = shape_elems
                    .iter()
                    .filter_map(|e| match e {
                        CompiledExpr::Integer(n) => Some(*n),
                        _ => None,
                    })
                    .collect();
                Ok(StableHLOType::f32_tensor(shape))
            } else {
                Ok(StableHLOType::scalar_f32())
            }
        }

        // sum: (sum x :axis N :keepdims bool) -> removes or keeps axis N
        "sum" | "mean" => {
            if args.is_empty() {
                return Ok(StableHLOType::scalar_f32());
            }
            let input_ty = infer_type(&args[0])?;
            let shape = input_ty.shape();
            if shape.is_empty() {
                return Ok(StableHLOType::scalar_f32());
            }
            // Parse :axis and :keepdims from args
            let mut axis: i64 = -1;
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
            let ndim = shape.len();
            let axis_usize = if axis < 0 {
                (ndim as i64 + axis) as usize
            } else {
                axis as usize
            };
            let axis_usize = axis_usize.min(ndim.saturating_sub(1));
            if keepdims {
                let mut out_shape = shape.clone();
                out_shape[axis_usize] = 1;
                Ok(StableHLOType::f32_tensor(out_shape))
            } else {
                let out_shape: Vec<i64> = shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != axis_usize)
                    .map(|(_, &d)| d)
                    .collect();
                if out_shape.is_empty() {
                    Ok(StableHLOType::scalar_f32())
                } else {
                    Ok(StableHLOType::f32_tensor(out_shape))
                }
            }
        }

        // Unknown function: assume scalar
        _ => Ok(StableHLOType::scalar_f32()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SourceLocation;

    fn make_compiled_int(n: i64) -> CompiledExpr {
        CompiledExpr::Integer(n)
    }

    fn make_compiled_float(x: f64) -> CompiledExpr {
        CompiledExpr::Float(x)
    }

    fn make_compiled_vector(elems: Vec<CompiledExpr>) -> CompiledExpr {
        CompiledExpr::Vector(elems)
    }

    fn make_compiled_call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: name.to_string(),
            args,
        }
    }

    #[test]
    fn test_infer_scalar() {
        let expr = make_compiled_float(42.0);
        let ty = infer_type(&expr).unwrap();
        assert_eq!(ty, StableHLOType::scalar_f32());
    }

    #[test]
    fn test_infer_add() {
        // (+ 1.0 2.0) -> tensor<f32>
        let expr = make_compiled_call(
            "+",
            vec![make_compiled_float(1.0), make_compiled_float(2.0)],
        );
        let ty = infer_type(&expr).unwrap();
        assert_eq!(ty, StableHLOType::scalar_f32());
    }

    #[test]
    fn test_infer_matrix() {
        // [[1.0 2.0] [3.0 4.0]] -> tensor<2x2xf32>
        let expr = make_compiled_vector(vec![
            make_compiled_vector(vec![make_compiled_float(1.0), make_compiled_float(2.0)]),
            make_compiled_vector(vec![make_compiled_float(3.0), make_compiled_float(4.0)]),
        ]);
        let ty = infer_type(&expr).unwrap();
        assert_eq!(ty, StableHLOType::f32_tensor(vec![2, 2]));
    }

    #[test]
    fn test_infer_matmul() {
        // [[1.0 2.0]] @ [[3.0] [4.0]] -> tensor<1x1xf32>
        let lhs = make_compiled_vector(vec![make_compiled_vector(vec![
            make_compiled_float(1.0),
            make_compiled_float(2.0),
        ])]);
        let rhs = make_compiled_vector(vec![
            make_compiled_vector(vec![make_compiled_float(3.0)]),
            make_compiled_vector(vec![make_compiled_float(4.0)]),
        ]);
        let expr = make_compiled_call("@", vec![lhs, rhs]);

        let ty = infer_type(&expr).unwrap();
        assert_eq!(ty, StableHLOType::f32_tensor(vec![1, 1]));
    }

    #[test]
    fn test_infer_signature() {
        let compiler = CompilerContext::new();
        let params = vec!["x".to_string(), "y".to_string()];

        // Body: (+ x y) - returns scalar
        let body = make_compiled_call(
            "+",
            vec![
                CompiledExpr::Symbol("x".to_string()),
                CompiledExpr::Symbol("y".to_string()),
            ],
        );

        let sig = infer_function_signature(&compiler, &params, &body).unwrap();

        assert_eq!(sig.param_types.len(), 2);
        assert_eq!(sig.param_types[0], StableHLOType::scalar_f32());
        assert_eq!(sig.param_types[1], StableHLOType::scalar_f32());
        assert_eq!(sig.return_type, StableHLOType::scalar_f32());
    }

    #[test]
    fn test_infer_matmul_signature() {
        let compiler = CompilerContext::new();
        let params = vec!["A".to_string(), "B".to_string()];

        // Body: (@ A B) where A is 2x3, B is 3x4 -> should return 2x4
        // For now, we'll simulate with literals
        let a_matrix = make_compiled_vector(vec![
            make_compiled_vector(vec![
                make_compiled_float(1.0),
                make_compiled_float(2.0),
                make_compiled_float(3.0),
            ]),
            make_compiled_vector(vec![
                make_compiled_float(4.0),
                make_compiled_float(5.0),
                make_compiled_float(6.0),
            ]),
        ]);
        let b_matrix = make_compiled_vector(vec![
            make_compiled_vector(vec![
                make_compiled_float(1.0),
                make_compiled_float(2.0),
                make_compiled_float(3.0),
                make_compiled_float(4.0),
            ]),
            make_compiled_vector(vec![
                make_compiled_float(5.0),
                make_compiled_float(6.0),
                make_compiled_float(7.0),
                make_compiled_float(8.0),
            ]),
            make_compiled_vector(vec![
                make_compiled_float(9.0),
                make_compiled_float(10.0),
                make_compiled_float(11.0),
                make_compiled_float(12.0),
            ]),
        ]);

        let body = make_compiled_call("@", vec![a_matrix, b_matrix]);
        let sig = infer_function_signature(&compiler, &params, &body).unwrap();

        // Return should be 2x4
        assert_eq!(sig.return_type, StableHLOType::f32_tensor(vec![2, 4]));
    }
}
