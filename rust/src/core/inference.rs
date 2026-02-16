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
pub fn infer_function_signature(
    compiler: &CompilerContext,
    params: &[String],
    body_expr: &CompiledExpr,
) -> SheafResult<FunctionSignature> {
    // Build symbol table with inferred types
    let mut symbol_types = std::collections::HashMap::new();
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

/// Infer types for symbols by analyzing how they're used in the expression
fn infer_symbol_types(
    expr: &CompiledExpr,
    symbol_types: &mut std::collections::HashMap<String, StableHLOType>,
) -> SheafResult<()> {
    match expr {
        CompiledExpr::FunctionCall { name, args } => {
            // Infer types from function call context
            if name == "@" && args.len() == 2 {
                // Matrix multiply: first arg should be 2D, second arg should be 2D
                if let CompiledExpr::Symbol(sym) = &args[0] {
                    // Can't infer exact shape without more context, default to 2D
                    symbol_types
                        .entry(sym.clone())
                        .or_insert(StableHLOType::f32_tensor(vec![1, 1]));
                }
                if let CompiledExpr::Symbol(sym) = &args[1] {
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
        "relu" | "sigmoid" | "tanh" | "sqrt" | "exp" | "log" => {
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
