// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

use crate::core::error::{SheafError, SheafResult};
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::inference::{expr_is_tensor, infer_type_with_context};
use crate::lowering::stablehlo::StableHLOType;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static DESTRUCTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn fresh_temp() -> String {
    let n = DESTRUCTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("__dst{}", n)
}

pub fn classify_vectors(
    expr: CompiledExpr,
    param_shapes: &HashMap<String, Vec<i64>>,
) -> CompiledExpr {
    let mut symbol_types = HashMap::new();
    for (name, shape) in param_shapes {
        symbol_types.insert(name.clone(), StableHLOType::f32_tensor(shape.clone()));
    }
    classify_vectors_rec(expr, &mut symbol_types)
}

fn classify_vectors_rec(
    expr: CompiledExpr,
    symbol_types: &mut HashMap<String, StableHLOType>,
) -> CompiledExpr {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            let mut new_bindings = Vec::with_capacity(bindings.len());
            for (pattern, value) in bindings {
                let value_classified = classify_vectors_rec(value, symbol_types);
                if let BindingPattern::Simple(name) = &pattern {
                    let ty = infer_type_with_context(&value_classified, &*symbol_types)
                        .unwrap_or_else(|_| StableHLOType::scalar_f32());
                    symbol_types.insert(name.clone(), ty);
                }
                new_bindings.push((pattern.clone(), value_classified));
            }
            let body_classified = classify_vectors_rec(*body, symbol_types);
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(body_classified),
            }
        }
        CompiledExpr::Lambda { params, body, .. } => {
            let mut saved = Vec::new();
            for param in &params {
                let old_ty = symbol_types.remove(param);
                let active = old_ty
                    .clone()
                    .unwrap_or_else(StableHLOType::scalar_f32);
                symbol_types.insert(param.clone(), active);
                saved.push((param.clone(), old_ty));
            }
            let body_classified = classify_vectors_rec(std::rc::Rc::unwrap_or_clone(body), symbol_types);
            for (param, old_ty) in saved {
                match old_ty {
                    Some(old) => {
                        symbol_types.insert(param, old);
                    }
                    None => {
                        symbol_types.remove(&param);
                    }
                }
            }
            CompiledExpr::lambda(params, body_classified)
        }
        CompiledExpr::Vector(elems) => {
            let mut elems_classified = Vec::with_capacity(elems.len());
            for e in elems {
                elems_classified.push(classify_vectors_rec(e, symbol_types));
            }
            if crate::lowering::codegen::try_flatten_to_constant(&elems_classified).is_some() {
                return CompiledExpr::Vector(elems_classified);
            }
            let mut all_scalar = true;
            for e in &elems_classified {
                if expr_is_tensor(e, symbol_types) {
                    all_scalar = false;
                    break;
                }
            }
            if all_scalar {
                return CompiledExpr::Vector(elems_classified);
            }
            CompiledExpr::Tuple(elems_classified)
        }
        CompiledExpr::Tuple(elems) => {
            let mut elems_classified = Vec::with_capacity(elems.len());
            for e in elems {
                elems_classified.push(classify_vectors_rec(e, symbol_types));
            }
            CompiledExpr::Tuple(elems_classified)
        }
        CompiledExpr::FunctionCall { name, args, loc } => {
            let mut args_classified = Vec::with_capacity(args.len());
            for a in args {
                args_classified.push(classify_vectors_rec(a, symbol_types));
            }
            CompiledExpr::FunctionCall {
                name,
                args: args_classified,
                loc,
            }
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition_classified = classify_vectors_rec(*condition, symbol_types);
            let then_classified = classify_vectors_rec(*then_branch, symbol_types);
            let else_classified =
                else_branch.map(|e| Box::new(classify_vectors_rec(*e, symbol_types)));
            CompiledExpr::If {
                condition: Box::new(condition_classified),
                then_branch: Box::new(then_classified),
                else_branch: else_classified,
            }
        }
        CompiledExpr::Do(exprs) => {
            let mut exprs_classified = Vec::with_capacity(exprs.len());
            for e in exprs {
                exprs_classified.push(classify_vectors_rec(e, symbol_types));
            }
            CompiledExpr::Do(exprs_classified)
        }
        CompiledExpr::LambdaCall { callee, args } => {
            let callee_classified = classify_vectors_rec(*callee, symbol_types);
            let mut args_classified = Vec::with_capacity(args.len());
            for a in args {
                args_classified.push(classify_vectors_rec(a, symbol_types));
            }
            CompiledExpr::LambdaCall {
                callee: Box::new(callee_classified),
                args: args_classified,
            }
        }
        CompiledExpr::Repeat {
            index_var,
            count,
            acc_var,
            acc_init,
            body,
        } => {
            let count_classified = classify_vectors_rec(*count, symbol_types);
            let acc_init_classified = classify_vectors_rec(*acc_init, symbol_types);
            let body_classified = classify_vectors_rec(*body, symbol_types);
            CompiledExpr::Repeat {
                index_var,
                count: Box::new(count_classified),
                acc_var,
                acc_init: Box::new(acc_init_classified),
                body: Box::new(body_classified),
            }
        }
        CompiledExpr::Integer(_)
        | CompiledExpr::Float(_)
        | CompiledExpr::Boolean(_)
        | CompiledExpr::Nil
        | CompiledExpr::String(_)
        | CompiledExpr::Keyword(_)
        | CompiledExpr::Dict(_)
        | CompiledExpr::Quoted(_)
        | CompiledExpr::FunctionRef(_)
        | CompiledExpr::GlobalRef(_)
        | CompiledExpr::Symbol(_)
        | CompiledExpr::GetTupleElement { .. }
        | CompiledExpr::ValueAndGrad { .. }
        | CompiledExpr::Def { .. }
        | CompiledExpr::Guard { .. }
        | CompiledExpr::While { .. } => expr,
    }
}

/// Whether `expr` is an inline value-and-grad call, which always returns
/// the two-element tuple `[value gradient]`.
fn is_inline_value_and_grad(expr: &CompiledExpr) -> bool {
    matches!(
        expr,
        CompiledExpr::LambdaCall { callee, .. }
            if matches!(
                callee.as_ref(),
                CompiledExpr::FunctionCall { name, args, .. }
                    if name == "__value-and-grad-hof__" && args.len() == 1
            )
    )
}

/// Classify a destructuring source.
enum DestructKind {
    Tuple,
    Tensor1D,
    Vector1D, // literal Vector of scalars
    Unknown,
}

fn kind_of(expr: &CompiledExpr, symbol_types: &HashMap<String, StableHLOType>) -> DestructKind {
    match expr {
        CompiledExpr::Tuple(_) => DestructKind::Tuple,
        expr if is_inline_value_and_grad(expr) => DestructKind::Tuple,
        CompiledExpr::Vector(_) => DestructKind::Vector1D,
        CompiledExpr::Let { body, .. } => kind_of(body, symbol_types),
        CompiledExpr::Do(exprs) => exprs
            .last()
            .map_or(DestructKind::Unknown, |expr| kind_of(expr, symbol_types)),
        CompiledExpr::Symbol(s) => match symbol_types.get(s) {
            Some(StableHLOType::Tuple(_, _)) => DestructKind::Tuple,
            Some(ty) => {
                let sh = ty.shape();
                if sh.len() == 1 && !sh.is_empty() {
                    DestructKind::Tensor1D
                } else {
                    DestructKind::Unknown
                }
            }
            None => DestructKind::Unknown,
        },
        _ => DestructKind::Unknown,
    }
}

/// Return the statically known length of a destructuring source.
fn static_length_with_types(
    expr: &CompiledExpr,
    symbol_types: &HashMap<String, StableHLOType>,
) -> Option<usize> {
    match expr {
        CompiledExpr::Vector(elems) => Some(elems.len()),
        CompiledExpr::Tuple(elems) => Some(elems.len()),
        expr if is_inline_value_and_grad(expr) => Some(2),
        CompiledExpr::Let { body, .. } => static_length_with_types(body, symbol_types),
        CompiledExpr::Do(exprs) => exprs
            .last()
            .and_then(|expr| static_length_with_types(expr, symbol_types)),
        CompiledExpr::Symbol(s) => symbol_types.get(s).and_then(|ty| match ty {
            StableHLOType::Tuple(tys, _) => Some(tys.len()),
            other => {
                let sh = other.shape();
                if sh.len() == 1 && !sh.is_empty() {
                    Some(sh[0] as usize)
                } else {
                    None
                }
            }
        }),
        _ => None,
    }
}

/// Desugar destructuring `Let` bindings.
pub fn lower_tuples_and_destructuring(
    body: CompiledExpr,
    param_shapes: &HashMap<String, Vec<i64>>,
) -> SheafResult<CompiledExpr> {
    let mut symbol_types: HashMap<String, StableHLOType> = HashMap::new();
    for (name, shape) in param_shapes {
        symbol_types.insert(name.clone(), StableHLOType::f32_tensor(shape.clone()));
    }
    let classified = classify_vectors_rec(body, &mut symbol_types);
    desugar_destructuring_lets(classified, &symbol_types)
}

pub fn desugar_destructuring_lets(
    expr: CompiledExpr,
    symbol_types: &HashMap<String, StableHLOType>,
) -> SheafResult<CompiledExpr> {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            let mut new_bindings = Vec::new();
            let body = desugar_destructuring_lets(*body, symbol_types)?;

            for (pattern, value) in bindings {
                match pattern {
                    BindingPattern::Simple(_) => {
                        new_bindings
                            .push((pattern, desugar_destructuring_lets(value, symbol_types)?));
                    }
                    BindingPattern::Destructure(names) => {
                        for name in &names {
                            if let BindingPattern::Destructure(_) = name {
                                return Err(SheafError::Compile {
                                    message: "Nested destructuring patterns are not yet supported"
                                        .to_string(),
                                    location: crate::core::error::SourceLocation::unknown(),
                                });
                            }
                        }

                        let tmp = fresh_temp();
                        let value = desugar_destructuring_lets(value, symbol_types)?;

                        let kind = kind_of(&value, symbol_types);
                        let len =
                            static_length_with_types(&value, symbol_types).ok_or_else(|| {
                                SheafError::Compile {
                                    message:
                                        "Destructuring source has a dynamic length that cannot be \
                                          resolved statically: it is neither a literal nor a typed \
                                          parameter of known shape"
                                            .to_string(),
                                    location: crate::core::error::SourceLocation::unknown(),
                                }
                            })?;

                        if len != names.len() {
                            return Err(SheafError::Compile {
                                message: format!(
                                    "Destructuring arity mismatch: pattern has {} names, source has {} elements",
                                    names.len(),
                                    len
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }

                        new_bindings.push((BindingPattern::Simple(tmp.clone()), value));
                        for (i, name) in names.into_iter().enumerate() {
                            let extraction = match kind {
                                DestructKind::Tuple => CompiledExpr::GetTupleElement {
                                    param: tmp.clone(),
                                    indices: vec![i],
                                },
                                DestructKind::Vector1D | DestructKind::Tensor1D => {
                                    CompiledExpr::FunctionCall {
                                        name: "get".to_string(),
                                        args: vec![
                                            CompiledExpr::Symbol(tmp.clone()),
                                            CompiledExpr::Integer(i as i64),
                                        ],
                                        loc: None,
                                    }
                                }
                                DestructKind::Unknown => {
                                    return Err(SheafError::Compile {
                                        message: "Destructuring the result of a runtime tensor of unknown shape is not supported".to_string(),
                                        location: crate::core::error::SourceLocation::unknown(),
                                    });
                                }
                            };
                            new_bindings.push((name, extraction));
                        }
                    }
                }
            }

            Ok(CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(body),
            })
        }
        CompiledExpr::FunctionCall { name, args, loc } => {
            let mut new_args = Vec::new();
            for arg in args {
                new_args.push(desugar_destructuring_lets(arg, symbol_types)?);
            }
            Ok(CompiledExpr::FunctionCall {
                name,
                args: new_args,
                loc,
            })
        }
        CompiledExpr::Lambda { params, body, .. } => Ok(CompiledExpr::lambda(
            params, desugar_destructuring_lets(std::rc::Rc::unwrap_or_clone(body), symbol_types)?,
        )),
        CompiledExpr::LambdaCall { callee, args } => {
            let new_args: Vec<CompiledExpr> = args
                .into_iter()
                .map(|a| desugar_destructuring_lets(a, symbol_types))
                .collect::<SheafResult<_>>()?;
            Ok(CompiledExpr::LambdaCall {
                callee: Box::new(desugar_destructuring_lets(*callee, symbol_types)?),
                args: new_args,
            })
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => Ok(CompiledExpr::If {
            condition: Box::new(desugar_destructuring_lets(*condition, symbol_types)?),
            then_branch: Box::new(desugar_destructuring_lets(*then_branch, symbol_types)?),
            else_branch: match else_branch {
                Some(e) => Some(Box::new(desugar_destructuring_lets(*e, symbol_types)?)),
                None => None,
            },
        }),
        CompiledExpr::Do(exprs) => Ok(CompiledExpr::Do(
            exprs
                .into_iter()
                .map(|e| desugar_destructuring_lets(e, symbol_types))
                .collect::<SheafResult<_>>()?,
        )),
        CompiledExpr::Repeat {
            index_var,
            count,
            acc_var,
            acc_init,
            body,
        } => Ok(CompiledExpr::Repeat {
            index_var,
            count: Box::new(desugar_destructuring_lets(*count, symbol_types)?),
            acc_var,
            acc_init: Box::new(desugar_destructuring_lets(*acc_init, symbol_types)?),
            body: Box::new(desugar_destructuring_lets(*body, symbol_types)?),
        }),
        CompiledExpr::Vector(elems) => Ok(CompiledExpr::Vector(
            elems
                .into_iter()
                .map(|e| desugar_destructuring_lets(e, symbol_types))
                .collect::<SheafResult<_>>()?,
        )),
        CompiledExpr::Tuple(elems) => Ok(CompiledExpr::Tuple(
            elems
                .into_iter()
                .map(|e| desugar_destructuring_lets(e, symbol_types))
                .collect::<SheafResult<_>>()?,
        )),
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use super::desugar_destructuring_lets;
    use crate::core::expr::{BindingPattern, CompiledExpr};
    use std::collections::HashMap;

    #[test]
    fn destructures_a_tuple_returned_through_a_let() {
        let source = CompiledExpr::Let {
            bindings: vec![(
                BindingPattern::Simple("intermediate".to_string()),
                CompiledExpr::Float(1.0),
            )],
            body: Box::new(CompiledExpr::Tuple(vec![
                CompiledExpr::Symbol("intermediate".to_string()),
                CompiledExpr::Float(2.0),
            ])),
        };
        let expr = CompiledExpr::Let {
            bindings: vec![
                (
                    BindingPattern::Destructure(vec![
                        BindingPattern::Simple("left".to_string()),
                        BindingPattern::Simple("right".to_string()),
                    ]),
                    source,
                ),
                (
                    BindingPattern::Simple("total".to_string()),
                    CompiledExpr::FunctionCall {
                        name: "+".to_string(),
                        args: vec![
                            CompiledExpr::Symbol("left".to_string()),
                            CompiledExpr::Symbol("right".to_string()),
                        ],
                        loc: None,
                    },
                ),
            ],
            body: Box::new(CompiledExpr::Symbol("total".to_string())),
        };

        let CompiledExpr::Let { bindings, .. } =
            desugar_destructuring_lets(expr, &HashMap::new()).unwrap()
        else {
            panic!("destructuring should remain a let");
        };
        let names: Vec<_> = bindings
            .iter()
            .filter_map(|(pattern, _)| pattern.as_simple())
            .collect();
        assert_eq!(names[1..], ["left", "right", "total"]);
    }
}
