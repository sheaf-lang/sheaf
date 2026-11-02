// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Codegen helpers.

use crate::core::expr::CompiledExpr;
use crate::lowering::stablehlo::StableHLOType;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub(crate) struct TupleLeaf {
    pub(crate) indices: Vec<usize>,
    pub(crate) symbol: String,
}

pub(crate) fn collect_tuple_type_leaves(param_name: &str, ty: &StableHLOType) -> Vec<TupleLeaf> {
    fn collect(param_name: &str, ty: &StableHLOType, indices: &mut Vec<usize>, out: &mut Vec<TupleLeaf>) {
        match ty {
            StableHLOType::Tuple(elems, _) => {
                for (idx, elem) in elems.iter().enumerate() {
                    indices.push(idx);
                    collect(param_name, elem, indices, out);
                    indices.pop();
                }
            }
            _ => out.push(TupleLeaf {
                indices: indices.clone(),
                symbol: format!(
                    "{}_{}",
                    param_name,
                    indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("_")
                ),
            }),
        }
    }

    let mut leaves = Vec::new();
    collect(param_name, ty, &mut Vec::new(), &mut leaves);
    leaves
}

pub(crate) fn collect_tuple_references(expr: &CompiledExpr, param_name: &str) -> Vec<TupleLeaf> {
    let mut leaves = Vec::new();
    let mut seen = HashSet::new();
    fn walk(expr: &CompiledExpr, param_name: &str, out: &mut Vec<TupleLeaf>, seen: &mut HashSet<Vec<usize>>) {
        match expr {
            CompiledExpr::GetTupleElement { param, indices } if param == param_name && seen.insert(indices.clone()) => {
                out.push(TupleLeaf {
                    indices: indices.clone(),
                    symbol: format!("{}_{}", param_name, indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("_")),
                });
            }
            other => {
                other.map_children(|child| {
                    walk(child, param_name, out, seen);
                    child.clone()
                });
            }
        }
    }
    walk(expr, param_name, &mut leaves, &mut seen);
    leaves
}

pub(crate) fn expand_tuple_to_symbols(expr: &CompiledExpr, param_name: &str) -> CompiledExpr {
    match expr {
        CompiledExpr::GetTupleElement { param, indices } if param == param_name => {
            let symbol = format!("{}_{}", param_name, indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("_"));
            CompiledExpr::Symbol(symbol)
        }
        CompiledExpr::FunctionCall { name, args, loc } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(|a| expand_tuple_to_symbols(a, param_name)).collect(),
            loc: loc.clone(),
        },
        CompiledExpr::Let { bindings, body } => CompiledExpr::Let {
            bindings: bindings.iter().map(|(k, v)| (k.clone(), expand_tuple_to_symbols(v, param_name))).collect(),
            body: Box::new(expand_tuple_to_symbols(body, param_name)),
        },
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs.iter().map(|e| expand_tuple_to_symbols(e, param_name)).collect(),
        ),
        CompiledExpr::If { condition, then_branch, else_branch } => CompiledExpr::If {
            condition: Box::new(expand_tuple_to_symbols(condition, param_name)),
            then_branch: Box::new(expand_tuple_to_symbols(then_branch, param_name)),
            else_branch: else_branch.as_ref().map(|e| Box::new(expand_tuple_to_symbols(e, param_name))),
        },
        CompiledExpr::Lambda { params, body, .. } => CompiledExpr::lambda(
            params.clone(), expand_tuple_to_symbols(body, param_name),
        ),
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(expand_tuple_to_symbols(callee, param_name)),
            args: args.iter().map(|a| expand_tuple_to_symbols(a, param_name)).collect(),
        },
        other => other.clone(),
    }
}

pub fn try_flatten_to_constant(elements: &[CompiledExpr]) -> Option<(Vec<f64>, Vec<i64>)> {
    if elements.is_empty() {
        return None;
    }

    match &elements[0] {
        CompiledExpr::Float(_) | CompiledExpr::Integer(_) => {
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
