// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Codegen helpers.

use crate::core::expr::CompiledExpr;
use crate::lowering::stablehlo::StableHLOType;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

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

pub(crate) fn expand_tuple_to_symbols(
    expr: &CompiledExpr,
    param_name: &str,
    param_type: &StableHLOType,
) -> CompiledExpr {
    fn tree(name: &str, ty: &StableHLOType, indices: &[usize]) -> CompiledExpr {
        match ty {
            StableHLOType::Tuple(elements, keys) => {
                let elements = elements.iter().enumerate().map(|(index, ty)| {
                    let mut path = indices.to_vec();
                    path.push(index);
                    tree(name, ty, &path)
                }).collect::<Vec<_>>();
                match keys {
                    Some(keys) => CompiledExpr::Dict(keys.iter().cloned().map(CompiledExpr::Keyword)
                        .zip(elements).collect()),
                    None => CompiledExpr::Tuple(elements),
                }
            }
            _ => CompiledExpr::Symbol(format!("{}_{}", name,
                indices.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"))),
        }
    }
    fn select<'a>(value: &'a CompiledExpr, key: &CompiledExpr) -> Option<&'a CompiledExpr> {
        match (value, key) {
            (CompiledExpr::Tuple(values), CompiledExpr::Integer(index)) => {
                let index = if *index < 0 { values.len() as i64 + index } else { *index };
                usize::try_from(index).ok().and_then(|index| values.get(index))
            }
            (CompiledExpr::Dict(values), CompiledExpr::Keyword(key) | CompiledExpr::String(key)) => {
                values.iter().find_map(|(name, value)| match name {
                    CompiledExpr::Keyword(name) | CompiledExpr::String(name) if name == key => Some(value),
                    _ => None,
                })
            }
            _ => None,
        }
    }
    fn rewrite(expr: &CompiledExpr, locals: &HashMap<String, Rc<CompiledExpr>>) -> CompiledExpr {
        match expr {
            CompiledExpr::Symbol(name) => locals.get(name).map(|value| value.as_ref().clone())
                .unwrap_or_else(|| expr.clone()),
            CompiledExpr::GetTupleElement { param, indices } => {
                let mut value = locals.get(param).map(Rc::as_ref);
                for &index in indices {
                    value = value.and_then(|value| match value {
                        CompiledExpr::Tuple(values) => values.get(index),
                        CompiledExpr::Dict(values) => values.get(index).map(|(_, value)| value),
                        _ => None,
                    });
                }
                value.cloned().unwrap_or_else(|| expr.clone())
            }
            CompiledExpr::Let { bindings, body } => {
                let mut scoped = locals.clone();
                let bindings = bindings.iter().map(|(pattern, value)| {
                    let value = rewrite(value, &scoped);
                    let mut names = HashSet::new();
                    crate::core::bindings::pattern_names(pattern, &mut names);
                    for name in names { scoped.remove(&name); }
                    if let crate::core::expr::BindingPattern::Simple(name) = pattern
                        && matches!(&value, CompiledExpr::Tuple(_) | CompiledExpr::Dict(_)) {
                            scoped.insert(name.clone(), Rc::new(value.clone()));
                        }
                    (pattern.clone(), value)
                }).collect();
                CompiledExpr::Let { bindings, body: Box::new(rewrite(body, &scoped)) }
            }
            CompiledExpr::Lambda { params, body, .. } => {
                let mut scoped = locals.clone();
                for param in params { scoped.remove(param); }
                CompiledExpr::lambda(params.clone(), rewrite(body, &scoped))
            }
            CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => {
                let mut scoped = locals.clone();
                scoped.remove(index_var);
                scoped.remove(acc_var);
                CompiledExpr::Repeat {
                    index_var: index_var.clone(),
                    count: Box::new(rewrite(count, locals)),
                    acc_var: acc_var.clone(),
                    acc_init: Box::new(rewrite(acc_init, locals)),
                    body: Box::new(rewrite(body, &scoped)),
                }
            }
            CompiledExpr::While { condition, acc_var, acc_init, body } => {
                let mut scoped = locals.clone();
                scoped.remove(acc_var);
                CompiledExpr::While {
                    condition: Box::new(rewrite(condition, &scoped)),
                    acc_var: acc_var.clone(),
                    acc_init: Box::new(rewrite(acc_init, locals)),
                    body: Box::new(rewrite(body, &scoped)),
                }
            }
            _ => {
                let rewritten = expr.map_children(|child| rewrite(child, locals));
                if let CompiledExpr::FunctionCall { name, args, .. } = &rewritten {
                    if matches!(name.as_str(), "len" | "count") && args.len() == 1 {
                        let length = match &args[0] {
                            CompiledExpr::Tuple(values) | CompiledExpr::Vector(values) => Some(values.len()),
                            CompiledExpr::Dict(values) => Some(values.len()),
                            _ => None,
                        };
                        if let Some(length) = length {
                            return CompiledExpr::Integer(length as i64);
                        }
                    }
                    let key = match (name.as_str(), args.as_slice()) {
                        ("first", [_]) => Some(CompiledExpr::Integer(0)),
                        ("second", [_]) => Some(CompiledExpr::Integer(1)),
                        ("last", [_]) => Some(CompiledExpr::Integer(-1)),
                        ("get", [_, key]) => Some(key.clone()),
                        _ => None,
                    };
                    if let Some(key) = key
                        && let Some(value) = select(&args[0], &key) {
                            return value.clone();
                        }
                }
                rewritten
            }
        }
    }
    rewrite(expr, &HashMap::from([(
        param_name.to_string(), Rc::new(tree(param_name, param_type, &[])),
    )]))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::expr::BindingPattern;

    fn first(name: &str) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: "first".into(), args: vec![CompiledExpr::Symbol(name.into())], loc: None,
        }
    }

    #[test]
    fn aggregate_aliases_do_not_cross_binders() {
        let ty = StableHLOType::Tuple(vec![StableHLOType::f32_tensor(vec![2]); 2], None);
        let expression = CompiledExpr::Let {
            bindings: vec![(BindingPattern::Simple("alias".into()), CompiledExpr::Symbol("p".into()))],
            body: Box::new(first("alias")),
        };
        let CompiledExpr::Let { body, .. } = expand_tuple_to_symbols(&expression, "p", &ty)
            else { panic!() };
        assert!(matches!(body.as_ref(), CompiledExpr::Symbol(name) if name == "p_0"));

        let expression = CompiledExpr::Repeat {
            index_var: "i".into(), count: Box::new(CompiledExpr::Integer(1)),
            acc_var: "p".into(), acc_init: Box::new(CompiledExpr::Symbol("p".into())),
            body: Box::new(first("p")),
        };
        let CompiledExpr::Repeat { body, acc_init, .. } = expand_tuple_to_symbols(&expression, "p", &ty)
            else { panic!() };
        assert!(matches!(acc_init.as_ref(), CompiledExpr::Tuple(_)));
        assert!(matches!(body.as_ref(), CompiledExpr::FunctionCall { name, .. } if name == "first"));

        let expression = CompiledExpr::lambda(vec!["p".into()], first("p"));
        let CompiledExpr::Lambda { body, .. } = expand_tuple_to_symbols(&expression, "p", &ty)
            else { panic!() };
        assert!(matches!(body.as_ref(), CompiledExpr::FunctionCall { name, .. } if name == "first"));
    }
}
