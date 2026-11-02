// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::lowering::config::ParamIndexMap;
use std::collections::{BTreeMap, HashMap};

/// Lower dictionary accesses through `Let` aliases.
pub fn lower_inlined_gets(
    body: &CompiledExpr,
    index_maps: &[ParamIndexMap],
) -> CompiledExpr {
    let mut reverse: HashMap<(String, Vec<usize>), Vec<String>> = HashMap::new();
    for (param, imap) in index_maps {
        for (path, indices) in imap {
            reverse.insert((param.clone(), indices.clone()), path.clone());
        }
    }
    lower_inlined_gets_rec(body, index_maps, &mut HashMap::new(), &reverse)
}

fn lower_inlined_gets_rec(
    expr: &CompiledExpr,
    index_maps: &[ParamIndexMap],
    aliases: &mut HashMap<String, (String, Vec<String>)>,
    reverse: &HashMap<(String, Vec<usize>), Vec<String>>,
) -> CompiledExpr {
    match expr {
        CompiledExpr::GetTupleElement { param, indices } => {
            if let Some((root, prefix)) = aliases.get(param) {
                let mut full_indices = if prefix.is_empty() {
                    Vec::new()
                } else if let Some(indices) = index_maps.iter()
                    .find(|(name, _)| name == root).and_then(|(_, map)| map.get(prefix)) {
                    indices.clone()
                } else {
                    return expr.clone();
                };
                full_indices.extend(indices);
                CompiledExpr::GetTupleElement { param: root.clone(), indices: full_indices }
            } else {
                expr.clone()
            }
        }
        CompiledExpr::FunctionCall { name, args, loc }
            if (name == "get" || name == "get-in") && args.len() >= 2 =>
        {
            if let CompiledExpr::Symbol(alias) = &args[0]
                && let Some((root_param, prefix_path)) = aliases.get(alias)
            {
                    let mut path = prefix_path.clone();
                    let resolved = if name == "get-in" {
                        if let CompiledExpr::Vector(keys) = &args[1] {
                            let mut ok = true;
                            for k in keys {
                                match k {
                                    CompiledExpr::Keyword(s) => path.push(s.clone()),
                                    _ => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            ok
                        } else {
                            false
                        }
                    } else {
                        let mut ok = true;
                        for arg in &args[1..] {
                            match arg {
                                CompiledExpr::Keyword(k) => path.push(k.clone()),
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        ok
                    };
                    if resolved
                        && let Some(imap) = index_maps
                            .iter()
                            .find(|(p, _)| p == root_param)
                            .map(|(_, m)| m)
                        && let Some(indices) = imap.get(&path)
                    {
                        return CompiledExpr::GetTupleElement {
                            param: root_param.clone(),
                            indices: indices.clone(),
                        };
                    }
                }
            let resolved_args: Vec<_> = args
                .iter()
                .map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse))
                .collect();
            if let CompiledExpr::GetTupleElement { param, indices } = &resolved_args[0] {
                let key = (param.clone(), indices.clone());
                if let Some(prefix_path) = reverse.get(&key) {
                    let mut path = prefix_path.clone();
                    let keys_ok = if name == "get-in" {
                        if let CompiledExpr::Vector(keys) = &resolved_args[1] {
                            let mut ok = true;
                            for k in keys {
                                match k {
                                    CompiledExpr::Keyword(s) => path.push(s.clone()),
                                    _ => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            ok
                        } else {
                            false
                        }
                    } else {
                        let mut ok = true;
                        for arg in &resolved_args[1..] {
                            match arg {
                                CompiledExpr::Keyword(k) => path.push(k.clone()),
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        ok
                    };
                    if keys_ok
                        && let Some(imap) =
                            index_maps.iter().find(|(p, _)| p == param).map(|(_, m)| m)
                        && let Some(full_indices) = imap.get(&path)
                    {
                        return CompiledExpr::GetTupleElement {
                            param: param.clone(),
                            indices: full_indices.clone(),
                        };
                    }
                }
            }
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: resolved_args,
                loc: loc.clone(),
            }
        }
        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<_> = bindings
                .iter()
                .map(|(k, v)| {
                    let resolved = lower_inlined_gets_rec(v, index_maps, aliases, reverse);
                    match &resolved {
                        CompiledExpr::GetTupleElement { param, indices } => {
                            let key = (param.clone(), indices.clone());
                            if let Some(path) = reverse.get(&key)
                                && let BindingPattern::Simple(k_str) = k
                            {
                                aliases.insert(k_str.clone(), (param.clone(), path.clone()));
                            }
                        }
                        CompiledExpr::Symbol(s) => {
                            if let Some(existing) = aliases.get(s).cloned() {
                                if let BindingPattern::Simple(k_str) = k {
                                    aliases.insert(k_str.clone(), existing);
                                }
                            } else if index_maps.iter().any(|(p, _)| p == s)
                                && let BindingPattern::Simple(k_str) = k
                            {
                                aliases.insert(k_str.clone(), (s.clone(), vec![]));
                            }
                        }
                        _ => {}
                    }
                    (k.clone(), resolved)
                })
                .collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
            }
        }
        CompiledExpr::FunctionCall { name, args, loc } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse))
                .collect(),
            loc: loc.clone(),
        },
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs
                .iter()
                .map(|e| lower_inlined_gets_rec(e, index_maps, aliases, reverse))
                .collect(),
        ),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(lower_inlined_gets_rec(
                condition, index_maps, aliases, reverse,
            )),
            then_branch: Box::new(lower_inlined_gets_rec(
                then_branch,
                index_maps,
                aliases,
                reverse,
            )),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(lower_inlined_gets_rec(e, index_maps, aliases, reverse))),
        },
        CompiledExpr::Lambda { params, body, .. } => CompiledExpr::lambda(
            params.clone(), lower_inlined_gets_rec(body, index_maps, aliases, reverse),
        ),
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(lower_inlined_gets_rec(callee, index_maps, aliases, reverse)),
            args: args
                .iter()
                .map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse))
                .collect(),
        },
        CompiledExpr::Repeat {
            index_var,
            count,
            acc_var,
            acc_init,
            body,
        } => CompiledExpr::Repeat {
            index_var: index_var.clone(),
            count: Box::new(lower_inlined_gets_rec(count, index_maps, aliases, reverse)),
            acc_var: acc_var.clone(),
            acc_init: Box::new(lower_inlined_gets_rec(
                acc_init, index_maps, aliases, reverse,
            )),
            body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
        },
        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems
                .iter()
                .map(|e| lower_inlined_gets_rec(e, index_maps, aliases, reverse))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Propagate dictionary layouts through `Let` bindings.
pub fn propagate_let_layouts(
    expr: &CompiledExpr,
    idx_to_key: &HashMap<(String, usize), String>,
    layouts: &mut HashMap<String, BTreeMap<String, usize>>,
) {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            for (var_name, value_expr) in bindings {
                if let CompiledExpr::GetTupleElement { param, indices } = value_expr {
                    let mut cur = param.clone();
                    let mut resolved = true;
                    for &idx in indices {
                        if let Some(key) = idx_to_key.get(&(cur.clone(), idx)) {
                            cur = key.clone();
                        } else {
                            resolved = false;
                            break;
                        }
                    }
                    if resolved
                        && let Some(sub_layout) = layouts.get(&cur).cloned()
                        && let BindingPattern::Simple(var_name_str) = var_name
                    {
                        layouts.insert(var_name_str.clone(), sub_layout);
                    }
                }
                propagate_let_layouts(value_expr, idx_to_key, layouts);
            }
            propagate_let_layouts(body, idx_to_key, layouts);
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                propagate_let_layouts(a, idx_to_key, layouts);
            }
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                propagate_let_layouts(e, idx_to_key, layouts);
            }
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            propagate_let_layouts(condition, idx_to_key, layouts);
            propagate_let_layouts(then_branch, idx_to_key, layouts);
            if let Some(e) = else_branch {
                propagate_let_layouts(e, idx_to_key, layouts);
            }
        }
        CompiledExpr::Lambda { body, .. } => {
            propagate_let_layouts(body, idx_to_key, layouts);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{lower_inlined_gets, propagate_let_layouts};
    use crate::core::expr::{BindingPattern, CompiledExpr};
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn lower_get_through_inlined_nested_tuple_alias() {
        let body = CompiledExpr::Let {
            bindings: vec![(
                BindingPattern::Simple("layer".to_string()),
                CompiledExpr::GetTupleElement {
                    param: "params".to_string(),
                    indices: vec![0],
                },
            )],
            body: Box::new(CompiledExpr::FunctionCall {
                name: "get".to_string(),
                args: vec![
                    CompiledExpr::Symbol("layer".to_string()),
                    CompiledExpr::Keyword("weight".to_string()),
                ],
                loc: None,
            }),
        };
        let mut index_map = BTreeMap::new();
        index_map.insert(vec!["layer".to_string()], vec![0]);
        index_map.insert(
            vec!["layer".to_string(), "weight".to_string()],
            vec![0, 0],
        );

        let mut layouts = HashMap::from([
            ("nested".to_string(), BTreeMap::from([("weight".to_string(), 0)])),
        ]);
        let idx_to_key = HashMap::from([
            (("params".to_string(), 0), "nested".to_string()),
            (("nested".to_string(), 0), "weight".to_string()),
        ]);
        propagate_let_layouts(&body, &idx_to_key, &mut layouts);
        assert_eq!(layouts.get("layer"), layouts.get("nested"));

        let lowered = lower_inlined_gets(&body, &[("params".to_string(), index_map)]);
        assert!(matches!(
            lowered,
            CompiledExpr::Let { body, .. }
                if matches!(*body, CompiledExpr::GetTupleElement { ref param, ref indices }
                    if param == "params" && indices == &vec![0, 0])
        ));
    }
}
