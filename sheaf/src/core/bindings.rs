// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Lexical binding analysis.

use crate::core::expr::{BindingPattern, CompiledExpr};
use std::collections::{HashMap, HashSet};

pub fn pattern_names(pattern: &BindingPattern, names: &mut HashSet<String>) {
    match pattern {
        BindingPattern::Simple(name) => { names.insert(name.clone()); }
        BindingPattern::Destructure(patterns) => {
            for pattern in patterns { pattern_names(pattern, names); }
        }
    }
}

/// Collect local references that are not bound within the expression.
pub fn free_locals(expr: &CompiledExpr) -> Vec<String> {
    let mut names = HashSet::new();
    collect_locals(expr, &HashSet::new(), &mut names);
    let mut names: Vec<_> = names.into_iter().collect();
    names.sort();
    names
}

fn collect_locals(expr: &CompiledExpr, bound: &HashSet<String>, names: &mut HashSet<String>) {
    match expr {
        CompiledExpr::Symbol(name) | CompiledExpr::GetTupleElement { param: name, .. } => {
            if !bound.contains(name) { names.insert(name.clone()); }
        }
        CompiledExpr::Let { bindings, body } => {
            let mut scope = bound.clone();
            for (pattern, value) in bindings {
                collect_locals(value, &scope, names);
                pattern_names(pattern, &mut scope);
            }
            collect_locals(body, &scope, names);
        }
        CompiledExpr::Lambda { params, body, .. } => {
            let mut scope = bound.clone();
            scope.extend(params.iter().cloned());
            collect_locals(body, &scope, names);
        }
        CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => {
            collect_locals(count, bound, names);
            collect_locals(acc_init, bound, names);
            let mut scope = bound.clone();
            scope.insert(index_var.clone());
            scope.insert(acc_var.clone());
            collect_locals(body, &scope, names);
        }
        CompiledExpr::While { condition, acc_var, acc_init, body } => {
            collect_locals(acc_init, bound, names);
            let mut scope = bound.clone();
            scope.insert(acc_var.clone());
            collect_locals(condition, &scope, names);
            collect_locals(body, &scope, names);
        }
        _ => expr.for_each_child(|child| collect_locals(child, bound, names)),
    }
}

pub fn pattern_binds(pattern: &BindingPattern, name: &str) -> bool {
    match pattern {
        BindingPattern::Simple(binding) => binding == name,
        BindingPattern::Destructure(patterns) => patterns.iter().any(|pattern| pattern_binds(pattern, name)),
    }
}

pub(crate) type Replacements = HashMap<String, HashMap<Vec<usize>, CompiledExpr>>;

pub fn replace_local(expr: &CompiledExpr, name: &str, replacement: &CompiledExpr) -> CompiledExpr {
    replace_bindings(expr, &HashMap::from([
        (name.to_string(), HashMap::from([(Vec::new(), replacement.clone())])),
    ]))
}

pub(crate) fn replace_bindings(expr: &CompiledExpr, replacements: &Replacements) -> CompiledExpr {
    let scope = replacements.iter().map(|(name, values)| (name.as_str(), values)).collect();
    replace_in_scope(expr, &scope)
}

type Scope<'a> = HashMap<&'a str, &'a HashMap<Vec<usize>, CompiledExpr>>;

fn replace_in_scope(expr: &CompiledExpr, replacements: &Scope<'_>) -> CompiledExpr {
    let replace = |child: &CompiledExpr| replace_in_scope(child, replacements);
    let scoped = |names: &[String]| {
        let mut scoped = replacements.clone();
        for name in names { scoped.remove(name.as_str()); }
        scoped
    };
    match expr {
        CompiledExpr::Symbol(name) => replacements.get(name.as_str()).and_then(|values| values.get(&Vec::new()))
            .cloned().unwrap_or_else(|| expr.clone()),
        CompiledExpr::GetTupleElement { param, indices } => {
            let Some(values) = replacements.get(param.as_str()) else { return expr.clone() };
            if let Some(value) = values.get(indices) { return value.clone() }
            let Some(value) = values.get(&Vec::new()) else { return expr.clone() };
            if let CompiledExpr::Symbol(param) = value {
                return CompiledExpr::GetTupleElement { param: param.clone(), indices: indices.clone() };
            }
            indices.iter().fold(value.clone(), |value, &index| {
                if let CompiledExpr::Tuple(values) = &value
                    && let Some(element) = values.get(index) {
                    return element.clone();
                }
                CompiledExpr::FunctionCall {
                    name: "get".to_string(), args: vec![value, CompiledExpr::Integer(index as i64)], loc: None,
                }
            })
        }
        CompiledExpr::Let { bindings, body } => {
            let mut scope = replacements.clone();
            let bindings = bindings.iter().map(|(pattern, value)| {
                let value = replace_in_scope(value, &scope);
                scope.retain(|name, _| !pattern_binds(pattern, name));
                (pattern.clone(), value)
            }).collect();
            CompiledExpr::Let { bindings, body: Box::new(replace_in_scope(body, &scope)) }
        }
        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(), body: Box::new(replace_in_scope(body, &scoped(params))),
        },
        CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => CompiledExpr::Repeat {
            index_var: index_var.clone(), count: Box::new(replace(count)),
            acc_var: acc_var.clone(), acc_init: Box::new(replace(acc_init)),
            body: Box::new(replace_in_scope(body, &scoped(&[index_var.clone(), acc_var.clone()]))),
        },
        CompiledExpr::While { condition, acc_var, acc_init, body } => {
            let scope = scoped(std::slice::from_ref(acc_var));
            CompiledExpr::While {
                condition: Box::new(replace_in_scope(condition, &scope)),
                acc_var: acc_var.clone(), acc_init: Box::new(replace(acc_init)),
                body: Box::new(replace_in_scope(body, &scope)),
            }
        }
        _ => expr.map_children(replace),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbol(name: &str) -> CompiledExpr { CompiledExpr::Symbol(name.to_string()) }

    #[test]
    fn replacements_preserve_scalar_types_and_call_locations() {
        use crate::core::error::SourceLocation;
        let loc = SourceLocation::new(3, 5, "capture.shf".into());
        let call = CompiledExpr::FunctionCall {
            name: "identity".into(), args: vec![symbol("x")], loc: Some(loc),
        };
        for replacement in [CompiledExpr::Float(2.0), CompiledExpr::Integer(9_007_199_254_740_993)] {
            let result = replace_local(&call, "x", &replacement);
            let CompiledExpr::FunctionCall { args, loc: Some(loc), .. } = result else { panic!() };
            assert_eq!((loc.line, loc.column, loc.filename.as_ref()), (3, 5, "capture.shf"));
            match (&args[0], replacement) {
                (CompiledExpr::Float(a), CompiledExpr::Float(b)) => assert_eq!(*a, b),
                (CompiledExpr::Integer(a), CompiledExpr::Integer(b)) => assert_eq!(*a, b),
                _ => panic!("replacement changed scalar type"),
            }
        }
    }

    #[test]
    fn replacements_respect_destructuring_and_loop_scopes() {
        let replacement = CompiledExpr::Integer(7);
        let expr = CompiledExpr::Let {
            bindings: vec![(BindingPattern::Destructure(vec![BindingPattern::Simple("x".into())]),
                            CompiledExpr::Tuple(vec![symbol("x")]))],
            body: Box::new(symbol("x")),
        };
        let CompiledExpr::Let { bindings, body } = replace_local(&expr, "x", &replacement) else { panic!() };
        assert!(matches!(&bindings[0].1, CompiledExpr::Tuple(v) if matches!(v[0], CompiledExpr::Integer(7))));
        assert!(matches!(*body, CompiledExpr::Symbol(_)));
        let expr = CompiledExpr::Repeat {
            index_var: "x".into(), count: Box::new(symbol("x")), acc_var: "a".into(),
            acc_init: Box::new(symbol("x")), body: Box::new(symbol("x")),
        };
        let CompiledExpr::Repeat { count, acc_init, body, .. } = replace_local(&expr, "x", &replacement) else { panic!() };
        assert!(matches!(*count, CompiledExpr::Integer(7)));
        assert!(matches!(*acc_init, CompiledExpr::Integer(7)));
        assert!(matches!(*body, CompiledExpr::Symbol(_)));
        let expr = CompiledExpr::While {
            acc_var: "x".into(), acc_init: Box::new(symbol("x")),
            condition: Box::new(symbol("x")), body: Box::new(symbol("x")),
        };
        let CompiledExpr::While { acc_init, condition, body, .. } = replace_local(&expr, "x", &replacement) else { panic!() };
        assert!(matches!(*acc_init, CompiledExpr::Integer(7)));
        assert!(matches!(*condition, CompiledExpr::Symbol(_)));
        assert!(matches!(*body, CompiledExpr::Symbol(_)));
    }

    #[test]
    fn sequential_bindings_and_nested_patterns_hide_only_bound_references() {
        let body = CompiledExpr::Let {
            bindings: vec![
                (BindingPattern::Simple("x".into()), symbol("x")),
                (BindingPattern::Destructure(vec![BindingPattern::Simple("y".into()),
                    BindingPattern::Destructure(vec![BindingPattern::Simple("z".into())])]),
                 CompiledExpr::Tuple(vec![symbol("x"), symbol("outer")])),
            ],
            body: Box::new(CompiledExpr::Tuple(vec![symbol("x"), symbol("y"),
                CompiledExpr::GetTupleElement { param: "z".into(), indices: vec![0] }, symbol("external")])),
        };
        assert_eq!(free_locals(&body), ["external", "outer", "x"]);
    }

    #[test]
    fn lambda_parameters_do_not_hide_outer_container_inputs() {
        let body = CompiledExpr::Lambda { params: vec!["x".into()], body: Box::new(CompiledExpr::Tuple(vec![
            symbol("x"), CompiledExpr::Dict(vec![(symbol("key"), symbol("outer"))]),
            CompiledExpr::GetTupleElement { param: "params".into(), indices: vec![0] },
            CompiledExpr::FunctionRef("module-function".into()),
        ])) };
        assert_eq!(free_locals(&body), ["key", "outer", "params"]);
    }

    #[test]
    fn loop_initializers_and_bound_bodies_have_separate_scopes() {
        let repeat = CompiledExpr::Repeat {
            index_var: "i".into(), count: Box::new(symbol("i")), acc_var: "a".into(),
            acc_init: Box::new(symbol("a")),
            body: Box::new(CompiledExpr::Tuple(vec![symbol("i"), symbol("a"), symbol("outside")])),
        };
        assert_eq!(free_locals(&repeat), ["a", "i", "outside"]);
        let while_loop = CompiledExpr::While {
            acc_var: "a".into(), acc_init: Box::new(symbol("initial")),
            condition: Box::new(CompiledExpr::Tuple(vec![symbol("a"), symbol("condition")])),
            body: Box::new(CompiledExpr::Tuple(vec![symbol("a"), symbol("body")])),
        };
        assert_eq!(free_locals(&while_loop), ["body", "condition", "initial"]);
    }
}
