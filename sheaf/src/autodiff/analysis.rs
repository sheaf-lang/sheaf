// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Static analysis of expressions for autodiff compatibility.

use crate::core::expr::{BindingPattern, CompiledExpr};
use std::collections::{HashMap, HashSet};

/// Collect lexical inputs and module dependencies for derivative analysis.
pub fn collect_free_vars(expr: &CompiledExpr, out: &mut HashSet<String>) {
    out.extend(crate::core::bindings::free_locals(expr));
    out.extend(crate::core::bindings::global_names(expr));
}

/// Return whether an autodiff expression may depend on `symbol`.
///
/// This analysis models lexical bindings only. It does not model global
/// mutations, which are not admitted as intermediate autodiff expressions.
/// It does not infer whether an operation has an autodiff rule.
#[cfg(test)]
pub(crate) fn depends_on(expr: &CompiledExpr, symbol: &str) -> bool {
    let mut env = HashMap::new();
    env.insert(symbol.to_string(), true);
    depends_on_in_env(expr, &env)
}

fn bind_pattern(pattern: &BindingPattern, depends: bool, env: &mut HashMap<String, bool>) {
    match pattern {
        BindingPattern::Simple(name) => {
            env.insert(name.clone(), depends);
        }
        BindingPattern::Destructure(patterns) => {
            for pattern in patterns {
                bind_pattern(pattern, depends, env);
            }
        }
    }
}

pub(crate) fn depends_on_in_env(expr: &CompiledExpr, env: &HashMap<String, bool>) -> bool {
    match expr {
        CompiledExpr::Integer(_)
        | CompiledExpr::Float(_)
        | CompiledExpr::Boolean(_)
        | CompiledExpr::Nil
        | CompiledExpr::String(_)
        | CompiledExpr::Keyword(_)
        | CompiledExpr::Quoted(_)
        | CompiledExpr::FunctionRef(_)
        | CompiledExpr::GlobalRef(_) => false,
        // Deferred value-and-grad does not evaluate to a numeric expression here.
        CompiledExpr::ValueAndGrad { .. } => false,
        CompiledExpr::Def { value, .. } => depends_on_in_env(value, env),
        CompiledExpr::Symbol(name) => env.get(name).copied().unwrap_or(false),
        CompiledExpr::GetTupleElement { param, .. } => env.get(param).copied().unwrap_or(false),
        CompiledExpr::Vector(elems) | CompiledExpr::Tuple(elems) => {
            elems.iter().any(|elem| depends_on_in_env(elem, env))
        }
        CompiledExpr::Dict(pairs) => pairs
            .iter()
            .any(|(key, value)| depends_on_in_env(key, env) || depends_on_in_env(value, env)),
        CompiledExpr::FunctionCall { name, .. }
            if matches!(name.as_str(), "shape" | "ndim") => false,
        CompiledExpr::FunctionCall { args, .. } => {
            args.iter().any(|arg| depends_on_in_env(arg, env))
        }
        CompiledExpr::Let { bindings, body } => {
            let mut scoped = env.clone();
            for (pattern, value) in bindings {
                let value_depends = depends_on_in_env(value, &scoped);
                bind_pattern(pattern, value_depends, &mut scoped);
            }
            depends_on_in_env(body, &scoped)
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            depends_on_in_env(condition, env)
                || depends_on_in_env(then_branch, env)
                || else_branch
                    .as_ref()
                    .is_some_and(|branch| depends_on_in_env(branch, env))
        }
        CompiledExpr::Do(exprs) => exprs
            .last()
            .is_some_and(|expr| depends_on_in_env(expr, env)),
        CompiledExpr::Lambda { params, body, .. } => {
            let mut scoped = env.clone();
            for param in params {
                scoped.insert(param.clone(), false);
            }
            depends_on_in_env(body, &scoped)
        }
        CompiledExpr::LambdaCall { callee, args } => {
            if let CompiledExpr::Lambda { params, body, .. } = callee.as_ref() {
                // Do not truncate mismatched lambda arguments. Until arity is
                // rejected by the frontend, retain a conservative dependency.
                if params.len() != args.len() {
                    return depends_on_in_env(callee, env)
                        || args.iter().any(|arg| depends_on_in_env(arg, env));
                }

                let mut scoped = env.clone();
                for (param, arg) in params.iter().zip(args) {
                    scoped.insert(param.clone(), depends_on_in_env(arg, env));
                }
                depends_on_in_env(body, &scoped)
            } else {
                // An opaque callee may inspect any argument or capture.
                depends_on_in_env(callee, env) || args.iter().any(|arg| depends_on_in_env(arg, env))
            }
        }
        CompiledExpr::Repeat {
            index_var,
            count,
            acc_var,
            acc_init,
            body,
        } => {
            let mut scoped = env.clone();
            let init_depends = depends_on_in_env(acc_init, env);
            scoped.insert(index_var.clone(), false);
            scoped.insert(acc_var.clone(), init_depends);
            depends_on_in_env(count, env) || init_depends || depends_on_in_env(body, &scoped)
        }
        CompiledExpr::While {
            condition,
            acc_var,
            acc_init,
            body,
        } => {
            let mut scoped = env.clone();
            let init_depends = depends_on_in_env(acc_init, env);
            scoped.insert(acc_var.clone(), init_depends);
            depends_on_in_env(condition, &scoped)
                || init_depends
                || depends_on_in_env(body, &scoped)
        }
        CompiledExpr::Guard { expr, .. } => depends_on_in_env(expr, env),
    }
}

#[cfg(test)]
mod tests {
    use super::depends_on;
    use crate::core::expr::{BindingPattern, CompiledExpr};

    fn symbol(name: &str) -> CompiledExpr {
        CompiledExpr::Symbol(name.to_string())
    }

    fn call(args: Vec<CompiledExpr>) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: "sentinel".to_string(),
            args,
            loc: None,
        }
    }

    #[test]
    fn free_variable_collection_respects_local_bindings() {
        let expr = CompiledExpr::Let {
            bindings: vec![(BindingPattern::Simple("local".into()), CompiledExpr::Float(1.0))],
            body: Box::new(CompiledExpr::Do(vec![symbol("local"), symbol("outside")])),
        };
        let mut names = std::collections::HashSet::from(["caller".to_string()]);
        super::collect_free_vars(&expr, &mut names);
        assert_eq!(names, ["caller", "outside"].into_iter().map(str::to_string).collect());
    }

    #[test]
    fn shared_call_analysis_covers_containers_and_loops() {
        let target = |name: &str| CompiledExpr::FunctionCall {
            name: name.to_string(), args: vec![], loc: None,
        };
        let body = CompiledExpr::Tuple(vec![
            CompiledExpr::Dict(vec![(target("key"), target("value"))]),
            CompiledExpr::Repeat {
                index_var: "i".to_string(), count: Box::new(target("count")),
                acc_var: "a".to_string(), acc_init: Box::new(target("initial")),
                body: Box::new(target("step")),
            },
            CompiledExpr::While {
                condition: Box::new(target("condition")), acc_var: "a".to_string(),
                acc_init: Box::new(target("while_initial")), body: Box::new(target("while_step")),
            },
            CompiledExpr::FunctionRef("not_called".to_string()),
        ]);
        let expected = ["key", "value", "count", "initial", "step", "condition", "while_initial", "while_step"]
            .into_iter().map(str::to_string).collect::<std::collections::HashSet<_>>();
        assert_eq!(crate::lowering::call_graph::direct_callees(&body), expected);
    }

    #[test]
    fn tracks_each_if_position() {
        let constant = || CompiledExpr::Float(0.0);
        let condition = CompiledExpr::If {
            condition: Box::new(symbol("x")),
            then_branch: Box::new(constant()),
            else_branch: Some(Box::new(constant())),
        };
        let then_branch = CompiledExpr::If {
            condition: Box::new(CompiledExpr::Boolean(true)),
            then_branch: Box::new(symbol("x")),
            else_branch: Some(Box::new(constant())),
        };
        let else_branch = CompiledExpr::If {
            condition: Box::new(CompiledExpr::Boolean(true)),
            then_branch: Box::new(constant()),
            else_branch: Some(Box::new(symbol("x"))),
        };

        assert!(depends_on(&condition, "x"));
        assert!(depends_on(&then_branch, "x"));
        assert!(depends_on(&else_branch, "x"));
    }

    #[test]
    fn tracks_each_collection_position() {
        let key = CompiledExpr::Dict(vec![(symbol("x"), CompiledExpr::Float(0.0))]);
        let value = CompiledExpr::Dict(vec![(CompiledExpr::Keyword("k".to_string()), symbol("x"))]);
        let vector = CompiledExpr::Vector(vec![CompiledExpr::Float(0.0), symbol("x")]);
        let tuple = CompiledExpr::Tuple(vec![CompiledExpr::Float(0.0), symbol("x")]);

        assert!(depends_on(&key, "x"));
        assert!(depends_on(&value, "x"));
        assert!(depends_on(&vector, "x"));
        assert!(depends_on(&tuple, "x"));
    }

    #[test]
    fn tracks_calls_and_guard() {
        let args = call(vec![CompiledExpr::Float(0.0), symbol("x")]);
        let guard = CompiledExpr::Guard {
            check: crate::core::expr::GuardCheck::NoNan,
            expr: Box::new(symbol("x")),
        };

        assert!(depends_on(&args, "x"));
        assert!(depends_on(&guard, "x"));
    }

    #[test]
    fn get_tuple_element_respects_lexical_shadowing() {
        let expr = CompiledExpr::Let {
            bindings: vec![(
                BindingPattern::Simple("x".to_string()),
                CompiledExpr::Float(1.0),
            )],
            body: Box::new(CompiledExpr::GetTupleElement {
                param: "x".to_string(),
                indices: vec![0],
            }),
        };

        assert!(!depends_on(&expr, "x"));
    }

    #[test]
    fn tracks_sequential_and_destructured_bindings() {
        let sequential = CompiledExpr::Let {
            bindings: vec![
                (BindingPattern::Simple("a".to_string()), symbol("x")),
                (BindingPattern::Simple("b".to_string()), symbol("a")),
            ],
            body: Box::new(symbol("b")),
        };
        let destructured = CompiledExpr::Let {
            bindings: vec![(
                BindingPattern::Destructure(vec![
                    BindingPattern::Simple("a".to_string()),
                    BindingPattern::Simple("b".to_string()),
                ]),
                symbol("x"),
            )],
            body: Box::new(symbol("b")),
        };

        assert!(depends_on(&sequential, "x"));
        assert!(depends_on(&destructured, "x"));
    }

    #[test]
    fn respects_shadowing_and_do_result_semantics() {
        let shadowed = CompiledExpr::Let {
            bindings: vec![(
                BindingPattern::Simple("x".to_string()),
                CompiledExpr::Float(1.0),
            )],
            body: Box::new(symbol("x")),
        };
        let dead_do = CompiledExpr::Do(vec![call(vec![symbol("x")]), CompiledExpr::Float(1.0)]);
        let live_do = CompiledExpr::Do(vec![CompiledExpr::Float(1.0), call(vec![symbol("x")])]);

        assert!(!depends_on(&shadowed, "x"));
        assert!(!depends_on(&dead_do, "x"));
        assert!(depends_on(&live_do, "x"));
    }

    #[test]
    fn lambda_masks_parameters_and_preserves_captures() {
        let shadowed = CompiledExpr::lambda(vec!["x".to_string()], symbol("x"));
        let captured = CompiledExpr::lambda(vec!["y".to_string()], symbol("x"));

        assert!(!depends_on(&shadowed, "x"));
        assert!(depends_on(&captured, "x"));
    }

    #[test]
    fn direct_lambda_call_binds_parameter_dependencies() {
        let ignores_argument = CompiledExpr::LambdaCall {
            callee: Box::new(CompiledExpr::lambda(vec!["y".to_string()], CompiledExpr::Float(1.0))),
            args: vec![symbol("x")],
        };
        let uses_argument = CompiledExpr::LambdaCall {
            callee: Box::new(CompiledExpr::lambda(vec!["y".to_string()], symbol("y"))),
            args: vec![symbol("x")],
        };
        let opaque_callee = CompiledExpr::LambdaCall {
            callee: Box::new(symbol("x")),
            args: vec![CompiledExpr::Float(1.0)],
        };
        let mismatched_arity = CompiledExpr::LambdaCall {
            callee: Box::new(CompiledExpr::lambda(
                vec!["y".to_string(), "z".to_string()], CompiledExpr::Float(1.0),
            )),
            args: vec![symbol("x")],
        };

        assert!(!depends_on(&ignores_argument, "x"));
        assert!(depends_on(&uses_argument, "x"));
        assert!(depends_on(&opaque_callee, "x"));
        assert!(depends_on(&mismatched_arity, "x"));
    }

    #[test]
    fn tracks_each_loop_input() {
        let repeat_count = CompiledExpr::Repeat {
            index_var: "i".to_string(),
            count: Box::new(symbol("x")),
            acc_var: "acc".to_string(),
            acc_init: Box::new(CompiledExpr::Float(0.0)),
            body: Box::new(symbol("acc")),
        };
        let repeat_init = CompiledExpr::Repeat {
            index_var: "i".to_string(),
            count: Box::new(CompiledExpr::Integer(2)),
            acc_var: "acc".to_string(),
            acc_init: Box::new(symbol("x")),
            body: Box::new(symbol("acc")),
        };
        let repeat_body = CompiledExpr::Repeat {
            index_var: "i".to_string(),
            count: Box::new(CompiledExpr::Integer(2)),
            acc_var: "acc".to_string(),
            acc_init: Box::new(CompiledExpr::Float(0.0)),
            body: Box::new(symbol("x")),
        };
        let while_condition = CompiledExpr::While {
            condition: Box::new(symbol("x")),
            acc_var: "acc".to_string(),
            acc_init: Box::new(CompiledExpr::Float(0.0)),
            body: Box::new(symbol("acc")),
        };
        let while_init = CompiledExpr::While {
            condition: Box::new(CompiledExpr::Boolean(false)),
            acc_var: "acc".to_string(),
            acc_init: Box::new(symbol("x")),
            body: Box::new(symbol("acc")),
        };
        let while_body = CompiledExpr::While {
            condition: Box::new(CompiledExpr::Boolean(false)),
            acc_var: "acc".to_string(),
            acc_init: Box::new(CompiledExpr::Float(0.0)),
            body: Box::new(symbol("x")),
        };

        for expr in [
            repeat_count,
            repeat_init,
            repeat_body,
            while_condition,
            while_init,
            while_body,
        ] {
            assert!(depends_on(&expr, "x"));
        }
    }

    #[test]
    fn literals_and_deferred_value_and_grad_are_independent() {
        let value_and_grad = CompiledExpr::ValueAndGrad {
            fn_name: "grad-f".to_string(),
            src_fn_name: "f".to_string(),
            wrt_params: vec!["x".to_string()],
            shape_config: vec![],
        };

        assert!(!depends_on(&CompiledExpr::Float(1.0), "x"));
        assert!(!depends_on(&value_and_grad, "x"));
    }

    #[test]
    fn definition_returns_its_value_dependency() {
        let constant = CompiledExpr::Def {
            name: "y".to_string(),
            value: Box::new(CompiledExpr::Float(1.0)),
        };
        let dependent = CompiledExpr::Def {
            name: "y".to_string(),
            value: Box::new(symbol("x")),
        };

        assert!(!depends_on(&constant, "x"));
        assert!(depends_on(&dependent, "x"));
    }
}
