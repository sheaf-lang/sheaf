// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Automatic differentiation and gradient IR utilities.

pub mod analysis;
pub mod reverse;
pub mod trace;
pub mod transforms;

pub use analysis::collect_free_vars;
pub use transforms::{fold_dict_gets, inline_function_calls};

use crate::core::expr::CompiledExpr;

fn call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
    CompiledExpr::FunctionCall {
        name: name.to_string(),
        args,
        loc: None,
    }
}

fn float(v: f64) -> CompiledExpr {
    CompiledExpr::Float(v)
}

/// Inline let-bindings into the body by replacing symbols with their values.
///
pub(crate) fn replace_symbol(expr: &CompiledExpr, name: &str, replacement: &CompiledExpr) -> CompiledExpr {
    match expr {
        CompiledExpr::Symbol(s) if s == name => replacement.clone(),
        CompiledExpr::FunctionCall {
            name: fn_name,
            args,
            loc,
        } => CompiledExpr::FunctionCall {
            name: fn_name.clone(),
            args: args
                .iter()
                .map(|a| replace_symbol(a, name, replacement))
                .collect(),
            loc: loc.clone(),
        },
        CompiledExpr::Let { bindings, body } => {
            // Sequential scoping: once a binding shadows the name,
            // stop substituting in subsequent values AND the body.
            let mut new_bindings = Vec::new();
            let mut shadowed = false;
            for (k, v) in bindings {
                let new_v = if shadowed {
                    v.clone()
                } else {
                    replace_symbol(v, name, replacement)
                };
                new_bindings.push((k.clone(), new_v));
                if k.as_simple() == Some(name) {
                    shadowed = true;
                }
            }
            let new_body = if shadowed {
                body.clone()
            } else {
                Box::new(replace_symbol(body, name, replacement))
            };
            CompiledExpr::Let {
                bindings: new_bindings,
                body: new_body,
            }
        }
    CompiledExpr::Vector(elems) => CompiledExpr::Vector(
        elems
            .iter()
            .map(|e| replace_symbol(e, name, replacement))
            .collect(),
    ),
    CompiledExpr::Do(exprs) => CompiledExpr::Do(
        exprs
            .iter()
            .map(|e| replace_symbol(e, name, replacement))
            .collect(),
    ),
    other => other.clone(),
    }
}

/// Basic algebraic simplification for generated gradient expressions.
///
/// Rules:
///   0 + x  ->  x,  x + 0  ->  x
///   0 * x  ->  0,  x * 0  ->  0
///   1 * x  ->  x,  x * 1  ->  x
pub fn simplify(expr: CompiledExpr) -> CompiledExpr {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } => {
            let args: Vec<CompiledExpr> = args.into_iter().map(simplify).collect();
            match name.as_str() {
                "+" => match (&args[0], &args[1]) {
                    (CompiledExpr::Float(f), _) if *f == 0.0 => args.into_iter().nth(1).unwrap(),
                    (_, CompiledExpr::Float(f)) if *f == 0.0 => args.into_iter().next().unwrap(),
                    _ => call("+", args),
                },
                "*" => match (&args[0], &args[1]) {
                    (CompiledExpr::Float(f), _) if *f == 0.0 => float(0.0),
                    (_, CompiledExpr::Float(f)) if *f == 0.0 => float(0.0),
                    (CompiledExpr::Float(f), _) if *f == 1.0 => args.into_iter().nth(1).unwrap(),
                    (_, CompiledExpr::Float(f)) if *f == 1.0 => args.into_iter().next().unwrap(),
                    _ => call("*", args),
                },
                "-" => match (&args[0], &args[1]) {
                    (_, CompiledExpr::Float(f)) if *f == 0.0 => args.into_iter().next().unwrap(),
                    _ => call("-", args),
                },
                "transpose" | "tr" => {
                    // Detect: transpose(transpose(x)) → x (double transpose elimination)
                    if args.len() == 1
                        && let CompiledExpr::FunctionCall { name, args: inner_args, .. } = &args[0]
                        && (name == "transpose" || name == "tr")
                        && inner_args.len() == 1
                    {
                        // Double transpose cancels out
                        return inner_args[0].clone();
                    }
                    call(&name, args)
                },
                _ => call(&name, args),
            }
        }
        // Passthrough for all other variants
        other => other,
    }
}
