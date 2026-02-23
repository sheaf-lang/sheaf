// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Symbolic reverse-mode autodiff on `CompiledExpr`.

pub mod value_and_grad;

// `grad(expr, wrt)` returns a new `CompiledExpr` representing dL/d(wrt),
// assuming `expr` is the scalar loss (so the incoming gradient is 1.0).

use crate::core::compiler::{CompiledExpr, FunctionDef};
use std::collections::HashMap;

// helpers
fn call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
    CompiledExpr::FunctionCall {
        name: name.to_string(),
        args,
    }
}

fn float(v: f64) -> CompiledExpr {
    CompiledExpr::Float(v)
}

fn add(a: CompiledExpr, b: CompiledExpr) -> CompiledExpr {
    call("+", vec![a, b])
}

fn sub(a: CompiledExpr, b: CompiledExpr) -> CompiledExpr {
    call("-", vec![a, b])
}

fn mul(a: CompiledExpr, b: CompiledExpr) -> CompiledExpr {
    call("*", vec![a, b])
}

fn matmul(a: CompiledExpr, b: CompiledExpr) -> CompiledExpr {
    call("@", vec![a, b])
}

fn transpose(a: CompiledExpr) -> CompiledExpr {
    call("transpose", vec![a])
}

/// Inline let-bindings into the body by replacing symbols with their values.
fn substitute_bindings(body: &CompiledExpr, bindings: &[(String, CompiledExpr)]) -> CompiledExpr {
    let mut result = body.clone();
    for (name, value) in bindings {
        result = replace_symbol(&result, name, value);
    }
    result
}

fn replace_symbol(expr: &CompiledExpr, name: &str, replacement: &CompiledExpr) -> CompiledExpr {
    match expr {
        CompiledExpr::Symbol(s) if s == name => replacement.clone(),
        CompiledExpr::FunctionCall {
            name: fn_name,
            args,
        } => CompiledExpr::FunctionCall {
            name: fn_name.clone(),
            args: args
                .iter()
                .map(|a| replace_symbol(a, name, replacement))
                .collect(),
        },
        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<(String, CompiledExpr)> = bindings
                .iter()
                .map(|(k, v)| (k.clone(), replace_symbol(v, name, replacement)))
                .collect();
            // If a binding shadows the name, stop substituting in the body
            if bindings.iter().any(|(k, _)| k == name) {
                CompiledExpr::Let {
                    bindings: new_bindings,
                    body: body.clone(),
                }
            } else {
                CompiledExpr::Let {
                    bindings: new_bindings,
                    body: Box::new(replace_symbol(body, name, replacement)),
                }
            }
        }
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs
                .iter()
                .map(|e| replace_symbol(e, name, replacement))
                .collect(),
        ),
        other => other.clone(),
    }
}

//  simplify

/// Basic algebraic simplification to reduce the symbolic gradient expression.
///
/// Rules:
///   0 + x  →  x,  x + 0  →  x
///   0 * x  →  0,  x * 0  →  0
///   1 * x  →  x,  x * 1  →  x
pub fn simplify(expr: CompiledExpr) -> CompiledExpr {
    match expr {
        CompiledExpr::FunctionCall { name, args } => {
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
                _ => call(&name, args),
            }
        }
        // Passthrough for all other variants
        other => other,
    }
}

// grad

/// Compute the symbolic gradient of `expr` with respect to `wrt`.
///
/// `grad_output` is the upstream gradient (dL/d_expr). Pass `None` when
/// differentiating the loss itself — an implicit `1.0` is used.
///
/// Returns a `CompiledExpr` that can be fed to the code generator as-is.
pub fn grad(expr: &CompiledExpr, wrt: &str, grad_output: Option<CompiledExpr>) -> CompiledExpr {
    let g = grad_output.unwrap_or_else(|| float(1.0));
    grad_with(expr, wrt, g)
}

fn grad_with(expr: &CompiledExpr, wrt: &str, g: CompiledExpr) -> CompiledExpr {
    match expr {
        // Constants and irrelevant symbols → zero
        CompiledExpr::Float(_) | CompiledExpr::Integer(_) => float(0.0),

        CompiledExpr::Symbol(name) => {
            if name == wrt {
                g
            } else {
                float(0.0)
            }
        }

        // GetTupleElement represents a named parameter (e.g. W extracted from p)
        // Treat it like a variable: if it *is* `wrt`, the gradient is g; otherwise 0.
        CompiledExpr::GetTupleElement { param, .. } => {
            if param == wrt {
                g
            } else {
                float(0.0)
            }
        }

        CompiledExpr::FunctionCall { name, args } => grad_function_call(name, args, wrt, g),

        // Let: substitute bindings into body, then differentiate.
        CompiledExpr::Let { bindings, body } => {
            let expanded = substitute_bindings(body, bindings);
            grad_with(&expanded, wrt, g)
        }

        // Do: only the last expression matters
        CompiledExpr::Do(exprs) => {
            if let Some(last) = exprs.last() {
                grad_with(last, wrt, g)
            } else {
                float(0.0)
            }
        }

        _ => float(0.0),
    }
}

fn grad_function_call(
    name: &str,
    args: &[CompiledExpr],
    wrt: &str,
    g: CompiledExpr,
) -> CompiledExpr {
    match name {
        // Arithmetic
        "+" => {
            // d/dx (f + h) = df/dx + dh/dx
            let (lhs, rhs) = (&args[0], &args[1]);
            add(grad_with(lhs, wrt, g.clone()), grad_with(rhs, wrt, g))
        }

        "-" if args.len() == 2 => {
            // d/dx (f - h) = df/dx - dh/dx
            let (lhs, rhs) = (&args[0], &args[1]);
            sub(
                grad_with(lhs, wrt, g.clone()),
                grad_with(rhs, wrt, mul(float(-1.0), g)),
            )
        }

        "-" => {
            // Unary negation: d/dx (-f) = -df/dx
            grad_with(&args[0], wrt, mul(float(-1.0), g))
        }

        "*" => {
            // d/dx (f * h) = df/dx * h + f * dh/dx  (element-wise)
            let (lhs, rhs) = (&args[0], &args[1]);
            let g_lhs = mul(g.clone(), rhs.clone());
            let g_rhs = mul(g, lhs.clone());
            add(grad_with(lhs, wrt, g_lhs), grad_with(rhs, wrt, g_rhs))
        }

        "/" if args.len() == 2 => {
            // d/dx (f / h) = df/dx / h  (assuming h doesn't depend on wrt)
            let (lhs, rhs) = (&args[0], &args[1]);
            let g_lhs = call("/", vec![g.clone(), rhs.clone()]);
            // dh/dx term: -f / h^2 * dh/dx (usually h is constant w.r.t. wrt)
            let g_rhs_upstream = mul(
                float(-1.0),
                call(
                    "/",
                    vec![lhs.clone(), call("*", vec![rhs.clone(), rhs.clone()])],
                ),
            );
            add(
                grad_with(lhs, wrt, g_lhs),
                grad_with(rhs, wrt, mul(g, g_rhs_upstream)),
            )
        }

        // Matrix ops
        "@" => {
            // C = A @ B
            // dL/dA = dL/dC @ B^T
            // dL/dB = A^T @ dL/dC
            let (a, b) = (&args[0], &args[1]);
            let g_a = matmul(g.clone(), transpose(b.clone()));
            let g_b = matmul(transpose(a.clone()), g.clone());
            add(grad_with(a, wrt, g_a), grad_with(b, wrt, g_b))
        }

        "transpose" => {
            // d/dx transpose(f) = transpose(df/dx)
            grad_with(&args[0], wrt, transpose(g))
        }

        // Activations
        "relu" => {
            // d/dx relu(f) ≈ df/dx  (simplified; full version multiplies by (f > 0))
            grad_with(&args[0], wrt, g)
        }

        "sigmoid" => {
            // d/dx sigmoid(f) = sigmoid(f) * (1 - sigmoid(f)) * df/dx
            let sig = call("sigmoid", vec![args[0].clone()]);
            let local_g = mul(sig.clone(), sub(float(1.0), sig));
            grad_with(&args[0], wrt, mul(g, local_g))
        }

        "exp" => {
            // d/dx exp(f) = exp(f) * df/dx
            let local_g = call("exp", vec![args[0].clone()]);
            grad_with(&args[0], wrt, mul(g, local_g))
        }

        "log" => {
            // d/dx log(f) = (1/f) * df/dx
            let local_g = call("/", vec![float(1.0), args[0].clone()]);
            grad_with(&args[0], wrt, mul(g, local_g))
        }

        // Reductions
        "mean" => {
            // d/dx mean(f) = (1/N) * df/dx (broadcast back)
            // Simplified: pass gradient through (codegen handles broadcast)
            grad_with(&args[0], wrt, g)
        }

        "sum" => {
            // d/dx sum(f) = df/dx * ones (broadcast back)
            // Simplified: pass gradient through
            grad_with(&args[0], wrt, g)
        }

        "softmax" => {
            // d/dx softmax(f): complex, approximated as pass-through for now
            // Full Jacobian: diag(s) - s*s^T where s = softmax(f)
            grad_with(&args[0], wrt, g)
        }

        // Power
        "**" if args.len() == 2 => {
            // d/dx (f^n) = n * f^(n-1) * df/dx
            let (base, exp) = (&args[0], &args[1]);
            if let CompiledExpr::Float(n) = exp {
                let local_g = mul(float(*n), call("**", vec![base.clone(), float(n - 1.0)]));
                grad_with(base, wrt, mul(g, local_g))
            } else {
                // General case: d/dx f^g = f^g * (g/f * df/dx + log(f) * dg/dx)
                grad_with(base, wrt, g)
            }
        }

        // Unknown
        _ => float(0.0),
    }
}

/// Compute gradient and simplify in one step.
pub fn grad_simplified(expr: &CompiledExpr, wrt: &str) -> CompiledExpr {
    let g = grad(expr, wrt, None);
    simplify(g)
}

/// Inline user-defined function calls so that autodiff can see through them.
///
/// Replaces `FunctionCall("f", [a, b])` where `f` is in `registry` with:
///   `Let { bindings: [(p1, a), (p2, b)], body: f.body_compiled }`
///
/// Recurses into the result (the inlined body may itself contain calls).
/// `depth` guards against infinite recursion (mutual/self-recursive functions).
pub fn inline_function_calls(
    expr: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
) -> CompiledExpr {
    inline_calls_rec(expr, registry, 0)
}

const MAX_INLINE_DEPTH: usize = 16;

fn inline_calls_rec(
    expr: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
    depth: usize,
) -> CompiledExpr {
    if depth > MAX_INLINE_DEPTH {
        return expr.clone();
    }

    match expr {
        CompiledExpr::FunctionCall { name, args } => {
            // First, inline in arguments
            let inlined_args: Vec<CompiledExpr> = args
                .iter()
                .map(|a| inline_calls_rec(a, registry, depth))
                .collect();

            // Try to inline this call if it's a user-defined function
            if let Some(func_def) = registry.get(name.as_str()) {
                if let Some(body) = &func_def.body_compiled {
                    let bindings: Vec<(String, CompiledExpr)> = func_def
                        .params
                        .iter()
                        .zip(inlined_args.iter())
                        .map(|(p, a)| (p.clone(), a.clone()))
                        .collect();
                    let inlined = CompiledExpr::Let {
                        bindings,
                        body: Box::new(body.clone()),
                    };
                    // Recurse into the inlined body
                    return inline_calls_rec(&inlined, registry, depth + 1);
                }
            }

            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: inlined_args,
            }
        }

        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<(String, CompiledExpr)> = bindings
                .iter()
                .map(|(k, v)| (k.clone(), inline_calls_rec(v, registry, depth)))
                .collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(inline_calls_rec(body, registry, depth)),
            }
        }

        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs
                .iter()
                .map(|e| inline_calls_rec(e, registry, depth))
                .collect(),
        ),

        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(inline_calls_rec(condition, registry, depth)),
            then_branch: Box::new(inline_calls_rec(then_branch, registry, depth)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(inline_calls_rec(e, registry, depth))),
        },

        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(inline_calls_rec(body, registry, depth)),
        },

        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(inline_calls_rec(callee, registry, depth)),
            args: args
                .iter()
                .map(|a| inline_calls_rec(a, registry, depth))
                .collect(),
        },

        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems
                .iter()
                .map(|e| inline_calls_rec(e, registry, depth))
                .collect(),
        ),

        // Leaves: no recursion needed
        _ => expr.clone(),
    }
}

/// Common Subexpression Elimination.
///
/// Traverses the expression tree, finds structurally identical non-trivial
/// sub-expressions that appear more than once, and hoists them into `Let`
/// bindings so they are computed only once.
///
/// A sub-expression is "non-trivial" if it is a `FunctionCall` (not a leaf).
pub fn cse(expr: CompiledExpr) -> CompiledExpr {
    use std::collections::HashMap;

    // Count occurrences of each sub-expression (keyed by Debug string).
    let mut counts: HashMap<String, usize> = HashMap::new();
    count_exprs(&expr, &mut counts);

    // Collect sub-expressions that appear more than once, in discovery order.
    let mut seen_keys: Vec<String> = Vec::new();
    let mut bindings: Vec<(String, CompiledExpr)> = Vec::new();
    let mut subst: HashMap<String, String> = HashMap::new(); // key → binding name

    collect_cse_candidates(&expr, &counts, &mut seen_keys, &mut bindings, &mut subst);

    if bindings.is_empty() {
        return expr;
    }

    // Substitute repeated sub-expressions with their binding names.
    let body = substitute(expr, &subst);

    CompiledExpr::Let {
        bindings,
        body: Box::new(body),
    }
}

fn expr_key(expr: &CompiledExpr) -> String {
    format!("{:?}", expr)
}

fn is_trivial(expr: &CompiledExpr) -> bool {
    matches!(
        expr,
        CompiledExpr::Symbol(_)
            | CompiledExpr::Float(_)
            | CompiledExpr::Integer(_)
            | CompiledExpr::GetTupleElement { .. }
    )
}

fn count_exprs(expr: &CompiledExpr, counts: &mut std::collections::HashMap<String, usize>) {
    if is_trivial(expr) {
        return;
    }
    let key = expr_key(expr);
    *counts.entry(key).or_insert(0) += 1;

    match expr {
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                count_exprs(a, counts);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                count_exprs(v, counts);
            }
            count_exprs(body, counts);
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                count_exprs(e, counts);
            }
        }
        _ => {}
    }
}

fn collect_cse_candidates(
    expr: &CompiledExpr,
    counts: &std::collections::HashMap<String, usize>,
    seen_keys: &mut Vec<String>,
    bindings: &mut Vec<(String, CompiledExpr)>,
    subst: &mut std::collections::HashMap<String, String>,
) {
    if is_trivial(expr) {
        return;
    }
    let key = expr_key(expr);
    if counts.get(&key).copied().unwrap_or(0) > 1 {
        if !seen_keys.contains(&key) {
            seen_keys.push(key.clone());
            let name = format!("__cse{}", bindings.len());
            subst.insert(key, name.clone());
            bindings.push((name, expr.clone()));
        }
        // Don't recurse into already-hoisted expressions.
        return;
    }

    match expr {
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                collect_cse_candidates(a, counts, seen_keys, bindings, subst);
            }
        }
        CompiledExpr::Let { bindings: b, body } => {
            for (_, v) in b {
                collect_cse_candidates(v, counts, seen_keys, bindings, subst);
            }
            collect_cse_candidates(body, counts, seen_keys, bindings, subst);
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                collect_cse_candidates(e, counts, seen_keys, bindings, subst);
            }
        }
        _ => {}
    }
}

fn substitute(
    expr: CompiledExpr,
    subst: &std::collections::HashMap<String, String>,
) -> CompiledExpr {
    if is_trivial(&expr) {
        return expr;
    }
    let key = expr_key(&expr);
    if let Some(name) = subst.get(&key) {
        return CompiledExpr::Symbol(name.clone());
    }
    match expr {
        CompiledExpr::FunctionCall { name, args } => {
            let args = args.into_iter().map(|a| substitute(a, subst)).collect();
            CompiledExpr::FunctionCall { name, args }
        }
        CompiledExpr::Let { bindings, body } => {
            let bindings = bindings
                .into_iter()
                .map(|(k, v)| (k, substitute(v, subst)))
                .collect();
            CompiledExpr::Let {
                bindings,
                body: Box::new(substitute(*body, subst)),
            }
        }
        CompiledExpr::Do(exprs) => {
            CompiledExpr::Do(exprs.into_iter().map(|e| substitute(e, subst)).collect())
        }
        other => other,
    }
}
