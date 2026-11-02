// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Symbolic reverse-mode autodiff on `CompiledExpr`.

pub mod analysis;
pub mod reverse;
pub mod trace;
pub mod transforms;

// Re-export public API from submodules.
pub use analysis::{collect_free_vars, contains_undiffable_ops, find_undiffable_ops};
pub use transforms::{fold_dict_gets, inline_function_calls};

// `grad(expr, wrt)` returns a new `CompiledExpr` representing dL/d(wrt),
// assuming `expr` is the scalar loss (so the incoming gradient is 1.0).

use crate::autodiff::analysis::depends_on;
use crate::core::error::{SheafError, SheafResult};
use crate::core::expr::{BindingPattern, CompiledExpr};

// helpers
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

fn is_zero(expr: &CompiledExpr) -> bool {
    matches!(expr, CompiledExpr::Float(v) if *v == 0.0)
        || matches!(expr, CompiledExpr::Integer(0))
}

fn unsupported_gradient(
    expr: &CompiledExpr,
    wrt: &str,
    operation: &str,
    location: Option<crate::core::error::SourceLocation>,
) -> SheafResult<CompiledExpr> {
    if depends_on(expr, wrt) {
        Err(SheafError::AutodiffMissingRule {
            operation: operation.to_string(),
            location,
        })
    } else {
        Ok(float(0.0))
    }
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

fn transpose(a: CompiledExpr) -> CompiledExpr {
    call("transpose", vec![a])
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

//  simplify

/// Basic algebraic simplification to reduce the symbolic gradient expression.
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

// grad

/// Compute the symbolic gradient of `expr` with respect to `wrt`.
///
/// `grad_output` is the upstream gradient (dL/d_expr). Pass `None` when
/// differentiating the loss itself: an implicit `1.0` is used.
///
/// Returns a `CompiledExpr` that can be fed to the code generator as-is.
pub fn grad(
    expr: &CompiledExpr,
    wrt: &str,
    grad_output: Option<CompiledExpr>,
) -> SheafResult<CompiledExpr> {
    let g = grad_output.unwrap_or_else(|| float(1.0));
    grad_with(expr, wrt, g)
}

fn grad_with(expr: &CompiledExpr, wrt: &str, g: CompiledExpr) -> SheafResult<CompiledExpr> {
    if is_zero(&g) {
        return Ok(float(0.0));
    }

    let result = match expr {
        // Constants and irrelevant symbols -> zero
        CompiledExpr::Float(_) | CompiledExpr::Integer(_) | CompiledExpr::GlobalRef(_) => float(0.0),

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

        CompiledExpr::FunctionCall { name, args, loc } => {
            grad_function_call(name, args, wrt, g, loc.clone())?
        }

        // Let: forward-mode AD through bindings without exponential expansion.
        //
        // d/dwrt Let { x1=e1, ..., xn=en, body }
        // = Let { x1=e1, ..., xn=en,            -- forward values
        //         dx1 = de1/dwrt,                -- gradient seed
        // Reverse-mode AD through Let bindings.
        //
        // 1. Compute upstream gradient for each binding: dL/dxi
        // 2. Back-propagate each dL/dxi through its value expression
        //
        // Previous approach used float(1.0) as seed for per-binding Jacobians,
        // which breaks for matrix-valued bindings (matmul with scalar seed).
        // This approach threads the actual upstream gradient through each binding.
        CompiledExpr::Let { bindings, body } => {
            // Split shadowed bindings into nested Lets before differentiating.
            // as-> produces flat Lets like [(h, x), (h, f(h))] which confuse
            // gradient accumulation (duplicate variable names).
            if has_shadowed_bindings(bindings) {
                let nested = unshadow_let(bindings, body);
                return grad_with(&nested, wrt, g);
            }

            // Handle self-referencing bindings: h = f(h) where h comes from
            // outer scope. Rename the outer reference to __pre_h.
            let mut all_bindings: Vec<(BindingPattern, CompiledExpr)> = Vec::new();
            let effective_bindings: Vec<(String, CompiledExpr)> = bindings
                .iter()
                .filter_map(|(name, value)| {
                    let name_str = name.as_simple()?;
                    Some((name_str.to_string(), value.clone()))
                })
                .map(|(name, value)| {
                    if expr_contains_symbol(&value, &name) {
                        let alias = format!("__pre_{}", name);
                        all_bindings.push((BindingPattern::Simple(alias.clone()), CompiledExpr::Symbol(name.clone())));
                        let renamed = replace_symbol(&value, &name, &CompiledExpr::Symbol(alias));
                        all_bindings.push((BindingPattern::Simple(name.clone()), renamed.clone()));
                        (name, renamed)
                    } else {
                        all_bindings.push((BindingPattern::Simple(name.clone()), value.clone()));
                        (name, value)
                    }
                })
                .collect();

            let binding_names: Vec<&str> = effective_bindings
                .iter()
                .map(|(n, _)| n.as_str())
                .collect();
            let n = effective_bindings.len();

            // 1: Compute upstream gradient for each binding.
            // upstream[i] = dL/dxi, accumulated from body and later bindings.
            let mut upstream: Vec<CompiledExpr> = (0..n)
                .map(|i| grad_with(body, binding_names[i], g.clone()))
                .collect::<SheafResult<_>>()?;

            // Back-propagate through later bindings (reverse order)
            for j in (0..n).rev() {
                for i in 0..j {
                    let contrib = grad_with(
                        &effective_bindings[j].1,
                        binding_names[i],
                        upstream[j].clone(),
                    )?;
                    if !is_zero(&contrib) {
                        upstream[i] = add(upstream[i].clone(), contrib);
                    }
                }
            }

            // 2: Compute total gradient wrt `wrt`.
            //
            // If wrt is a binding name, body references go through the binding
            // (not directly to the outer variable), so the direct body term is 0.
            // The gradient flows only through the alias chain.
            let is_bound = binding_names.contains(&wrt);
            let mut total = if is_bound {
                float(0.0)
            } else {
                grad_with(body, wrt, g)?
            };

            for i in 0..n {
                let contrib = grad_with(
                    &effective_bindings[i].1,
                    wrt,
                    upstream[i].clone(),
                )?;
                if !is_zero(&contrib) {
                    total = add(total, contrib);
                }
            }

            // Propagate gradient through aliases back to outer scope
            for j in 0..n {
                let outer_name = bindings[j].0.as_simple();
                if outer_name.is_none() { continue; }
                let outer_name = outer_name.unwrap();
                if expr_contains_symbol(&bindings[j].1, outer_name) {
                    let alias = format!("__pre_{}", outer_name);
                    let contrib = grad_with(
                        &effective_bindings[j].1,
                        &alias,
                        upstream[j].clone(),
                    )?;
                    if !is_zero(&contrib) {
                        // __pre_h = Symbol(h_outer). Propagate: if wrt matches
                        // the outer name, the gradient flows through.
                        if outer_name == wrt {
                            total = add(total, contrib);
                        } else {
                            // For other wrt, propagate through the alias source
                            let outer_contrib = grad_with(
                                &CompiledExpr::Symbol(outer_name.to_string()),
                                wrt,
                                contrib,
                            )?;
                            if !is_zero(&outer_contrib) {
                                all_bindings.push((
                                    BindingPattern::Simple(format!("__d_alias_{}", outer_name)),
                                    simplify(outer_contrib.clone()),
                                ));
                                total = add(total, CompiledExpr::Symbol(
                                    format!("__d_alias_{}", outer_name),
                                ));
                            }
                        }
                    }
                }
            }

            CompiledExpr::Let {
                bindings: all_bindings,
                body: Box::new(simplify(total)),
            }
        }

        // Do: only the last expression matters
        CompiledExpr::Do(exprs) => {
            if let Some(last) = exprs.last() {
                grad_with(last, wrt, g)?
            } else {
                float(0.0)
            }
        }

        CompiledExpr::Boolean(_) => float(0.0),
        CompiledExpr::Nil => float(0.0),
        CompiledExpr::String(_) => float(0.0),
        CompiledExpr::Keyword(_) => float(0.0),
        CompiledExpr::Quoted(_) => float(0.0),
        CompiledExpr::FunctionRef(_) => float(0.0),
        CompiledExpr::Vector(_) => return unsupported_gradient(expr, wrt, "vector", None),
        CompiledExpr::Tuple(_) => return unsupported_gradient(expr, wrt, "tuple", None),
        CompiledExpr::Dict(_) => return unsupported_gradient(expr, wrt, "dict", None),
        CompiledExpr::If { .. } => return unsupported_gradient(expr, wrt, "if", None),
        CompiledExpr::Lambda { .. } => return unsupported_gradient(expr, wrt, "lambda", None),
        CompiledExpr::LambdaCall { .. } => {
            return unsupported_gradient(expr, wrt, "lambda-call", None);
        }
        CompiledExpr::ValueAndGrad { .. } => {
            return unsupported_gradient(expr, wrt, "value-and-grad", None);
        }
        CompiledExpr::Repeat { .. } => return unsupported_gradient(expr, wrt, "repeat", None),
        CompiledExpr::While { .. } => return unsupported_gradient(expr, wrt, "while", None),
        CompiledExpr::Guard { .. } => return unsupported_gradient(expr, wrt, "guard", None),
        CompiledExpr::Def { .. } => return unsupported_gradient(expr, wrt, "def", None),
    };
    Ok(result)
}

/// Check if a Let has duplicate binding names (e.g. from as-> threading).
fn has_shadowed_bindings(bindings: &[(BindingPattern, CompiledExpr)]) -> bool {
    let mut seen = std::collections::HashSet::new();
    bindings.iter().any(|(name, _)| {
        if let BindingPattern::Simple(n) = name {
            !seen.insert(n.as_str())
        } else {
            false
        }
    })
}

/// Convert a flat Let with shadowed bindings into nested Lets.
/// Splits at the first duplicate: [a=e1, a=e2, b=e3] -> Let{a=e1, Let{a=e2, b=e3, body}}
fn unshadow_let(bindings: &[(BindingPattern, CompiledExpr)], body: &CompiledExpr) -> CompiledExpr {
    let mut seen = std::collections::HashSet::new();
    for (i, (name, _)) in bindings.iter().enumerate() {
        if let BindingPattern::Simple(n) = name
            && !seen.insert(n.as_str()) 
        {
            let outer = bindings[..i].to_vec();
            let inner = unshadow_let(&bindings[i..], body);
            return CompiledExpr::Let {
                bindings: outer,
                body: Box::new(inner),
            };
        }
    }
    CompiledExpr::Let {
        bindings: bindings.to_vec(),
        body: Box::new(body.clone()),
    }
}

/// Check if an expression contains a free reference to Symbol(name).
fn expr_contains_symbol(expr: &CompiledExpr, name: &str) -> bool {
    match expr {
        CompiledExpr::Symbol(s) => s == name,
        CompiledExpr::FunctionCall { args, .. } => {
            args.iter().any(|a| expr_contains_symbol(a, name))
        }
        CompiledExpr::Let { bindings, body } => {
            for (k, v) in bindings {
                if expr_contains_symbol(v, name) {
                    return true;
                }
                if k.as_simple() == Some(name) {
                    return false; // shadowed from here on
                }
            }
            expr_contains_symbol(body, name)
        }
        CompiledExpr::Do(exprs) => exprs.iter().any(|e| expr_contains_symbol(e, name)),
        CompiledExpr::GetTupleElement { param, .. } => param == name,
        CompiledExpr::Vector(elems) => elems.iter().any(|e| expr_contains_symbol(e, name)),
        _ => false,
    }
}

fn grad_function_call(
    name: &str,
    args: &[CompiledExpr],
    wrt: &str,
    g: CompiledExpr,
    location: Option<crate::core::error::SourceLocation>,
) -> SheafResult<CompiledExpr> {
    let result = match name {
        // stop-gradient: forward is identity, backward is zero (blocks backprop)
        "stop-gradient" => float(0.0),

        // Arithmetic
        "+" => {
            // d/dx (f + h) = df/dx + dh/dx
            let (lhs, rhs) = (&args[0], &args[1]);
            add(grad_with(lhs, wrt, g.clone())?, grad_with(rhs, wrt, g)?)
        }

        "-" if args.len() == 2 => {
            // d/dx (f - h) = df/dx + d(-h)/dx
            // We pass -g as upstream to rhs (the negative side), then ADD.
            // Do NOT sub here: the negation is already in the upstream.
            let (lhs, rhs) = (&args[0], &args[1]);
            add(
                grad_with(lhs, wrt, g.clone())?,
                grad_with(rhs, wrt, mul(float(-1.0), g))?,
            )
        }

        "-" => {
            // Unary negation: d/dx (-f) = -df/dx
            grad_with(&args[0], wrt, mul(float(-1.0), g))?
        }

        "*" => {
            // d/dx (f * h) = df/dx * h + f * dh/dx  (element-wise)
            let (lhs, rhs) = (&args[0], &args[1]);
            let g_lhs = mul(g.clone(), rhs.clone());
            let g_rhs = mul(g, lhs.clone());
            add(grad_with(lhs, wrt, g_lhs)?, grad_with(rhs, wrt, g_rhs)?)
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
                grad_with(lhs, wrt, g_lhs)?,
                grad_with(rhs, wrt, mul(g, g_rhs_upstream))?,
            )
        }

        // Matrix ops
        "@" => {
            // C = A @ B
            // dL/dA = dL/dC @ B^T  (via @-grad-lhs)
            // dL/dB = A^T @ dL/dC  (via @-grad-rhs)
            // These builtins handle 1D edge cases with proper reshape.
            let (a, b) = (&args[0], &args[1]);
            let g_a = call("@-grad-lhs", vec![a.clone(), b.clone(), g.clone()]);
            let g_b = call("@-grad-rhs", vec![a.clone(), b.clone(), g.clone()]);
            add(grad_with(a, wrt, g_a)?, grad_with(b, wrt, g_b)?)
        }

        "einsum" => {
            // einsum("lhs_sub,rhs_sub->out_sub", A, B)
            // dL/dA = einsum("out_sub,rhs_sub->lhs_sub", grad, B)
            // dL/dB = einsum("lhs_sub,out_sub->rhs_sub", A, grad)
            //
            // When grad is scalar (e.g. from sum), we multiply after einsum
            // with identity subscripts to avoid shape mismatch.
            if args.len() == 3
                && let CompiledExpr::String(ref sub) = args[0]
                && let Some((inputs, output)) = sub.split_once("->")
            {
                let parts: Vec<&str> = inputs.split(',').collect();
                if parts.len() == 2 {
                    let lhs_sub = parts[0].trim();
                    let rhs_sub = parts[1].trim();
                    let out_sub = output.trim();

                    // For dA: use identity einsum on B to get lhs shape, then multiply by grad
                    let grad_lhs_sub = format!("{},{}->{}", out_sub, rhs_sub, lhs_sub);
                    let grad_rhs_sub = format!("{},{}->{}", lhs_sub, out_sub, rhs_sub);

                    // Broadcast grad to output shape: g * ones(einsum_output)
                    let fwd = call("einsum", vec![args[0].clone(), args[1].clone(), args[2].clone()]);
                    let g_broadcast = add(mul(fwd.clone(), CompiledExpr::Float(0.0)), g.clone());

                    let g_a = call("einsum", vec![
                        CompiledExpr::String(grad_lhs_sub),
                        g_broadcast.clone(),
                        args[2].clone(),
                    ]);
                    let g_b = call("einsum", vec![
                        CompiledExpr::String(grad_rhs_sub),
                        args[1].clone(),
                        g_broadcast,
                    ]);
                    return Ok(add(
                        grad_with(&args[1], wrt, g_a)?,
                        grad_with(&args[2], wrt, g_b)?,
                    ));
                }
            }
            let function_call = CompiledExpr::FunctionCall {
                name: name.to_string(),
                args: args.to_vec(),
                loc: location.clone(),
            };
            return unsupported_gradient(&function_call, wrt, name, location);
        }

        "transpose" | "tr" => {
            // d/dx transpose(f) = transpose(df/dx)
            grad_with(&args[0], wrt, transpose(g))?
        }

        // Activations
        "relu" => {
            // d/dx relu(f) = (f > 0) * df/dx
            let mask = call("where", vec![
                call(">", vec![args[0].clone(), float(0.0)]),
                float(1.0),
                float(0.0),
            ]);
            grad_with(&args[0], wrt, mul(g, mask))?
        }

        "maximum" => {
            // d/dx maximum(a, b): gradient flows to a where a >= b, to b otherwise
            // For relu case (maximum(x, 0)), this passes gradient through to x
            let g_a = call("where", vec![
                call(">=", vec![args[0].clone(), args[1].clone()]),
                g.clone(),
                float(0.0),
            ]);
            let g_b = call("where", vec![
                call(">=", vec![args[0].clone(), args[1].clone()]),
                float(0.0),
                g,
            ]);
            add(
                grad_with(&args[0], wrt, g_a)?,
                grad_with(&args[1], wrt, g_b)?,
            )
        }

        "minimum" => {
            // d/dx minimum(a, b): gradient flows to a where a <= b, to b otherwise
            let g_a = call("where", vec![
                call("<=", vec![args[0].clone(), args[1].clone()]),
                g.clone(),
                float(0.0),
            ]);
            let g_b = call("where", vec![
                call("<=", vec![args[0].clone(), args[1].clone()]),
                float(0.0),
                g,
            ]);
            add(
                grad_with(&args[0], wrt, g_a)?,
                grad_with(&args[1], wrt, g_b)?,
            )
        }

        "sigmoid" => {
            // d/dx sigmoid(f) = sigmoid(f) * (1 - sigmoid(f)) * df/dx
            let sig = call("sigmoid", vec![args[0].clone()]);
            let local_g = mul(sig.clone(), sub(float(1.0), sig));
            grad_with(&args[0], wrt, mul(g, local_g))?
        }

        "exp" => {
            // d/dx exp(f) = exp(f) * df/dx
            let local_g = call("exp", vec![args[0].clone()]);
            grad_with(&args[0], wrt, mul(g, local_g))?
        }

        "log" => {
            // d/dx log(f) = (1/f) * df/dx
            let local_g = call("/", vec![float(1.0), args[0].clone()]);
            grad_with(&args[0], wrt, mul(g, local_g))?
        }

        "sqrt" => {
            // d/dx sqrt(f) = 1/(2*sqrt(f)) * df/dx
            let local_g = call("/", vec![
                float(1.0),
                mul(float(2.0), call("sqrt", vec![args[0].clone()])),
            ]);
            grad_with(&args[0], wrt, mul(g, local_g))?
        }

        "tanh" => {
            // d/dx tanh(f) = (1 - tanh(f)^2) * df/dx
            let t = call("tanh", vec![args[0].clone()]);
            let local_g = sub(float(1.0), mul(t.clone(), t));
            grad_with(&args[0], wrt, mul(g, local_g))?
        }

        "gelu" => {
            // GELU tanh approximation: gelu(x) = 0.5 * x * (1 + tanh(k))
            // where k = sqrt(2/pi) * (x + 0.044715 * x^3)
            // gelu'(x) = 0.5*(1+tanh(k)) + 0.5*x*sech^2(k)*k'
            // k' = sqrt(2/pi) * (1 + 3*0.044715*x^2)
            let x = args[0].clone();
            let x2 = mul(x.clone(), x.clone());
            let x3 = mul(x2.clone(), x.clone());
            let k = mul(float(0.7978845608), add(x.clone(), mul(float(0.044715), x3)));
            let tanh_k = call("tanh", vec![k.clone()]);
            let sech2_k = sub(float(1.0), mul(tanh_k.clone(), tanh_k.clone()));
            let dk = mul(float(0.7978845608), add(float(1.0), mul(float(0.134145), x2)));
            let local_g = add(
                mul(float(0.5), add(float(1.0), tanh_k)),
                mul(mul(float(0.5), x), mul(sech2_k, dk)),
            );
            grad_with(&args[0], wrt, mul(g, local_g))?
        }

        "log-softmax" => {
            // d/dx log_softmax(f) = g - softmax(f) * sum(g)
            let sm = call("softmax", vec![args[0].clone(), CompiledExpr::Keyword("axis".to_string()), CompiledExpr::Integer(-1)]);
            let sum_g = call("sum", vec![g.clone(), CompiledExpr::Keyword("axis".to_string()), CompiledExpr::Integer(-1), CompiledExpr::Keyword("keepdims".to_string())]);
            let local_g = sub(g.clone(), mul(sm, sum_g));
            grad_with(&args[0], wrt, local_g)?
        }

        "reshape" => {
            // Flatten the gradient and let downstream ops handle shape alignment.
            grad_with(&args[0], wrt, g)?
        }

        "swapaxes" => {
            // Shape-manipulation ops: gradient passes through.
            grad_with(&args[0], wrt, g)?
        }

        "where" if args.len() == 3 => {
            // where(cond, a, b): gradient flows to a where cond is true, to b where false
            let g_a = call("where", vec![args[0].clone(), g.clone(), float(0.0)]);
            let g_b = call("where", vec![args[0].clone(), float(0.0), g]);
            add(
                grad_with(&args[1], wrt, g_a)?,
                grad_with(&args[2], wrt, g_b)?,
            )
        }

        "slice" if args.len() >= 3 => {
            // slice is a view: gradient flows through the slice
            // (simplified: treat as pass-through, codegen handles the shape)
            grad_with(&args[0], wrt, g)?
        }

        "neg" => {
            grad_with(&args[0], wrt, mul(float(-1.0), g))?
        }

        "abs" => {
            let local_g = call("where", vec![
                call(">=", vec![args[0].clone(), float(0.0)]),
                float(1.0),
                float(-1.0),
            ]);
            grad_with(&args[0], wrt, mul(g, local_g))?
        }

        // Reductions
        "mean" => {
            // d/dx mean(f) = (1/N) * df/dx
            let scaled_g = call("/", vec![g, call("len", vec![args[0].clone()])]);
            grad_with(&args[0], wrt, scaled_g)?
        }

        "sum" => {
            // d/dx sum(f) = df/dx (gradient broadcasts back automatically)
            grad_with(&args[0], wrt, g)?
        }

        "softmax" => {
            // d/dx softmax(f): g * s - s * sum(g * s)
            // where s = softmax(f)
            let sm = call("softmax", vec![args[0].clone(), CompiledExpr::Keyword("axis".to_string()), CompiledExpr::Integer(-1)]);
            let gs = mul(g, sm.clone());
            let sum_gs = call("sum", vec![gs.clone(), CompiledExpr::Keyword("axis".to_string()), CompiledExpr::Integer(-1), CompiledExpr::Keyword("keepdims".to_string())]);
            let local_g = sub(gs, mul(sm, sum_gs));
            grad_with(&args[0], wrt, local_g)?
        }

        // Cross-entropy loss
        "cross-entropy-loss" if args.len() == 2 => {
            // d/d(logits) cross-entropy-loss(logits, targets) = softmax(logits) - one_hot(targets, V)
            // Scaled by -1/N from the mean in the loss definition.
            // Since cross-entropy-loss = -mean(sum(one_hot * log_softmax)),
            // the gradient w.r.t. logits is: (softmax(logits) - one_hot(targets)) / N
            let sm = call("softmax", vec![args[0].clone(), CompiledExpr::Keyword("axis".to_string()), CompiledExpr::Integer(-1)]);
            let n_classes = call("last", vec![call("shape", vec![args[0].clone()])]);
            let oh = call("one-hot", vec![args[1].clone(), n_classes]);
            let local_g = sub(sm, oh);
            grad_with(&args[0], wrt, mul(g, local_g))?
        }

        // Power
        "**" if args.len() == 2 => {
            // d/dx (f^n) = n * f^(n-1) * df/dx
            let (base, exp) = (&args[0], &args[1]);
            let n = match exp {
                CompiledExpr::Float(n) => Some(*n),
                CompiledExpr::Integer(n) => Some(*n as f64),
                _ => None,
            };
            if let Some(n) = n {
                let local_g = mul(float(n), call("**", vec![base.clone(), float(n - 1.0)]));
                grad_with(base, wrt, mul(g, local_g))?
            } else {
                let function_call = CompiledExpr::FunctionCall {
                    name: name.to_string(),
                    args: args.to_vec(),
                    loc: location.clone(),
                };
                return unsupported_gradient(&function_call, wrt, name, location);
            }
        }

        // An unknown operation can be treated as constant only after its
        // arguments are proven independent of the differentiated value.
        _ if args.iter().any(|arg| depends_on(arg, wrt)) => {
            return Err(SheafError::AutodiffMissingRule {
                operation: name.to_string(),
                location,
            });
        }
        _ => float(0.0),
    };
    Ok(result)
}

/// Compute gradient and simplify in one step.
pub fn grad_simplified(expr: &CompiledExpr, wrt: &str) -> SheafResult<CompiledExpr> {
    let g = grad(expr, wrt, None)?;
    Ok(simplify(g))
}

#[cfg(test)]
mod tests {
    use super::{call, float, grad_simplified, CompiledExpr, SheafError};

    fn missing_rule(error: SheafError, operation: &str) {
        match error {
            SheafError::AutodiffMissingRule {
                operation: actual, ..
            } => assert_eq!(actual, operation),
            other => panic!("expected missing gradient rule, got {other:?}"),
        }
    }

    #[test]
    fn rejects_dependent_unknown_operation() {
        let expr = call(
            "sum",
            vec![call(
                "__missing-gradient-rule__",
                vec![CompiledExpr::Symbol("x".into())],
            )],
        );
        missing_rule(
            grad_simplified(&expr, "x").unwrap_err(),
            "__missing-gradient-rule__",
        );
    }

    #[test]
    fn rejects_nonliteral_power_exponent() {
        let expr = call(
            "**",
            vec![
                CompiledExpr::Symbol("x".into()),
                CompiledExpr::Symbol("exponent".into()),
            ],
        );
        missing_rule(grad_simplified(&expr, "x").unwrap_err(), "**");
    }

    #[test]
    fn accepts_constant_unknown_operation() {
        let expr = call(
            "sum",
            vec![call("__missing-gradient-rule__", vec![float(1.0)])],
        );
        assert!(matches!(
            grad_simplified(&expr, "x").unwrap(),
            CompiledExpr::Float(0.0)
        ));
    }

    #[test]
    fn preserves_stop_gradient_zero() {
        let expr = call(
            "sum",
            vec![call(
                "stop-gradient",
                vec![CompiledExpr::Symbol("x".into())],
            )],
        );
        assert!(matches!(
            grad_simplified(&expr, "x").unwrap(),
            CompiledExpr::Float(0.0)
        ));
    }

    #[test]
    fn skips_dead_let_binding() {
        let expr = CompiledExpr::Let {
            bindings: vec![(
                crate::core::expr::BindingPattern::Simple("unused".into()),
                call("__missing-gradient-rule__", vec![CompiledExpr::Symbol("x".into())]),
            )],
            body: Box::new(float(1.0)),
        };
        let gradient = grad_simplified(&expr, "x").unwrap();
        assert!(matches!(
            gradient,
            CompiledExpr::Let { body, .. } if matches!(*body, CompiledExpr::Float(0.0))
        ));
    }

    #[test]
    fn skips_dead_do_expression() {
        let expr = CompiledExpr::Do(vec![
            call("__missing-gradient-rule__", vec![CompiledExpr::Symbol("x".into())]),
            float(1.0),
        ]);
        assert!(matches!(
            grad_simplified(&expr, "x").unwrap(),
            CompiledExpr::Float(0.0)
        ));
    }

    #[test]
    fn retains_existing_rules() {
        let expr = call("sum", vec![call("exp", vec![CompiledExpr::Symbol("x".into())])]);
        assert!(matches!(
            grad_simplified(&expr, "x").unwrap(),
            CompiledExpr::FunctionCall { ref name, ref args, .. }
                if name == "exp" && matches!(args.as_slice(), [CompiledExpr::Symbol(symbol)] if symbol == "x")
        ));
    }

    #[test]
    fn rejects_dependent_unsupported_constructs() {
        let x = || CompiledExpr::Symbol("x".into());
        let constructs = vec![
            (CompiledExpr::If {
                condition: Box::new(CompiledExpr::Boolean(true)),
                then_branch: Box::new(x()),
                else_branch: None,
            }, "if"),
            (CompiledExpr::Tuple(vec![x()]), "tuple"),
            (
                CompiledExpr::Dict(vec![(CompiledExpr::Keyword("x".into()), x())]),
                "dict",
            ),
            (CompiledExpr::LambdaCall {
                callee: Box::new(CompiledExpr::lambda(vec!["y".into()], CompiledExpr::Symbol("y".into()))),
                args: vec![x()],
            }, "lambda-call"),
        ];

        for (expr, operation) in constructs {
            missing_rule(grad_simplified(&expr, "x").unwrap_err(), operation);
        }
    }

}
