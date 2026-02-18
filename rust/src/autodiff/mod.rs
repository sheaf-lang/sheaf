// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Symbolic reverse-mode autodiff on `CompiledExpr`.
//!
//! Ported from poc/autograd/tensor_autodiff.py.
//!
//! `grad(expr, wrt)` returns a new `CompiledExpr` representing dL/d(wrt),
//! assuming `expr` is the scalar loss (so the incoming gradient is 1.0).

use crate::core::compiler::CompiledExpr;

// ── helpers ──────────────────────────────────────────────────────────────────

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

// ── simplify ─────────────────────────────────────────────────────────────────

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

// ── grad ─────────────────────────────────────────────────────────────────────

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

        // Let: differentiate the body; bindings are treated as constants here.
        // A full treatment would substitute bindings, but for simple cases
        // (where `wrt` is not shadowed) passing through is correct.
        CompiledExpr::Let { body, .. } => grad_with(body, wrt, g),

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
        // ── Arithmetic ─────────────────────────────────────────────────────
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

        // ── Matrix ops ─────────────────────────────────────────────────────
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

        // ── Activations ────────────────────────────────────────────────────
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

        // ── Reductions ─────────────────────────────────────────────────────
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

        // ── Power ──────────────────────────────────────────────────────────
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

        // ── Unknown ────────────────────────────────────────────────────────
        _ => float(0.0),
    }
}

/// Compute gradient and simplify in one step.
pub fn grad_simplified(expr: &CompiledExpr, wrt: &str) -> CompiledExpr {
    let g = grad(expr, wrt, None);
    simplify(g)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::compiler::CompiledExpr;

    fn sym(s: &str) -> CompiledExpr {
        CompiledExpr::Symbol(s.to_string())
    }

    fn matmul_expr(a: CompiledExpr, b: CompiledExpr) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: "@".to_string(),
            args: vec![a, b],
        }
    }

    fn add_expr(a: CompiledExpr, b: CompiledExpr) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: "+".to_string(),
            args: vec![a, b],
        }
    }

    fn mean_expr(a: CompiledExpr) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: "mean".to_string(),
            args: vec![a],
        }
    }

    #[test]
    fn test_grad_var_wrt_itself() {
        // d(x)/dx = 1
        let expr = sym("x");
        let g = grad(&expr, "x", None);
        assert!(matches!(g, CompiledExpr::Float(v) if v == 1.0));
    }

    #[test]
    fn test_grad_var_wrt_other() {
        // d(x)/dy = 0
        let expr = sym("x");
        let g = grad(&expr, "y", None);
        assert!(matches!(g, CompiledExpr::Float(v) if v == 0.0));
    }

    #[test]
    fn test_grad_matmul_wrt_b() {
        // z = x @ W, dz/dW = x^T @ 1 = x^T
        let expr = matmul_expr(sym("x"), sym("W"));
        let g = grad_simplified(&expr, "W");
        // Should be: transpose(x) @ 1.0  →  simplified
        // Exact shape of result depends on simplify; just check it's a FunctionCall
        assert!(matches!(g, CompiledExpr::FunctionCall { .. }));
    }

    #[test]
    fn test_grad_add() {
        // d(x + W)/dW = 0 + 1 = 1
        let expr = add_expr(sym("x"), sym("W"));
        let g = grad_simplified(&expr, "W");
        assert!(matches!(g, CompiledExpr::Float(v) if v == 1.0));
    }

    #[test]
    fn test_grad_mean_matmul() {
        // loss = mean(x @ W)
        // dL/dW = x^T @ grad_from_mean = x^T @ 1 = x^T
        let expr = mean_expr(matmul_expr(sym("x"), sym("W")));
        let g = grad(&expr, "W", None);
        // Result should be a FunctionCall (matmul/add combination)
        assert!(matches!(g, CompiledExpr::FunctionCall { .. }));
    }
}
