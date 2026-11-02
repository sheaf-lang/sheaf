// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Value-and-grad: automatic differentiation.

use crate::core::expr::CompiledExpr;
use crate::core::error::SheafError;
use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::Value;
use ndarray::{ArrayD, IxDyn};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::{eval, call_function};

/// (value-and-grad f) -> returns a function that, given params, returns [loss, grad_params].
pub(super) fn eval_value_and_grad_hof(args: &[Value], _env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 1 {
        return Err(runtime_error("value-and-grad: expected exactly 1 argument (the function)"));
    }
    let func = args[0].clone();

    Ok(Value::Function {
        kind: super::value::FunctionKind::ValueAndGrad,
        name: None,
        params: vec!["__vag_params__".to_string()],
        body: crate::core::expr::CompiledExpr::Symbol("__vag_params__".to_string()).into(),
        closure: vec![("__vag_fn__".to_string(), func)],
    })
}

/// Evaluate a value-and-grad HOF call.
///
/// Tries JIT first. Falls back to tracing (resolves non-tensor ops, then
/// symbolic AD) for functions the JIT cannot compile (e.g. string captures).
pub(super) fn eval_value_and_grad_call(func: &Value, params: &Value, env: &mut Env) -> Result<Value, SheafError> {
    let prepared = super::specialize::prepare_vag(func, env)?.ready();
    let func = prepared.as_ref().unwrap_or(func);
    #[cfg(iree_runtime)]
    if env.tracer.is_none() && prepared.is_some() {
        use crate::runtime::jit::JitVagOutcome;
        match super::iree_dispatch::try_jit_vag(func, params, env) {
            JitVagOutcome::Success(result) => return result,
            JitVagOutcome::Unsupported => {}
            JitVagOutcome::Bug(reason) => {
                return Err(runtime_error(format!(
                    "value-and-grad: JIT compilation failed: {} \
                     (run with -vv for details). This is a bug in Sheaf.",
                    reason
                )));
            }
        }
    }

    use crate::autodiff::{collect_free_vars, contains_undiffable_ops, grad_simplified, inline_function_calls};

    if let Value::Function { params: fn_params, body, closure, .. } = func
        && fn_params.len() == 1 {
            let param_name = &fn_params[0];

        let mut aug_registry = env.registry.clone();

        // Collect free variables and function call targets from the body and
        // augment from the env so that inlining can resolve calls like (model x p).
        let mut free_set = std::collections::HashSet::new();
        collect_free_vars(body, &mut free_set);
        let call_names = crate::lowering::call_graph::direct_callees(body);
        // Function calls that are NOT in the registry may be closure variables
        for name in &call_names {
            if !env.registry.contains_key(name.as_str()) {
                free_set.insert(name.clone());
            }
        }
        for p in fn_params {
            free_set.remove(p.as_str());
        }
        for (k, _) in closure {
            free_set.remove(k.as_str());
        }
        let mut free: Vec<&str> = free_set.iter().map(|s| s.as_str()).collect();
        free.sort();

        // Build augmented closure: existing closure + free vars from env.
        // Only capture user-defined values (tensors, functions, etc.).
        // Builtins (BuiltinFn) and registry defns are already resolvable.
        let mut aug_closure = closure.clone();
        for name in &free {
            if !env.registry.contains_key(*name)
                && let Ok(val) = env.get(name)
                && !matches!(val, Value::BuiltinFn { .. }) {
                        aug_closure.push((name.to_string(), val.clone()));
                    }
        }

        for (name, val) in &aug_closure {
            if let Value::Function { params: fp, body: fb, .. } = val {
                aug_registry.entry(name.clone()).or_insert_with(|| {
                    crate::core::expr::FunctionDef {
                        name: name.clone(),
                        params: fp.clone(),
                        body: crate::core::ast::SheafValue::Nil(
                            crate::core::error::SourceLocation::new(0, 0, "".into()),
                        ),
                        body_compiled: Some(fb.clone()),
                        requires_initialization: false,
                        signature: None,
                        vmfb_module_name: None,
                        known_param_types: Vec::new(),
                        compile_error: None,
                    }
                });
            }
        }
        let inlined = inline_function_calls(body, &aug_registry);

        // Direct symbolic path for pure-tensor expressions
        if !contains_undiffable_ops(&inlined) {
            env.push_scope();
            for (name, val) in &aug_closure {
                env.set(name, val.clone());
            }
            env.set(param_name, params.clone());

            let loss_val = eval(&inlined, env)?;
            let loss = scalar_from_value(&loss_val)?;
            let grad_expr = match grad_simplified(&inlined, param_name) {
                Ok(expr) => expr,
                Err(error) => {
                    env.pop_scope();
                    return Err(error);
                }
            };
            let grad_val = eval(&grad_expr, env)?;

            env.pop_scope();
            return Ok(Value::List(vec![Value::Float(loss), grad_val]));
        }

        // Tracing path: evaluate non-tensor ops (string gets, comparisons)
        // with the interpreter, keep tensor ops symbolic, then differentiate.
        {
            use crate::autodiff::trace::{trace_expr, LeafMap};

            env.push_scope();
            for (name, val) in &aug_closure {
                env.set(name, val.clone());
            }
            env.set(param_name, params.clone());

            let mut leaf_map = LeafMap::new();
            leaf_map.track_input(param_name);
        match trace_expr(&inlined, env, &mut leaf_map) {
            Ok(traced) => {
                if !contains_undiffable_ops(&traced) {
                        env.pop_scope();
                        let loss_val = call_function(func, std::slice::from_ref(params), env)?;
                        let loss = scalar_from_value(&loss_val)?;

                        env.push_scope();
                        for (name, val) in &aug_closure {
                            env.set(name, val.clone());
                        }
                        env.set(param_name, params.clone());
                        for (sym, val) in &leaf_map.leaves {
                            env.set(sym, val.clone());
                        }

                            let grad_tree = match build_grad_from_leaves(
                                &traced, params, &leaf_map.leaves, env,
                            ) {
                                Ok(tree) => tree,
                                Err(error) => {
                                    env.pop_scope();
                                    return Err(error);
                                }
                            };
                            env.pop_scope();
                            return Ok(Value::List(vec![Value::Float(loss), grad_tree]));
                        }
                        env.pop_scope();
                    }
                    Err(e) => {
                        env.pop_scope();
                        return Err(runtime_error(format!(
                            "value-and-grad -> {}", e.short_message()
                        )));
                    }
                }
            }
        }

    Err(runtime_error("value-and-grad: cannot differentiate this function"))
}

fn scalar_from_value(val: &Value) -> Result<f32, SheafError> {
    let val = val.ensure_host_cow()?;
    match &*val {
        Value::Float(x) => Ok(*x),
        Value::Int(n) => Ok(*n as f32),
        Value::Tensor { data, .. } => data.first().copied()
            .ok_or_else(|| runtime_error("value-and-grad: empty tensor result")),
        _ => Err(runtime_error("value-and-grad: loss must return a scalar")),
    }
}

fn build_grad_from_leaves(
    traced_expr: &CompiledExpr,
    params: &Value,
    leaves: &[(String, Value)],
    env: &mut Env,
) -> Result<Value, SheafError> {
    use crate::autodiff::grad_simplified;
    use std::collections::HashMap;

    let mut leaf_grads: HashMap<String, Value> = HashMap::new();
    for (sym, _val) in leaves {
        let grad_expr = grad_simplified(traced_expr, sym)?;
        let grad_val = eval(&grad_expr, env)?;
        leaf_grads.insert(sym.clone(), grad_val);
    }

    build_grad_tree_by_value(params, leaves, &leaf_grads)
}

fn values_equal(a: &Value, b: &Value) -> bool {
    let a_cow = a.ensure_host_cow().ok();
    let b_cow = b.ensure_host_cow().ok();
    let a = a_cow.as_deref().unwrap_or(a);
    let b = b_cow.as_deref().unwrap_or(b);
    match (a, b) {
        (Value::Tensor { data: da, .. }, Value::Tensor { data: db, .. }) => {
            da.shape() == db.shape() && da.iter().zip(db.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
        }
        (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
        (Value::Int(a), Value::Int(b)) => a == b,
        _ => false,
    }
}

fn build_grad_tree_by_value(
    params: &Value,
    leaves: &[(String, Value)],
    leaf_grads: &std::collections::HashMap<String, Value>,
) -> Result<Value, SheafError> {
    match params {
        Value::Tensor { .. } | Value::Float(_) | Value::Int(_) | Value::DeviceBuffer(_) => {
            let params_host = params.ensure_host_cow().unwrap_or_else(|_| Cow::Borrowed(params));
            for (sym, leaf_val) in leaves {
                if values_equal(&params_host, leaf_val)
                    && let Some(grad_val) = leaf_grads.get(sym) {
                        return reduce_grad_to_param_shape(grad_val, &params_host);
                    }
            }
            Ok(zeros_like(&params_host))
        }
        Value::Dict(map) => {
            let mut grad_map = BTreeMap::new();
            for (k, v) in map {
                let g = build_grad_tree_by_value(v, leaves, leaf_grads)?;
                grad_map.insert(k.clone(), g);
            }
            Ok(Value::Dict(grad_map))
        }
        Value::List(items) => {
            let mut grad_items = Vec::new();
            for item in items {
                let g = build_grad_tree_by_value(item, leaves, leaf_grads)?;
                grad_items.push(g);
            }
            Ok(Value::List(grad_items))
        }
        _ => Ok(Value::Nil),
    }
}

fn zeros_like(val: &Value) -> Value {
    match val {
        Value::Tensor { data, dtype } => {
            Value::Tensor {
                data: Arc::new(ArrayD::zeros(IxDyn(data.shape()))),
                dtype: *dtype,
            }
        }
        Value::Float(_) => Value::Float(0.0),
        Value::Int(_) => Value::Int(0),
        _ => Value::Nil,
    }
}

fn reduce_grad_to_param_shape(grad: &Value, param: &Value) -> Result<Value, SheafError> {
    match (grad, param) {
        (Value::Tensor { data: g_data, dtype }, Value::Tensor { data: p_data, .. }) => {
            let g_shape = g_data.shape();
            let p_shape = p_data.shape();
            if g_shape == p_shape {
                return Ok(grad.clone());
            }
            let g_ndim = g_shape.len();
            let p_ndim = p_shape.len();
            if g_ndim > p_ndim {
                let extra = g_ndim - p_ndim;
                let mut reduced = (**g_data).clone();
                for _ in 0..extra {
                    reduced = reduced.sum_axis(ndarray::Axis(0));
                }
                let r_shape = reduced.shape().to_vec();
                for (i, (&rd, &pd)) in r_shape.iter().zip(p_shape.iter()).enumerate() {
                    if pd == 1 && rd > 1 {
                        reduced = reduced.sum_axis(ndarray::Axis(i));
                        reduced = reduced.insert_axis(ndarray::Axis(i));
                    }
                }
                Ok(Value::Tensor { data: Arc::new(reduced), dtype: *dtype })
            } else if g_ndim == p_ndim {
                let mut reduced = (**g_data).clone();
                for (i, (&gd, &pd)) in g_shape.iter().zip(p_shape.iter()).enumerate() {
                    if pd == 1 && gd > 1 {
                        reduced = reduced.sum_axis(ndarray::Axis(i));
                        reduced = reduced.insert_axis(ndarray::Axis(i));
                    }
                }
                Ok(Value::Tensor { data: Arc::new(reduced), dtype: *dtype })
            } else {
                Ok(grad.clone())
            }
        }
        (Value::Float(_), Value::Float(_)) => Ok(grad.clone()),
        (Value::Tensor { data, dtype: _ }, Value::Float(_)) => {
            let sum: f32 = data.iter().sum();
            Ok(Value::Float(sum))
        }
        _ => Ok(grad.clone()),
    }
}
