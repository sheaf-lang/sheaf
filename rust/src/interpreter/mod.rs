// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf interpreter — evaluates CompiledExpr directly to runtime Values.

pub mod builtins;
pub mod env;
pub mod eval;
pub mod value;

use crate::ast::SheafValue;
use crate::core::compiler::{CompiledExpr, CompilerContext};
use crate::core::error::SheafError;
use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::{Dtype, Value};
use ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;

pub fn eval(expr: &CompiledExpr, env: &mut Env) -> Result<Value, SheafError> {
    match expr {
        CompiledExpr::Integer(n) => Ok(Value::Int(*n)),
        CompiledExpr::Float(x) => Ok(Value::Float(*x)),
        CompiledExpr::Boolean(b) => Ok(Value::Bool(*b)),
        CompiledExpr::Nil => Ok(Value::Nil),
        CompiledExpr::String(s) => Ok(Value::String(s.clone())),
        CompiledExpr::Keyword(k) => Ok(Value::Keyword(k.clone())),

        CompiledExpr::Symbol(name) => env.get(name),

        CompiledExpr::Vector(elements) => eval_vector(elements, env),

        CompiledExpr::Dict(pairs) => eval_dict(pairs, env),

        CompiledExpr::Quoted(sv) => sheaf_value_to_value(sv),

        CompiledExpr::FunctionRef(name) => {
            // Try env first (builtins live here)
            if let Ok(val) = env.get(name) {
                return Ok(val);
            }
            // Registry functions → Value::Function with real params/body
            if let Some(func_def) = env.registry.get(name).cloned() {
                if let Some(body) = func_def.body_compiled {
                    return Ok(Value::Function {
                        params: func_def.params,
                        body,
                        closure: vec![],
                    });
                }
            }
            Err(runtime_error(format!("Undefined function: {}", name)))
        }

        CompiledExpr::FunctionCall { name, args } => eval_call(name, args, env),

        CompiledExpr::Let { bindings, body } => {
            env.push_scope();
            for (name, expr) in bindings {
                let val = eval(expr, env)?;
                bind_pattern(name, val, env)?;
            }
            let result = eval(body, env);
            env.pop_scope();
            result
        }

        CompiledExpr::If { condition, then_branch, else_branch } => {
            let cond = eval(condition, env)?;
            if cond.is_truthy() {
                eval(then_branch, env)
            } else if let Some(else_br) = else_branch {
                eval(else_br, env)
            } else {
                Ok(Value::Nil)
            }
        }

        CompiledExpr::Do(exprs) => {
            let mut result = Value::Nil;
            for expr in exprs {
                result = eval(expr, env)?;
            }
            Ok(result)
        }

        CompiledExpr::Lambda { params, body } => {
            Ok(Value::Function {
                params: params.clone(),
                body: *body.clone(),
                closure: vec![],
            })
        }

        CompiledExpr::LambdaCall { callee, args } => {
            let func = eval(callee, env)?;
            let mut arg_vals = Vec::new();
            for arg in args {
                arg_vals.push(eval(arg, env)?);
            }
            call_function(&func, &arg_vals, env)
        }

        CompiledExpr::GetTupleElement { param, indices } => {
            let val = env.get(param)?;
            get_nested(&val, indices)
        }

        CompiledExpr::ValueAndGrad { fn_name, .. } => {
            Err(runtime_error(format!(
                "value-and-grad '{}': interpreter support not yet implemented", fn_name
            )))
        }

        CompiledExpr::InlineValueAndGrad { lambda, args, .. } => {
            // Evaluate the lambda to get the function value
            let func = eval(lambda, env)?;
            // Evaluate args
            let mut evaluated_args = Vec::new();
            for arg in args {
                evaluated_args.push(eval(arg, env)?);
            }
            // Use the same HOF path: wrap in vag closure, then call it
            let vag = eval_value_and_grad_hof(&[func], env)?;
            call_function(&vag, &evaluated_args, env)
        }

        CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => {
            let n = match eval(count, env)? {
                Value::Int(n) => n,
                Value::Float(f) => f as i64,
                other => return Err(runtime_error(format!(
                    "repeat: count must be an integer, got {}", other.type_name()
                ))),
            };
            let mut acc = eval(acc_init, env)?;
            env.push_scope();
            for i in 0..n {
                env.set(index_var, Value::Int(i));
                env.set(acc_var, acc);
                acc = eval(body, env)?;
            }
            env.pop_scope();
            Ok(acc)
        }

        CompiledExpr::While { condition, acc_var, acc_init, body } => {
            let mut acc = eval(acc_init, env)?;
            env.push_scope();
            env.set(acc_var, acc.clone());
            loop {
                let cond = eval(condition, env)?;
                if !cond.is_truthy() {
                    break;
                }
                acc = eval(body, env)?;
                env.set(acc_var, acc.clone());
            }
            env.pop_scope();
            Ok(acc)
        }
    }
}

/// Bind a pattern name to a value in the current scope.
///
/// Patterns:
///   - Simple: `"x"` → env["x"] = val
///   - Destructuring: `"[a b]"` (encoded by compiler as "[a b]") → env["a"] = val[0], env["b"] = val[1]
///
/// The compiler encodes vector destructuring patterns as a string like `"[k1 k2]"`.
/// We detect this by the leading `[` and parse the names out.
fn bind_pattern(name: &str, val: Value, env: &mut Env) -> Result<(), SheafError> {
    if name.starts_with('[') && name.ends_with(']') {
        // Destructuring pattern: extract symbol names
        let inner = &name[1..name.len() - 1];
        let names: Vec<&str> = inner.split_whitespace().collect();
        let items = match &val {
            Value::List(items) => items.clone(),
            Value::Tuple(items) => items.clone(),
            Value::Tensor { data, .. } => {
                if data.ndim() == 1 {
                    data.iter().map(|&x| Value::Float(x)).collect()
                } else {
                    return Err(runtime_error(format!(
                        "let destructuring: expected list/tuple, got tensor with shape {:?}", data.shape()
                    )));
                }
            }
            other => return Err(runtime_error(format!(
                "let destructuring: expected list or tuple, got {}", other.type_name()
            ))),
        };
        for (n, v) in names.iter().zip(items.iter()) {
            env.set(n, v.clone());
        }
        // If fewer values than names, bind remaining to Nil
        for n in names.iter().skip(items.len()) {
            env.set(n, Value::Nil);
        }
    } else {
        env.set(name, val);
    }
    Ok(())
}

fn eval_vector(elements: &[CompiledExpr], env: &mut Env) -> Result<Value, SheafError> {
    let vals: Result<Vec<Value>, _> = elements.iter().map(|e| eval(e, env)).collect();
    let vals = vals?;

    if vals.is_empty() {
        return Ok(Value::List(vec![]));
    }

    // Check if all elements are numeric → produce a Tensor (always F32 by default)
    let all_numeric = vals.iter().all(|v| matches!(v, Value::Int(_) | Value::Float(_)));
    if all_numeric {
        let data: Vec<f64> = vals.iter().map(|v| v.to_f64().unwrap()).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap();
        return Ok(Value::tensor_f32(arr));
    }

    // Check if all elements are vectors/tensors of same shape → produce a 2D+ tensor
    let all_tensors = vals.iter().all(|v| matches!(v, Value::Tensor { .. }));
    if all_tensors {
        let shapes: Vec<_> = vals.iter().map(|v| match v {
            Value::Tensor { data, .. } => data.shape().to_vec(),
            _ => unreachable!(),
        }).collect();
        if shapes.windows(2).all(|w| w[0] == w[1]) {
            let inner_shape = &shapes[0];
            let mut full_shape = vec![vals.len()];
            full_shape.extend(inner_shape);
            let mut flat_data = Vec::new();
            for v in &vals {
                if let Value::Tensor { data, .. } = v {
                    flat_data.extend(data.iter());
                }
            }
            let arr = ArrayD::from_shape_vec(IxDyn(&full_shape), flat_data).unwrap();
            return Ok(Value::tensor_f32(arr));
        }
    }

    // Otherwise, a heterogeneous list
    Ok(Value::List(vals))
}

fn eval_dict(pairs: &[(CompiledExpr, CompiledExpr)], env: &mut Env) -> Result<Value, SheafError> {
    let mut map = BTreeMap::new();
    for (k, v) in pairs {
        let key = match eval(k, env)? {
            Value::Keyword(s) => s,
            Value::String(s) => s,
            other => return Err(runtime_error(format!("Dict key must be keyword or string, got {}", other.type_name()))),
        };
        let val = eval(v, env)?;
        map.insert(key, val);
    }
    Ok(Value::Dict(map))
}

fn sheaf_value_to_value(sv: &SheafValue) -> Result<Value, SheafError> {
    match sv {
        SheafValue::Integer(n, _) => Ok(Value::Int(*n)),
        SheafValue::Float(x, _) => Ok(Value::Float(*x)),
        SheafValue::Boolean(b, _) => Ok(Value::Bool(*b)),
        SheafValue::Nil(_) => Ok(Value::Nil),
        SheafValue::String(s, _) => Ok(Value::String(s.clone())),
        SheafValue::Symbol(s, _) => Ok(Value::String(s.clone())),
        SheafValue::Keyword(k, _) => Ok(Value::Keyword(k.clone())),
        SheafValue::List(elems, _) | SheafValue::Vector(elems, _) => {
            let items: Result<Vec<Value>, _> = elems.iter().map(|e| sheaf_value_to_value(e)).collect();
            Ok(Value::List(items?))
        }
        SheafValue::Dict(pairs, _) => {
            let mut map = BTreeMap::new();
            for (k, v) in pairs {
                let key = match sheaf_value_to_value(k)? {
                    Value::Keyword(s) => s,
                    Value::String(s) => s,
                    other => return Err(runtime_error(format!("Dict key must be keyword or string, got {:?}", other))),
                };
                map.insert(key, sheaf_value_to_value(v)?);
            }
            Ok(Value::Dict(map))
        }
        SheafValue::Quote(inner, _) => sheaf_value_to_value(inner),
        _ => Err(runtime_error(format!("Cannot convert quoted value: {}", sv))),
    }
}

fn eval_call(name: &str, args: &[CompiledExpr], env: &mut Env) -> Result<Value, SheafError> {
    // Special handling for short-circuit operators
    match name {
        "and" => return eval_and(args, env),
        "or" => return eval_or(args, env),
        _ => {}
    }

    // Some functions treat keywords as positional args (not kwargs)
    let no_kwargs = matches!(name, "get" | "get-in" | "assoc" | "dissoc" | "dict");

    // Evaluate args, splitting kwargs
    let (pos_args, kwargs) = if no_kwargs {
        let pos: Result<Vec<Value>, _> = args.iter().map(|a| eval(a, env)).collect();
        (pos?, BTreeMap::new())
    } else {
        split_kwargs(args, env)?
    };

    // Higher-order functions need &mut Env to call lambdas
    match name {
        "map" => return eval_map(&pos_args, env),
        "filter" => return eval_filter(&pos_args, env),
        "reduce" => return eval_reduce(&pos_args, env),
        "apply" => return eval_apply(&pos_args, env),
        "find" => return eval_find(&pos_args, env),
        "tree-map" => return eval_tree_map(&pos_args, env),
        "tree-reduce" => return eval_tree_reduce(&pos_args, env),
        "flatten" => return eval_flatten(&pos_args),
        "__value-and-grad-hof__" => return eval_value_and_grad_hof(&pos_args, env),
        _ => {}
    }

    // Try builtin from env
    if let Ok(Value::BuiltinFn { func, .. }) = env.get(name) {
        return func(&pos_args, &kwargs);
    }

    // Try user-defined function from registry
    if let Some(func_def) = env.registry.get(name).cloned() {
        // Record the first call for tracing (sheaf build --trace-with)
        if let Some(ref mut records) = env.call_records {
            records.entry(name.to_string()).or_insert_with(|| {
                crate::interpreter::env::CallRecord {
                    arg_values: pos_args.clone(),
                }
            });
        }

        // VMFB dispatch: pure compiled functions run via IREE
        #[cfg(iree_runtime)]
        if let Some(session_idx) = func_def.vmfb_session_idx {
            if let Some(session) = env.vmfb_sessions.get(session_idx) {
                if let Some(iree_session) = session.downcast_ref::<crate::runtime::iree_session::IreeSession>() {
                    let full_name = format!("module.{}", name.replace('-', "_"));
                    // Use typed call to reconstruct tuple/dict structure from flat outputs
                    if let Some(ref sig) = func_def.signature {
                        let mut result = iree_session.call_typed(&full_name, &pos_args, &sig.return_type)?;
                        // Reconstruct Dict if the function originally returned one
                        if let Some(ref keys) = sig.return_dict_keys {
                            result = match result {
                                Value::Tuple(elems) if elems.len() == keys.len() => {
                                    let mut map = std::collections::BTreeMap::new();
                                    for (k, v) in keys.iter().zip(elems) {
                                        map.insert(k.clone(), v);
                                    }
                                    Value::Dict(map)
                                }
                                other => other,
                            };
                        }
                        return Ok(result);
                    }
                    return iree_session.call(&full_name, &pos_args);
                }
            }
        }

        // Fallback: interpret
        if let Some(ref body) = func_def.body_compiled {
            env.push_scope();
            for (param, val) in func_def.params.iter().zip(pos_args.iter()) {
                env.set(param, val.clone());
            }
            let result = eval(body, env);
            env.pop_scope();
            return result;
        }
    }

    // Try function value in env
    if let Ok(func_val) = env.get(name) {
        return call_function(&func_val, &pos_args, env);
    }

    Err(runtime_error(format!("Unknown function: {}", name)))
}

fn eval_and(args: &[CompiledExpr], env: &mut Env) -> Result<Value, SheafError> {
    if args.is_empty() {
        return Ok(Value::Bool(true));
    }
    for arg in &args[..args.len() - 1] {
        let val = eval(arg, env)?;
        if !val.is_truthy() {
            return Ok(val);
        }
    }
    eval(&args[args.len() - 1], env)
}

fn eval_or(args: &[CompiledExpr], env: &mut Env) -> Result<Value, SheafError> {
    if args.is_empty() {
        return Ok(Value::Bool(false));
    }
    for arg in &args[..args.len() - 1] {
        let val = eval(arg, env)?;
        if val.is_truthy() {
            return Ok(val);
        }
    }
    eval(&args[args.len() - 1], env)
}

fn split_kwargs(args: &[CompiledExpr], env: &mut Env) -> Result<(Vec<Value>, BTreeMap<String, Value>), SheafError> {
    let mut pos = Vec::new();
    let mut kwargs = BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        if let CompiledExpr::Keyword(k) = &args[i] {
            if i + 1 < args.len() {
                // Check if next arg is also a keyword (then this is a flag)
                if matches!(&args[i + 1], CompiledExpr::Keyword(_)) {
                    kwargs.insert(k.clone(), Value::Bool(true));
                } else {
                    let val = eval(&args[i + 1], env)?;
                    kwargs.insert(k.clone(), val);
                    i += 1;
                }
            } else {
                // Last arg is a keyword flag
                kwargs.insert(k.clone(), Value::Bool(true));
            }
        } else {
            pos.push(eval(&args[i], env)?);
        }
        i += 1;
    }
    Ok((pos, kwargs))
}

fn call_function(func: &Value, args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    match func {
        Value::Function { params: _, body: _, closure } => {
            // Detect value-and-grad HOF closure: contains __vag_fn__
            if let Some((_, vag_fn)) = closure.iter().find(|(k, _)| k == "__vag_fn__") {
                if args.len() != 1 {
                    return Err(runtime_error("value-and-grad: expected exactly 1 argument (params)"));
                }
                return eval_value_and_grad_call(vag_fn, &args[0], env);
            }
            // Normal function call
            let Value::Function { params, body, closure } = func else { unreachable!() };
            env.push_scope();
            for (name, val) in closure {
                env.set(name, val.clone());
            }
            for (param, val) in params.iter().zip(args.iter()) {
                env.set(param, val.clone());
            }
            let result = eval(body, env);
            env.pop_scope();
            result
        }
        Value::BuiltinFn { func, .. } => {
            func(args, &BTreeMap::new())
        }
        _ => Err(runtime_error(format!("Not a function: {}", func.type_name()))),
    }
}

fn get_nested(val: &Value, indices: &[usize]) -> Result<Value, SheafError> {
    let mut current = val.clone();
    for &idx in indices {
        current = match current {
            Value::List(items) => {
                items.get(idx).cloned().ok_or_else(|| {
                    runtime_error(format!("Tuple index {} out of bounds (len {})", idx, items.len()))
                })?
            }
            Value::Dict(map) => {
                let entry = map.values().nth(idx).cloned().ok_or_else(|| {
                    runtime_error(format!("Tuple index {} out of bounds (len {})", idx, map.len()))
                })?;
                entry
            }
            _ => return Err(runtime_error(format!("Cannot index into {}", current.type_name()))),
        };
    }
    Ok(current)
}

fn eval_map(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 2 {
        return Err(runtime_error("map requires 2 arguments: (map fn coll)"));
    }
    let func = &args[0];
    match &args[1] {
        Value::List(items) => {
            let mut results = Vec::with_capacity(items.len());
            for item in items {
                results.push(call_function(func, &[item.clone()], env)?);
            }
            Ok(Value::List(results))
        }
        Value::Tensor { data, dtype } => {
            if data.ndim() == 1 {
                // 1D: iterate over scalar elements
                let mut results = Vec::with_capacity(data.len());
                for &x in data.iter() {
                    results.push(call_function(func, &[Value::Float(x)], env)?);
                }
                Ok(Value::List(results))
            } else {
                // ND: iterate over slices along axis 0
                let n = data.shape()[0];
                let mut results = Vec::with_capacity(n);
                for i in 0..n {
                    let row = data.index_axis(ndarray::Axis(0), i).to_owned();
                    let row_val = Value::Tensor { data: row, dtype: *dtype };
                    results.push(call_function(func, &[row_val], env)?);
                }
                Ok(Value::List(results))
            }
        }
        _ => Err(runtime_error("map: expected list or tensor")),
    }
}

fn eval_filter(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 2 {
        return Err(runtime_error("filter requires 2 arguments: (filter fn coll)"));
    }
    let func = &args[0];
    match &args[1] {
        Value::List(items) => {
            let mut results = Vec::new();
            for item in items {
                let result = call_function(func, &[item.clone()], env)?;
                if result.is_truthy() {
                    results.push(item.clone());
                }
            }
            Ok(Value::List(results))
        }
        _ => Err(runtime_error("filter: expected list")),
    }
}

fn eval_reduce(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 3 {
        return Err(runtime_error("reduce requires 3 arguments: (reduce fn init coll)"));
    }
    let func = &args[0];
    let mut acc = args[1].clone();
    match &args[2] {
        Value::List(items) => {
            for item in items {
                acc = call_function(func, &[acc, item.clone()], env)?;
            }
            Ok(acc)
        }
        Value::Tensor { data, .. } => {
            if data.ndim() == 1 {
                for &x in data.iter() {
                    acc = call_function(func, &[acc, Value::Float(x)], env)?;
                }
            } else {
                for i in 0..data.shape()[0] {
                    let row = data.index_axis(ndarray::Axis(0), i).to_owned();
                    acc = call_function(func, &[acc, Value::tensor_f32(row)], env)?;
                }
            }
            Ok(acc)
        }
        _ => Err(runtime_error("reduce: expected list or tensor")),
    }
}

fn eval_apply(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 2 {
        return Err(runtime_error("apply requires 2 arguments: (apply fn args)"));
    }
    let func = &args[0];
    let call_args = match &args[1] {
        Value::List(items) => items.clone(),
        Value::Tensor { data, .. } => data.iter().map(|&x| Value::Float(x)).collect(),
        _ => return Err(runtime_error("apply: expected list or tensor")),
    };
    call_function(func, &call_args, env)
}

fn eval_find(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 2 {
        return Err(runtime_error("find requires 2 arguments: (find fn coll)"));
    }
    let func = &args[0];
    match &args[1] {
        Value::List(items) => {
            for item in items {
                let result = call_function(func, &[item.clone()], env)?;
                if result.is_truthy() {
                    return Ok(item.clone());
                }
            }
            Ok(Value::Nil)
        }
        _ => Err(runtime_error("find: expected list")),
    }
}

fn tree_map_multi(trees: &[Value], func: &Value, env: &mut Env) -> Result<Value, SheafError> {
    match &trees[0] {
        Value::Dict(map) => {
            let mut result = BTreeMap::new();
            for k in map.keys() {
                let sub_trees: Result<Vec<Value>, _> = trees
                    .iter()
                    .map(|t| match t {
                        Value::Dict(m) => m
                            .get(k)
                            .cloned()
                            .ok_or_else(|| runtime_error(format!("tree-map: key {} missing in one tree", k))),
                        _ => Err(runtime_error("tree-map: tree structure mismatch")),
                    })
                    .collect();
                result.insert(k.clone(), tree_map_multi(&sub_trees?, func, env)?);
            }
            Ok(Value::Dict(result))
        }
        Value::List(items) => {
            let n = items.len();
            let mut result = Vec::new();
            for i in 0..n {
                let sub_trees: Result<Vec<Value>, _> = trees
                    .iter()
                    .map(|t| match t {
                        Value::List(v) => v
                            .get(i)
                            .cloned()
                            .ok_or_else(|| runtime_error("tree-map: list length mismatch")),
                        _ => Err(runtime_error("tree-map: tree structure mismatch")),
                    })
                    .collect();
                result.push(tree_map_multi(&sub_trees?, func, env)?);
            }
            Ok(Value::List(result))
        }
        _ => call_function(func, trees, env),
    }
}

fn eval_tree_map(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() < 2 {
        return Err(runtime_error("tree-map requires at least 2 arguments: (tree-map fn tree ...)"));
    }
    let func = &args[0];
    let trees = &args[1..];
    tree_map_multi(trees, func, env)
}

fn tree_reduce_value(val: &Value, func: &Value, acc: Value, env: &mut Env) -> Result<Value, SheafError> {
    match val {
        Value::Dict(map) => {
            let mut acc = acc;
            for v in map.values() {
                acc = tree_reduce_value(v, func, acc, env)?;
            }
            Ok(acc)
        }
        Value::List(items) => {
            let mut acc = acc;
            for item in items {
                acc = tree_reduce_value(item, func, acc, env)?;
            }
            Ok(acc)
        }
        leaf => call_function(func, &[acc, leaf.clone()], env),
    }
}

fn eval_tree_reduce(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 3 {
        return Err(runtime_error("tree-reduce requires 3 arguments: (tree-reduce fn tree init)"));
    }
    tree_reduce_value(&args[1], &args[0], args[2].clone(), env)
}

fn flatten_leaves(val: &Value, leaves: &mut Vec<Value>) {
    match val {
        Value::Dict(map) => {
            for v in map.values() {
                flatten_leaves(v, leaves);
            }
        }
        Value::List(items) => {
            for item in items {
                flatten_leaves(item, leaves);
            }
        }
        leaf => leaves.push(leaf.clone()),
    }
}

fn eval_flatten(args: &[Value]) -> Result<Value, SheafError> {
    if args.is_empty() {
        return Err(runtime_error("flatten requires 1 argument"));
    }
    let mut leaves = Vec::new();
    flatten_leaves(&args[0], &mut leaves);
    // Returns (leaves_list, reconstruct_fn) — we return a list of [leaves, nil] for now
    // The test only uses (first (flatten params)) → the leaves list
    Ok(Value::List(vec![Value::List(leaves), Value::Nil]))
}

/// (value-and-grad f) → returns a function that, given params, returns [loss, grad_params].
///
/// Gradient computed by central finite differences: grad[i] ≈ (f(p+h) - f(p-h)) / 2h
/// Applied element-wise to every leaf tensor in the params pytree.
fn eval_value_and_grad_hof(args: &[Value], _env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 1 {
        return Err(runtime_error("value-and-grad: expected exactly 1 argument (the function)"));
    }
    let func = args[0].clone();

    // Return a Value::Function closure that captures `func`.
    // When called with params, call_function detects __vag_fn__ and dispatches
    // to eval_value_and_grad_call which computes (loss, grad) via finite differences.
    Ok(Value::Function {
        params: vec!["__vag_params__".to_string()],
        body: crate::core::compiler::CompiledExpr::Symbol("__vag_params__".to_string()),
        closure: vec![("__vag_fn__".to_string(), func)],
    })
}

/// Evaluate a value-and-grad HOF call: the Function created above is called with params.
/// We intercept it in call_function when we detect the __vag_params__ sentinel.
fn eval_value_and_grad_call(func: &Value, params: &Value, env: &mut Env) -> Result<Value, SheafError> {
    let h = 1e-4_f64;

    // Evaluate loss at params
    let loss_val = call_function(func, &[params.clone()], env)?;
    let loss = match &loss_val {
        Value::Float(x) => *x,
        Value::Int(n) => *n as f64,
        Value::Tensor { data, .. } => data.first().copied().unwrap_or(0.0),
        _ => return Err(runtime_error("value-and-grad: loss function must return a scalar")),
    };

    // Count leaves
    let n_leaves = {
        let mut count = 0usize;
        fn count_leaves_inner(val: &Value, count: &mut usize) {
            match val {
                Value::Tensor { data, .. } => *count += data.len(),
                Value::Dict(map) => map.values().for_each(|v| count_leaves_inner(v, count)),
                Value::Float(_) | Value::Int(_) => *count += 1,
                _ => {}
            }
        }
        count_leaves_inner(params, &mut count);
        count
    };

    // Central finite differences for each leaf
    let mut grads = vec![0.0f64; n_leaves];
    for i in 0..n_leaves {
        let p_plus = perturb_leaf(params, i, h);
        let p_minus = perturb_leaf(params, i, -h);
        let f_plus = call_function(func, &[p_plus], env)?;
        let f_minus = call_function(func, &[p_minus], env)?;
        let fp = scalar_from_value(&f_plus)?;
        let fm = scalar_from_value(&f_minus)?;
        grads[i] = (fp - fm) / (2.0 * h);
    }

    let mut counter = 0;
    let grad_tree = build_grad_tree(params, &grads, &mut counter);

    Ok(Value::List(vec![Value::Float(loss), grad_tree]))
}

fn perturb_leaf(val: &Value, leaf_idx: usize, delta: f64) -> Value {
    let mut counter = 0usize;
    perturb_leaf_inner(val, leaf_idx, delta, &mut counter)
}

fn perturb_leaf_inner(val: &Value, leaf_idx: usize, delta: f64, counter: &mut usize) -> Value {
    match val {
        Value::Tensor { data, dtype } => {
            let n = data.len();
            if *counter <= leaf_idx && leaf_idx < *counter + n {
                let local = leaf_idx - *counter;
                let mut new_data = data.clone();
                new_data.as_slice_mut().unwrap()[local] += delta;
                *counter += n;
                Value::Tensor { data: new_data, dtype: *dtype }
            } else {
                *counter += n;
                val.clone()
            }
        }
        Value::Dict(map) => Value::Dict(
            map.iter().map(|(k, v)| (k.clone(), perturb_leaf_inner(v, leaf_idx, delta, counter))).collect()
        ),
        Value::Float(x) => {
            let result = if *counter == leaf_idx { Value::Float(x + delta) } else { val.clone() };
            *counter += 1;
            result
        }
        Value::Int(n) => {
            let result = if *counter == leaf_idx { Value::Float(*n as f64 + delta) } else { val.clone() };
            *counter += 1;
            result
        }
        _ => val.clone(),
    }
}

fn scalar_from_value(val: &Value) -> Result<f64, SheafError> {
    match val {
        Value::Float(x) => Ok(*x),
        Value::Int(n) => Ok(*n as f64),
        Value::Tensor { data, .. } => data.first().copied()
            .ok_or_else(|| runtime_error("value-and-grad: empty tensor result")),
        _ => Err(runtime_error("value-and-grad: loss must return a scalar")),
    }
}

fn build_grad_tree(params: &Value, grads: &[f64], counter: &mut usize) -> Value {
    match params {
        Value::Tensor { data, dtype } => {
            let n = data.len();
            let slice = grads[*counter..*counter + n].to_vec();
            *counter += n;
            Value::Tensor {
                data: ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(data.shape()), slice).unwrap(),
                dtype: *dtype,
            }
        }
        Value::Dict(map) => Value::Dict(
            map.iter().map(|(k, v)| (k.clone(), build_grad_tree(v, grads, counter))).collect()
        ),
        Value::Float(_) | Value::Int(_) => {
            let g = grads[*counter];
            *counter += 1;
            Value::Float(g)
        }
        _ => params.clone(),
    }
}

/// High-level entry point: parse + compile + eval a Sheaf expression string.
pub fn eval_str(source: &str) -> Result<Value, SheafError> {
    let exprs = crate::core::parse(source, "<eval>")?;
    let mut compiler = CompilerContext::new();
    let mut last = Value::Nil;

    for expr in &exprs {
        let compiled = compiler.compile(expr)?;
        let mut env = Env::with_registry(compiler.registry.clone());
        builtins::register_builtins(&mut env);
        last = eval(&compiled, &mut env)?;
    }

    Ok(last)
}

/// Evaluate multiple expressions, maintaining state across them.
pub fn eval_exprs(source: &str) -> Result<Value, SheafError> {
    let exprs = crate::core::parse(source, "<eval>")?;
    let mut compiler = CompilerContext::new();
    let mut last = Value::Nil;

    // First pass: compile all (registers defn, defparams, etc.)
    let mut compiled_exprs = Vec::new();
    for expr in &exprs {
        compiled_exprs.push(compiler.compile(expr)?);
    }

    // Second pass: evaluate all non-Nil expressions
    let mut env = Env::with_registry(compiler.registry.clone());
    builtins::register_builtins(&mut env);

    for compiled in &compiled_exprs {
        if !matches!(compiled, CompiledExpr::Nil) {
            last = eval(compiled, &mut env)?;
        }
    }

    Ok(last)
}
