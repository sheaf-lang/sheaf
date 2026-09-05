// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf interpreter, evaluates CompiledExpr directly to runtime Values.

pub mod builtins;
pub mod env;
pub mod eval;
#[cfg(iree_runtime)]
mod iree_dispatch;
pub(crate) mod hof;
pub mod profiler;
pub mod mem_profile;
mod vag;
pub mod tracer;
pub mod value;

use crate::core::ast::SheafValue;
use crate::core::expr::{BindingPattern, CompiledExpr, CompilerContext};
use crate::core::error::SheafError;
use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::Value;
use ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;

fn is_stdlib_location(loc: &crate::core::ast::SourceLocation) -> bool {
    let f = loc.filename.as_ref();
    f.ends_with("nn.shf") || f.ends_with("optim.shf") || f.ends_with("macros.shf") || f.ends_with("misc.shf")
}

pub fn eval(expr: &CompiledExpr, env: &mut Env) -> Result<Value, SheafError> {
    match expr {
        CompiledExpr::Integer(n) => Ok(Value::Int(*n)),
        CompiledExpr::Float(x) => Ok(Value::Float(*x as f32)),
        CompiledExpr::Boolean(b) => Ok(Value::Bool(*b)),
        CompiledExpr::Nil => Ok(Value::Nil),
        CompiledExpr::String(s) => Ok(Value::String(s.clone())),
        CompiledExpr::Keyword(k) => Ok(Value::Keyword(k.clone())),

        CompiledExpr::Symbol(name) => {
            if name == "..." { return Ok(Value::Keyword("...".to_string())); }
            env.get(name)
        }

        CompiledExpr::Vector(elements) => eval_vector(elements, env),

        CompiledExpr::Dict(pairs) => eval_dict(pairs, env),

        CompiledExpr::Quoted(sv) => sheaf_value_to_value(sv),

        CompiledExpr::FunctionRef(name) => {
            if let Ok(val) = env.get(name) {
                return Ok(val);
            }
            if let Some(func_def) = env.registry.get(name) {
                if let Some(ref body) = func_def.body_compiled {
                    return Ok(Value::Function {
                        name: Some(name.to_string()),
                        params: func_def.params.clone(),
                        body: body.clone(),
                        closure: vec![],
                    });
                }
                return Ok(Value::Nil);
            }
            Err(runtime_error(format!("Undefined function: {}", name)))
        }

        CompiledExpr::FunctionCall { name, args, loc } => {
            eval_call(name, args, env).map_err(|e| match e {
                SheafError::Runtime { message, location: None } => SheafError::Runtime {
                    message,
                    location: loc.clone(),
                },
                SheafError::Runtime { message, location: Some(ref err_loc) }
                    if is_stdlib_location(err_loc)
                        && loc.is_some()
                        && !is_stdlib_location(loc.as_ref().unwrap())
                        && !message.contains("called from") =>
                {
                    // Error in stdlib, called from user code: show function name + call site
                    let call_loc = loc.as_ref().unwrap();
                    SheafError::Runtime {
                        message: format!(
                            "{} -> {}\n  = called from {}:{}",
                            name, message, call_loc.filename, call_loc.line
                        ),
                        location: Some(err_loc.clone()),
                    }
                }
                other => other,
            })
        }

        CompiledExpr::Def { name, value } => {
            let val = eval(value, env)?;
            env.set_global(name, val.clone());
            Ok(val)
        }

        CompiledExpr::Let { bindings, body } => {
            env.push_scope();
            for (pattern, expr) in bindings {
                let val = eval(expr, env)?;
                bind_pattern(pattern, val, env)?;
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
                name: None,
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

        CompiledExpr::Tuple(elements) => {
            let mut evaled = Vec::new();
            for elem in elements {
                evaled.push(eval(elem, env)?);
            }
            Ok(Value::Tuple(evaled))
        }
        CompiledExpr::Guard { check, expr } => {
            let val = eval(expr, env)?;
            if let Err(msg) = apply_guard_check(check, &val) {
                eprintln!("\x1b[91m/!\\ Guard Breached: {:?}\x1b[0m", check);
                eprintln!("{}", msg);
                if let Some(ref tracer) = env.tracer {
                    tracer.dump_ring_buffer();
                }
                std::process::exit(1);
            }
            Ok(val)
        }
    }
}

/// Check a guard condition against a value.
/// Returns Ok(()) if the check passes, Err(message) if it fails.
pub fn apply_guard_check(
    check: &crate::core::expr::GuardCheck,
    val: &Value,
) -> Result<(), String> {
    use crate::core::expr::GuardCheck;
    match check {
        GuardCheck::NoNan => {
            match val {
                Value::Tensor { data, .. }
                    if data.iter().any(|x| !x.is_finite()) => {
                        let stats = format_value_brief(val);
                        return Err(format!("Tensor contains NaN or Inf values: {}", stats));
                    }
                Value::Float(f)
                    if !f.is_finite() => {
                        return Err(format!("Value is {}", f));
                    }
                Value::Dict(map) => {
                    for (k, v) in map {
                        if let Err(e) = apply_guard_check(check, v) {
                            return Err(format!(":{} -> {}", k, e));
                        }
                    }
                }
                Value::List(items) => {
                    for (i, v) in items.iter().enumerate() {
                        if let Err(e) = apply_guard_check(check, v) {
                            return Err(format!("[{}] -> {}", i, e));
                        }
                    }
                }
                _ => {}
            }
            Ok(())
        }
        GuardCheck::Range { lo, hi } => {
            if let Value::Tensor { data, .. } = val {
                let v_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let v_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if (v_min as f64) < *lo || (v_max as f64) > *hi {
                    return Err(format!(
                        "Value range [{:.2e}, {:.2e}] outside allowed [{}, {}]",
                        v_min, v_max, lo, hi
                    ));
                }
            }
            Ok(())
        }
        GuardCheck::Shape(expected) => {
            if let Value::Tensor { data, .. } = val {
                let actual: Vec<i64> = data.shape().iter().map(|&d| d as i64).collect();
                if actual != *expected {
                    return Err(format!(
                        "Shape mismatch: expected {:?}, got {:?}",
                        expected, actual
                    ));
                }
            }
            Ok(())
        }
    }
}

fn format_value_brief(val: &Value) -> String {
    match val {
        Value::Tensor { data, .. } => {
            let shape: Vec<usize> = data.shape().to_vec();
            let shape_str: String = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
            let v_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let v_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            format!("f32[{}] [min:{:.2e} max:{:.2e}]", shape_str, v_min, v_max)
        }
        other => format!("{}", other),
    }
}

/// Bind a pattern name to a value in the current scope.
///
/// Patterns:
///   - Simple: `"x"` -> env["x"] = val
fn bind_pattern(pattern: &BindingPattern, val: Value, env: &mut Env) -> Result<(), SheafError> {
    match pattern {
        BindingPattern::Simple(name) => {
            env.set(name, val);
        }
        BindingPattern::Destructure(names) => {
            let items = match val {
                Value::List(items) | Value::Tuple(items) => items,
                Value::Tensor { ref data, .. } => {
                    if data.ndim() == 1 {
                        data.iter().map(|&x| Value::Float(x)).collect()
                    } else {
                        return Err(runtime_error(format!(
                            "let destructuring: expected a list or tuple, got tensor with shape {:?}", data.shape()
                        )));
                    }
                }
                other => return Err(runtime_error(format!(
                    "let destructuring: expected a list or tuple, got {}", other.type_name()
                ))),
            };
            let mut items_iter = items.into_iter();
            for name in names {
                match items_iter.next() {
                    Some(v) => bind_pattern(name, v, env)?,
                    None => return Err(runtime_error("let destructuring: arity mismatch".to_string())),
                }
            }
        }
    }
    Ok(())
}

fn eval_vector(elements: &[CompiledExpr], env: &mut Env) -> Result<Value, SheafError> {
    // Mirror of classify_vectors in lowering/transforms.rs.
    // Stacking rule (Decision D1, revisited):
    //   - All numeric scalars -> 1-D Tensor (matches classify rule 1/2).
    //   - All Tensors of the same inner shape (i.e. nested literal like [[1 2] [3 4]]) -> stack.
    //   - Heterogeneous or runtime expressions -> Value::List (classify rule 3).
    let vals: Result<Vec<Value>, _> = elements.iter().map(|e| eval(e, env)).collect();
    let vals = vals?;

    if vals.is_empty() {
        return Ok(Value::List(vec![]));
    }

    // Rule 1 (mirror of classify rule 2): all scalar -> 1-D tensor.
    let all_scalar = vals.iter().all(|v| matches!(v, Value::Int(_) | Value::Float(_)));
    if all_scalar {
        let data: Vec<f32> = vals.iter().map(|v| v.to_f64().unwrap() as f32).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap();
        return Ok(Value::tensor_f32(arr));
    }

    // Rule 2 (literal stacking only): stack nested *literal* vectors such as
    // `[[1 2] [3 4]]` into a 2-D tensor. Runtime vectors of tensors (e.g. `[a b]`
    // where a, b are tensors) are NOT stacked: they become a List, matching
    // classify_vectors which turns them into a Tuple (Decision D1: no implicit
    // stacking of runtime tensors).
    let all_literal_vectors = elements
        .iter()
        .all(|e| matches!(e, CompiledExpr::Vector(_)));
    let all_tensors = vals.iter().all(|v| matches!(v, Value::Tensor { .. }));
    if all_tensors && all_literal_vectors {
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

    // Otherwise: heterogeneous list (1+ tensor or tuple) -> List.
    // rule 3 of classify: at least one non-scalar -> List, not stacked tensor.
    Ok(Value::List(vals))
}

fn eval_dict(pairs: &[(CompiledExpr, CompiledExpr)], env: &mut Env) -> Result<Value, SheafError> {
    let mut map = BTreeMap::new();
    for (k, v) in pairs {
        let key = match eval(k, env)? {
            Value::Keyword(s) => s,
            Value::String(s) => s,
            other => return Err(runtime_error(format!("Dict key must be a keyword or a string, got {}", other.type_name()))),
        };
        let val = eval(v, env)?;
        map.insert(key, val);
    }
    Ok(Value::Dict(map))
}

fn sheaf_value_to_value(sv: &SheafValue) -> Result<Value, SheafError> {
    match sv {
        SheafValue::Integer(n, _) => Ok(Value::Int(*n)),
        SheafValue::Float(x, _) => Ok(Value::Float(*x as f32)),
        SheafValue::Boolean(b, _) => Ok(Value::Bool(*b)),
        SheafValue::Nil(_) => Ok(Value::Nil),
        SheafValue::String(s, _) => Ok(Value::String(s.clone())),
        SheafValue::Symbol(s, _) => Ok(Value::String(s.clone())),
        SheafValue::Keyword(k, _) => Ok(Value::Keyword(k.clone())),
        SheafValue::List(elems, _) | SheafValue::Vector(elems, _) => {
            let items: Result<Vec<Value>, _> = elems.iter().map(sheaf_value_to_value).collect();
            Ok(Value::List(items?))
        }
        SheafValue::Dict(pairs, _) => {
            let mut map = BTreeMap::new();
            for (k, v) in pairs {
                let key = match sheaf_value_to_value(k)? {
                    Value::Keyword(s) => s,
                    Value::String(s) => s,
                    other => return Err(runtime_error(format!("Dict key must be a keyword or a string, got {:?}", other))),
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

    // Only functions that declare keyword params consume :kw val pairs.
    // All others treat keywords as positional values (Clojure semantics).
let has_kwargs = matches!(name,
            "softmax" | "log-softmax" | "sum" | "mean" | "product"
            | "min" | "max" | "argmax" | "argmin" | "concat"
            | "leaky-relu" | "celu" | "var" | "normalize"
            | "range" | "tensor-split" | "slice" | "sort"
            | "print" | "choice" | "flip"
        );

    // Evaluate args, splitting kwargs only for functions that use them
    let (pos_args, kwargs) = if has_kwargs {
        split_kwargs(args, env)?
    } else {
        let pos: Result<Vec<Value>, _> = args.iter().map(|a| eval(a, env)).collect();
        (pos?, BTreeMap::new())
    };

    // Higher-order functions need &mut Env to call lambdas
    match name {
        "map" | "filter" | "reduce" | "scan" | "apply" | "find"
        | "tree-map" | "tree-reduce" | "flatten" | "vmap"
        | "__value-and-grad-hof__" => {
            if let Some(ref mut p) = env.profiler { p.enter(name); }
            let result = match name {
                "map" => hof::eval_map(&pos_args, env),
                "filter" => hof::eval_filter(&pos_args, env),
                "reduce" => hof::eval_reduce(&pos_args, env),
                "scan" => hof::eval_scan(&pos_args, env),
                "apply" => hof::eval_apply(&pos_args, env),
                "find" => hof::eval_find(&pos_args, env),
                "tree-map" => hof::eval_tree_map(&pos_args, env),
                "tree-reduce" => hof::eval_tree_reduce(&pos_args, env),
                "flatten" => hof::eval_flatten(&pos_args),
                "vmap" => hof::eval_vmap(&pos_args, env),
                "__value-and-grad-hof__" => vag::eval_value_and_grad_hof(&pos_args, env),
                _ => unreachable!(),
            };
            if let Some(ref mut p) = env.profiler { p.exit(); }
            return result;
        }
        _ => {}
    }

    // Try builtin from env
    if let Ok(Value::BuiltinFn { func, .. }) = env.get(name) {
        if let Some(ref mut p) = env.profiler { p.enter(name); }
        let result = func(&pos_args, &kwargs);
        if let Some(ref mut p) = env.profiler { p.exit(); }
        return result;
    }

    // Try user-defined function from registry
    if let Some(func_def) = env.registry.get(name).cloned() {
        // Arity check before any JIT or interpreter dispatch
        if pos_args.len() != func_def.params.len() {
            let got = pos_args.iter().map(|a| a.short_desc()).collect::<Vec<_>>().join(", ");
            let params = func_def.params.join(", ");
            return Err(runtime_error(format!(
                "{} expects {} arguments ({}), got {}.\n  Called with: ({})",
                name, func_def.params.len(), params, pos_args.len(), got
            )));
        }

        // Check evaluation deadline (used by auto-trace to avoid running forever)
        if let Some(deadline) = env.eval_deadline
            && std::time::Instant::now() > deadline {
                return Err(SheafError::Runtime {
                    message: "auto-trace timeout".to_string(),
                    location: None,
                });
            }

        // Record the first call for tracing (sheaf build --trace-with)
        if let Some(ref mut records) = env.call_records {
            let is_new = !records.contains_key(name);
            records.entry(name.to_string()).or_insert_with(|| {
                crate::interpreter::env::CallRecord {
                    arg_values: pos_args.clone(),
                }
            });
            if is_new {
                env.trace_stale_calls = 0;
                // Check if all target functions have been observed
                if let Some(ref targets) = env.trace_targets
                    && targets.iter().all(|t| records.contains_key(t.as_str())) {
                        return Err(SheafError::Runtime {
                            message: "trace complete".to_string(),
                            location: None,
                        });
                    }
            } else {
                env.trace_stale_calls += 1;
                // No new recordings for many calls, we're in a loop, stop
                if env.trace_stale_calls > 20 {
                    return Err(SheafError::Runtime {
                        message: "trace complete".to_string(),
                        location: None,
                    });
                }
            }
        }

        if let Some(ref mut p) = env.profiler { p.enter(name); }

        // VMFB/JIT dispatch: skip when tracing so the interpreter runs
        // and exposes the full call tree
        #[cfg(iree_runtime)]
        if env.tracer.is_none() {
            if let Some(result) = iree_dispatch::try_iree_dispatch(&func_def, &pos_args, env) {
                if let Some(ref mut p) = env.profiler { p.exit(); }
                return result;
            }

            let mut recompiled = false;
            if let Some(jit) = &mut env.jit_compiler {
                let shared_session = match crate::runtime::iree_session::shared_session() {
                    Ok(session) => session,
                    Err(error) => {
                        if let Some(ref mut p) = env.profiler { p.exit(); }
                        return Err(error);
                    }
                };
                if jit.try_jit_compile(&func_def, &pos_args, &env.registry, &shared_session).is_some() {
                    recompiled = true;
                    if let Some(result) = iree_dispatch::try_iree_dispatch(&func_def, &pos_args, env) {
                        if let Some(ref mut p) = env.profiler { p.exit(); }
                        return result;
                    }
                }
            }

            if recompiled {
                if let Some(ref mut p) = env.profiler { p.exit(); }
                let got = pos_args.iter().map(|a| a.short_desc()).collect::<Vec<_>>().join(", ");
                return Err(runtime_error(format!(
                    "{}: runtime error.\n  Called with: ({})\n  This is a bug in Sheaf. Please report it at https://github.com/sheaf-lang/sheaf/issues",
                    func_def.name, got
                )));
            }
        }

        // Interpret (only reached when no VMFB exists yet, i.e. first-time tracing)
        if let Some(ref body) = func_def.body_compiled {
            let tracing = env.tracer.as_ref().is_some_and(|t| t.is_active(name));
            if tracing {
                let mut tracer = env.tracer.take().unwrap();
                tracer.log_call(name, &pos_args);
                env.tracer = Some(tracer);
            }

            env.push_scope();
            for (param, val) in func_def.params.iter().zip(pos_args.iter()) {
                env.set(param, val.clone());
            }
            let result = eval(body, env);
            env.pop_scope();

            if tracing {
                let mut tracer = env.tracer.take().unwrap();
                if let Ok(ref val) = result {
                    tracer.log_return(name, val);
                    tracer.check_cli_guards(name, val);
                }
                env.tracer = Some(tracer);
            }

            if let Some(ref mut p) = env.profiler { p.exit(); }
            return result;
        }

        // AST-only function (macro helper): no compiled body available
        if func_def.body_compiled.is_none() {
            if let Some(ref mut p) = env.profiler { p.exit(); }
            return Err(runtime_error(format!(
                "{}: cannot call at runtime (compile-time only function)", name
            )));
        }

        if let Some(ref mut p) = env.profiler { p.exit(); }
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

pub(crate) fn call_function(func: &Value, args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    match func {
        Value::Function { closure, .. } => {
            // Detect vmap HOF closure: contains __vmap_fn__
            if let Some((_, vmap_fn)) = closure.iter().find(|(k, _)| k == "__vmap_fn__") {
                let axes = closure.iter().find(|(k, _)| k == "__vmap_axes__").map(|(_, v)| v.clone());
                return hof::eval_vmap_call(vmap_fn, axes.as_ref(), args, env);
            }
            // Detect value-and-grad HOF closure: contains __vag_fn__
            if let Some((_, vag_fn)) = closure.iter().find(|(k, _)| k == "__vag_fn__") {
                if args.len() != 1 {
                    return Err(runtime_error("value-and-grad: expected exactly 1 argument (params)"));
                }
                return vag::eval_value_and_grad_call(vag_fn, &args[0], env);
            }
            // Normal function call
            let Value::Function { params, body, closure, .. } = func else { unreachable!() };
            if let Some(ref mut p) = env.profiler { p.enter("<lambda>"); }
            env.push_scope();
            for (name, val) in closure {
                env.set(name, val.clone());
            }
            for (param, val) in params.iter().zip(args.iter()) {
                env.set(param, val.clone());
            }
            let result = eval(body, env);
            env.pop_scope();
            if let Some(ref mut p) = env.profiler { p.exit(); }
            result
        }
        Value::BuiltinFn { name, func } => {
            if let Some(ref mut p) = env.profiler { p.enter(name); }
            let result = func(args, &BTreeMap::new());
            if let Some(ref mut p) = env.profiler { p.exit(); }
            result
        }
        _ => Err(runtime_error(format!("Expected a function, got {}", func.short_desc()))),
    }
}

fn get_nested(val: &Value, indices: &[usize]) -> Result<Value, SheafError> {
    let mut current = val.clone();
    for &idx in indices {
        current = match current {
            Value::List(items) | Value::Tuple(items) => {
                items.get(idx).cloned().ok_or_else(|| {
                    runtime_error(format!("Tuple index {} out of bounds (len {})", idx, items.len()))
                })?
            }
            Value::Dict(map) => {

                map.values().nth(idx).cloned().ok_or_else(|| {
                    runtime_error(format!("Tuple index {} out of bounds (len {})", idx, map.len()))
                })?
            }
            _ => return Err(runtime_error(format!("Cannot index into {}", current.type_name()))),
        };
    }
    Ok(current)
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

    // First pass: compile all (registers defn, etc.)
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
