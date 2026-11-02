// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Simplifies an expression before symbolic autodiff.

use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::error::SheafError;
use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::Value;
use crate::interpreter::eval;
use crate::interpreter::hof::{dict_scan_length, slice_dict};
use std::collections::{HashMap, HashSet};

/// Values stored under temporary names while tracing.
pub struct LeafMap {
    pub leaves: Vec<(String, Value)>,
    inputs: HashSet<String>,
    counter: usize,
}

impl Default for LeafMap {
    fn default() -> Self {
        Self::new()
    }
}

impl LeafMap {
    pub fn new() -> Self {
        Self { leaves: Vec::new(), inputs: HashSet::new(), counter: 0 }
    }

    pub fn track_input(&mut self, name: &str) {
        self.inputs.insert(name.to_string());
    }

    pub fn register(&mut self, val: Value) -> String {
        let name = format!("__leaf_{}__", self.counter);
        self.counter += 1;
        // Later comparisons need the tensor data on the host.
        let host_val = val.ensure_host().unwrap_or(val);
        self.leaves.push((name.clone(), host_val));
        name
    }
}

fn is_tensor_leaf(val: &Value) -> bool {
    matches!(val, Value::Tensor { .. } | Value::Float(_) | Value::Int(_) | Value::DeviceBuffer(_))
}

/// Returns whether autodiff needs to see this operation.
fn is_differentiable_op(name: &str) -> bool {
    matches!(name,
        "+" | "-" | "*" | "/" | "@" | "**"
        | "relu" | "sigmoid" | "exp" | "log" | "sqrt" | "tanh"
        | "sum" | "mean" | "softmax" | "transpose" | "tr"
        | "maximum" | "minimum" | "where" | "neg"
        | "gelu" | "abs" | "@-grad-lhs" | "@-grad-rhs"
    )
}

/// Local names and the expressions they stand for.
type SymEnv = HashMap<String, CompiledExpr>;

/// Simplifies collection operations but leaves tensor math unchanged.
pub fn trace_expr(
    expr: &CompiledExpr,
    env: &mut Env,
    leaf_map: &mut LeafMap,
) -> Result<CompiledExpr, SheafError> {
    let mut sym_env = SymEnv::new();
    trace_rec(expr, env, leaf_map, &mut sym_env)
}

fn trace_rec(
    expr: &CompiledExpr,
    env: &mut Env,
    leaf_map: &mut LeafMap,
    sym_env: &mut SymEnv,
) -> Result<CompiledExpr, SheafError> {
    match expr {
        CompiledExpr::Float(_) | CompiledExpr::Integer(_) | CompiledExpr::Boolean(_) => {
            Ok(expr.clone())
        }
        CompiledExpr::Keyword(_) | CompiledExpr::String(_) | CompiledExpr::Nil => {
            Ok(expr.clone())
        }

        CompiledExpr::Symbol(name) => {
            if let Some(expr) = sym_env.get(name) {
                return Ok(expr.clone());
            }
            if leaf_map.inputs.contains(name)
                && let Ok(value) = env.get(name)
                && is_tensor_leaf(&value) {
                    let symbol = CompiledExpr::Symbol(leaf_map.register(value));
                    sym_env.insert(name.clone(), symbol.clone());
                    return Ok(symbol);
                }
            Ok(CompiledExpr::Symbol(name.clone()))
        }

        CompiledExpr::FunctionCall { name, args, .. } => {
            if name == "mean" && args.len() == 1 {
                // Express mean using operations understood by symbolic autodiff.
                let traced_arg = trace_rec(&args[0], env, leaf_map, sym_env)?;
                let concrete = eval(&args[0], env)?;
                let n = match &concrete {
                    Value::Tensor { data, .. } => data.len() as f64,
                    Value::Float(_) | Value::Int(_) => 1.0,
                    _ => 1.0,
                };
                return Ok(CompiledExpr::FunctionCall {
                    name: "/".to_string(),
                    args: vec![
                        CompiledExpr::FunctionCall {
                            name: "sum".to_string(),
                            args: vec![traced_arg],
                            loc: None,
                        },
                        CompiledExpr::Float(n),
                    ],
                    loc: None,
                });
            }

            if is_differentiable_op(name) {
                let traced_args: Result<Vec<CompiledExpr>, _> =
                    args.iter().map(|a| trace_rec(a, env, leaf_map, sym_env)).collect();
                return Ok(CompiledExpr::FunctionCall {
                    name: name.clone(),
                    args: traced_args?,
                    loc: None,
                });
            }

            // `(first (scan ...))` only needs the final carry.
            if name == "first" && args.len() == 1
                && let CompiledExpr::FunctionCall { name: inner, args: inner_args, .. } = &args[0]
                && inner == "scan"
            {
                return trace_reduce(inner_args, env, leaf_map, sym_env, true);
            }

            match name.as_str() {
                "get" => trace_get(args, env, leaf_map, sym_env),
                "reduce" => trace_reduce(args, env, leaf_map, sym_env, false),
                "scan" => trace_reduce(args, env, leaf_map, sym_env, true),
                _ => {
                    let val = eval(expr, env)?;
                    if is_tensor_leaf(&val) {
                        let sym = leaf_map.register(val);
                        Ok(CompiledExpr::Symbol(sym))
                    } else {
                        Ok(expr.clone())
                    }
                }
            }
        }

        CompiledExpr::Let { bindings, body } => {
            env.push_scope();
            let outer_sym_env = sym_env.clone();

            let result = (|| {
                for (bname, bval) in bindings {
                    match bname {
                        BindingPattern::Destructure(names) => {
                            let concrete = eval(bval, env)?;
                            let items = match concrete {
                                Value::List(items) | Value::Tuple(items) => items,
                                other => return Err(runtime_error(format!(
                                    "trace: destructuring expected list/tuple, got {}", other.type_name()
                                ))),
                            };
                            for (i, v) in items.into_iter().enumerate() {
                                if let Some(BindingPattern::Simple(n)) = names.get(i) {
                                    if is_tensor_leaf(&v) {
                                        let sym = leaf_map.register(v.clone());
                                        sym_env.insert(n.clone(), CompiledExpr::Symbol(sym));
                                    }
                                    env.set(n, v);
                                }
                            }
                        }
                        BindingPattern::Simple(name) => {
                            let traced_val = trace_rec(bval, env, leaf_map, sym_env)?;
                            let concrete = eval(bval, env)?;
                            env.set(name, concrete.clone());

                            if is_tensor_leaf(&concrete) {
                                sym_env.insert(name.clone(), traced_val);
                            }
                        }
                    }
                }

                trace_rec(body, env, leaf_map, sym_env)
            })();

            env.pop_scope();
            *sym_env = outer_sym_env;
            result
        }

        CompiledExpr::Do(exprs) => {
            if let Some(last) = exprs.last() {
                for e in &exprs[..exprs.len() - 1] {
                    let _ = eval(e, env);
                }
                trace_rec(last, env, leaf_map, sym_env)
            } else {
                Ok(CompiledExpr::Float(0.0))
            }
        }

        _ => {
            let val = eval(expr, env)?;
            if is_tensor_leaf(&val) {
                let sym = leaf_map.register(val);
                Ok(CompiledExpr::Symbol(sym))
            } else {
                Ok(expr.clone())
            }
        }
    }
}

/// Resolves a collection lookup and registers tensor results as leaves.
fn trace_get(
    args: &[CompiledExpr],
    env: &mut Env,
    leaf_map: &mut LeafMap,
    _sym_env: &mut SymEnv,
) -> Result<CompiledExpr, SheafError> {
    let get_expr = CompiledExpr::FunctionCall {
        name: "get".to_string(),
        args: args.to_vec(),
        loc: None,
    };
    let val = eval(&get_expr, env)?;

    if is_tensor_leaf(&val) {
        let sym = leaf_map.register(val);
        Ok(CompiledExpr::Symbol(sym))
    } else {
        Ok(get_expr)
    }
}

/// Unrolls a reduction or scan into a flat expression.
fn trace_reduce(
    args: &[CompiledExpr],
    env: &mut Env,
    leaf_map: &mut LeafMap,
    sym_env: &mut SymEnv,
    is_scan: bool,
) -> Result<CompiledExpr, SheafError> {
    if args.len() != 3 {
        return Err(runtime_error("trace: reduce requires 3 args (f init coll)"));
    }

    let lambda = &args[0];
    let init = &args[1];
    let coll = &args[2];

    let (lambda_params, lambda_body) = match lambda {
        CompiledExpr::Lambda { params, body, .. } => (params.clone(), body.as_ref().clone()),
        _ => return Err(runtime_error("trace: reduce function must be a lambda")),
    };

    if lambda_params.len() != 2 {
        return Err(runtime_error("trace: reduce lambda must take 2 params (acc, item)"));
    }

    let acc_param = &lambda_params[0];
    let item_param = &lambda_params[1];

    let coll_val = eval(coll, env)?;
    let items = match &coll_val {
        Value::List(items) => items.clone(),
        Value::Dict(map) => {
            let n = dict_scan_length(map)?;
            (0..n)
                .map(|i| slice_dict(map, i))
                .collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err(runtime_error("trace: reduce collection must be a list or dict of tensors")),
    };

    let mut acc_expr = trace_rec(init, env, leaf_map, sym_env)?;
    let mut acc_val = eval(init, env)?;

    for item_val in items.iter() {
        env.push_scope();
        env.set(acc_param, acc_val.clone());
        env.set(item_param, item_val.clone());

        let saved_acc = sym_env.insert(acc_param.clone(), acc_expr.clone());
        let saved_item = if is_tensor_leaf(item_val) {
            let leaf_sym = leaf_map.register(item_val.clone());
            env.set(&leaf_sym, item_val.clone());
            sym_env.insert(item_param.clone(), CompiledExpr::Symbol(leaf_sym))
        } else {
            sym_env.remove(item_param)
        };

        let step_expr = trace_rec(&lambda_body, env, leaf_map, sym_env)?;

        let step_val = eval(&lambda_body, env)?;
        env.pop_scope();

        if is_scan {
            // A scan result can be a pair or two tensors stacked along a new axis.
            let extract_carry = |val: &Value| -> Option<Value> {
                match val {
                    Value::List(items) | Value::Tuple(items) if items.len() == 2 => {
                        Some(items[0].clone())
                    }
                    Value::Tensor { data, .. } if data.shape()[0] == 2 => {
                        Some(Value::tensor_f32(data.index_axis(ndarray::Axis(0), 0).to_owned()))
                    }
                    _ => None,
                }
            };

            if let Some(carry_val) = extract_carry(&step_val) {
                acc_val = carry_val;
                acc_expr = match step_expr {
                    CompiledExpr::Vector(ref elems) if elems.len() == 2 => {
                        elems[0].clone()
                    }
                    _ => {
                        // Keep the concrete carry when tracing lost the pair structure.
                        let sym = leaf_map.register(acc_val.clone());
                        env.set(&sym, acc_val.clone());
                        CompiledExpr::Symbol(sym)
                    }
                };
            } else {
                return Err(runtime_error("trace: scan fn must return [carry output]"));
            }
        } else {
            acc_val = step_val;
            acc_expr = step_expr;
        }

        match saved_acc {
            Some(v) => { sym_env.insert(acc_param.clone(), v); }
            None => { sym_env.remove(acc_param); }
        }
        match saved_item {
            Some(v) => { sym_env.insert(item_param.clone(), v); }
            None => { sym_env.remove(item_param); }
        }
    }

    Ok(acc_expr)
}
