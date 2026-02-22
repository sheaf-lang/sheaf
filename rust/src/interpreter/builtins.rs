// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Built-in functions for the Sheaf interpreter.

use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::{Dtype, Value};
use ndarray::{ArrayD, Dimension, IxDyn};
use std::collections::BTreeMap;

type R = Result<Value, crate::core::error::SheafError>;

pub fn register_builtins(env: &mut Env) {
    env.set_builtin("+", builtin_add);
    env.set_builtin("-", builtin_sub);
    env.set_builtin("*", builtin_mul);
    env.set_builtin("/", builtin_div);
    env.set_builtin("//", builtin_floor_div);
    env.set_builtin("mod", builtin_mod);
    env.set_builtin("%", builtin_mod);
    env.set_builtin("**", builtin_pow);
    env.set_builtin("abs", builtin_abs);
    env.set_builtin("exp", builtin_exp);
    env.set_builtin("log", builtin_log);
    env.set_builtin("sqrt", builtin_sqrt);
    env.set_builtin("@", builtin_matmul);
    env.set_builtin("=", builtin_eq);
    env.set_builtin("==", builtin_elem_eq);
    env.set_builtin("!=", builtin_neq);
    env.set_builtin("<", builtin_lt);
    env.set_builtin(">", builtin_gt);
    env.set_builtin("<=", builtin_le);
    env.set_builtin(">=", builtin_ge);
    env.set_builtin("not", builtin_not);
    env.set_builtin("shape", builtin_shape);
    env.set_builtin("ndim", builtin_ndim);
    env.set_builtin("len", builtin_len);
    env.set_builtin("count", builtin_len);
    env.set_builtin("int", builtin_int);
    env.set_builtin("float", builtin_float);
    // Phase 2: Activations
    env.set_builtin("relu", builtin_relu);
    env.set_builtin("leaky-relu", builtin_leaky_relu);
    env.set_builtin("sigmoid", builtin_sigmoid);
    env.set_builtin("tanh", builtin_tanh);
    env.set_builtin("gelu", builtin_gelu);
    env.set_builtin("selu", builtin_selu);
    env.set_builtin("celu", builtin_celu);
    env.set_builtin("silu", builtin_silu);
    env.set_builtin("softmax", builtin_softmax);
    env.set_builtin("log-softmax", builtin_log_softmax);
    // Phase 2: Reductions
    env.set_builtin("sum", builtin_sum);
    env.set_builtin("mean", builtin_mean);
    env.set_builtin("product", builtin_product);
    env.set_builtin("min", builtin_min);
    env.set_builtin("max", builtin_max);
    env.set_builtin("minimum", builtin_minimum);
    env.set_builtin("maximum", builtin_maximum);
    env.set_builtin("argmax", builtin_argmax);
    env.set_builtin("argmin", builtin_argmin);
    // Phase 2: Tensor construction
    env.set_builtin("zeros", builtin_zeros);
    env.set_builtin("ones", builtin_ones);
    env.set_builtin("arange", builtin_arange);
    env.set_builtin("eye", builtin_eye);
    env.set_builtin("one-hot", builtin_one_hot);
    env.set_builtin("tril", builtin_tril);
    // Phase 2: Tensor ops
    env.set_builtin("reshape", builtin_reshape);
    env.set_builtin("transpose", builtin_transpose);
    env.set_builtin("concat", builtin_concat);
    env.set_builtin("slice", builtin_slice);
    env.set_builtin("get", builtin_get);
    env.set_builtin("where", builtin_where);
    env.set_builtin("roll", builtin_roll);
    env.set_builtin("index-update", builtin_index_update);
    // Phase 2: List ops
    env.set_builtin("first", builtin_first);
    env.set_builtin("second", builtin_second);
    env.set_builtin("last", builtin_last);
    env.set_builtin("rest", builtin_rest);
    env.set_builtin("nth", builtin_nth);
    env.set_builtin("cons", builtin_cons);
    env.set_builtin("append", builtin_append);
    env.set_builtin("empty?", builtin_empty);
    // Phase 2: Dict ops
    env.set_builtin("get-in", builtin_get_in);
    env.set_builtin("assoc", builtin_assoc);
    env.set_builtin("dissoc", builtin_dissoc);
    env.set_builtin("merge", builtin_merge);
    env.set_builtin("keys", builtin_keys);
    env.set_builtin("vals", builtin_vals);
    env.set_builtin("dict", builtin_dict);
    // Phase 2: String
    env.set_builtin("str", builtin_str);
    // Phase 3: Easy builtins
    env.set_builtin("tensor", builtin_tensor);
    env.set_builtin("range", builtin_range);
    env.set_builtin("swapaxes", builtin_swapaxes);
    env.set_builtin("var", builtin_var);
    env.set_builtin("normalize", builtin_normalize);
    env.set_builtin("index-of", builtin_index_of);
    env.set_builtin("gensym", builtin_gensym);
    env.set_builtin("symbol?", builtin_symbol_q);
    // Phase 3: Medium builtins
    env.set_builtin("einsum", builtin_einsum);
    env.set_builtin("append-and-roll", builtin_append_and_roll);
    env.set_builtin("dynamic-slice", builtin_dynamic_slice);
    env.set_builtin("mse-loss", builtin_mse_loss);
    env.set_builtin("mae-loss", builtin_mae_loss);
    env.set_builtin("sparse-cross-entropy", builtin_sparse_cross_entropy);
    env.set_builtin("tree-map-zeros", builtin_tree_map_zeros);
    env.set_builtin("print", builtin_print);
    env.set_builtin("io", builtin_io);
    env.set_builtin("random-key", builtin_random_key);
    env.set_builtin("random-split", builtin_random_split);
    env.set_builtin("random-normal", builtin_random_normal);
    env.set_builtin("random-uniform", builtin_random_uniform);
}

fn to_array(val: &Value) -> Result<(ArrayD<f64>, Dtype), crate::core::error::SheafError> {
    match val {
        Value::Int(n) => Ok((ArrayD::from_elem(IxDyn(&[]), *n as f64), Dtype::I32)),
        Value::Float(f) => Ok((ArrayD::from_elem(IxDyn(&[]), *f), Dtype::F32)),
        Value::Bool(b) => Ok((ArrayD::from_elem(IxDyn(&[]), if *b { 1.0 } else { 0.0 }), Dtype::I32)),
        Value::Tensor { data, dtype } => Ok((data.clone(), *dtype)),
        _ => Err(runtime_error(format!("Expected numeric value, got {}", val.type_name()))),
    }
}

fn result_dtype(a: Dtype, b: Dtype) -> Dtype {
    if a == Dtype::F32 || b == Dtype::F32 { Dtype::F32 } else if a == Dtype::Bool && b == Dtype::Bool { Dtype::Bool } else { Dtype::I32 }
}

fn binary_op(args: &[Value], op: fn(f64, f64) -> f64) -> R {
    if args.len() < 2 {
        return Err(runtime_error("Binary operation requires at least 2 arguments"));
    }
    let (mut acc, mut dt) = to_array(&args[0])?;
    let mut any_tensor = !matches!(&args[0], Value::Int(_) | Value::Float(_) | Value::Bool(_));
    for arg in &args[1..] {
        let (b, bdt) = to_array(arg)?;
        dt = result_dtype(dt, bdt);
        if !matches!(arg, Value::Int(_) | Value::Float(_) | Value::Bool(_)) {
            any_tensor = true;
        }
        if acc.shape() == &[] && b.shape() != &[] {
            let scalar = *acc.first().unwrap();
            acc = b.mapv(|x| op(scalar, x));
        } else if b.shape() == &[] && acc.shape() != &[] {
            let scalar = *b.first().unwrap();
            acc = acc.mapv(|x| op(x, scalar));
        } else if acc.shape() == b.shape() {
            acc = ndarray::Zip::from(&acc).and(&b).map_collect(|&a, &b| op(a, b));
        } else {
            // Try broadcasting a to b's shape, then b to a's shape
            if let Some(a_bc) = acc.broadcast(b.shape()) {
                let b_bc = b.broadcast(a_bc.shape()).ok_or_else(|| {
                    runtime_error(format!("Cannot broadcast shapes {:?} and {:?}", acc.shape(), b.shape()))
                })?.to_owned();
                acc = ndarray::Zip::from(&a_bc.to_owned()).and(&b_bc).map_collect(|&a, &b| op(a, b));
            } else if let Some(b_bc) = b.broadcast(acc.shape()) {
                let a_bc = &acc;
                acc = ndarray::Zip::from(a_bc).and(&b_bc.to_owned()).map_collect(|&a, &b| op(a, b));
            } else {
                return Err(runtime_error(format!("Cannot broadcast shapes {:?} and {:?}", acc.shape(), b.shape())));
            }
        }
    }
    // Tensor arithmetic always produces F32 (matches JAX behavior)
    if any_tensor {
        dt = Dtype::F32;
    }
    if acc.shape() == &[] {
        let x = *acc.first().unwrap();
        if dt == Dtype::I32 && x == x.floor() {
            Ok(Value::Int(x as i64))
        } else {
            Ok(Value::Float(x))
        }
    } else {
        Ok(Value::Tensor { data: acc, dtype: dt })
    }
}

fn unary_op(args: &[Value], op: fn(f64) -> f64) -> R {
    if args.is_empty() {
        return Err(runtime_error("Unary operation requires at least 1 argument"));
    }
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(op);
    if result.shape() == &[] {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: result, dtype: Dtype::F32 })
    }
}

// Like unary_op but casts through f32 for JAX-matching precision
fn unary_op_f32(args: &[Value], op: fn(f32) -> f32) -> R {
    if args.is_empty() {
        return Err(runtime_error("Unary operation requires at least 1 argument"));
    }
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(|x| op(x as f32) as f64);
    if result.shape() == &[] {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: result, dtype: Dtype::F32 })
    }
}

fn builtin_add(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    binary_op(args, |a, b| a + b)
}

fn builtin_sub(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() == 1 {
        // Unary negation
        let (arr, dt) = to_array(&args[0])?;
        let result = arr.mapv(|x| -x);
        if result.shape() == &[] {
            let x = *result.first().unwrap();
            if dt == Dtype::I32 { return Ok(Value::Int(x as i64)); }
            return Ok(Value::Float(x));
        }
        return Ok(Value::Tensor { data: result, dtype: dt });
    }
    binary_op(args, |a, b| a - b)
}

fn builtin_mul(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, |a, b| a * b)
}

fn builtin_div(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    // Division always returns float
    let result = binary_op(args, |a, b| a / b)?;
    match result {
        Value::Int(n) => Ok(Value::Float(n as f64)),
        Value::Tensor { data, .. } => Ok(Value::Tensor { data, dtype: Dtype::F32 }),
        other => Ok(other),
    }
}

fn builtin_floor_div(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, |a, b| (a / b).floor())
}

fn builtin_mod(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    // Python-style modulo: result has same sign as divisor
    binary_op(args, |a, b| ((a % b) + b) % b)
}

fn builtin_pow(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, |a, b| a.powf(b))
}

fn builtin_abs(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f64::abs)
}

fn builtin_exp(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f64::exp)
}

fn builtin_log(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op_f32(args, f32::ln)
}

fn builtin_sqrt(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f64::sqrt)
}

fn builtin_matmul(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("@ requires exactly 2 arguments"));
    }
    let (a, _) = to_array(&args[0])?;
    let (b, _) = to_array(&args[1])?;

    match (a.ndim(), b.ndim()) {
        (1, 1) => {
            // Dot product
            let a1 = a.into_dimensionality::<ndarray::Ix1>().map_err(|e| runtime_error(e.to_string()))?;
            let b1 = b.into_dimensionality::<ndarray::Ix1>().map_err(|e| runtime_error(e.to_string()))?;
            Ok(Value::Float(a1.dot(&b1)))
        }
        (2, 2) => {
            let a2 = a.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            let b2 = b.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            let c = a2.dot(&b2);
            Ok(Value::tensor_f32(c.into_dyn()))
        }
        (2, 1) => {
            let a2 = a.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            let b1 = b.into_dimensionality::<ndarray::Ix1>().map_err(|e| runtime_error(e.to_string()))?;
            let c = a2.dot(&b1);
            Ok(Value::tensor_f32(c.into_dyn()))
        }
        (1, 2) => {
            // Row vector @ matrix: [n] @ [n, m] → [m]
            let a1 = a.into_dimensionality::<ndarray::Ix1>().map_err(|e| runtime_error(e.to_string()))?;
            let b2 = b.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            let c = a1.dot(&b2);
            Ok(Value::tensor_f32(c.into_dyn()))
        }
        _ => Err(runtime_error(format!(
            "@ not supported for {}D x {}D", a.ndim(), b.ndim()
        ))),
    }
}

fn builtin_einsum(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 3 {
        return Err(runtime_error("einsum requires exactly 3 arguments: subscript, a, b"));
    }
    let subscript = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(runtime_error("einsum: first argument must be a subscript string")),
    };
    let (a, _) = to_array(&args[1])?;
    let (b, _) = to_array(&args[2])?;

    // Normalise ellipsis — for "...i,...i->..." with 1D inputs, ellipsis covers
    // zero batch dims, reducing to "i,i->"
    let subscript = subscript.replace("...", "");

    let arrow = subscript.find("->")
        .ok_or_else(|| runtime_error("einsum: subscript must contain '->'"))?;
    let lhs = &subscript[..arrow];
    let rhs = &subscript[arrow + 2..];
    let parts: Vec<&str> = lhs.split(',').collect();
    if parts.len() != 2 {
        return Err(runtime_error("einsum: only two-operand einsum is supported"));
    }
    let idx_a: Vec<char> = parts[0].chars().collect();
    let idx_b: Vec<char> = parts[1].chars().collect();
    let idx_out: Vec<char> = rhs.chars().collect();

    // Map each label → its dimension size
    let mut sizes: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (&label, &dim) in idx_a.iter().zip(a.shape().iter()) {
        sizes.insert(label, dim);
    }
    for (&label, &dim) in idx_b.iter().zip(b.shape().iter()) {
        sizes.insert(label, dim);
    }

    // Output shape
    let out_shape: Vec<usize> = idx_out.iter()
        .map(|c| *sizes.get(c).unwrap_or(&1))
        .collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut result = vec![0.0f64; out_len];

    // All labels in stable order: output labels first, then contracted
    let mut all_labels: Vec<char> = idx_out.clone();
    for &c in idx_a.iter().chain(idx_b.iter()) {
        if !all_labels.contains(&c) {
            all_labels.push(c);
        }
    }

    let label_sizes: Vec<usize> = all_labels.iter()
        .map(|c| *sizes.get(c).unwrap_or(&1))
        .collect();

    let label_pos: std::collections::HashMap<char, usize> = all_labels.iter()
        .enumerate().map(|(i, &c)| (c, i)).collect();

    let out_strides: Vec<usize> = (0..out_shape.len()).map(|i| {
        out_shape[i + 1..].iter().product::<usize>().max(1)
    }).collect();

    // Iterate over all combinations of label values via a carry counter
    let total: usize = label_sizes.iter().product::<usize>().max(1);
    let mut coords = vec![0usize; all_labels.len()];

    for _ in 0..total {
        let a_idx: Vec<usize> = idx_a.iter().map(|c| coords[label_pos[c]]).collect();
        let b_idx: Vec<usize> = idx_b.iter().map(|c| coords[label_pos[c]]).collect();
        let flat_out: usize = idx_out.iter().enumerate()
            .map(|(i, c)| coords[label_pos[c]] * out_strides[i])
            .sum();
        result[flat_out] += a[IxDyn(&a_idx)] * b[IxDyn(&b_idx)];

        // Increment coords (little-endian carry)
        for k in (0..coords.len()).rev() {
            coords[k] += 1;
            if coords[k] < label_sizes[k] { break; }
            coords[k] = 0;
        }
    }

    // Cast via f32 for JAX-matching precision
    let result_f32: Vec<f64> = result.iter().map(|&x| (x as f32) as f64).collect();

    if out_shape.is_empty() {
        Ok(Value::Float(result_f32[0]))
    } else {
        let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), result_f32)
            .map_err(|e| runtime_error(e.to_string()))?;
        Ok(Value::tensor_f32(arr))
    }
}

fn builtin_append_and_roll(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("append-and-roll requires 2 arguments: tensor, new-element"));
    }
    let (arr, _) = to_array(&args[0])?;
    if arr.ndim() != 1 {
        return Err(runtime_error("append-and-roll: first argument must be a 1D tensor"));
    }
    let new_val = args[1].to_f64()
        .ok_or_else(|| runtime_error("append-and-roll: second argument must be a number"))?;
    let n = arr.shape()[0];
    // Shift left by 1, append new value at the end
    let mut data: Vec<f64> = arr.iter().skip(1).copied().collect();
    data.push(new_val);
    let result = ArrayD::from_shape_vec(IxDyn(&[n]), data).unwrap();
    Ok(Value::tensor_f32(result))
}

fn builtin_eq(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 { return Err(runtime_error("= requires 2 arguments")); }
    let (a, _) = to_array(&args[0])?;
    let (b, _) = to_array(&args[1])?;
    Ok(Value::Bool(a == b))
}

fn builtin_elem_eq(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    cmp_op(args, |a, b| if (a - b).abs() < 1e-10 { 1.0 } else { 0.0 }, Dtype::I32)
}

fn builtin_neq(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    // For scalars: return Bool
    if args.len() == 2 {
        if let (Value::Int(_) | Value::Float(_) | Value::Bool(_), Value::Int(_) | Value::Float(_) | Value::Bool(_)) = (&args[0], &args[1]) {
            let a = args[0].to_f64().unwrap();
            let b = args[1].to_f64().unwrap();
            return Ok(Value::Bool((a - b).abs() > 1e-10));
        }
    }
    cmp_op(args, |a, b| if (a - b).abs() > 1e-10 { 1.0 } else { 0.0 }, Dtype::I32)
}

fn cmp_op(args: &[Value], op: fn(f64, f64) -> f64, _dt: Dtype) -> R {
    if args.len() != 2 { return Err(runtime_error("Comparison requires 2 arguments")); }
    let (a, _) = to_array(&args[0])?;
    let (b, _) = to_array(&args[1])?;
    // Broadcast
    if a.shape() == &[] && b.shape() != &[] {
        let scalar = *a.first().unwrap();
        let result = b.mapv(|x| op(scalar, x));
        return Ok(bool_tensor(result));
    }
    if b.shape() == &[] && a.shape() != &[] {
        let scalar = *b.first().unwrap();
        let result = a.mapv(|x| op(x, scalar));
        return Ok(bool_tensor(result));
    }
    if a.shape() == &[] && b.shape() == &[] {
        let r = op(*a.first().unwrap(), *b.first().unwrap());
        return Ok(Value::Bool(r != 0.0));
    }
    let result = ndarray::Zip::from(&a).and(&b).map_collect(|&a, &b| op(a, b));
    Ok(bool_tensor(result))
}

fn bool_tensor(data: ArrayD<f64>) -> Value {
    Value::Tensor { data, dtype: Dtype::Bool }
}

fn builtin_lt(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    cmp_op(args, |a, b| if a < b { 1.0 } else { 0.0 }, Dtype::I32)
}

fn builtin_gt(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() == 2 {
        if let (Some(a), Some(b)) = (args[0].to_f64(), args[1].to_f64()) {
            return Ok(Value::Bool(a > b));
        }
    }
    cmp_op(args, |a, b| if a > b { 1.0 } else { 0.0 }, Dtype::I32)
}

fn builtin_le(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    cmp_op(args, |a, b| if a <= b { 1.0 } else { 0.0 }, Dtype::I32)
}

fn builtin_ge(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    cmp_op(args, |a, b| if a >= b { 1.0 } else { 0.0 }, Dtype::I32)
}

fn builtin_not(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 1 { return Err(runtime_error("not requires 1 argument")); }
    Ok(Value::Bool(!args[0].is_truthy()))
}

fn builtin_shape(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 1 { return Err(runtime_error("shape requires 1 argument")); }
    match &args[0] {
        Value::Tensor { data, .. } => {
            let shape: Vec<f64> = data.shape().iter().map(|&s| s as f64).collect();
            Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[shape.len()]), shape).unwrap()))
        }
        Value::List(items) => Ok(Value::Int(items.len() as i64)),
        _ => Err(runtime_error(format!("shape: expected tensor, got {}", args[0].type_name()))),
    }
}

fn builtin_ndim(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 1 { return Err(runtime_error("ndim requires 1 argument")); }
    match &args[0] {
        Value::Tensor { data, .. } => Ok(Value::Int(data.ndim() as i64)),
        _ => Err(runtime_error(format!("ndim: expected tensor, got {}", args[0].type_name()))),
    }
}

fn builtin_len(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 1 { return Err(runtime_error("len requires 1 argument")); }
    match &args[0] {
        Value::Tensor { data, .. } => Ok(Value::Int(data.shape()[0] as i64)),
        Value::List(items) => Ok(Value::Int(items.len() as i64)),
        Value::Dict(map) => Ok(Value::Int(map.len() as i64)),
        Value::String(s) => Ok(Value::Int(s.len() as i64)),
        _ => Err(runtime_error(format!("len: expected collection, got {}", args[0].type_name()))),
    }
}

fn builtin_int(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Err(runtime_error("int requires at least 1 argument")); }
    match &args[0] {
        Value::Float(f) => Ok(Value::Int(*f as i64)),
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
        Value::Tensor { data, .. } => {
            Ok(Value::tensor_i32(data.mapv(|x| x.floor())))
        }
        _ => Err(runtime_error(format!("int: cannot convert {}", args[0].type_name()))),
    }
}

fn builtin_float(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Err(runtime_error("float requires at least 1 argument")); }
    match &args[0] {
        Value::Int(n) => Ok(Value::Float(*n as f64)),
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
        Value::Tensor { data, .. } => {
            Ok(Value::tensor_f32(data.clone()))
        }
        _ => Err(runtime_error(format!("float: cannot convert {}", args[0].type_name()))),
    }
}

fn get_axis(kw: &BTreeMap<String, Value>) -> Option<i64> {
    kw.get("axis").and_then(|v| match v {
        Value::Int(n) => Some(*n),
        Value::Float(f) => Some(*f as i64),
        _ => None,
    })
}

fn reduce_along_axis(arr: &ArrayD<f64>, axis: usize, op: fn(&[f64]) -> f64) -> ArrayD<f64> {
    let shape = arr.shape();
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        let data: Vec<f64> = arr.iter().copied().collect();
        return ArrayD::from_elem(IxDyn(&[]), op(&data));
    }
    let total = new_shape.iter().product::<usize>();
    let mut result_data = Vec::with_capacity(total);
    let n_axis = shape[axis];
    for idx in ndarray::indices(&*new_shape) {
        let mut vals = Vec::with_capacity(n_axis);
        for k in 0..n_axis {
            let mut full_idx: Vec<usize> = idx.as_array_view().to_vec();
            full_idx.insert(axis, k);
            vals.push(arr[IxDyn(&full_idx)]);
        }
        result_data.push(op(&vals));
    }
    ArrayD::from_shape_vec(IxDyn(&new_shape), result_data).unwrap()
}

fn argreduce_along_axis(arr: &ArrayD<f64>, axis: usize, cmp: fn(f64, f64) -> bool) -> ArrayD<f64> {
    let shape = arr.shape();
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        let data: Vec<f64> = arr.iter().copied().collect();
        let idx = data.iter().enumerate().fold(0, |best, (i, &x)| if cmp(x, data[best]) { i } else { best });
        return ArrayD::from_elem(IxDyn(&[]), idx as f64);
    }
    let total = new_shape.iter().product::<usize>();
    let mut result_data = Vec::with_capacity(total);
    let n_axis = shape[axis];
    for idx in ndarray::indices(&*new_shape) {
        let mut best_idx = 0;
        let mut full_idx: Vec<usize> = idx.as_array_view().to_vec();
        full_idx.insert(axis, 0);
        let mut best_val = arr[IxDyn(&full_idx)];
        for k in 1..n_axis {
            full_idx[axis] = k;
            let v = arr[IxDyn(&full_idx)];
            if cmp(v, best_val) {
                best_val = v;
                best_idx = k;
            }
        }
        result_data.push(best_idx as f64);
    }
    ArrayD::from_shape_vec(IxDyn(&new_shape), result_data).unwrap()
}

fn builtin_relu(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, |x| if x > 0.0 { x } else { 0.0 })
}

fn builtin_leaky_relu(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let slope = kw.get("negative_slope")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.01) as f32;
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(|x| {
        let xf = x as f32;
        (if xf > 0.0 { xf } else { slope * xf }) as f64
    });
    if result.shape() == &[] {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: result, dtype: Dtype::F32 })
    }
}

fn builtin_sigmoid(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, |x| {
        let r = 1.0 / (1.0 + (-x).exp());
        // Clamp to f32 precision range
        if r < 1e-7 { 0.0 } else if r > 1.0 - 1e-7 { 1.0 } else { r }
    })
}

fn builtin_tanh(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f64::tanh)
}

fn builtin_gelu(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, |x| {
        0.5 * x * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 * (x + 0.044715 * x * x * x)).tanh())
    })
}

fn builtin_selu(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let alpha = 1.6732632423543772_f32;
    let scale = 1.0507009873554805_f32;
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(|x| {
        let xf = x as f32;
        (if xf > 0.0 { scale * xf } else { scale * alpha * (xf.exp() - 1.0) }) as f64
    });
    if result.shape() == &[] {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: result, dtype: Dtype::F32 })
    }
}

fn builtin_celu(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let alpha = kw.get("alpha").and_then(|v| v.to_f64()).unwrap_or(1.0) as f32;
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(|x| {
        let xf = x as f32;
        (if xf > 0.0 { xf } else { alpha * ((xf / alpha).exp() - 1.0) }) as f64
    });
    if result.shape() == &[] {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: result, dtype: Dtype::F32 })
    }
}

fn builtin_silu(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, |x| x / (1.0 + (-x).exp()))
}

fn builtin_softmax(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let axis = get_axis(kw).unwrap_or(-1);
    let ndim = arr.ndim();
    let ax = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };
    // Compute in f32 for JAX-matching precision
    let arr_f32 = arr.mapv(|x| x as f32);
    let max_arr = reduce_along_axis(&arr_f32.mapv(|x| x as f64), ax, |v| v.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let max_bc = max_arr.insert_axis(ndarray::Axis(ax));
    let shifted = (&arr - &max_bc).mapv(|x| (x as f32) as f64);
    let exp_arr = shifted.mapv(|x| (x as f32).exp() as f64);
    let sum_arr = reduce_along_axis(&exp_arr, ax, |v| v.iter().sum::<f64>());
    let sum_bc = sum_arr.insert_axis(ndarray::Axis(ax));
    let result = (&exp_arr / &sum_bc).mapv(|x| (x as f32) as f64);
    Ok(Value::tensor_f32(result))
}

fn builtin_log_softmax(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let axis = get_axis(kw).unwrap_or(-1);
    let ndim = arr.ndim();
    let ax = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };
    let max_arr = reduce_along_axis(&arr, ax, |v| v.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let max_bc = max_arr.insert_axis(ndarray::Axis(ax));
    let shifted = (&arr - &max_bc).mapv(|x| (x as f32) as f64);
    let exp_arr = shifted.mapv(|x| (x as f32).exp() as f64);
    let sum_arr = reduce_along_axis(&exp_arr, ax, |v| v.iter().sum::<f64>());
    let log_sum = sum_arr.mapv(|x| (x as f32).ln() as f64);
    let log_sum_bc = log_sum.insert_axis(ndarray::Axis(ax));
    let result = (&shifted - &log_sum_bc).mapv(|x| (x as f32) as f64);
    Ok(Value::tensor_f32(result))
}

fn builtin_sum(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = if axis < 0 { (arr.ndim() as i64 + axis) as usize } else { axis as usize };
        let result = reduce_along_axis(&arr, ax, |v| v.iter().sum());
        Ok(Value::tensor_f32(result))
    } else {
        Ok(Value::Float(arr.iter().sum()))
    }
}

fn builtin_mean(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = if axis < 0 { (arr.ndim() as i64 + axis) as usize } else { axis as usize };
        let result = reduce_along_axis(&arr, ax, |v| v.iter().sum::<f64>() / v.len() as f64);
        Ok(Value::tensor_f32(result))
    } else {
        let n = arr.len() as f64;
        Ok(Value::Float(arr.iter().sum::<f64>() / n))
    }
}

fn builtin_product(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = if axis < 0 { (arr.ndim() as i64 + axis) as usize } else { axis as usize };
        let result = reduce_along_axis(&arr, ax, |v| v.iter().product());
        if result.shape().is_empty() {
            Ok(Value::Float(*result.first().unwrap()))
        } else {
            Ok(Value::tensor_f32(result))
        }
    } else {
        Ok(Value::Float(arr.iter().product()))
    }
}

fn builtin_min(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = if axis < 0 { (arr.ndim() as i64 + axis) as usize } else { axis as usize };
        let result = reduce_along_axis(&arr, ax, |v| v.iter().copied().fold(f64::INFINITY, f64::min));
        Ok(Value::tensor_f32(result))
    } else {
        Ok(Value::Float(arr.iter().copied().fold(f64::INFINITY, f64::min)))
    }
}

fn builtin_max(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    // Variadic: (max a b c) → max of scalars
    if args.len() > 1 {
        let vals: Result<Vec<f64>, _> = args.iter().map(|a| a.to_f64().ok_or_else(|| runtime_error("max: expected number"))).collect();
        return Ok(Value::Float(vals?.into_iter().fold(f64::NEG_INFINITY, f64::max)));
    }
    // List: (max [a b c]) → max of list elements
    if let Value::List(items) = &args[0] {
        let vals: Result<Vec<f64>, _> = items.iter().map(|a| a.to_f64().ok_or_else(|| runtime_error("max: list must contain numbers"))).collect();
        return Ok(Value::Float(vals?.into_iter().fold(f64::NEG_INFINITY, f64::max)));
    }
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = if axis < 0 { (arr.ndim() as i64 + axis) as usize } else { axis as usize };
        let result = reduce_along_axis(&arr, ax, |v| v.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        Ok(Value::tensor_f32(result))
    } else {
        Ok(Value::Float(arr.iter().copied().fold(f64::NEG_INFINITY, f64::max)))
    }
}

fn builtin_minimum(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, f64::min)
}

fn builtin_maximum(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, f64::max)
}

fn builtin_argmax(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = if axis < 0 { (arr.ndim() as i64 + axis) as usize } else { axis as usize };
        let result = argreduce_along_axis(&arr, ax, |a, b| a > b);
        Ok(Value::tensor_i32(result))
    } else {
        let data: Vec<f64> = arr.iter().copied().collect();
        let idx = data.iter().enumerate().fold(0, |best, (i, &x)| if x > data[best] { i } else { best });
        Ok(Value::Int(idx as i64))
    }
}

fn builtin_argmin(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = if axis < 0 { (arr.ndim() as i64 + axis) as usize } else { axis as usize };
        let result = argreduce_along_axis(&arr, ax, |a, b| a < b);
        Ok(Value::tensor_i32(result))
    } else {
        let data: Vec<f64> = arr.iter().copied().collect();
        let idx = data.iter().enumerate().fold(0, |best, (i, &x)| if x < data[best] { i } else { best });
        Ok(Value::Int(idx as i64))
    }
}

fn shape_from_value(val: &Value) -> Result<Vec<usize>, crate::core::error::SheafError> {
    match val {
        Value::List(items) => {
            items.iter().map(|v| match v {
                Value::Int(n) => Ok(*n as usize),
                Value::Float(f) => Ok(*f as usize),
                _ => Err(runtime_error("shape must contain integers")),
            }).collect()
        }
        Value::Tensor { data, .. } => {
            Ok(data.iter().map(|&x| x as usize).collect())
        }
        _ => Err(runtime_error(format!("Expected shape list, got {}", val.type_name()))),
    }
}

fn builtin_zeros(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let shape = shape_from_value(&args[0])?;
    Ok(Value::tensor_f32(ArrayD::zeros(IxDyn(&shape))))
}

fn builtin_ones(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let shape = shape_from_value(&args[0])?;
    Ok(Value::tensor_f32(ArrayD::ones(IxDyn(&shape))))
}

fn builtin_arange(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (start, stop, step) = match args.len() {
        1 => (0i64, args[0].to_f64().unwrap() as i64, 1i64),
        2 => (args[0].to_f64().unwrap() as i64, args[1].to_f64().unwrap() as i64, 1),
        _ => (args[0].to_f64().unwrap() as i64, args[1].to_f64().unwrap() as i64, args[2].to_f64().unwrap() as i64),
    };
    let mut data = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        data.push(i as f64);
        i += step;
    }
    Ok(Value::tensor_i32(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap()))
}

fn builtin_eye(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let n = args[0].to_f64().unwrap() as usize;
    let m = if args.len() > 1 { args[1].to_f64().unwrap() as usize } else { n };
    let mut data = vec![0.0; n * m];
    for i in 0..n.min(m) {
        data[i * m + i] = 1.0;
    }
    Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[n, m]), data).unwrap()))
}

fn builtin_one_hot(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let num_classes = args[1].to_f64().unwrap() as usize;
    match &args[0] {
        Value::Int(idx) => {
            let mut data = vec![0.0; num_classes];
            data[*idx as usize] = 1.0;
            Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[num_classes]), data).unwrap()))
        }
        Value::Tensor { data: indices, .. } => {
            let n = indices.len();
            let mut result = vec![0.0; n * num_classes];
            for (i, &idx) in indices.iter().enumerate() {
                result[i * num_classes + idx as usize] = 1.0;
            }
            Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[n, num_classes]), result).unwrap()))
        }
        _ => Err(runtime_error("one-hot: expected int or tensor")),
    }
}

fn builtin_tril(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if arr.ndim() != 2 { return Err(runtime_error("tril: expected 2D tensor")); }
    let shape = arr.shape();
    let (n, m) = (shape[0], shape[1]);
    let mut result = arr.clone();
    for i in 0..n {
        for j in (i + 1)..m {
            result[IxDyn(&[i, j])] = 0.0;
        }
    }
    Ok(Value::tensor_f32(result))
}

fn builtin_reshape(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, dt) = to_array(&args[0])?;
    let shape_val = &args[1];
    let raw_shape: Vec<i64> = match shape_val {
        Value::List(items) => items.iter().map(|v| match v {
            Value::Int(n) => *n,
            Value::Float(f) => *f as i64,
            _ => -999,
        }).collect(),
        Value::Tensor { data, .. } => data.iter().map(|&x| x as i64).collect(),
        _ => return Err(runtime_error("reshape: expected shape list")),
    };
    let total = arr.len() as i64;
    let neg_idx = raw_shape.iter().position(|&x| x < 0);
    let new_shape: Vec<usize> = if let Some(_ni) = neg_idx {
        let known: i64 = raw_shape.iter().filter(|&&x| x > 0).product();
        let inferred = total / known;
        raw_shape.iter().map(|&x| if x < 0 { inferred as usize } else { x as usize }).collect()
    } else {
        raw_shape.iter().map(|&x| x as usize).collect()
    };
    let result = arr.into_shape_with_order(IxDyn(&new_shape)).map_err(|e| runtime_error(e.to_string()))?;
    Ok(Value::Tensor { data: result, dtype: dt })
}

fn builtin_transpose(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if arr.ndim() == 2 {
        Ok(Value::tensor_f32(arr.t().to_owned()))
    } else if arr.ndim() == 1 {
        Ok(Value::tensor_f32(arr))
    } else {
        let mut axes: Vec<usize> = (0..arr.ndim()).rev().collect();
        if args.len() > 1 {
            if let Value::List(items) = &args[1] {
                axes = items.iter().map(|v| v.to_f64().unwrap() as usize).collect();
            }
        }
        Ok(Value::tensor_f32(arr.permuted_axes(IxDyn(&axes))))
    }
}

fn list_to_tensor(v: &Value) -> Option<(ArrayD<f64>, Dtype)> {
    match v {
        Value::Tensor { data, dtype } => Some((data.clone(), *dtype)),
        Value::List(items) => {
            // 1D list of numbers
            let all_int = items.iter().all(|x| matches!(x, Value::Int(_)));
            let nums: Option<Vec<f64>> = items.iter().map(|x| x.to_f64()).collect();
            if let Some(data) = nums {
                let dtype = if all_int { Dtype::I32 } else { Dtype::F32 };
                return ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).ok()
                    .map(|a| (a, dtype));
            }
            // 2D list of lists of numbers
            let rows: Option<Vec<(ArrayD<f64>, Dtype)>> = items.iter().map(|x| list_to_tensor(x)).collect();
            if let Some(rows) = rows {
                let all_i32 = rows.iter().all(|(_, dt)| *dt == Dtype::I32);
                let dtype = if all_i32 { Dtype::I32 } else { Dtype::F32 };
                let stacked: Option<ArrayD<f64>> = ndarray::concatenate(
                    ndarray::Axis(0),
                    &rows.iter().map(|(r, _)| r.view().insert_axis(ndarray::Axis(0))).collect::<Vec<_>>()
                ).ok();
                stacked.map(|a| (a, dtype))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn builtin_concat(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let axis = get_axis(kw).unwrap_or(0) as usize;
    let has_axis_kw = kw.contains_key("axis");

    // Try to convert all args to tensors (works for both Tensor and numeric List)
    let maybe_arrays: Option<Vec<(ArrayD<f64>, Dtype)>> = args.iter().map(|a| list_to_tensor(a)).collect();

    if let Some(arrays) = maybe_arrays {
        if has_axis_kw || args.iter().any(|a| matches!(a, Value::Tensor { .. })) {
            // Tensor concat along axis
            let all_i32 = arrays.iter().all(|(_, dt)| *dt == Dtype::I32);
            let dtype = if all_i32 { Dtype::I32 } else { Dtype::F32 };
            let views: Vec<ndarray::ArrayViewD<f64>> = arrays.iter().map(|(a, _)| a.view()).collect();
            let result = ndarray::concatenate(ndarray::Axis(axis), &views)
                .map_err(|e| runtime_error(e.to_string()))?;
            return Ok(Value::Tensor { data: result, dtype });
        }
    }

    // Flat list concat (no axis, lists of non-numeric or heterogeneous)
    if matches!(&args[0], Value::List(_)) {
        let mut all_items = Vec::new();
        for arg in args {
            match arg {
                Value::List(items) => all_items.extend(items.clone()),
                _ => all_items.push(arg.clone()),
            }
        }
        return Ok(Value::List(all_items));
    }

    // Tensor args
    let arrays: Vec<ArrayD<f64>> = args.iter().map(|a| {
        to_array(a).map(|(arr, _)| arr)
    }).collect::<Result<Vec<_>, _>>()?;
    let views: Vec<ndarray::ArrayViewD<f64>> = arrays.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(ndarray::Axis(axis), &views)
        .map_err(|e| runtime_error(e.to_string()))?;
    Ok(Value::tensor_f32(result))
}

fn builtin_slice(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let start = args[1].to_f64().unwrap() as usize;
    let end = args[2].to_f64().unwrap() as usize;
    let sliced = arr.slice_axis(ndarray::Axis(0), ndarray::Slice::from(start..end));
    Ok(Value::tensor_f32(sliced.to_owned()))
}

fn builtin_get(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            let key = match &args[1] {
                Value::Keyword(k) => k.clone(),
                Value::String(s) => s.clone(),
                _ => return Err(runtime_error("get: key must be keyword or string")),
            };
            match map.get(&key) {
                Some(v) => Ok(v.clone()),
                None => {
                    if args.len() > 2 { Ok(args[2].clone()) }
                    else if let Some(default) = kw.get("default") { Ok(default.clone()) }
                    else { Ok(Value::Nil) }
                }
            }
        }
        Value::Tensor { data, .. } => {
            let idx = args[1].to_f64().unwrap() as usize;
            let sliced = data.index_axis(ndarray::Axis(0), idx).to_owned();
            if sliced.shape().is_empty() {
                Ok(Value::Float(*sliced.first().unwrap()))
            } else {
                Ok(Value::tensor_f32(sliced))
            }
        }
        Value::List(items) => {
            let idx = args[1].to_f64().unwrap() as usize;
            items.get(idx).cloned().ok_or_else(|| runtime_error("get: index out of bounds"))
        }
        _ => Err(runtime_error(format!("get: expected dict/tensor/list, got {}", args[0].type_name()))),
    }
}

fn builtin_where(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (cond, _) = to_array(&args[0])?;
    let (on_true, _) = to_array(&args[1])?;
    let (on_false, _) = to_array(&args[2])?;
    let on_true_bc = if on_true.shape() == &[] {
        ArrayD::from_elem(cond.raw_dim(), *on_true.first().unwrap())
    } else { on_true };
    let on_false_bc = if on_false.shape() == &[] {
        ArrayD::from_elem(cond.raw_dim(), *on_false.first().unwrap())
    } else { on_false };
    let result = ndarray::Zip::from(&cond).and(&on_true_bc).and(&on_false_bc)
        .map_collect(|&c, &t, &f| if c != 0.0 { t } else { f });
    Ok(Value::tensor_f32(result))
}

fn builtin_roll(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let shift = args[1].to_f64().unwrap() as i64;
    let data: Vec<f64> = arr.iter().copied().collect();
    let n = data.len() as i64;
    let shift = ((shift % n) + n) % n;
    let mut result = vec![0.0; data.len()];
    for (i, &v) in data.iter().enumerate() {
        let new_i = ((i as i64 + shift) % n) as usize;
        result[new_i] = v;
    }
    Ok(Value::tensor_f32(ArrayD::from_shape_vec(arr.raw_dim(), result).unwrap()))
}

fn builtin_index_update(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (mut arr, _dt) = to_array(&args[0])?;
    let idx = args[1].to_f64().unwrap() as usize;
    match &args[2] {
        Value::Tensor { data: new_val, .. } => {
            let mut slice = arr.index_axis_mut(ndarray::Axis(0), idx);
            slice.assign(new_val);
        }
        other => {
            let v = other.to_f64().unwrap();
            arr[IxDyn(&[idx])] = v;
        }
    }
    Ok(Value::tensor_f32(arr))
}

fn builtin_first(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => items.first().cloned().ok_or_else(|| runtime_error("first: empty list")),
        Value::Tensor { data, .. } => {
            let sliced = data.index_axis(ndarray::Axis(0), 0).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(*sliced.first().unwrap())) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        _ => Err(runtime_error("first: expected list or tensor")),
    }
}

fn builtin_second(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => items.get(1).cloned().ok_or_else(|| runtime_error("second: list too short")),
        Value::Tensor { data, .. } => {
            let sliced = data.index_axis(ndarray::Axis(0), 1).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(*sliced.first().unwrap())) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        _ => Err(runtime_error("second: expected list or tensor")),
    }
}

fn builtin_last(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => items.last().cloned().ok_or_else(|| runtime_error("last: empty list")),
        Value::Tensor { data, .. } => {
            let n = data.shape()[0];
            let sliced = data.index_axis(ndarray::Axis(0), n - 1).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(*sliced.first().unwrap())) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        _ => Err(runtime_error("last: expected list or tensor")),
    }
}

fn builtin_rest(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            if items.is_empty() { Ok(Value::List(vec![])) }
            else { Ok(Value::List(items[1..].to_vec())) }
        }
        _ => Err(runtime_error("rest: expected list")),
    }
}

fn builtin_nth(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let idx = args[1].to_f64().unwrap() as usize;
    match &args[0] {
        Value::List(items) => items.get(idx).cloned().ok_or_else(|| runtime_error("nth: index out of bounds")),
        Value::Tensor { data, .. } => {
            let sliced = data.index_axis(ndarray::Axis(0), idx).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(*sliced.first().unwrap())) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        _ => Err(runtime_error("nth: expected list or tensor")),
    }
}

fn builtin_cons(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let head = args[0].clone();
    match &args[1] {
        Value::List(items) => {
            let mut new = vec![head];
            new.extend(items.clone());
            Ok(Value::List(new))
        }
        _ => Err(runtime_error("cons: second arg must be list")),
    }
}

fn builtin_append(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            let mut new = items.clone();
            new.push(args[1].clone());
            Ok(Value::List(new))
        }
        _ => Err(runtime_error("append: first arg must be list")),
    }
}

fn builtin_empty(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => Ok(Value::Bool(items.is_empty())),
        Value::Tensor { data, .. } => Ok(Value::Bool(data.is_empty())),
        Value::Dict(map) => Ok(Value::Bool(map.is_empty())),
        _ => Ok(Value::Bool(false)),
    }
}

fn builtin_get_in(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let path = match &args[1] {
        Value::List(items) => items.clone(),
        Value::Tensor { data, .. } => data.iter().map(|&x| Value::Int(x as i64)).collect(),
        _ => return Err(runtime_error("get-in: path must be a list")),
    };
    let default = if args.len() > 2 { Some(args[2].clone()) } else { None };
    let mut current = args[0].clone();
    for key in &path {
        current = match (&current, key) {
            (Value::Dict(map), Value::Keyword(k)) | (Value::Dict(map), Value::String(k)) => {
                match map.get(k) {
                    Some(v) => v.clone(),
                    None => return Ok(default.unwrap_or(Value::Nil)),
                }
            }
            (Value::Tensor { data, .. }, Value::Int(idx)) => {
                let sliced = data.index_axis(ndarray::Axis(0), *idx as usize).to_owned();
                if sliced.shape().is_empty() { Value::Float(*sliced.first().unwrap()) }
                else { Value::tensor_f32(sliced) }
            }
            (Value::List(items), Value::Int(idx)) => {
                match items.get(*idx as usize) {
                    Some(v) => v.clone(),
                    None => return Ok(default.unwrap_or(Value::Nil)),
                }
            }
            _ => return Ok(default.unwrap_or(Value::Nil)),
        };
    }
    Ok(current)
}

fn builtin_assoc(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            let mut new = map.clone();
            let key = match &args[1] {
                Value::Keyword(k) => k.clone(),
                Value::String(s) => s.clone(),
                _ => return Err(runtime_error("assoc: key must be keyword or string")),
            };
            new.insert(key, args[2].clone());
            Ok(Value::Dict(new))
        }
        _ => Err(runtime_error("assoc: expected dict")),
    }
}

fn builtin_dissoc(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            let keys_to_remove: Vec<String> = match &args[1] {
                Value::List(items) => items.iter().filter_map(|v| match v {
                    Value::Keyword(k) | Value::String(k) => Some(k.clone()),
                    _ => None,
                }).collect(),
                _ => return Err(runtime_error("dissoc: keys must be a list")),
            };
            let new: BTreeMap<String, Value> = map.iter()
                .filter(|(k, _)| !keys_to_remove.contains(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Ok(Value::Dict(new))
        }
        _ => Err(runtime_error("dissoc: expected dict")),
    }
}

fn builtin_merge(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let mut result = BTreeMap::new();
    for arg in args {
        if let Value::Dict(map) = arg {
            result.extend(map.clone());
        } else {
            return Err(runtime_error("merge: expected dicts"));
        }
    }
    Ok(Value::Dict(result))
}

fn builtin_keys(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            Ok(Value::List(map.keys().map(|k| Value::String(k.clone())).collect()))
        }
        _ => Err(runtime_error("keys: expected dict")),
    }
}

fn builtin_vals(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            Ok(Value::List(map.values().cloned().collect()))
        }
        _ => Err(runtime_error("vals: expected dict")),
    }
}

fn builtin_dict(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let mut map = BTreeMap::new();
    let mut i = 0;
    while i + 1 < args.len() {
        let key = match &args[i] {
            Value::Keyword(k) => k.clone(),
            Value::String(s) => s.clone(),
            _ => return Err(runtime_error("dict: key must be keyword or string")),
        };
        map.insert(key, args[i + 1].clone());
        i += 2;
    }
    Ok(Value::Dict(map))
}

fn builtin_str(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Ok(Value::String(String::new())); }
    Ok(Value::String(format!("{}", args[0])))
}

fn builtin_tensor(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            let all_numeric = items.iter().all(|v| matches!(v, Value::Int(_) | Value::Float(_)));
            if all_numeric && !items.is_empty() {
                let data: Vec<f64> = items.iter().map(|v| v.to_f64().unwrap()).collect();
                let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap();
                Ok(Value::tensor_f32(arr))
            } else {
                Err(runtime_error("tensor: expected list of numbers"))
            }
        }
        Value::Tensor { .. } => Ok(args[0].clone()),
        _ => Err(runtime_error("tensor: expected list or tensor")),
    }
}

fn builtin_range(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    builtin_arange(args, kw)
}

fn builtin_swapaxes(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let ax0 = args[1].to_f64().unwrap() as usize;
    let ax1 = args[2].to_f64().unwrap() as usize;
    let mut axes: Vec<usize> = (0..arr.ndim()).collect();
    axes[ax0] = ax1;
    axes[ax1] = ax0;
    Ok(Value::tensor_f32(arr.permuted_axes(IxDyn(&axes))))
}

fn builtin_var(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = if axis < 0 { (arr.ndim() as i64 + axis) as usize } else { axis as usize };
        let mean_arr = reduce_along_axis(&arr, ax, |v| v.iter().sum::<f64>() / v.len() as f64);
        let mean_bc = mean_arr.insert_axis(ndarray::Axis(ax));
        let diff = &arr - &mean_bc;
        let sq = &diff * &diff;
        let result = reduce_along_axis(&sq, ax, |v| v.iter().sum::<f64>() / v.len() as f64);
        Ok(Value::tensor_f32(result))
    } else {
        let n = arr.len() as f64;
        let mean = arr.iter().sum::<f64>() / n;
        let var = arr.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
        Ok(Value::Float(var))
    }
}

fn builtin_normalize(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = if axis < 0 { (arr.ndim() as i64 + axis) as usize } else { axis as usize };
        let sum_arr = reduce_along_axis(&arr, ax, |v| v.iter().sum());
        let sum_bc = sum_arr.insert_axis(ndarray::Axis(ax));
        Ok(Value::tensor_f32((&arr / &sum_bc).mapv(|x| (x as f32) as f64)))
    } else {
        let total: f64 = arr.iter().sum();
        Ok(Value::tensor_f32(arr.mapv(|x| ((x / total) as f32) as f64)))
    }
}

fn builtin_index_of(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            let target = &args[1];
            for (i, item) in items.iter().enumerate() {
                let eq = match (item, target) {
                    (Value::Int(a), Value::Int(b)) => a == b,
                    (Value::Float(a), Value::Float(b)) => (a - b).abs() < 1e-10,
                    (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => (*a as f64 - b).abs() < 1e-10,
                    (Value::String(a), Value::String(b)) => a == b,
                    (Value::Keyword(a), Value::Keyword(b)) => a == b,
                    (Value::Bool(a), Value::Bool(b)) => a == b,
                    _ => false,
                };
                if eq { return Ok(Value::Int(i as i64)); }
            }
            Ok(Value::Int(-1))
        }
        _ => Err(runtime_error("index-of: expected list")),
    }
}

fn builtin_gensym(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let prefix = if args.is_empty() {
        "g".to_string()
    } else {
        match &args[0] {
            Value::String(s) => s.clone(),
            _ => format!("{}", args[0]),
        }
    };
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let hash = format!("{:08x}", (t.as_nanos() & 0xFFFFFFFF) as u32);
    Ok(Value::String(format!("{}{}", prefix, hash)))
}

fn builtin_symbol_q(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Err(runtime_error("symbol? requires 1 argument")); }
    match &args[0] {
        Value::String(_) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

fn builtin_dynamic_slice(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let start = args[1].to_f64().unwrap() as usize;
    let end = args[2].to_f64().unwrap() as usize;
    // end is inclusive
    let sliced = arr.slice_axis(ndarray::Axis(0), ndarray::Slice::from(start..=end));
    Ok(Value::tensor_i32(sliced.to_owned()))
}

fn builtin_mse_loss(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (pred, _) = to_array(&args[0])?;
    let (target, _) = to_array(&args[1])?;
    let diff = &pred - &target;
    let mse_f64 = diff.iter().map(|&x| x * x).sum::<f64>() / pred.len() as f64;
    // Round to f32 precision to match JAX output
    Ok(Value::Float((mse_f64 as f32) as f64))
}

fn builtin_mae_loss(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (pred, _) = to_array(&args[0])?;
    let (target, _) = to_array(&args[1])?;
    let diff = &pred - &target;
    let mae_f64 = diff.iter().map(|&x| x.abs()).sum::<f64>() / pred.len() as f64;
    Ok(Value::Float((mae_f64 as f32) as f64))
}

fn builtin_sparse_cross_entropy(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    // (sparse-cross-entropy logits labels :i32)
    // logits: [batch, num_classes], labels: [batch] integer indices
    let (logits, _) = to_array(&args[0])?;
    let (labels, _) = to_array(&args[1])?;
    if logits.ndim() != 2 {
        return Err(runtime_error("sparse-cross-entropy: logits must be 2D [batch, classes]"));
    }
    let batch = logits.shape()[0];
    let num_classes = logits.shape()[1];
    let mut total_loss = 0.0f64;
    for i in 0..batch {
        let class_idx = labels[IxDyn(&[i])] as usize;
        if class_idx >= num_classes {
            return Err(runtime_error(format!(
                "sparse-cross-entropy: label {} out of range [0, {})", class_idx, num_classes
            )));
        }
        // log-softmax of the correct class
        let row: Vec<f64> = (0..num_classes).map(|j| logits[IxDyn(&[i, j])]).collect();
        let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<f64> = row.iter().map(|&x| x - max_val).collect();
        let log_sum = shifted.iter().map(|&x| x.exp()).sum::<f64>().ln();
        let log_prob = shifted[class_idx] - log_sum;
        total_loss += -log_prob;
    }
    let mean_loss = total_loss / batch as f64;
    Ok(Value::Float((mean_loss as f32) as f64))
}

/// Recursively zero-fill a pytree (nested dicts / tensors / scalars).
pub fn tree_zeros(val: &Value) -> Value {
    match val {
        Value::Dict(map) => {
            Value::Dict(map.iter().map(|(k, v)| (k.clone(), tree_zeros(v))).collect())
        }
        Value::Tensor { data, dtype } => {
            Value::Tensor { data: ArrayD::zeros(data.raw_dim()), dtype: *dtype }
        }
        Value::Float(_) => Value::Float(0.0),
        Value::Int(_) => Value::Int(0),
        Value::List(items) => Value::List(items.iter().map(tree_zeros).collect()),
        other => other.clone(),
    }
}

fn builtin_tree_map_zeros(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Err(runtime_error("tree-map-zeros requires 1 argument")); }
    Ok(tree_zeros(&args[0]))
}

/// print - Formatted output: (print "Epoch {} | loss: {:.6f}" epoch loss)
///
/// Supports Python-style {} and {:.Nf} format specifiers.
fn builtin_print(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() {
        println!();
        return Ok(Value::Nil);
    }
    let fmt = match &args[0] {
        Value::String(s) => s.clone(),
        other => {
            println!("{}", other);
            return Ok(Value::Nil);
        }
    };
    let vals = &args[1..];
    let result = format_string(&fmt, vals);
    println!("{}", result);
    Ok(Value::Nil)
}

fn format_string(fmt: &str, vals: &[Value]) -> String {
    let mut result = String::new();
    let mut val_idx = 0;
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                result.push('{');
                continue;
            }
            // Collect until '}'
            let mut spec = String::new();
            let mut closed = false;
            for ch in chars.by_ref() {
                if ch == '}' { closed = true; break; }
                spec.push(ch);
            }
            if !closed {
                result.push('{');
                result.push_str(&spec);
                continue;
            }
            if let Some(val) = vals.get(val_idx) {
                val_idx += 1;
                result.push_str(&format_value_with_spec(val, &spec));
            } else {
                result.push_str("{}");
            }
        } else if c == '}' && chars.peek() == Some(&'}') {
            chars.next();
            result.push('}');
        } else {
            result.push(c);
        }
    }
    result
}

fn format_value_with_spec(val: &Value, spec: &str) -> String {
    // Parse spec: e.g. ".6f", ":.6f", ".3f", ":.3f", "" (default)
    if spec.is_empty() {
        return format!("{}", val);
    }
    // Strip optional leading ':' (Python-style {:.3f} → spec = ":.3f")
    let spec = spec.strip_prefix(':').unwrap_or(spec);
    // Try to extract .<n>f
    if let Some(rest) = spec.strip_prefix('.') {
        if let Some(prec_str) = rest.strip_suffix('f') {
            if let Ok(prec) = prec_str.parse::<usize>() {
                let f = match val {
                    Value::Float(x) => *x,
                    Value::Int(n) => *n as f64,
                    Value::Tensor { data, .. } => data.first().copied().unwrap_or(0.0),
                    _ => return format!("{}", val),
                };
                return format!("{:.prec$}", f, prec = prec);
            }
        }
    }
    format!("{}", val)
}

/// io - System I/O: (io "entropy") → random seed as Int
///
/// Reads 8 bytes from the OS CSPRNG (/dev/urandom on Unix, BCryptGenRandom on Windows).
fn builtin_io(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match args.first() {
        Some(Value::String(s)) if s == "entropy" => {
            let mut bytes = [0u8; 8];
            getrandom::getrandom(&mut bytes).map_err(|e| {
                runtime_error(format!("io: entropy: {}", e))
            })?;
            let seed = u64::from_le_bytes(bytes) as i64;
            Ok(Value::Int(seed))
        }
        _ => Err(runtime_error("io: only (io \"entropy\") is supported")),
    }
}

/// random-key - Create a PRNG key from a seed: (random-key seed)
///
/// Returns an opaque key (stored as a List of two u32s, like JAX).
fn builtin_random_key(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let seed = match args.first() {
        Some(Value::Int(n)) => *n as u64,
        Some(Value::Float(f)) => *f as u64,
        _ => return Err(runtime_error("random-key: expected integer seed")),
    };
    // Represent key as [low32, high32]
    let lo = (seed & 0xFFFFFFFF) as i64;
    let hi = ((seed >> 32) & 0xFFFFFFFF) as i64;
    Ok(Value::List(vec![Value::Int(lo), Value::Int(hi)]))
}

/// Derive a u64 seed from a key value.
fn key_to_seed(key: &Value) -> u64 {
    match key {
        Value::List(items) => {
            let lo = items.first().and_then(|v| if let Value::Int(n) = v { Some(*n as u64) } else { None }).unwrap_or(0);
            let hi = items.get(1).and_then(|v| if let Value::Int(n) = v { Some(*n as u64) } else { None }).unwrap_or(0);
            lo | (hi << 32)
        }
        Value::Int(n) => *n as u64,
        _ => 42,
    }
}

/// random-split - Split a key into n subkeys: (random-split key n)
fn builtin_random_split(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("random-split: expected (random-split key n)"));
    }
    let seed = key_to_seed(&args[0]);
    let n = match &args[1] {
        Value::Int(n) => *n as usize,
        _ => return Err(runtime_error("random-split: n must be an integer")),
    };
    let mut keys = Vec::with_capacity(n);
    for i in 0..n {
        let child_seed = seed.wrapping_add(i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let lo = (child_seed & 0xFFFFFFFF) as i64;
        let hi = ((child_seed >> 32) & 0xFFFFFFFF) as i64;
        keys.push(Value::List(vec![Value::Int(lo), Value::Int(hi)]));
    }
    Ok(Value::List(keys))
}

/// SplitMix64 PRNG — high quality, fast, recommended for weight initialization.
/// Returns a uniform f64 in [0, 1).
fn splitmix64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// random-normal - Sample from N(0,1): (random-normal key shape)
fn builtin_random_normal(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("random-normal: expected (random-normal key shape)"));
    }
    let mut state = key_to_seed(&args[0]);
    let shape = parse_shape(&args[1])?;
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    // Box-Muller transform
    let mut i = 0;
    while i < n {
        let u1 = splitmix64(&mut state).max(1e-10);
        let u2 = splitmix64(&mut state);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push((r * theta.cos()) as f64);
        if i + 1 < n { data.push((r * theta.sin()) as f64); }
        i += 2;
    }
    data.truncate(n);
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| runtime_error(format!("random-normal: shape error: {}", e)))?;
    Ok(Value::tensor_f32(arr))
}

/// random-uniform - Sample from U(0,1): (random-uniform key shape)
fn builtin_random_uniform(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("random-uniform: expected (random-uniform key shape)"));
    }
    let mut state = key_to_seed(&args[0]);
    let shape = parse_shape(&args[1])?;
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|_| splitmix64(&mut state)).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| runtime_error(format!("random-uniform: shape error: {}", e)))?;
    Ok(Value::tensor_f32(arr))
}

fn parse_shape(val: &Value) -> Result<Vec<usize>, crate::core::error::SheafError> {
    match val {
        Value::List(items) => items.iter().map(|v| match v {
            Value::Int(n) => Ok(*n as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err(runtime_error("shape element must be integer")),
        }).collect(),
        Value::Tensor { data, .. } => data.iter().map(|&x| Ok(x as usize)).collect(),
        _ => Err(runtime_error("shape must be a list or quoted vector")),
    }
}
