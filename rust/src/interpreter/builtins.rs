// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Built-in functions for the Sheaf interpreter.

use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::{Dtype, Value};
use ndarray::{ArrayD, IxDyn};
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
    if a == Dtype::F32 || b == Dtype::F32 { Dtype::F32 } else { Dtype::I32 }
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
            let a_bc = acc.broadcast(b.shape()).ok_or_else(|| {
                runtime_error(format!("Cannot broadcast shapes {:?} and {:?}", acc.shape(), b.shape()))
            })?.to_owned();
            let b_bc = b.broadcast(a_bc.shape()).ok_or_else(|| {
                runtime_error(format!("Cannot broadcast shapes {:?} and {:?}", acc.shape(), b.shape()))
            })?.to_owned();
            acc = ndarray::Zip::from(&a_bc).and(&b_bc).map_collect(|&a, &b| op(a, b));
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
        // Math operations always produce F32
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
    unary_op(args, f64::ln)
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
        _ => Err(runtime_error(format!(
            "@ not supported for {}D x {}D", a.ndim(), b.ndim()
        ))),
    }
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
    // Check if all values are 0 or 1 — display as bool tensor
    Value::Tensor { data, dtype: Dtype::I32 }
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
            Ok(Value::tensor_i32(ArrayD::from_shape_vec(IxDyn(&[shape.len()]), shape).unwrap()))
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
