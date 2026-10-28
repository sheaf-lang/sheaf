// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Built-in functions for the Sheaf interpreter.

mod arithmetic;
mod activations;
pub(crate) mod pickle;
mod safetensors;
mod comparison;
mod reductions;
mod tensor_ops;
mod tensor_create;
mod collections;
mod io;
mod random;
mod losses;

use crate::core::error::SheafError;
use crate::core::signature::OpId;
use crate::interpreter::env::{arity_error, runtime_error, Env};
use crate::interpreter::value::{BuiltinFnPtr, Dtype, Value};
use ndarray::{ArrayD, Dimension, IxDyn};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;

type R = Result<Value, crate::core::error::SheafError>;

pub use losses::tree_zeros;

pub fn register_builtins(env: &mut Env) {
    arithmetic::register(env);
    activations::register(env);
    comparison::register(env);
    reductions::register(env);
    tensor_ops::register(env);
    tensor_create::register(env);
    collections::register(env);
    io::register(env);
    random::register(env);
    losses::register(env);
    register_native_builtin(env, OpId::StopGradient, builtin_stop_gradient);
}

fn register_native_builtin(env: &mut Env, op: OpId, func: BuiltinFnPtr) {
    for name in std::iter::once(op.name()).chain(op.aliases().iter().copied()) {
        env.set_builtin(name, func);
    }
}

fn builtin_stop_gradient(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 1 {
        return Err(runtime_error("stop-gradient expects exactly 1 argument"));
    }
    Ok(args[0].clone())
}

/// Resolve a potentially negative index against a dimension length.
fn resolve_idx(f: f64, len: usize) -> Result<usize, crate::core::error::SheafError> {
    if f.is_nan() {
        return Err(runtime_error("index cannot be NaN"));
    }
    let idx = if f < 0.0 {
        let resolved = len as i64 + f as i64;
        if resolved < 0 {
            return Err(runtime_error(format!(
                "index {} out of bounds for length {}", f, len
            )));
        }
        resolved as usize
    } else {
        f as usize
    };
    if idx >= len {
        return Err(runtime_error(format!(
            "index {} out of bounds for length {}", f, len
        )));
    }
    Ok(idx)
}

fn to_array(val: &Value) -> Result<(Cow<'_, ArrayD<f32>>, Dtype), crate::core::error::SheafError> {
    match val {
        Value::Int(n) => Ok((Cow::Owned(ArrayD::from_elem(IxDyn(&[]), *n as f32)), Dtype::I32)),
        Value::Float(f) => Ok((Cow::Owned(ArrayD::from_elem(IxDyn(&[]), *f)), Dtype::F32)),
        Value::Bool(b) => Ok((Cow::Owned(ArrayD::from_elem(IxDyn(&[]), if *b { 1.0f32 } else { 0.0f32 })), Dtype::I32)),
        Value::Tensor { data, dtype } => Ok((Cow::Borrowed(&**data), *dtype)),
        Value::DeviceBuffer(db) => {
            let data = db.to_host().map_err(|e| runtime_error(format!("materialize: {}", e)))?;
            Ok((Cow::Owned(data), db.dtype))
        }
        _ => Err(runtime_error(format!("Expected a number, got {}", val.short_desc()))),
    }
}

pub(crate) fn as_scalar(arr: &ArrayD<f32>) -> f32 {
    arr.first().copied().unwrap()
}

fn tensor_with_dtype(data: ArrayD<f32>, dtype: Dtype) -> Value {
    let data = data.mapv(|value| crate::core::dtype::quantize_f32(value, dtype));
    Value::Tensor { data: Arc::new(data), dtype }
}

fn indexed_tensor_value(data: ArrayD<f32>, dtype: Dtype) -> Value {
    if data.shape().is_empty() {
        Value::tensor_element(as_scalar(&data), dtype)
    } else {
        tensor_with_dtype(data, dtype)
    }
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    crate::core::shape::broadcast_shapes(a, b).ok()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BinaryDtypePolicy<'a> {
    Legacy,
    WeakScalars(&'a str),
}

fn result_dtype(a: Dtype, b: Dtype) -> Dtype {
    if a == b { return a; }
    if a == Dtype::Bool { return b; }
    if b == Dtype::Bool { return a; }
    match (a, b) {
        (Dtype::F32, _) | (_, Dtype::F32) => Dtype::F32,
        (Dtype::BF16, _) | (_, Dtype::BF16) => Dtype::BF16,
        _ => Dtype::I32,
    }
}

fn binary_op(args: &[Value], op: fn(f32, f32) -> f32) -> R {
    if args.len() < 2 {
        return Err(runtime_error(format!("+: expected at least 2 arguments, got {}", args.len())));
    }
    if args.len() == 2 {
        return binary_op_two(&args[0], &args[1], op, BinaryDtypePolicy::Legacy);
    }
    let (cow, mut dt) = to_array(&args[0])?;
    let mut acc = cow.into_owned();
    for arg in &args[1..] {
        match arg {
            Value::Int(n) => {
                let s = *n as f32;
                dt = result_dtype(dt, Dtype::I32);
                if acc.ndim() == 0 {
                    acc = ArrayD::from_elem(IxDyn(&[]), op(as_scalar(&acc), s));
                } else {
                    acc = acc.mapv(|x| op(x, s));
                }
            }
            Value::Float(f) => {
                let s = *f;
                dt = result_dtype(dt, Dtype::F32);
                if acc.ndim() == 0 {
                    acc = ArrayD::from_elem(IxDyn(&[]), op(as_scalar(&acc), s));
                } else {
                    acc = acc.mapv(|x| op(x, s));
                }
            }
            Value::Bool(b) => {
                let s = if *b { 1.0 } else { 0.0 };
                dt = result_dtype(dt, Dtype::I32);
                if acc.ndim() == 0 {
                    acc = ArrayD::from_elem(IxDyn(&[]), op(as_scalar(&acc), s));
                } else {
                    acc = acc.mapv(|x| op(x, s));
                }
            }
            Value::Tensor { data: b, dtype: bdt } => {
                dt = result_dtype(dt, *bdt);
                if acc.ndim() == 0 && b.ndim() != 0 {
                    let scalar = as_scalar(&acc);
                    acc = b.mapv(|x| op(scalar, x));
                } else if b.ndim() == 0 && acc.ndim() != 0 {
                    let scalar = as_scalar(b);
                    acc = acc.mapv(|x| op(x, scalar));
                } else if acc.shape() == b.shape() {
                    acc = ndarray::Zip::from(&acc).and(b.as_ref()).map_collect(|&a, &b| op(a, b));
                } else {
                    let out_shape = broadcast_shape(acc.shape(), b.shape()).ok_or_else(|| {
                        runtime_error(format!("Cannot broadcast shapes {:?} and {:?}", acc.shape(), b.shape()))
                    })?;
                    let a_bc = acc.broadcast(&out_shape[..]).ok_or_else(|| runtime_error("broadcast failed"))?;
                    let b_bc = b.broadcast(&out_shape[..]).ok_or_else(|| runtime_error("broadcast failed"))?;
                    acc = ndarray::Zip::from(&a_bc).and(&b_bc).map_collect(|&a, &b| op(a, b));
                }
            }
            _ => return Err(runtime_error(format!("Expected a number, got {}", arg.short_desc()))),
        }
    }
    if acc.ndim() == 0 {
        let x = as_scalar(&acc);
        if dt == Dtype::I32 && x == x.floor() {
            Ok(Value::Int(x as i64))
        } else {
            Ok(Value::Float(x))
        }
    } else {
        Ok(Value::Tensor { data: Arc::new(acc), dtype: dt })
    }
}

fn arithmetic_with_dtype_policy(
    name: &str,
    args: &[Value],
    op: fn(f32, f32) -> f32,
) -> R {
    if args.len() < 2 {
        return Err(runtime_error(format!(
            "{}: expected at least 2 arguments, got {}",
            name,
            args.len(),
        )));
    }
    let mut result = args[0].clone();
    for arg in &args[1..] {
        result = binary_op_two(
            &result,
            arg,
            op,
            BinaryDtypePolicy::WeakScalars(name),
        )?;
    }
    Ok(result)
}

fn binary_op_two(
    a: &Value,
    b: &Value,
    op: fn(f32, f32) -> f32,
    policy: BinaryDtypePolicy<'_>,
) -> R {
    let a_host;
    let b_host;
    let a = if matches!(a, Value::DeviceBuffer(_)) { a_host = a.ensure_host_cow()?; &*a_host } else { a };
    let b = if matches!(b, Value::DeviceBuffer(_)) { b_host = b.ensure_host_cow()?; &*b_host } else { b };

    let (dt, has_strong_operand) = match policy {
        BinaryDtypePolicy::Legacy => (result_dtype(dtype_of(a), dtype_of(b)), false),
        BinaryDtypePolicy::WeakScalars(name) => weak_scalar_result_dtype(name, a, b)?,
    };
    let quantize = |value| {
        if has_strong_operand {
            crate::core::dtype::quantize_f32(value, dt)
        } else {
            value
        }
    };
    let a_scalar = scalar_of(a).map(quantize);
    let b_scalar = scalar_of(b).map(quantize);

    if let (Some(sa), Some(sb)) = (a_scalar, b_scalar) {
        let x = quantize(op(sa, sb));
        return if dt == Dtype::I32 && x == x.floor() {
            Ok(Value::Int(x as i64))
        } else {
            Ok(Value::Float(x))
        };
    }

    if let (Some(s), Value::Tensor { data, .. }) = (a_scalar, b) {
        let result = if data.ndim() == 0 && policy == BinaryDtypePolicy::Legacy {
            let x = op(s, as_scalar(data));
            return Ok(Value::Float(x));
        } else {
            data.mapv(|x| quantize(op(s, x)))
        };
        return Ok(Value::Tensor { data: Arc::new(result), dtype: dt });
    }

    if let (Value::Tensor { data, .. }, Some(s)) = (a, b_scalar) {
        let result = if data.ndim() == 0 && policy == BinaryDtypePolicy::Legacy {
            let x = op(as_scalar(data), s);
            return Ok(Value::Float(x));
        } else {
            data.mapv(|x| quantize(op(x, s)))
        };
        return Ok(Value::Tensor { data: Arc::new(result), dtype: dt });
    }

    if let (Value::Tensor { data: ad, .. }, Value::Tensor { data: bd, .. }) = (a, b) {
        let result = if ad.ndim() == 0 {
            let s = as_scalar(ad);
            if bd.ndim() == 0 && policy == BinaryDtypePolicy::Legacy {
                return Ok(Value::Float(op(s, as_scalar(bd))));
            }
            bd.mapv(|x| quantize(op(s, x)))
        } else if bd.ndim() == 0 {
            let s = as_scalar(bd);
            ad.mapv(|x| quantize(op(x, s)))
        } else if ad.shape() == bd.shape() {
            ndarray::Zip::from(ad.as_ref())
                .and(bd.as_ref())
                .map_collect(|&a, &b| quantize(op(a, b)))
        } else {
            let out_shape = broadcast_shape(ad.shape(), bd.shape()).ok_or_else(|| {
                runtime_error(format!("Cannot broadcast shapes {:?} and {:?}", ad.shape(), bd.shape()))
            })?;
            let a_bc = ad.broadcast(&out_shape[..]).ok_or_else(|| runtime_error("broadcast failed"))?;
            let b_bc = bd.broadcast(&out_shape[..]).ok_or_else(|| runtime_error("broadcast failed"))?;
            ndarray::Zip::from(&a_bc)
                .and(&b_bc)
                .map_collect(|&a, &b| quantize(op(a, b)))
        };
        return Ok(Value::Tensor { data: Arc::new(result), dtype: dt });
    }

    Err(runtime_error(format!("Expected numbers, got {} and {}", a.short_desc(), b.short_desc())))
}

fn weak_scalar_result_dtype(
    name: &str,
    a: &Value,
    b: &Value,
) -> Result<(Dtype, bool), SheafError> {
    arithmetic_result_dtype(name, &[a, b])
}

fn arithmetic_result_dtype(
    name: &str,
    values: &[&Value],
) -> Result<(Dtype, bool), SheafError> {
    use crate::core::dtype::{DtypeOperand, DtypeStrength, resolve_arithmetic_dtype};

    let operand = |value: &Value| match value {
        Value::Int(_) => Some(DtypeOperand::weak(Dtype::I32)),
        Value::Float(_) => Some(DtypeOperand::weak(Dtype::F32)),
        Value::Bool(_) => Some(DtypeOperand::weak(Dtype::Bool)),
        Value::Tensor { dtype, .. } => Some(DtypeOperand::strong(*dtype)),
        Value::DeviceBuffer(buffer) => Some(DtypeOperand::strong(buffer.dtype)),
        _ => None,
    };
    let mut result = operand(values[0]).ok_or_else(|| {
        runtime_error(format!(
            "{}: expected a number, got {}",
            name,
            values[0].short_desc(),
        ))
    })?;
    for value in &values[1..] {
        let rhs = operand(value).ok_or_else(|| {
            runtime_error(format!(
                "{}: expected a number, got {}",
                name,
                value.short_desc(),
            ))
        })?;
        result.dtype = resolve_arithmetic_dtype(result, rhs)
            .map_err(|error| runtime_error(format!("{}: {}", name, error)))?;
        if rhs.strength == DtypeStrength::Strong {
            result.strength = DtypeStrength::Strong;
        }
    }
    Ok((result.dtype, result.strength == DtypeStrength::Strong))
}

fn scalar_of(v: &Value) -> Option<f32> {
    match v {
        Value::Int(n) => Some(*n as f32),
        Value::Float(f) => Some(*f),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn dtype_of(v: &Value) -> Dtype {
    match v {
        Value::Tensor { dtype, .. } => *dtype,
        Value::Float(_) => Dtype::F32,
        _ => Dtype::I32,
    }
}

fn unary_op(args: &[Value], op: fn(f32) -> f32) -> R {
    if args.is_empty() {
        return Err(runtime_error("Expected at least 1 argument, got 0"));
    }
    let (arr, dtype) = to_array(&args[0])?;
    let result = arr.mapv(op);
    if matches!(&args[0], Value::Tensor { .. } | Value::DeviceBuffer(_)) {
        let dtype = if dtype.is_float() { dtype } else { Dtype::F32 };
        Ok(tensor_with_dtype(result, dtype))
    } else {
        Ok(Value::Float(as_scalar(&result)))
    }
}

fn get_axis(kw: &BTreeMap<String, Value>) -> Option<i64> {
    kw.get("axis").and_then(|v| match v {
        Value::Int(n) => Some(*n),
        Value::Float(f) => Some(*f as i64),
        _ => None,
    })
}

fn get_dtype_kwarg(kw: &BTreeMap<String, Value>) -> Option<Dtype> {
    for key in kw.keys() {
        if let Some(dt) = Dtype::from_keyword(key) {
            return Some(dt);
        }
    }
    None
}

fn with_dtype_kwarg(result: R, kw: &BTreeMap<String, Value>) -> R {
    if let Some(dt) = get_dtype_kwarg(kw) {
        match result? {
            Value::Tensor { data, .. } => Ok(Value::Tensor { data, dtype: dt }),
            other => Ok(other),
        }
    } else {
        result
    }
}

fn reduce_along_axis(arr: &ArrayD<f32>, axis: usize, op: fn(&[f32]) -> f32) -> ArrayD<f32> {
    let shape = arr.shape();
    if shape.is_empty() || axis >= shape.len() {
        let data: Vec<f32> = arr.iter().copied().collect();
        return ArrayD::from_elem(IxDyn(&[]), op(&data));
    }
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        let data: Vec<f32> = arr.iter().copied().collect();
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

fn argreduce_along_axis(arr: &ArrayD<f32>, axis: usize, cmp: fn(f32, f32) -> bool) -> ArrayD<f32> {
    let shape = arr.shape();
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        let data: Vec<f32> = arr.iter().copied().collect();
        let idx = data.iter().enumerate().fold(0, |best, (i, &x)| if cmp(x, data[best]) { i } else { best });
        return ArrayD::from_elem(IxDyn(&[]), idx as f32);
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
        result_data.push(best_idx as f32);
    }
    ArrayD::from_shape_vec(IxDyn(&new_shape), result_data).unwrap()
}

fn checked_dimension(number: f64) -> Result<usize, crate::core::error::SheafError> {
    if !number.is_finite() || number.fract() != 0.0 || number < 0.0 {
        return Err(runtime_error(format!(
            "invalid shape dimension {number}: expected a nonnegative integer"
        )));
    }
    if number >= isize::MAX as f64 {
        return Err(runtime_error("shape dimension exceeds addressable memory"));
    }
    Ok(number as usize)
}

fn integer_dimensions(val: &Value) -> Result<Vec<i64>, SheafError> {
    match val {
        Value::List(items) | Value::Tuple(items) => items.iter().map(|value| match value {
            Value::Int(value) => Ok(*value),
            _ => Err(runtime_error(format!(
                "shape dimension must be an integer, got {}", value.type_name(),
            ))),
        }).collect(),
        Value::Tensor { data, dtype } if data.ndim() == 1 && dtype.is_integer() => {
            data.iter().map(|&value| {
                if value.is_finite() && value.fract() == 0.0
                    && (value as f64) >= i64::MIN as f64 && (value as f64) < i64::MAX as f64 {
                    Ok(value as i64)
                } else {
                    Err(runtime_error("invalid integer shape dimension"))
                }
            }).collect()
        }
        _ => Err(runtime_error(format!(
            "shape dimensions must be integers, got {}", val.type_name(),
        ))),
    }
}

fn shape_from_value(val: &Value) -> Result<Vec<usize>, crate::core::error::SheafError> {
    let shape: Vec<usize> = match val {
        Value::List(items) | Value::Tuple(items) => {
            items.iter().map(|v| match v {
                Value::Int(n) => usize::try_from(*n)
                    .map_err(|_| runtime_error(format!("invalid shape dimension {n}"))),
                Value::BuiltinFn { name, .. } => Err(runtime_error(format!(
                    "shape must contain integers, got function '{}'.\n  = hint: Arithmetic inside vectors needs parentheses: e.g, [(* 2 n)], not [* 2 n]",
                    name
                ))),
                Value::String(s) if s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) => Err(runtime_error(format!(
                    "shape must contain integers, got symbol '{}'.\n  = hint: Variables inside quotes are never evaluated. Use [{}] (unquoted) or extract from tensor with (shape t).",
                    s, s
                ))),
                _ => Err(runtime_error(format!(
                    "shape dimension must be an integer, got {}",
                    v.type_name()
                ))),
            }).collect()
        }
        Value::Tensor { data, dtype } if data.ndim() == 1 && dtype.is_integer() => {
            data.iter().map(|&x| checked_dimension(x as f64)).collect()
        }
        _ => Err(runtime_error(format!("shape dimensions must be integers, got {}", val.type_name()))),
    }?;
    let max_elements = isize::MAX as usize / std::mem::size_of::<f32>();
    shape.iter().try_fold(1usize, |size, &dimension| {
        if dimension > max_elements {
            return Err(runtime_error("shape dimension exceeds addressable memory"));
        }
        // Empty arrays still need strides that fit in addressable memory.
        size.checked_mul(dimension.max(1))
            .filter(|&size| size <= max_elements)
            .ok_or_else(|| runtime_error("shape exceeds addressable memory"))
    })?;
    Ok(shape)
}

fn list_to_tensor(v: &Value) -> Option<(ArrayD<f32>, Dtype)> {
    match v {
        Value::Tensor { data, dtype } => Some(((**data).clone(), *dtype)),
        Value::List(items) => {
            let all_int = items.iter().all(|x| matches!(x, Value::Int(_)));
            let nums: Option<Vec<f32>> = items.iter().map(|x| x.to_f64().map(|v| v as f32)).collect();
            if let Some(data) = nums {
                let dtype = if all_int { Dtype::I32 } else { Dtype::F32 };
                return ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).ok()
                    .map(|a| (a, dtype));
            }
            let rows: Option<Vec<(ArrayD<f32>, Dtype)>> = items.iter().map(list_to_tensor).collect();
            if let Some(rows) = rows {
                let all_i32 = rows.iter().all(|(_, dt)| *dt == Dtype::I32);
                let dtype = if all_i32 { Dtype::I32 } else { Dtype::F32 };
                let stacked: Option<ArrayD<f32>> = ndarray::concatenate(
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

fn cmp_op(args: &[Value], op: fn(f32, f32) -> f32, _dt: Dtype) -> R {
    if args.len() != 2 { return Err(runtime_error(format!("Comparison: expected 2 arguments, got {}", args.len()))); }
    let (a, _) = to_array(&args[0])?;
    let (b, _) = to_array(&args[1])?;
    if a.ndim() == 0 && b.ndim() == 0 {
        return Ok(Value::Bool(op(as_scalar(&a), as_scalar(&b)) != 0.0));
    }
    let shape = crate::core::shape::broadcast_shapes(a.shape(), b.shape())
        .map_err(|_| runtime_error(format!("comparison: incompatible shapes {:?} and {:?}", a.shape(), b.shape())))?;
    let a = a.broadcast(IxDyn(&shape)).ok_or_else(|| runtime_error("comparison: cannot broadcast left operand"))?;
    let b = b.broadcast(IxDyn(&shape)).ok_or_else(|| runtime_error("comparison: cannot broadcast right operand"))?;
    let result = ndarray::Zip::from(a).and(b).map_collect(|&a, &b| op(a, b));
    Ok(bool_tensor(result))
}

fn bool_tensor(data: ArrayD<f32>) -> Value {
    Value::Tensor { data: Arc::new(data), dtype: Dtype::Bool }
}

#[cfg(test)]
#[test]
fn comparisons_broadcast_without_panicking() {
    let mut env = Env::new();
    register_builtins(&mut env);
    for name in ["==", "!=", "<", ">", "<=", ">="] {
        let Value::BuiltinFn { func, .. } = env.get(name).unwrap() else { panic!("expected builtin") };
        for (lhs_shape, lhs_data, rhs_shape, rhs_data, shape) in [
            (vec![2, 1], vec![0., 2.], vec![1, 3], vec![0., 1., 2.], vec![2, 3]),
            (vec![2, 1], vec![0., 2.], vec![3], vec![0., 1., 2.], vec![2, 3]),
            (vec![], vec![0.], vec![3], vec![0., 1., 2.], vec![3]),
            (vec![3], vec![0., 1., 2.], vec![], vec![0.], vec![3]),
            (vec![3], vec![0., 1., 2.], vec![3], vec![2., 1., 0.], vec![3]),
            (vec![0, 3], vec![], vec![1, 3], vec![0., 1., 2.], vec![0, 3]),
        ] {
            let lhs = ArrayD::from_shape_vec(IxDyn(&lhs_shape), lhs_data).unwrap();
            let rhs = ArrayD::from_shape_vec(IxDyn(&rhs_shape), rhs_data).unwrap();
            let args = [Value::tensor_f32(lhs.clone()), Value::tensor_f32(rhs.clone())];
            let Value::Tensor { data, dtype } = func(&args, &BTreeMap::new()).unwrap()
                else { panic!("{name}: expected boolean tensor") };
            assert_eq!(dtype, Dtype::Bool);
            assert_eq!(data.shape(), shape);
            let lhs = lhs.broadcast(IxDyn(&shape)).unwrap();
            let rhs = rhs.broadcast(IxDyn(&shape)).unwrap();
            for ((actual, a), b) in data.iter().zip(lhs.iter()).zip(rhs.iter()) {
                let expected = match name {
                    "==" => a == b, "!=" => a != b, "<" => a < b,
                    ">" => a > b, "<=" => a <= b, ">=" => a >= b,
                    _ => unreachable!(),
                };
                assert_eq!(*actual, if expected { 1. } else { 0. }, "{name}");
            }
        }
        let args = [Value::Int(0), Value::Int(1)];
        assert!(matches!(func(&args, &BTreeMap::new()).unwrap(), Value::Bool(_)));
        let lhs = Value::tensor_f32(ArrayD::zeros(IxDyn(&[2])));
        let rhs = Value::tensor_f32(ArrayD::zeros(IxDyn(&[3])));
        for args in [[lhs.clone(), rhs.clone()], [rhs, lhs]] {
            let error = func(&args, &BTreeMap::new()).unwrap_err().to_string();
            assert!(error.contains("comparison: incompatible shapes"), "{name}: {error}");
        }
    }
}

#[cfg(test)]
#[test]
fn indexed_tensors_preserve_scalar_types_and_row_dtypes() {
    let mut env = Env::new();
    register_builtins(&mut env);
    for dtype in [Dtype::I32, Dtype::I64, Dtype::Bool, Dtype::F16, Dtype::BF16, Dtype::F32, Dtype::F64] {
        for shape in [&[3][..], &[3, 1][..]] {
            let source = Value::tensor(ArrayD::from_shape_vec(IxDyn(shape), vec![0., 1., 0.]).unwrap(), dtype);
            for (name, index, expected) in [
                ("first", None, 0.), ("second", None, 1.), ("last", None, 0.),
                ("nth", Some(1), 1.), ("nth", Some(-1), 0.),
                ("get", Some(1), 1.), ("get", Some(-1), 0.),
                ("get-in", Some(1), 1.), ("get-in", Some(-1), 0.),
            ] {
                let mut args = vec![source.clone()];
                if let Some(index) = index {
                    let index = Value::Int(index);
                    args.push(if name == "get-in" { Value::List(vec![index]) } else { index });
                }
                let Value::BuiltinFn { func, .. } = env.get(name).unwrap() else { panic!("expected builtin") };
                let result = func(&args, &BTreeMap::new()).unwrap();
                match (shape.len(), &result) {
                    (1, Value::Int(value)) if dtype.is_integer() => assert_eq!(*value, expected as i64),
                    (1, Value::Bool(value)) if dtype == Dtype::Bool => assert_eq!(*value, expected != 0.),
                    (1, Value::Float(value)) if dtype.is_float() => assert_eq!(*value, expected),
                    (2, Value::Tensor { data, dtype: actual }) => {
                        assert_eq!(*actual, dtype);
                        assert_eq!(data.shape(), &[1]);
                        assert_eq!(data[[0]], expected);
                    }
                    _ => panic!("{name} {dtype} {shape:?}: unexpected result {result:?}"),
                }
            }
        }
    }
}
