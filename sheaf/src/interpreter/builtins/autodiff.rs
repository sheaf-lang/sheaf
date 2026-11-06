// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use super::*;

pub(super) fn register(env: &mut Env) {
    register_native_builtin(env, OpId::OnesLike, ones_like);
    register_native_builtin(env, OpId::Broadcast, broadcast);
    register_native_builtin(env, OpId::SumToShape, sum_to_shape);
    register_native_builtin(env, OpId::SliceGrad, slice_grad);
}

fn ones_like(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 1 { return Err(runtime_error("__ones-like expects one argument")); }
    match &args[0] {
        Value::Float(_) | Value::Int(_) => Ok(Value::Float(1.0)),
        value => {
            let (data, dtype) = to_array(value)?;
            Ok(tensor_with_dtype(ArrayD::ones(data.raw_dim()), dtype))
        }
    }
}

fn broadcast(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 { return Err(runtime_error("broadcast expects a value and shape")); }
    let shape = shape_from_value(&args[1])?;
    let (data, dtype) = to_array(&args[0])?;
    let result = data.broadcast(IxDyn(&shape)).ok_or_else(|| runtime_error(format!(
        "broadcast: cannot broadcast {:?} to {shape:?}", data.shape(),
    )))?;
    Ok(tensor_with_dtype(result.to_owned(), dtype))
}

fn sum_to_shape(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 { return Err(runtime_error("sum_to_shape expects a value and shape")); }
    let shape = shape_from_value(&args[1])?;
    let (data, dtype) = to_array(&args[0])?;
    let axes = crate::core::shape::sum_to_shape_axes(data.shape(), &shape)
        .map_err(|_| runtime_error(format!(
            "sum_to_shape: cannot reduce {:?} to {shape:?}", data.shape(),
        )))?;
    let extra = data.ndim() - shape.len();
    let mut result = data.into_owned();
    for axis in axes.into_iter().rev() {
        result = result.sum_axis(ndarray::Axis(axis));
        if axis >= extra {
            result = result.insert_axis(ndarray::Axis(axis));
        }
    }
    if shape.is_empty() && dtype == Dtype::F32 {
        Ok(Value::Float(result[IxDyn(&[])]))
    } else { Ok(tensor_with_dtype(result, dtype)) }
}

fn slice_grad(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 3 {
        return Err(runtime_error("slice_grad expects a gradient, shape and start"));
    }
    let shape = shape_from_value(&args[1])?;
    let (gradient, dtype) = to_array(&args[0])?;
    let start = args[2].to_f64().ok_or_else(|| runtime_error("slice_grad: start must be an integer"))?;
    if !start.is_finite() || start.fract() != 0.0 || start < 0.0 || start > usize::MAX as f64 {
        return Err(runtime_error("slice_grad: invalid start"));
    }
    let axis = gradient.shape().iter().zip(&shape).position(|(gradient, target)| gradient != target).unwrap_or(0);
    if axis >= shape.len() { return Err(runtime_error("slice_grad: scalar target")); }
    let start = start as usize;
    if gradient.ndim() != shape.len() || start > shape[axis]
        || gradient.shape()[axis] > shape[axis] - start
        || shape.iter().zip(gradient.shape()).enumerate().any(|(i, (a, b))| i != axis && a != b) {
        return Err(runtime_error("slice_grad: gradient shape does not fit target"));
    }
    let mut result = ArrayD::zeros(IxDyn(&shape));
    result.slice_axis_mut(ndarray::Axis(axis), ndarray::Slice::from(start..start + gradient.shape()[axis]))
        .assign(&gradient);
    Ok(tensor_with_dtype(result, dtype))
}
