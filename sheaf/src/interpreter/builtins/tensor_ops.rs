use super::*;
use std::sync::Arc;

pub(super) fn register(env: &mut Env) {
    register_native_builtin(env, OpId::Reshape, builtin_reshape);
    register_native_builtin(env, OpId::Transpose, builtin_transpose);
    register_native_builtin(env, OpId::Concat, builtin_concat);
    register_native_builtin(env, OpId::Slice, builtin_slice);
    register_native_builtin(env, OpId::Get, builtin_get);
    register_native_builtin(env, OpId::Where, builtin_where);
    register_native_builtin(env, OpId::Roll, builtin_roll);
    register_native_builtin(env, OpId::IndexUpdate, builtin_index_update);
    register_native_builtin(env, OpId::Swapaxes, builtin_swapaxes);
    register_native_builtin(env, OpId::DynamicSlice, builtin_dynamic_slice);
    register_native_builtin(env, OpId::DynamicUpdateSlice, builtin_dynamic_update_slice);
    register_native_builtin(env, OpId::TensorSplit, builtin_tensor_split);
    register_native_builtin(env, OpId::Flip, builtin_flip);
}

fn builtin_reshape(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, dt) = to_array(&args[0])?;
    let raw_shape = integer_dimensions(&args[1])?;
    let total = arr.len() as i64;
    let neg_idx = raw_shape.iter().position(|&x| x < 0);
    let new_shape: Vec<usize> = if let Some(_ni) = neg_idx {
        let known: i64 = raw_shape.iter().filter(|&&x| x > 0).product();
        let inferred = total / known;
        raw_shape.iter().map(|&x| if x < 0 { inferred as usize } else { x as usize }).collect()
    } else {
        raw_shape.iter().map(|&x| x as usize).collect()
    };
    let arr = arr.as_standard_layout().into_owned();
    let result = arr.into_shape_with_order(IxDyn(&new_shape)).map_err(|_| runtime_error(format!(
        "Cannot reshape: input has {} elements, but target shape {:?} expects {}.",
        total, new_shape, new_shape.iter().product::<usize>()
    )))?;
    Ok(Value::Tensor { data: Arc::new(result), dtype: dt })
}

fn builtin_transpose(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, dtype) = to_array(&args[0])?;
    if arr.ndim() == 2 {
        Ok(tensor_with_dtype(arr.t().to_owned(), dtype))
    } else if arr.ndim() == 1 {
        Ok(tensor_with_dtype(arr.into_owned(), dtype))
    } else {
        let mut axes: Vec<usize> = (0..arr.ndim()).rev().collect();
        if args.len() > 1
            && let Value::List(items) = &args[1]
        {
            let mut resolved_axes = Vec::with_capacity(items.len());
                for v in items {
                    let ax = v.to_f64()
                        .ok_or_else(|| runtime_error(format!("transpose: expected numeric axis, got {}", v.type_name())))?;
                    if ax >= arr.ndim() as f64 {
                        return Err(runtime_error(format!("transpose: axis {} out of bounds for {}D tensor", ax, arr.ndim())));
                    }
                    resolved_axes.push(ax as usize);
            }
            axes = resolved_axes;
        }
        Ok(tensor_with_dtype(
            arr.into_owned().permuted_axes(IxDyn(&axes)),
            dtype,
        ))
    }
}

fn builtin_concat(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let axis_raw = get_axis(kw).unwrap_or(0);
    let has_axis_kw = kw.contains_key("axis");
    let resolve_axis = |rank: usize| -> Result<usize, crate::core::error::SheafError> {
        let axis = if axis_raw < 0 { rank as i64 + axis_raw } else { axis_raw };
        if axis < 0 || axis >= rank as i64 {
            return Err(runtime_error(format!(
                "concat: axis {axis_raw} out of bounds for {rank}D tensor"
            )));
        }
        Ok(axis as usize)
    };

    let maybe_arrays: Option<Vec<(ArrayD<f32>, Dtype)>> = args.iter().map(list_to_tensor).collect();

    if let Some(arrays) = maybe_arrays
        && (has_axis_kw || args.iter().any(|a| matches!(a, Value::Tensor { .. })))
    {
        let dtype = arrays[0].1;
        let axis = resolve_axis(arrays[0].0.ndim())?;
        if arrays.iter().any(|(_, candidate)| *candidate != dtype) {
            return Err(runtime_error("concat: dtype mismatch"));
        }
        let views: Vec<ndarray::ArrayViewD<f32>> =
            arrays.iter().map(|(array, _)| array.view()).collect();
        let result = ndarray::concatenate(ndarray::Axis(axis), &views)
            .map_err(|error| runtime_error(error.to_string()))?;
        return Ok(tensor_with_dtype(result, dtype));
    }

    if matches!(&args[0], Value::String(_)) {
        let mut s = String::new();
        for arg in args {
            match arg {
                Value::String(st) => s.push_str(st),
                other => s.push_str(&format!("{}", other)),
            }
        }
        return Ok(Value::String(s));
    }

    if matches!(&args[0], Value::List(_)) {
        let mut all_items = Vec::new();
        for arg in args {
            match arg {
                Value::List(items) => all_items.extend_from_slice(items),
                _ => all_items.push(arg.clone()),
            }
        }
        return Ok(Value::List(all_items));
    }

    let arrays: Vec<ArrayD<f32>> = args.iter().map(|a| {
        to_array(a).map(|(arr, _)| arr.into_owned())
    }).collect::<Result<Vec<_>, _>>()?;
    let axis = resolve_axis(arrays[0].ndim())?;
    let views: Vec<ndarray::ArrayViewD<f32>> = arrays.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(ndarray::Axis(axis), &views)
        .map_err(|e| runtime_error(e.to_string()))?;
    Ok(Value::tensor_f32(result))
}

fn builtin_slice(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    if let Value::String(s) = &args[0] {
        let start = args[1].to_f64().ok_or_else(|| runtime_error("slice: start must be a number"))? as usize;
        let end = if args.len() > 2 { args[2].to_f64().ok_or_else(|| runtime_error("slice: end must be a number"))? as usize } else { s.len() };
        return Ok(Value::String(s[start..end.min(s.len())].to_string()));
    }
    let (arr, dtype) = to_array(&args[0])?;
    if arr.ndim() == 0 {
        return Err(runtime_error("slice: cannot slice a 0-dimensional tensor"));
    }
    let axis_raw = kw.get("axis").and_then(|v| v.to_f64()).map(|v| v as i64).unwrap_or(0);
    let axis = if axis_raw < 0 { (arr.ndim() as i64 + axis_raw) as usize } else { axis_raw as usize };
    if axis >= arr.ndim() {
        return Err(runtime_error(format!("slice: axis {} out of bounds for tensor with {} dimensions", axis_raw, arr.ndim())));
    }
    let axis_len = arr.shape()[axis];
    let start = args[1].to_f64().ok_or_else(|| runtime_error("slice: start must be a number"))? as usize;
    let end = if args.len() > 2 { args[2].to_f64().ok_or_else(|| runtime_error("slice: end must be a number"))? as usize } else { axis_len };
    if start > end {
        return Err(runtime_error(format!(
            "slice: start ({}) > end ({})", start, end
        )));
    }
    if end > axis_len {
        return Err(runtime_error(format!(
            "slice: end ({}) out of bounds for axis {} with size {}", end, axis, axis_len
        )));
    }
    let sliced = arr.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(start..end));
    Ok(tensor_with_dtype(sliced.to_owned(), dtype))
}

fn builtin_get(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    if args.len() >= 2 && matches!(&args[1], Value::DeviceBuffer(_)) {
        let mut host_args = args.to_vec();
        host_args[1] = args[1].ensure_host()?;
        return builtin_get(&host_args, kw);
    }
    match &args[0] {
        Value::Dict(map) => {
            let key = match &args[1] {
                Value::Keyword(k) => k.clone(),
                Value::String(s) => s.clone(),
                _ => return Err(runtime_error("get: key must be keyword or string")),
            };
            match map.get(&key) {
                Some(v) => {
                    if args.len() > 2 {
                        let mut cur = v.clone();
                        for extra in &args[2..] {
                            match (&cur, extra) {
                                (Value::Dict(m), Value::Keyword(k)) | (Value::Dict(m), Value::String(k)) => {
                                    cur = m.get(k).cloned().unwrap_or(Value::Nil);
                                }
                                _ => break,
                            }
                        }
                        Ok(cur)
                    } else {
                        Ok(v.clone())
                    }
                }
                None => {
                    if args.len() > 2 { Ok(args[2].clone()) }
                    else if let Some(default) = kw.get("default") { Ok(default.clone()) }
                    else { Ok(Value::Nil) }
                }
            }
        }
        Value::Tensor { data, dtype } => {
            if matches!(&args[1], Value::Keyword(k) if k == "...") {
                if args.len() < 3 { return Err(runtime_error("get: ... requires an index argument")); }
                let last_axis = ndarray::Axis(data.ndim() - 1);
                return match &args[2] {
                    v if v.to_f64().is_some() => {
                        let raw = v.to_f64().unwrap();
                        let idx = resolve_idx(raw, data.shape()[data.ndim() - 1])?;
                        let sliced = data.index_axis(last_axis, idx).to_owned();
                        Ok(tensor_with_dtype(sliced, *dtype))
                    }
                    Value::Tensor { data: range_t, .. } if range_t.ndim() == 1 && !range_t.is_empty() => {
                        let start = as_scalar(range_t) as usize;
                        let end = *range_t.iter().last().unwrap() as usize + 1;
                        let last_dim = data.shape()[data.ndim() - 1];
                        if end > last_dim {
                            return Err(runtime_error(format!("get: range end {} out of bounds for last axis with size {}", end, last_dim)));
                    }
                    let sliced = data.slice_axis(last_axis, ndarray::Slice::from(start..end));
                    Ok(tensor_with_dtype(sliced.to_owned(), *dtype))
                }
                    other => Err(runtime_error(format!("get: ... index must be int or range, got {}", other.type_name()))),
                };
            }
            if let Some(f) = args[1].to_f64() {
                let idx = resolve_idx(f, data.shape()[0])?;
                let sliced = data.index_axis(ndarray::Axis(0), idx).to_owned();
                return Ok(indexed_tensor_value(sliced, *dtype));
            }
            if let Value::Tensor { data: idx_data, .. } = &args[1] {
                let row_shape = &data.shape()[1..];
                let idx_shape = idx_data.shape();
                let mut out_shape = Vec::with_capacity(idx_shape.len() + row_shape.len());
                out_shape.extend_from_slice(idx_shape);
                out_shape.extend_from_slice(row_shape);
                let row_size: usize = row_shape.iter().product::<usize>().max(1);
                let total = idx_data.len() * row_size;
                let mut result = Vec::with_capacity(total);
                let dim = data.shape()[0];
                for &idx_f in idx_data.iter() {
                    let idx = idx_f as usize;
                    if idx >= dim {
                        return Err(runtime_error(format!(
                            "get: index {} is out of bounds for tensor with {} rows",
                            idx, dim
                        )));
                    }
                    let row = data.index_axis(ndarray::Axis(0), idx);
                    result.extend(row.iter());
                }
                let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), result)
                    .map_err(|e| runtime_error(format!("get: gather reshape: {}", e)))?;
                return Ok(tensor_with_dtype(arr, *dtype));
            }
            if let Value::List(items) = &args[1]
                && items.iter().all(|v| matches!(v, Value::Int(_) | Value::Float(_)))
            {
                return Err(runtime_error(
                    "get: expected tensor index, got quoted list.\n  = hint: use [...] (tensor) instead of '[...] (quoted list)."
                ));
            }
            Err(runtime_error(format!("get: expected integer or tensor index, got {}", args[1].type_name())))
        }
        Value::List(items) => {
            let f = args[1].to_f64()
                .ok_or_else(|| runtime_error(format!("get: expected integer index, got {}", args[1].type_name())))?;
            let idx = resolve_idx(f, items.len())?;
            Ok(items[idx].clone())
        }
        Value::String(s) => {
            let f = args[1].to_f64()
                .ok_or_else(|| runtime_error(format!("get: expected integer index, got {}", args[1].type_name())))?;
            let idx = resolve_idx(f, s.chars().count())?;
            Ok(Value::String(s.chars().nth(idx).unwrap().to_string()))
        }
        Value::DeviceBuffer(_) => {
            let mut host_args = args.to_vec();
            host_args[0] = args[0].ensure_host()?;
            builtin_get(&host_args, kw)
        }
        _ => Err(runtime_error(format!("get: expected a dict, a tensor, or a list, got {} (key: {})", args[0].type_name(), args.get(1).map(|v| format!("{}", v)).unwrap_or_default()))),
    }
}

fn broadcast_shape(shapes: &[&[usize]]) -> Result<Vec<usize>, crate::core::error::SheafError> {
    let mut result = Vec::new();
    for shape in shapes {
        result = crate::core::shape::broadcast_shapes(&result, shape).map_err(|error| {
            let rank = result.len().max(shape.len());
            let axis = rank - error.axis_from_right - 1;
            runtime_error(format!(
                "where: cannot broadcast shapes, dimension mismatch {} vs {} at axis {}",
                error.lhs, error.rhs, axis
            ))
        })?;
    }
    Ok(result)
}

fn builtin_where(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (dtype, _) = arithmetic_result_dtype("where", &[&args[1], &args[2]])?;
    let (cond, _) = to_array(&args[0])?;
    let (on_true, _) = to_array(&args[1])?;
    let (on_false, _) = to_array(&args[2])?;
    let target = broadcast_shape(&[cond.shape(), on_true.shape(), on_false.shape()])?;
    let cond_bc = cond.broadcast(IxDyn(&target)).ok_or_else(|| {
        runtime_error(format!("where: cannot broadcast cond {:?} to {:?}", cond.shape(), target))
    })?.to_owned();
    let true_bc = on_true.broadcast(IxDyn(&target)).ok_or_else(|| {
        runtime_error(format!("where: cannot broadcast on_true {:?} to {:?}", on_true.shape(), target))
    })?.to_owned();
    let false_bc = on_false.broadcast(IxDyn(&target)).ok_or_else(|| {
        runtime_error(format!("where: cannot broadcast on_false {:?} to {:?}", on_false.shape(), target))
    })?.to_owned();
    let result = ndarray::Zip::from(&cond_bc).and(&true_bc).and(&false_bc)
        .map_collect(|&c, &t, &f| if c != 0.0 { t } else { f });
    Ok(tensor_with_dtype(result, dtype))
}

fn builtin_roll(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, dtype) = to_array(&args[0])?;
    let shift = args[1].to_f64().ok_or_else(|| runtime_error("roll: shift must be a number"))? as i64;
    if arr.is_empty() {
        return Ok(tensor_with_dtype(arr.into_owned(), dtype));
    }
    let data: Vec<f32> = arr.iter().copied().collect();
    let n = data.len() as i64;
    let shift = ((shift % n) + n) % n;
    let mut result = vec![0.0; data.len()];
    for (i, &v) in data.iter().enumerate() {
        let new_i = ((i as i64 + shift) % n) as usize;
        result[new_i] = v;
    }
    Ok(tensor_with_dtype(
        ArrayD::from_shape_vec(arr.raw_dim(), result).unwrap(),
        dtype,
    ))
}

fn builtin_index_update(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (cow, dtype) = to_array(&args[0])?;
    let mut arr = cow.into_owned();
    let idx = args[1].to_f64().ok_or_else(|| runtime_error("index-update: index must be a number"))? as usize;
    let dim = arr.shape()[0];
    if idx >= dim {
        return Err(runtime_error(format!(
            "index-update: index {} is out of bounds for axis 0 with size {}",
            idx, dim
        )));
    }
    match &args[2] {
        Value::Tensor { data: new_val, .. } => {
            let mut slice = arr.index_axis_mut(ndarray::Axis(0), idx);
            slice.assign(new_val);
        }
        other => {
            let v = other.to_f64().ok_or_else(|| runtime_error(format!("index-update: value must be a number, got {}", other.type_name())))? as f32;
            arr[IxDyn(&[idx])] = v;
        }
    }
    Ok(tensor_with_dtype(arr, dtype))
}

fn builtin_swapaxes(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, dtype) = to_array(&args[0])?;
    let ndim = arr.ndim();
    if ndim < 2 {
        return Err(runtime_error(format!("swapaxes: tensor must have at least 2 dimensions, got {}", ndim)));
    }
    let ax0 = resolve_idx(
        args[1].to_f64().ok_or_else(|| runtime_error("swapaxes: first axis must be a number"))?,
        ndim
    )?;
    let ax1 = resolve_idx(
        args[2].to_f64().ok_or_else(|| runtime_error("swapaxes: second axis must be a number"))?,
        ndim
    )?;
    let mut axes: Vec<usize> = (0..arr.ndim()).collect();
    axes[ax0] = ax1;
    axes[ax1] = ax0;
    Ok(tensor_with_dtype(
        arr.into_owned().permuted_axes(IxDyn(&axes)),
        dtype,
    ))
}

fn builtin_tensor_split(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, dtype) = to_array(&args[0])?;
    let num = args[1].to_f64()
        .ok_or_else(|| runtime_error("tensor-split: num-sections must be a number"))? as usize;
    if num == 0 {
        return Err(runtime_error("tensor-split: num-sections must be > 0"));
    }
    let ax = get_axis(kw).unwrap_or(0);
    let axis = resolve_idx(ax as f64, arr.ndim())?;
    let dim = arr.shape()[axis];
    if dim % num != 0 {
        return Err(runtime_error(format!(
            "tensor-split: dimension {} ({}) not divisible by {}", axis, dim, num
        )));
    }
    let chunk = dim / num;
    let chunks: Vec<Value> = (0..num).map(|i| {
        let start = i * chunk;
        let end = start + chunk;
        let sliced = arr.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(start..end));
        tensor_with_dtype(sliced.to_owned(), dtype)
    }).collect();
    Ok(Value::List(chunks))
}

fn numeric_vector(
    value: &Value,
    operation: &str,
) -> Result<Vec<i64>, crate::core::error::SheafError> {
    let (array, _) = to_array(value)?;
    if array.ndim() != 1 {
        return Err(runtime_error(format!(
            "{}: expected a one-dimensional numeric tensor",
            operation
        )));
    }
    Ok(array.iter().map(|value| *value as i64).collect())
}

fn dynamic_starts(
    value: &Value,
    rank: usize,
    operation: &str,
    limits: &[usize],
) -> Result<Vec<usize>, crate::core::error::SheafError> {
    let starts = numeric_vector(value, operation)?;
    if starts.len() != rank {
        return Err(runtime_error(format!(
            "{}: starts has length {}, expected {}",
            operation, starts.len(), rank
        )));
    }
    Ok(starts
        .into_iter()
        .zip(limits)
        .map(|(start, &limit)| {
            if start < 0 { 0 } else { (start as usize).min(limit) }
        })
        .collect())
}

fn static_sizes(
    value: &Value,
    rank: usize,
    operation: &str,
) -> Result<Vec<usize>, crate::core::error::SheafError> {
    let values: Vec<f64> = match value {
        Value::List(items) => items
            .iter()
            .map(|item| {
                item.to_f64().ok_or_else(|| {
                    runtime_error(format!("{}: sizes must be numeric", operation))
                })
            })
            .collect::<Result<_, _>>()?,
        Value::Tensor { data, .. } if data.ndim() == 1 => {
            data.iter().map(|&value| value as f64).collect()
        }
        _ => {
            return Err(runtime_error(format!(
                "{}: sizes must be a one-dimensional sequence",
                operation
            )));
        }
    };
    if values.len() != rank {
        return Err(runtime_error(format!(
            "{}: sizes has length {}, expected {}",
            operation,
            values.len(),
            rank
        )));
    }
    values
        .into_iter()
        .map(|size| {
            if size.is_finite() && size >= 0.0 && size.fract() == 0.0 {
                Ok(size as usize)
            } else {
                Err(runtime_error(format!(
                    "{}: sizes must contain non-negative integers",
                    operation
                )))
            }
        })
        .collect()
}

fn builtin_dynamic_slice(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 3 {
        return Err(runtime_error(format!(
            "dynamic-slice expects 3 arguments, got {}",
            args.len()
        )));
    }
    let (operand, dtype) = to_array(&args[0])?;
    let rank = operand.ndim();
    if rank == 0 {
        return Err(runtime_error(
            "dynamic-slice: operand must have non-zero rank",
        ));
    }
    let sizes = static_sizes(&args[2], rank, "dynamic-slice")?;
    if sizes
        .iter()
        .zip(operand.shape())
        .any(|(&size, &dimension)| size > dimension)
    {
        return Err(runtime_error(
            "dynamic-slice: sizes exceed operand dimensions",
        ));
    }
    let limits: Vec<usize> = operand
        .shape()
        .iter()
        .zip(&sizes)
        .map(|(&dimension, &size)| dimension - size)
        .collect();
    let starts = dynamic_starts(&args[1], rank, "dynamic-slice", &limits)?;
    let mut result = operand;
    for (axis, (&start, &size)) in starts.iter().zip(&sizes).enumerate() {
        result = std::borrow::Cow::Owned(
            result
                .slice_axis(
                    ndarray::Axis(axis),
                    ndarray::Slice::from(start..start + size),
                )
                .to_owned(),
        );
    }
    Ok(tensor_with_dtype(result.into_owned(), dtype))
}

fn builtin_dynamic_update_slice(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 3 {
        return Err(runtime_error(format!(
            "dynamic-update-slice expects 3 arguments, got {}",
            args.len()
        )));
    }
    let (operand, dtype) = to_array(&args[0])?;
    let (update, update_dtype) = to_array(&args[1])?;
    let rank = operand.ndim();
    if rank == 0 || update.ndim() != rank {
        return Err(runtime_error(
            "dynamic-update-slice: operand and update must have the same non-zero rank",
        ));
    }
    if dtype != update_dtype {
        return Err(runtime_error(
            "dynamic-update-slice: operand and update dtypes must match",
        ));
    }
    if update
        .shape()
        .iter()
        .zip(operand.shape())
        .any(|(&update_size, &operand_size)| update_size > operand_size)
    {
        return Err(runtime_error(
            "dynamic-update-slice: update exceeds operand dimensions",
        ));
    }
    let limits: Vec<usize> = operand
        .shape()
        .iter()
        .zip(update.shape())
        .map(|(&dimension, &size)| dimension - size)
        .collect();
    let starts = dynamic_starts(&args[2], rank, "dynamic-update-slice", &limits)?;
    let mut result = operand.into_owned();
    let update_shape = update.shape().to_vec();
    for (flat_index, &value) in update.iter().enumerate() {
        let mut remainder = flat_index;
        let mut result_index = vec![0; rank];
        for axis in (0..rank).rev() {
            result_index[axis] = remainder % update_shape[axis];
            remainder /= update_shape[axis];
            result_index[axis] += starts[axis];
        }
        result[ndarray::IxDyn(&result_index)] = value;
    }
    Ok(tensor_with_dtype(result, dtype))
}

fn builtin_flip(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    if let Value::List(items) = &args[0] {
        let mut reversed = items.clone();
        reversed.reverse();
        return Ok(Value::List(reversed));
    }
    let (arr, dt) = to_array(&args[0])?;
    if arr.is_empty() {
        return Ok(Value::Tensor { data: Arc::new(arr.into_owned()), dtype: dt });
    }
    let axis = if let Some(ax) = get_axis(kw)
        .or_else(|| args.get(1).and_then(Value::to_f64).map(|axis| axis as i64))
    {
        let ndim = arr.ndim();
        if ndim == 0 {
            return Err(runtime_error("flip: cannot flip a 0-dimensional tensor along an axis"));
        }
        resolve_idx(ax as f64, ndim)?
    } else {
        0
    };
    if arr.ndim() == 0 {
        return Ok(Value::Tensor { data: Arc::new(arr.into_owned()), dtype: dt });
    }
    let dim = arr.shape()[axis];
    let data: Vec<f32> = arr.iter().copied().collect();
    let shape: Vec<usize> = arr.shape().to_vec();
    let _ndim = shape.len();
    let axis_stride: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let axis_stride_outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let mut result = vec![0.0f32; data.len()];
    for outer in 0..axis_stride_outer {
        for i in 0..dim {
            let src_offset = outer * dim * axis_stride + i * axis_stride;
            let dst_offset = outer * dim * axis_stride + (dim - 1 - i) * axis_stride;
            let src_slice = &data[src_offset..src_offset + axis_stride];
            result[dst_offset..dst_offset + axis_stride].copy_from_slice(src_slice);
        }
    }
    let result_arr = ArrayD::from_shape_vec(IxDyn(&shape), result).unwrap();
    Ok(Value::Tensor { data: Arc::new(result_arr), dtype: dt })
}
