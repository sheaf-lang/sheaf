use super::*;

pub(super) fn register(env: &mut Env) {
    register_native_builtin(env, OpId::Zeros, builtin_zeros);
    register_native_builtin(env, OpId::Ones, builtin_ones);
    register_native_builtin(env, OpId::Arange, builtin_arange);
    register_native_builtin(env, OpId::Eye, builtin_eye);
    register_native_builtin(env, OpId::OneHot, builtin_one_hot);
    register_native_builtin(env, OpId::Tril, builtin_tril);
    register_native_builtin(env, OpId::Tensor, builtin_tensor);
    register_native_builtin(env, OpId::Range, builtin_range);
    register_native_builtin(env, OpId::Cast, builtin_cast);
    register_native_builtin(env, OpId::CastLike, builtin_cast_like);
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
        1 => (0i64, args[0].to_f64().ok_or_else(|| runtime_error("arange: argument must be a number"))? as i64, 1i64),
        2 => (
            args[0].to_f64().ok_or_else(|| runtime_error("arange: start must be a number"))? as i64,
            args[1].to_f64().ok_or_else(|| runtime_error("arange: stop must be a number"))? as i64,
            1,
        ),
        _ => (
            args[0].to_f64().ok_or_else(|| runtime_error("arange: start must be a number"))? as i64,
            args[1].to_f64().ok_or_else(|| runtime_error("arange: stop must be a number"))? as i64,
            args[2].to_f64().ok_or_else(|| runtime_error("arange: step must be a number"))? as i64,
        ),
    };
    let mut data = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        data.push(i as f32);
        i += step;
    }
    Ok(Value::tensor_i32(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap()))
}

fn builtin_eye(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let dimensions = shape_from_value(&Value::List(args.to_vec()))?;
    let n = dimensions[0];
    let m = dimensions.get(1).copied().unwrap_or(n);
    let mut data = vec![0.0; n * m];
    for i in 0..n.min(m) {
        data[i * m + i] = 1.0;
    }
    Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[n, m]), data).unwrap()))
}

fn builtin_one_hot(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error(format!(
            "one-hot expects 2 arguments (one-hot indices num_classes), got {}",
            args.len()
        )));
    }
    let num_classes = shape_from_value(&Value::List(vec![args[1].clone()]))?[0];
    match &args[0] {
        Value::Int(idx) => {
            let idx_u = *idx as usize;
            if idx_u >= num_classes {
                return Err(runtime_error(format!(
                    "one-hot: index {} is out of bounds for {} classes",
                    idx_u, num_classes
                )));
            }
            let mut data = vec![0.0; num_classes];
            data[idx_u] = 1.0;
            Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[num_classes]), data).unwrap()))
        }
        Value::Tensor { data: indices, .. } => {
            let n = indices.len();
            let mut result = vec![0.0; n * num_classes];
            for (i, &idx) in indices.iter().enumerate() {
                let idx_u = idx as usize;
                if idx_u >= num_classes {
                    return Err(runtime_error(format!(
                        "one-hot: index {} is out of bounds for {} classes",
                        idx_u, num_classes
                    )));
                }
                result[i * num_classes + idx_u] = 1.0;
            }
            // Preserve input shape: [...indices_shape, num_classes]
            let mut out_shape: Vec<usize> = indices.shape().to_vec();
            out_shape.push(num_classes);
            Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&out_shape), result).unwrap()))
        }
        other => Err(runtime_error(format!("one-hot: expected an integer or a tensor, got {}", other.type_name()))),
    }
}

fn builtin_tril(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() {
        return Err(runtime_error("tril: missing argument, expected a 2D tensor"));
    }
    let (arr, _dt) = to_array(&args[0])?;
    if arr.ndim() != 2 { return Err(runtime_error(format!("tril: input must be 2D (matrix), got {}D", arr.ndim()))); }
    let shape = arr.shape();
    let (n, m) = (shape[0], shape[1]);
    let mut result = arr.into_owned();
    for i in 0..n {
        for j in (i + 1)..m {
            result[IxDyn(&[i, j])] = 0.0;
        }
    }
    Ok(Value::tensor_f32(result))
}

fn builtin_tensor(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            let all_numeric = items.iter().all(|v| matches!(v, Value::Int(_) | Value::Float(_)));
            if all_numeric && !items.is_empty() {
                let data: Vec<f32> = items.iter().map(|v| v.to_f64().unwrap() as f32).collect();
                let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap();
                Ok(Value::tensor_f32(arr))
            } else if items.is_empty() {
                Err(runtime_error("tensor: cannot create tensor from empty list"))
            } else {
                let first_bad = items.iter().find(|v| !matches!(v, Value::Int(_) | Value::Float(_))).unwrap();
                let hint = if matches!(first_bad, Value::Tensor { .. }) {
                    "\n  = hint: To combine a list of tensors, use (apply concat list-of-tensors)."
                } else {
                    ""
                };
                Err(runtime_error(format!(
                    "tensor: expected a list of numbers, got a list containing {}.{}",
                    first_bad.type_name(), hint
                )))
            }
        }
        Value::Tensor { .. } => Ok(args[0].clone()),
        other => Err(runtime_error(format!("tensor: expected a list or a tensor, got {}", other.type_name()))),
    }
}

fn builtin_range(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    builtin_arange(args, kw)
}

fn cast_value(value: &Value, target: Dtype) -> R {
    match value {
        Value::Tensor { data, dtype } => {
            if *dtype == target {
                return Ok(value.clone());
            }
            Ok(Value::Tensor { data: data.clone(), dtype: target })
        }
        Value::DeviceBuffer(db) => {
            let host_data = db.to_host()?;
            Ok(Value::Tensor { data: std::sync::Arc::new(host_data), dtype: target })
        }
        Value::Float(f) => {
            Ok(Value::Tensor {
                data: std::sync::Arc::new(ArrayD::from_elem(IxDyn(&[]), *f)),
                dtype: target,
            })
        }
        Value::Int(n) => {
            Ok(Value::Tensor {
                data: std::sync::Arc::new(ArrayD::from_elem(IxDyn(&[]), *n as f32)),
                dtype: target,
            })
        }
        other => Err(runtime_error(format!(
            "cast expects a tensor, float, or int as 1st argument, got {}",
            other.type_name()
        ))),
    }
}

fn builtin_cast(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    use crate::interpreter::value::Dtype;
    if args.len() != 2 {
        return Err(runtime_error(format!(
            "cast expects 2 arguments (cast tensor :dtype), got {}",
            args.len()
        )));
    }
    let target = match &args[1] {
        Value::Keyword(k) => Dtype::from_keyword(k).ok_or_else(|| {
            runtime_error(format!(
                "cast: unknown dtype :{k}. Valid dtypes: :f32, :f16, :bf16, :i32"
            ))
        })?,
        other => return Err(runtime_error(format!(
            "cast expects a dtype keyword as 2nd argument, got {}. Example: (cast x :f32)",
            other.type_name()
        ))),
    };
    cast_value(&args[0], target)
}

fn builtin_cast_like(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("__cast-like expects two arguments"));
    }
    cast_like(&args[0], &args[1])
}

pub(crate) fn cast_like(value: &Value, reference: &Value) -> R {
    let target = match reference {
        Value::Tensor { dtype, .. } => *dtype,
        Value::DeviceBuffer(buffer) => buffer.dtype,
        Value::Float(_) => Dtype::F32,
        Value::Int(_) => Dtype::I32,
        other => return Err(runtime_error(format!(
            "__cast-like expects a numeric reference, got {}",
            other.type_name()
        ))),
    };
    cast_value(value, target)
}
