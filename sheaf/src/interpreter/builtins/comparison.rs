use super::*;

pub(super) fn register(env: &mut Env) {
    register_native_builtin(env, OpId::Equal, builtin_eq);
    register_native_builtin(env, OpId::ElementEqual, builtin_elem_eq);
    register_native_builtin(env, OpId::NotEqual, builtin_neq);
    register_native_builtin(env, OpId::Less, builtin_lt);
    register_native_builtin(env, OpId::Greater, builtin_gt);
    register_native_builtin(env, OpId::LessEqual, builtin_le);
    register_native_builtin(env, OpId::GreaterEqual, builtin_ge);
    register_native_builtin(env, OpId::Not, builtin_not);
    register_native_builtin(env, OpId::Shape, builtin_shape);
    register_native_builtin(env, OpId::Ndim, builtin_ndim);
    register_native_builtin(env, OpId::Len, builtin_len);
    register_native_builtin(env, OpId::Int, builtin_int);
    register_native_builtin(env, OpId::Float, builtin_float);
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Nil, Value::Nil) => true,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Keyword(a), Value::Keyword(b)) => a == b,
        (Value::List(a), Value::List(b)) => lists_equal(a, b),
        _ => false,
    }
}

fn lists_equal(a: &[Value], b: &[Value]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| values_equal(x, y))
}

fn builtin_eq(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 { return Err(runtime_error(format!("=: expected 2 arguments, got {}", args.len()))); }
    fn str_val(v: &Value) -> Option<&str> {
        match v {
            Value::String(s) | Value::Keyword(s) => Some(s.as_str()),
            _ => None,
        }
    }
    match (&args[0], &args[1]) {
        (a, b) if str_val(a).is_some() && str_val(b).is_some() =>
            return Ok(Value::Bool(str_val(a) == str_val(b))),
        (Value::Bool(a), Value::Bool(b)) => return Ok(Value::Bool(a == b)),
        (Value::Nil, Value::Nil) => return Ok(Value::Bool(true)),
        (Value::Nil, _) | (_, Value::Nil) => return Ok(Value::Bool(false)),
        (a, _) if str_val(a).is_some() => return Ok(Value::Bool(false)),
        (_, b) if str_val(b).is_some() => return Ok(Value::Bool(false)),
        (Value::List(a), Value::List(b)) => return Ok(Value::Bool(lists_equal(a, b))),
        (Value::List(_), _) | (_, Value::List(_)) => return Ok(Value::Bool(false)),
        _ => {}
    }
    let (a, _) = to_array(&args[0])?;
    let (b, _) = to_array(&args[1])?;
    Ok(Value::Bool(a == b))
}

fn builtin_elem_eq(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() == 2 {
        fn str_val_eq(v: &Value) -> Option<&str> {
            match v {
                Value::String(s) | Value::Keyword(s) => Some(s.as_str()),
                _ => None,
            }
        }
        match (&args[0], &args[1]) {
            (a, b) if str_val_eq(a).is_some() && str_val_eq(b).is_some() =>
                return Ok(Value::Bool(str_val_eq(a) == str_val_eq(b))),
            (a, _) if str_val_eq(a).is_some() => return Ok(Value::Bool(false)),
            (_, b) if str_val_eq(b).is_some() => return Ok(Value::Bool(false)),
            _ => {}
        }
    }
    cmp_op(args, |a, b| if (a - b).abs() < 1e-10 { 1.0 } else { 0.0 }, Dtype::I32)
}

fn builtin_neq(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 { return Err(runtime_error(format!("!=: expected 2 arguments, got {}", args.len()))); }
    match (&args[0], &args[1]) {
        (Value::String(a), Value::String(b)) => return Ok(Value::Bool(a != b)),
        (Value::Keyword(a), Value::Keyword(b)) => return Ok(Value::Bool(a != b)),
        (Value::Bool(a), Value::Bool(b)) => return Ok(Value::Bool(a != b)),
        (Value::Nil, Value::Nil) => return Ok(Value::Bool(false)),
        (Value::Nil, _) | (_, Value::Nil) => return Ok(Value::Bool(true)),
        (Value::String(_), _) | (_, Value::String(_)) => return Ok(Value::Bool(true)),
        (Value::Keyword(_), _) | (_, Value::Keyword(_)) => return Ok(Value::Bool(true)),
        _ => {}
    }
    if let (Value::Int(_) | Value::Float(_) | Value::Bool(_), Value::Int(_) | Value::Float(_) | Value::Bool(_)) = (&args[0], &args[1]) {
        let a = args[0].to_f64().ok_or_else(|| runtime_error(format!("!=: expected number, got {}", args[0].type_name())))?;
        let b = args[1].to_f64().ok_or_else(|| runtime_error(format!("!=: expected number, got {}", args[1].type_name())))?;
        return Ok(Value::Bool((a - b).abs() > 1e-10));
    }
    cmp_op(args, |a, b| if (a - b).abs() > 1e-10 { 1.0 } else { 0.0 }, Dtype::I32)
}

fn builtin_lt(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    cmp_op(args, |a, b| if a < b { 1.0 } else { 0.0 }, Dtype::I32)
}

fn builtin_gt(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() == 2
       && let (Some(a), Some(b)) = (args[0].to_f64(), args[1].to_f64()) {
            return Ok(Value::Bool(a > b));
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
    if args.len() != 1 { return Err(runtime_error(format!("not: expected 1 argument, got {}", args.len()))); }
    Ok(Value::Bool(!args[0].is_truthy()))
}

fn builtin_shape(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Err(runtime_error("shape requires at least 1 argument")); }
    match &args[0] {
        Value::Tensor { data, .. } => {
            if args.len() >= 2 {
                let axis_f = args[1].to_f64().ok_or_else(|| runtime_error(format!("shape: axis must be a number, got {}", args[1].type_name())))?;
                let axis = resolve_idx(axis_f, data.ndim())?;
                Ok(Value::Int(data.shape()[axis] as i64))
            } else {
                let shape: Vec<f32> = data.shape().iter().map(|&s| s as f32).collect();
                Ok(Value::tensor_i32(ArrayD::from_shape_vec(IxDyn(&[shape.len()]), shape).unwrap()))
        }
        }
        Value::DeviceBuffer(db) => {
            if args.len() >= 2 {
                let axis_f = args[1].to_f64().ok_or_else(|| runtime_error(format!("shape: axis must be a number, got {}", args[1].type_name())))?;
                let axis = resolve_idx(axis_f, db.shape.len())?;
                Ok(Value::Int(db.shape[axis] as i64))
            } else {
                let shape: Vec<f32> = db.shape.iter().map(|&s| s as f32).collect();
                Ok(Value::tensor_i32(ArrayD::from_shape_vec(IxDyn(&[shape.len()]), shape).unwrap()))
        }
        }
        Value::List(items) => Ok(Value::Int(items.len() as i64)),
        _ => Err(runtime_error(format!("shape: expected tensor, got {}", args[0].type_name()))),
    }
}

fn builtin_ndim(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 1 { return Err(runtime_error(format!("ndim: expected 1 argument, got {}", args.len()))); }
    match &args[0] {
        Value::Tensor { data, .. } => Ok(Value::Int(data.ndim() as i64)),
        Value::DeviceBuffer(db) => Ok(Value::Int(db.shape.len() as i64)),
        _ => Err(runtime_error(format!("ndim: expected tensor, got {}", args[0].type_name()))),
    }
}

fn builtin_len(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 1 { return Err(runtime_error(format!("len: expected 1 argument, got {}", args.len()))); }
    match &args[0] {
        Value::Tensor { data, .. } => Ok(Value::Int(data.shape()[0] as i64)),
        Value::DeviceBuffer(db) => Ok(Value::Int(db.shape.first().copied().unwrap_or(1) as i64)),
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
            if data.ndim() == 0 {
                // Scalar tensor: extract as integer
                Ok(Value::Int(as_scalar(data) as i64))
            } else {
                Ok(Value::tensor_i32(data.mapv(|x| x.floor())))
            }
        }
        Value::DeviceBuffer(db) => {
            let data = db.to_host().map_err(|e| runtime_error(format!("int: {}", e)))?;
            if data.ndim() == 0 {
                Ok(Value::Int(as_scalar(&data) as i64))
            } else {
                Ok(Value::tensor_i32(data.mapv(|x| x.floor())))
            }
        }
        _ => Err(runtime_error(format!(
            "int expects a scalar or tensor, got {}",
            args[0].type_name()
        ))),
    }
}

fn builtin_float(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Err(runtime_error("float requires at least 1 argument")); }
    match &args[0] {
        Value::Int(n) => Ok(Value::Float(*n as f32)),
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
        Value::Tensor { data, .. } => {
            if data.ndim() == 0 {
                Ok(Value::Float(as_scalar(data)))
            } else {
                Ok(Value::Tensor { data: data.clone(), dtype: Dtype::F32 })
            }
        }
        Value::DeviceBuffer(db) => {
            let data = db
                .to_host()
                .map_err(|e| runtime_error(format!("float: {}", e)))?;
            if data.ndim() == 0 {
                Ok(Value::Float(as_scalar(&data)))
            } else {
                Ok(Value::tensor_f32(data))
            }
        }
        _ => Err(runtime_error(format!(
            "float expects a scalar or tensor, got {}. Use (sum x) to reduce a tensor to a scalar.",
            args[0].type_name()
        ))),
    }
}
