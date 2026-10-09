use super::*;
use crate::core::prng::{self, Host};
use std::sync::Arc;

pub(super) fn register(env: &mut Env) {
    register_native_builtin(env, OpId::RandomKey, builtin_random_key);
    register_native_builtin(env, OpId::RandomSplit, builtin_random_split);
    register_native_builtin(env, OpId::RandomNormal, builtin_random_normal);
    register_native_builtin(env, OpId::RandomUniform, builtin_random_uniform);
    register_native_builtin(env, OpId::RandomRandint, builtin_random_randint);
    register_native_builtin(env, OpId::Choice, builtin_choice);
    register_native_builtin(env, OpId::TopK, builtin_top_k);
}

fn scalar_integer(value: &Value, name: &str) -> Result<i64, SheafError> {
    let value = value.ensure_host_cow()?;
    if let Value::Int(n) = &*value { return Ok(*n); }
    if !matches!(&*value, Value::Tensor { dtype: Dtype::Bool, .. })
        && let Some(n) = value.to_f64()
        && n.is_finite() && n.fract() == 0.0
        && (-9223372036854775808.0..9223372036854775808.0).contains(&n) {
        return Ok(n as i64);
    }
    Err(runtime_error(format!("{name}: expected an integer")))
}

fn words_to_key(words: [u32; 2]) -> Value {
    let limbs = prng::encode(&mut Host, words);
    Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[prng::KEY_SIZE]), limbs.to_vec()).unwrap())
}

fn normalize_key(key: &Value) -> R {
    let key = key.ensure_host()?;
    match &key {
        Value::Int(seed) => Ok(words_to_key(prng::seed(*seed))),
        Value::Tensor { data, dtype: Dtype::F32 } if data.shape() == [prng::KEY_SIZE] => {
            let limbs = std::array::from_fn::<_, { prng::KEY_SIZE }, _>(|i| data[IxDyn(&[i])]);
            if prng::valid_key(&limbs) { return Ok(key); }
            Err(runtime_error("expected a PRNG key with four unsigned 16-bit limbs"))
        }
        Value::List(items) if matches!(items.as_slice(), [Value::Int(_), Value::Int(_)]) => {
            let [Value::Int(lo), Value::Int(hi)] = items.as_slice() else { unreachable!() };
            match (u32::try_from(*lo), u32::try_from(*hi)) {
                (Ok(lo), Ok(hi)) => Ok(words_to_key([lo, hi])),
                _ => Err(runtime_error("expected a PRNG key with two unsigned 32-bit words")),
            }
        }
        _ => Err(runtime_error(format!("expected a PRNG key, got {}", key.type_name()))),
    }
}

fn key_words(key: &Value) -> Result<[u32; 2], SheafError> {
    let Value::Tensor { data, .. } = normalize_key(key)? else { unreachable!() };
    let limbs = std::array::from_fn(|i| data[IxDyn(&[i])]);
    Ok(prng::decode(&mut Host, limbs))
}

fn builtin_random_key(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 1 { return Err(arity_error("random-key", 1, args.len())); }
    let seed = scalar_integer(&args[0], "random-key")?;
    Ok(words_to_key(prng::seed(seed)))
}

fn builtin_random_split(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() || args.len() > 2 {
        return Err(runtime_error("random-split: expected (random-split key) or (random-split key n)"));
    }
    let key = key_words(&args[0])?;
    let n = match args.get(1) {
        Some(Value::Int(n)) => usize::try_from(*n)
            .map_err(|_| runtime_error("random-split: count must be nonnegative"))?,
        Some(Value::Float(n)) => checked_dimension(*n as f64)
            .map_err(|error| runtime_error(format!("random-split: count: {error}")))?,
        None => 2,
        _ => return Err(runtime_error("random-split: count must be an integer")),
    };
    if n > isize::MAX as usize / std::mem::size_of::<Value>() {
        return Err(runtime_error("random-split: count exceeds addressable memory"));
    }
    if n as u128 > 1u128 << 32 {
        return Err(runtime_error("random-split: count exceeds PRNG counter capacity"));
    }
    let mut keys = Vec::new();
    keys.try_reserve_exact(n).map_err(|_| runtime_error("random-split: cannot allocate keys"))?;
    keys.extend((0..n).map(|i| words_to_key(prng::split(&mut Host, key, i as u32))));
    Ok(Value::List(keys))
}

fn random_tensor(
    args: &[Value], dtype: Dtype, mut sample: impl FnMut([u32; 2]) -> f32,
) -> R {
    let key = key_words(&args[0])?;
    let shape = shape_from_value(&args[1])?;
    let n: usize = shape.iter().product();
    if n as u128 > 1u128 << 32 {
        return Err(runtime_error("random tensor exceeds PRNG counter capacity"));
    }
    let data = (0..n).map(|i| sample(prng::bits(&mut Host, key, i as u32))).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| runtime_error(format!("random tensor: shape error: {e}")))?;
    Ok(Value::tensor(arr, dtype))
}

fn builtin_random_normal(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 { return Err(arity_error("random-normal", 2, args.len())); }
    random_tensor(args, Dtype::F32, |bits| prng::normal(&mut Host, bits))
}

fn builtin_random_uniform(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 { return Err(arity_error("random-uniform", 2, args.len())); }
    random_tensor(args, Dtype::F32, |bits| prng::uniform(&mut Host, bits[0]))
}

fn builtin_random_randint(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 4 { return Err(arity_error("random-randint", 4, args.len())); }
    let low = scalar_integer(&args[2], "random-randint: low")?;
    let high = scalar_integer(&args[3], "random-randint: high")?;
    if high <= low { return Err(runtime_error("random-randint: high must be > low")); }
    random_tensor(args, Dtype::I32, |bits| prng::randint(&mut Host, bits[0], low, high))
}

fn builtin_choice(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    if !(2..=3).contains(&args.len()) {
        return Err(runtime_error("choice: expected (choice key n :p probs)"));
    }
    let key = key_words(&args[0])?;
    let bits = prng::bits(&mut Host, key, 0);
    let u = prng::uniform(&mut Host, bits[0]);
    let n = args[1].to_f64().ok_or_else(|| runtime_error("choice: n must be integer"))? as usize;
    let Some(probs) = kw.get("p").or_else(|| args.get(2)) else {
        return Ok(Value::Int((u * n as f32) as i64));
    };
    let probs = probs.ensure_host()?;
    let values: Vec<f32> = match &probs {
        Value::Tensor { data, .. } => data.iter().copied().collect(),
        Value::List(items) => items.iter().map(|v| {
            v.to_f32().ok_or_else(|| runtime_error("choice: probabilities must be numbers"))
        }).collect::<Result<_, _>>()?,
        _ => return Err(runtime_error("choice: :p must be a tensor or list of probabilities")),
    };
    if values.is_empty() { return Err(runtime_error("choice: empty probability tensor")); }
    let mut cumsum = 0.0f32;
    for (i, p) in values.iter().enumerate() {
        cumsum += p;
        if u < cumsum { return Ok(Value::Int(i as i64)); }
    }
    Ok(Value::Int((values.len() - 1) as i64))
}

fn builtin_top_k(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() < 2 {
        return Err(runtime_error("top_k: expected (top_k tensor k)"));
    }
    let (arr, dtype) = to_array(&args[0])?;
    let k = match &args[1] {
        Value::Int(n) => *n as usize,
        Value::Float(f) => *f as usize,
        Value::Tensor { data, .. } if data.ndim() == 0 => as_scalar(data) as usize,
        _ => return Err(runtime_error("top_k: k must be integer")),
    };
    let flat: Vec<f32> = arr.iter().copied().collect();
    let mut indexed: Vec<(usize, f32)> = flat.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = k.min(indexed.len());
    let top_vals: Vec<f32> = indexed[..k].iter().map(|(_, v)| *v).collect();
    let top_idxs: Vec<f32> = indexed[..k].iter().map(|(i, _)| *i as f32).collect();
    let vals = ArrayD::from_shape_vec(IxDyn(&[k]), top_vals)
        .map_err(|e| runtime_error(format!("top_k: {}", e)))?;
    let idxs = ArrayD::from_shape_vec(IxDyn(&[k]), top_idxs)
        .map_err(|e| runtime_error(format!("top_k: {}", e)))?;
    Ok(Value::Tuple(vec![
        Value::Tensor { data: Arc::new(vals), dtype },
        Value::tensor_i32(idxs),
    ]))
}
