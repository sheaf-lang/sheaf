// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! IREE dispatch.

#![cfg(iree_runtime)]

use crate::sheaf_msg;
use crate::runtime::jit::JitVagOutcome;
use crate::core::error::SheafError;
use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::{Dtype, Value};
use std::collections::BTreeMap;

pub(super) fn validate_device_dtypes(args: &[Value]) -> Result<(), SheafError> {
    let is_metal = crate::core::config::device_override() == Some("metal")
        || crate::runtime::iree_session::IreeSession::cached_target_backend()
            == Some("metal-spirv");
    validate_backend_dtypes(is_metal, args)
}

fn validate_backend_dtypes(is_metal: bool, args: &[Value]) -> Result<(), SheafError> {
    if is_metal && args.iter().any(contains_bf16) {
        return Err(runtime_error(
            "bf16 is not supported on Metal\n  = hint: use f16, or select a CUDA/CPU device",
        ));
    }
    Ok(())
}

fn contains_bf16(value: &Value) -> bool {
    match value {
        Value::Tensor { dtype, .. } => *dtype == Dtype::BF16,
        Value::DeviceBuffer(buffer) => buffer.dtype == Dtype::BF16,
        Value::Dict(values) => values.values().any(contains_bf16),
        Value::List(values) | Value::Tuple(values) => values.iter().any(contains_bf16),
        _ => false,
    }
}

/// Dispatches a compiled function when available.
pub(super) fn try_iree_dispatch(
    func_def: &crate::core::expr::FunctionDef,
    args: &[Value],
    env: &mut Env,
) -> Option<Result<Value, SheafError>> {
    if let Err(error) = validate_device_dtypes(args) {
        return Some(Err(error));
    }
    let aot_variant = match func_def.signature.as_ref() {
        Some(signature)
            if crate::runtime::iree_session::args_match_signature(args, &signature.param_types)
                && crate::runtime::iree_session::check_shapes_match(
                    args,
                    &signature.param_types,
                )
                .is_ok() =>
        {
            match crate::runtime::iree_session::initialized_shared_session() {
                Some(session) => {
                    match session.precompiled_module_for(&func_def.name, &func_def.body_hash()) {
                        Ok(module_name) => module_name.map(|name| (name, signature.clone())),
                        Err(e) => return Some(Err(e)),
                    }
                }
                None => None,
            }
        }
        _ => None,
    };
    let (full_name, sig) = match aot_variant {
        Some((module_name, signature)) => (dispatch_name(&module_name, &func_def.name), signature),
        None => {
            let jit = env.jit_compiler.as_mut()?;
            let info = if let Some(info) = jit.module_for_args(func_def, args) {
                info
            } else {
                let key = crate::runtime::jit::cache_key_for_function(
                    func_def,
                    args,
                    &env.registry,
                )?;
                jit.module_for_key(&key)?
            };
            (jit_dispatch_name(&info, &func_def.name), info.sig)
        }
    };

    let iree_session = match crate::runtime::iree_session::shared_session() {
        Ok(session) => session,
        Err(e) => return Some(Err(e)),
    };
    let result = match iree_session.call_typed_device(&full_name, args, &sig.return_type) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };

    let result = if !sig.arg_type_layouts.is_empty() {
        crate::core::inference::reconstruct_jit_result(result, &sig.return_type, &sig.arg_type_layouts)
    } else {
        result
    };
    let result = if let Some(layout) = &sig.return_layout {
        layout.reconstruct(result)
    } else {
        result
    };
    let result = match (&sig.return_dict_keys, result) {
        (Some(keys), Value::Tuple(elems)) if elems.len() == keys.len() => {
            let map = keys.iter().cloned().zip(elems).collect();
            Value::Dict(map)
        }
        (_, other) => other,
    };

    Some(Ok(result))
}

fn dispatch_name(module_name: &str, function_name: &str) -> String {
    format!(
        "{}.{}",
        module_name,
        function_name
            .replace('-', "_")
            .replace('?', "_q")
            .replace('!', "_b")
    )
}

fn jit_dispatch_name(
    module: &crate::runtime::jit::CompiledModuleInfo,
    function_name: &str,
) -> String {
    dispatch_name(&module.module_name, function_name)
}

/// Compiles and dispatches value-and-grad.
pub(super) fn try_jit_vag(
    func: &Value,
    params: &Value,
    env: &mut Env,
) -> JitVagOutcome {
    if let Err(error) = validate_device_dtypes(std::slice::from_ref(params)) {
        return JitVagOutcome::Success(Err(error));
    }
    let augmented_func = match augment_closure_with_free_vars(func, env) {
        Some(f) => f,
        None => return JitVagOutcome::Unsupported,
    };

    let jit = match env.jit_compiler.as_mut() {
        Some(j) => j,
        None => return JitVagOutcome::Unsupported,
    };

    let iree_session = match crate::runtime::iree_session::shared_session() {
        Ok(session) => session,
        Err(e) => return JitVagOutcome::Success(Err(e)),
    };

    let (module_name, sig, param_names) = match jit.try_jit_value_and_grad(
        &augmented_func,
        params,
        &env.registry,
        &iree_session,
    ) {
        Some(x) => x,
        None => return jit.classify_vag_skip(),
    };

    if let Some(ref mut mp) = env.mem_profiler {
        mp.sample("after VAG compile");
    }

    let (fn_params, closure) = match &augmented_func {
        Value::Function { params: p, closure: c, .. } => (p, c),
        _ => return JitVagOutcome::Unsupported,
    };

    let mut args: Vec<Value> = Vec::new();
    for name in &param_names {
        if fn_params.contains(name) {
            args.push(params.clone());
        } else if let Some((_, val)) = closure.iter().find(|(k, _)| k == name) {
            args.push(val.clone());
        } else {
            continue;
        }
    }

    let full_name = format!("{}.value_and_grad", module_name);
    let result = match iree_session.call_typed_device(&full_name, &args, &sig.return_type) {
        Ok(v) => v,
        Err(e) => return JitVagOutcome::Success(Err(e)),
    };

    if let Some(ref mut mp) = env.mem_profiler {
        mp.sample("after VAG dispatch");
    }

    if crate::core::config::verbosity() >= 2 {
        let desc = match &result {
            Value::Tuple(elems) => format!("Tuple(len={})", elems.len()),
            other => other.type_name().to_string(),
        };
        sheaf_msg!("jit: [vag] result structure: {}", desc);
        if let Value::Tuple(elems) = &result {
            for (i, e) in elems.iter().enumerate() {
                sheaf_msg!("jit: [vag]   elem[{}]: {}", i, e.type_name());
            }
        }
    }

    let unpacked = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unpack_vag_result(&result, params)
    })) {
        Ok(Some(v)) => v,
        Ok(None) => {
            return JitVagOutcome::Success(Err(runtime_error(
                "value-and-grad: result unpacking failed".to_string(),
            )));
        }
        Err(e) => {
            let detail = e.downcast_ref::<String>().map(|s| s.as_str())
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("unknown");
            return JitVagOutcome::Success(Err(runtime_error(format!(
                "value-and-grad: result unpacking panicked: {}", detail
            ))));
        }
    };

    JitVagOutcome::Success(Ok(unpacked))
}

fn augment_closure_with_free_vars(func: &Value, env: &Env) -> Option<Value> {
    let (fn_params, body, closure) = match func {
        Value::Function { params, body, closure, .. } => (params, body, closure),
        _ => return None,
    };

    let mut free_set = std::collections::HashSet::new();
    crate::autodiff::collect_free_vars(body, &mut free_set);

    for p in fn_params {
        free_set.remove(p.as_str());
    }
    for (k, _) in closure {
        free_set.remove(k.as_str());
    }

    let mut free: Vec<&str> = free_set.iter().map(|s| s.as_str()).collect();
    free.sort();

    let mut augmented_closure = closure.clone();
    for name in &free {
        if let Ok(val) = env.get(name) {
            augmented_closure.push((name.to_string(), val.clone()));
        }
    }

    Some(Value::Function {
        name: None,
        params: fn_params.clone(),
        body: body.clone(),
        closure: augmented_closure,
    })
}

fn unpack_vag_result(result: &Value, original_params: &Value) -> Option<Value> {
    let elems = match result {
        Value::Tuple(elems) => elems,
        _ => return None,
    };

    if elems.len() < 2 {
        return None;
    }

    let loss = match &elems[0] {
        Value::Tensor { data, .. } if data.len() == 1 => {
            Value::Float(crate::interpreter::builtins::as_scalar(data))
        }
        Value::DeviceBuffer(db) if db.shape.is_empty() || db.shape.iter().product::<usize>() == 1 => {
            match db.to_host() {
                Ok(data) => Value::Float(*data.iter().next().unwrap_or(&0.0)),
                Err(_) => return None,
            }
        }
        Value::Float(f) => Value::Float(*f),
        _ => return None,
    };

    let grad = if elems.len() == 2 {
        match original_params {
            Value::Dict(map) => tuple_to_dict(&elems[1], map)?,
            _ => elems[1].clone(),
        }
    } else {
        Value::Tuple(elems[1..].to_vec())
    };

    Some(Value::List(vec![loss, grad]))
}

fn tuple_to_dict(tuple_val: &Value, original: &BTreeMap<String, Value>) -> Option<Value> {
    let elems = match tuple_val {
        Value::Tuple(elems) => elems,
        _ => {
            return Some(tuple_val.clone());
        }
    };

    if elems.len() != original.len() {
        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: [vag] tuple_to_dict: tuple len {} != dict keys {} ({:?})",
                elems.len(), original.len(), original.keys().collect::<Vec<_>>());
        }
        return None;
    }

    let mut result = BTreeMap::new();
    for ((key, orig_val), elem) in original.iter().zip(elems.iter()) {
        let val = match orig_val {
            Value::Dict(sub_map) => tuple_to_dict(elem, sub_map)?,
            Value::List(orig_list) => tuple_to_list(elem, orig_list)?,
            _ => elem.clone(),
        };
        result.insert(key.clone(), val);
    }
    Some(Value::Dict(result))
}

fn tuple_to_list(tuple_val: &Value, original: &[Value]) -> Option<Value> {
    let elems = match tuple_val {
        Value::Tuple(elems) => elems,
        _ => {
            return None;
        }
    };

    if elems.len() != original.len() {
        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: [vag] tuple_to_list: tuple len {} != list len {}",
                elems.len(), original.len());
        }
        return None;
    }

    let mut result = Vec::with_capacity(original.len());
    for (orig_val, elem) in original.iter().zip(elems.iter()) {
        let val = match orig_val {
            Value::Dict(sub_map) => tuple_to_dict(elem, sub_map)?,
            Value::List(sub_list) => tuple_to_list(elem, sub_list)?,
            _ => elem.clone(),
        };
        result.push(val);
    }
    Some(Value::List(result))
}

#[cfg(test)]
mod tests {
    use super::{dispatch_name, jit_dispatch_name, validate_backend_dtypes};
    use crate::core::inference::FunctionSignature;
    use crate::runtime::jit::CompiledModuleInfo;

    fn signature() -> FunctionSignature {
        FunctionSignature {
            param_types: vec![crate::StableHLOType::scalar_f32()],
            return_type: crate::StableHLOType::scalar_f32(),
            return_dict_keys: None,
            return_layout: None,
            arg_type_layouts: Vec::new(),
            captured_scalars: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn metal_rejects_nested_bf16_values() {
        let value = crate::interpreter::value::Value::Dict(
            std::iter::once((
                "weight".to_string(),
                crate::interpreter::value::Value::tensor_bf16(
                    ndarray::ArrayD::zeros(ndarray::IxDyn(&[2])),
                ),
            ))
            .collect(),
        );
        let error = validate_backend_dtypes(true, std::slice::from_ref(&value)).unwrap_err();
        assert!(error.to_string().contains("bf16 is not supported on Metal"));
        assert!(validate_backend_dtypes(false, std::slice::from_ref(&value)).is_ok());
    }

    #[test]
    fn dispatch_name_uses_registered_module_name() {
        assert_eq!(
            dispatch_name("aot_model_42", "predict-value?"),
            "aot_model_42.predict_value_q"
        );
    }

    #[test]
    fn jit_dispatch_name_uses_catalogued_variant_namespace() {
        let module = CompiledModuleInfo {
            static_inputs: Vec::new(),
            function_name: "predict-value?".to_string(),
            module_name: "jit_variant_42".to_string(),
            sig: signature(),
            argument_layouts: vec![crate::core::inference::ValueLayout::Leaf],
        };

        assert_eq!(
            jit_dispatch_name(&module, "predict-value?"),
            "jit_variant_42.predict_value_q"
        );
    }
}
