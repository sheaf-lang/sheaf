// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Tracing compiler: discover function return types by interpreting with dummy data.
//!
//! Instead of static type inference (which can't see through value-and-grad, tree-map,
//! etc.), we run the function body through the interpreter with zero-filled tensors
//! of the declared shapes. The output Value is then converted to a StableHLOType.
//!
//! This is the same approach as JAX's abstract interpretation: trace once, observe
//! shapes, use those for compilation/dispatch.

use crate::compiler::stablehlo::StableHLOType;
use crate::core::compiler::{CompilerContext, FunctionDef, ParamField, ParamLayout};
use crate::core::error::{SheafError, SheafResult};
use crate::core::inference::FunctionSignature;
use crate::interpreter::builtins::register_builtins;
use crate::interpreter::env::Env;
use crate::interpreter::value::{Dtype, Value};
use ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;

/// Trace a function by executing it with dummy inputs to discover the concrete
/// return type. Creates zero-filled tensors matching each parameter's declared type,
/// runs the function body through the interpreter, and converts the output Value
/// back to a StableHLOType.
pub fn trace_function_signature(
    compiler: &CompilerContext,
    func_def: &FunctionDef,
) -> SheafResult<FunctionSignature> {
    let body = func_def.body_compiled.as_ref().ok_or_else(|| SheafError::Runtime {
        message: format!("trace: function '{}' has no compiled body", func_def.name),
        location: None,
    })?;

    // Build param types: use known_param_types, default to scalar
    let param_types: Vec<StableHLOType> = func_def
        .params
        .iter()
        .map(|p| {
            func_def
                .known_param_types
                .iter()
                .find(|(name, _)| name == p)
                .map(|(_, ty)| ty.clone())
                .or_else(|| {
                    func_def
                        .signature
                        .as_ref()
                        .and_then(|sig| {
                            func_def
                                .params
                                .iter()
                                .position(|n| n == p)
                                .map(|i| sig.param_types[i].clone())
                        })
                })
                .unwrap_or(StableHLOType::scalar_f32())
        })
        .collect();

    // Create dummy inputs
    let dummy_inputs: Vec<Value> = func_def
        .params
        .iter()
        .zip(param_types.iter())
        .map(|(name, ty)| {
            // For defparams-typed params, look up the ParamLayout to build a Dict
            if let Some(layout) = find_param_layout(compiler, func_def, name) {
                param_layout_to_dummy_value(layout)
            } else {
                stablehlo_to_dummy_value(ty)
            }
        })
        .collect();

    // Set up interpreter environment WITHOUT vmfb_sessions
    // (forces all function calls to interpret, not dispatch to IREE)
    let mut env = Env::with_registry(compiler.registry.clone());
    register_builtins(&mut env);
    register_trace_overrides(&mut env);

    // Bind params and execute
    env.push_scope();
    for (param, val) in func_def.params.iter().zip(dummy_inputs.iter()) {
        env.set(param, val.clone());
    }
    let result = crate::interpreter::eval(body, &mut env)?;
    env.pop_scope();

    // Extract return type and dict keys
    let return_type = value_to_stablehlo_type(&result)?;
    let return_dict_keys = match &result {
        Value::Dict(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            Some(keys)
        }
        _ => None,
    };

    Ok(FunctionSignature {
        param_types,
        return_type,
        return_dict_keys,
    })
}

/// Convert a StableHLOType to a dummy Value (zero-filled).
fn stablehlo_to_dummy_value(ty: &StableHLOType) -> Value {
    match ty {
        StableHLOType::ScalarF32 | StableHLOType::ScalarF64 => Value::Float(0.0),
        StableHLOType::ScalarI64 => Value::Int(0),
        StableHLOType::ScalarI1 => Value::Bool(false),
        StableHLOType::Tensor { shape, .. } => {
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let data = ArrayD::zeros(IxDyn(&dims));
            Value::Tensor {
                data,
                dtype: Dtype::F32,
            }
        }
        StableHLOType::Tuple(elems) => {
            Value::Tuple(elems.iter().map(stablehlo_to_dummy_value).collect())
        }
    }
}

/// Create a dummy Value::Dict matching a ParamLayout structure.
/// The interpreter uses Dicts (not Tuples) for defparams-typed params at runtime.
fn param_layout_to_dummy_value(layout: &ParamLayout) -> Value {
    // Group fields by top-level key
    let mut top_keys: Vec<String> = Vec::new();
    for field in &layout.fields {
        if let Some(k) = field.path.first() {
            if !top_keys.contains(k) {
                top_keys.push(k.clone());
            }
        }
    }

    let mut result = BTreeMap::new();
    for key in &top_keys {
        let children: Vec<_> = layout
            .fields
            .iter()
            .filter(|f| f.path.first().map(|k| k == key).unwrap_or(false))
            .collect();

        if children.len() == 1 && children[0].path.len() == 1 {
            // Leaf tensor
            let shape: Vec<usize> = children[0].shape.iter().map(|&d| d as usize).collect();
            let data = ArrayD::zeros(IxDyn(&shape));
            result.insert(
                key.clone(),
                Value::Tensor {
                    data,
                    dtype: Dtype::F32,
                },
            );
        } else {
            // Nested group: build sub-dict
            let mut sub = BTreeMap::new();
            for field in &children {
                if let Some(sub_key) = field.path.get(1) {
                    let shape: Vec<usize> = field.shape.iter().map(|&d| d as usize).collect();
                    let data = ArrayD::zeros(IxDyn(&shape));
                    sub.insert(
                        sub_key.clone(),
                        Value::Tensor {
                            data,
                            dtype: Dtype::F32,
                        },
                    );
                }
            }
            result.insert(key.clone(), Value::Dict(sub));
        }
    }

    Value::Dict(result)
}

/// Convert a runtime Value back to a StableHLOType (shape extraction).
pub fn value_to_stablehlo_type(val: &Value) -> SheafResult<StableHLOType> {
    match val {
        Value::Float(_) => Ok(StableHLOType::scalar_f32()),
        Value::Int(_) => Ok(StableHLOType::ScalarI64),
        Value::Bool(_) => Ok(StableHLOType::ScalarI1),
        Value::Tensor { data, .. } => {
            let shape: Vec<i64> = data.shape().iter().map(|&d| d as i64).collect();
            if shape.is_empty() {
                Ok(StableHLOType::scalar_f32())
            } else {
                Ok(StableHLOType::f32_tensor(shape))
            }
        }
        Value::Tuple(elems) => {
            let tys: SheafResult<Vec<StableHLOType>> =
                elems.iter().map(value_to_stablehlo_type).collect();
            Ok(StableHLOType::Tuple(tys?))
        }
        Value::Dict(map) => {
            // Dict -> Tuple sorted by key (matching codegen convention)
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let tys: SheafResult<Vec<StableHLOType>> = keys
                .iter()
                .map(|k| value_to_stablehlo_type(&map[*k]))
                .collect();
            Ok(StableHLOType::Tuple(tys?))
        }
        Value::List(items) => {
            // List of scalars -> 1D tensor
            if items
                .iter()
                .all(|v| matches!(v, Value::Float(_) | Value::Int(_)))
            {
                Ok(StableHLOType::f32_tensor(vec![items.len() as i64]))
            } else {
                // List of structured values -> Tuple
                let tys: SheafResult<Vec<StableHLOType>> =
                    items.iter().map(value_to_stablehlo_type).collect();
                Ok(StableHLOType::Tuple(tys?))
            }
        }
        Value::Nil => Ok(StableHLOType::scalar_f32()),
        _ => Err(SheafError::Runtime {
            message: format!(
                "trace: cannot convert {} to StableHLOType",
                val.type_name()
            ),
            location: None,
        }),
    }
}

/// Find the ParamLayout for a defparams-typed parameter.
/// Checks if the function has a type annotation for this param and looks it up
/// in the compiler's param_types registry.
fn find_param_layout<'a>(
    compiler: &'a CompilerContext,
    func_def: &FunctionDef,
    param_name: &str,
) -> Option<&'a ParamLayout> {
    // Type annotations are stored as known_param_types with Tuple types,
    // but we need the original ParamLayout for Dict reconstruction.
    // Check if the param's known type is a Tuple (from defparams).
    let has_tuple_type = func_def
        .known_param_types
        .iter()
        .find(|(n, _)| n == param_name)
        .map(|(_, ty)| matches!(ty, StableHLOType::Tuple(_)))
        .unwrap_or(false);

    if !has_tuple_type {
        return None;
    }

    // Search through all param layouts for one whose StableHLO type matches
    for layout in compiler.param_types.values() {
        let layout_ty = crate::forms::ml::param_layout_to_stablehlo_type(layout);
        if func_def
            .known_param_types
            .iter()
            .any(|(n, ty)| n == param_name && ty == &layout_ty)
        {
            return Some(layout);
        }
    }
    None
}

/// Replace effectful builtins with trace-safe no-ops.
fn register_trace_overrides(env: &mut Env) {
    env.set_builtin("print", |_args, _kwargs| Ok(Value::Nil));
    env.set_builtin("io", |_args, _kwargs| Ok(Value::Int(42)));
}

/// Convert a runtime `Value::Dict` (observed from tracing) to a `ParamLayout`.
/// This is the inverse of `param_layout_to_dummy_value`: given a nested Dict of
/// tensors, produce the same layout that `defparams` would create.
///
/// Keys are sorted alphabetically at each level (BTreeMap order), and tuple
/// indices are assigned in that order — matching the codegen convention.
pub fn value_to_param_layout(name: &str, val: &Value) -> Option<ParamLayout> {
    let dict = match val {
        Value::Dict(map) => map,
        _ => return None,
    };

    let mut fields = Vec::new();
    let top_keys: Vec<&String> = dict.keys().collect(); // BTreeMap: already sorted

    for (top_idx, key) in top_keys.iter().enumerate() {
        match &dict[*key] {
            // Nested dict: two-level path
            Value::Dict(sub) => {
                let sub_keys: Vec<&String> = sub.keys().collect();
                for (sub_idx, sub_key) in sub_keys.iter().enumerate() {
                    let shape = extract_shape(&sub[*sub_key])?;
                    fields.push(ParamField {
                        path: vec![(*key).clone(), (*sub_key).clone()],
                        shape,
                        tuple_index: vec![top_idx, sub_idx],
                    });
                }
            }
            // Leaf tensor: one-level path
            other => {
                let shape = extract_shape(other)?;
                fields.push(ParamField {
                    path: vec![(*key).clone()],
                    shape,
                    tuple_index: vec![top_idx],
                });
            }
        }
    }

    Some(ParamLayout {
        name: name.to_string(),
        fields,
    })
}

/// Extract shape from a Value (tensor or scalar).
fn extract_shape(val: &Value) -> Option<Vec<i64>> {
    match val {
        Value::Tensor { data, .. } => Some(data.shape().iter().map(|&d| d as i64).collect()),
        Value::Float(_) | Value::Int(_) => Some(vec![]),
        _ => None,
    }
}
