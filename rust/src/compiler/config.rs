// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Dict-to-Tuple lowering: converts a JSON shape config into StableHLO types
//! and rewrites `(get p :key)` chains into `GetTupleElement` nodes.
//!
//! Keys are ordered alphabetically (BTreeMap) at every level for a stable,
//! deterministic mapping between dict keys and tuple indices.
//! The interpreter's FFI flatten pass uses the same order when packing dicts
//! into IREE tensor lists.

use std::collections::BTreeMap;

use serde_json::Value as JsonValue;

use crate::compiler::stablehlo::StableHLOType;
use crate::core::compiler::CompiledExpr;
use crate::core::error::{SheafError, SheafResult, SourceLocation};

/// Convert a JSON config dict to a `StableHLOType::Tuple` (recursive).
///
/// Keys are sorted alphabetically at each level (BTreeMap order).
///
/// JSON schema:
/// - `{"W": [2, 8], "b": [8]}` → `Tuple([tensor<2x8xf32>, tensor<8xf32>])`
/// - `{"l1": {"W": [2,8], "b": [8]}, "l2": {...}}` → `Tuple([Tuple([...]), Tuple([...])])`
/// - scalar (empty array `[]`) → `tensor<f32>` (0D)
/// - float number → `tensor<f32>` (0D scalar)
pub fn json_to_stablehlo_type(val: &JsonValue) -> SheafResult<StableHLOType> {
    match val {
        JsonValue::Object(map) => {
            // Sort keys alphabetically — BTreeMap preserves insertion order for serde_json,
            // so we collect into a BTreeMap first.
            let sorted: BTreeMap<&str, &JsonValue> =
                map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            let elems: SheafResult<Vec<StableHLOType>> =
                sorted.values().map(|v| json_to_stablehlo_type(v)).collect();
            Ok(StableHLOType::Tuple(elems?))
        }
        JsonValue::Array(dims) => {
            if dims.is_empty() {
                Ok(StableHLOType::ScalarF32)
            } else {
                let shape: SheafResult<Vec<i64>> = dims
                    .iter()
                    .map(|d| {
                        d.as_i64().ok_or_else(|| SheafError::Compile {
                            message: format!("config: shape dimension must be integer, got {}", d),
                            location: SourceLocation::unknown(),
                        })
                    })
                    .collect();
                Ok(StableHLOType::f32_tensor(shape?))
            }
        }
        JsonValue::Number(_) => Ok(StableHLOType::ScalarF32),
        JsonValue::Bool(_) => Ok(StableHLOType::ScalarI1),
        other => Err(SheafError::Compile {
            message: format!("config: unsupported JSON value {}", other),
            location: SourceLocation::unknown(),
        }),
    }
}

/// Build a flat index map from a JSON config dict.
///
/// Returns `BTreeMap<Vec<String>, Vec<usize>>`:
///   key path (e.g. `["l1", "W"]`) → tuple indices (e.g. `[0, 0]`)
///
/// Example for `{"l1": {"W": [2,8], "b": [8]}, "l2": {"W": [8,1], "b": [1]}}`:
///   `["l1"]`     → `[0]`
///   `["l1","W"]` → `[0, 0]`
///   `["l1","b"]` → `[0, 1]`
///   `["l2"]`     → `[1]`
///   `["l2","W"]` → `[1, 0]`
///   `["l2","b"]` → `[1, 1]`
pub fn build_index_map(val: &JsonValue) -> BTreeMap<Vec<String>, Vec<usize>> {
    let mut map = BTreeMap::new();
    build_index_map_rec(val, &[], &[], &mut map);
    map
}

fn build_index_map_rec(
    val: &JsonValue,
    path: &[String],
    indices: &[usize],
    map: &mut BTreeMap<Vec<String>, Vec<usize>>,
) {
    if !path.is_empty() {
        map.insert(path.to_vec(), indices.to_vec());
    }
    if let JsonValue::Object(obj) = val {
        let sorted: BTreeMap<&str, &JsonValue> = obj.iter().map(|(k, v)| (k.as_str(), v)).collect();
        for (i, (key, child)) in sorted.iter().enumerate() {
            let mut child_path = path.to_vec();
            child_path.push(key.to_string());
            let mut child_indices = indices.to_vec();
            child_indices.push(i);
            build_index_map_rec(child, &child_path, &child_indices, map);
        }
    }
}

/// Build an index map from a `ParamLayout` (same format as `build_index_map`).
///
/// Converts each field's `path` and `tuple_index` into the expected mapping:
///   `["l1", "W"]` → `[0, 0]`, etc.
///
/// Also inserts prefix paths for intermediate levels:
///   `["l1"]` → `[0]` (inferred from children with path starting with "l1")
pub fn layout_to_index_map(layout: &crate::core::compiler::ParamLayout) -> BTreeMap<Vec<String>, Vec<usize>> {
    let mut map = BTreeMap::new();
    for field in &layout.fields {
        // Insert the full path
        map.insert(field.path.clone(), field.tuple_index.clone());
        // Insert all prefix paths (for intermediate tuple access)
        for depth in 1..field.path.len() {
            let prefix: Vec<String> = field.path[..depth].to_vec();
            let idx_prefix: Vec<usize> = field.tuple_index[..depth].to_vec();
            map.entry(prefix).or_insert(idx_prefix);
        }
    }
    map
}

/// Dict-to-tuple lowering pass.
///
/// Rewrites `FunctionCall("get", [expr, Keyword(k)])` chains rooted at `Symbol(param)`
/// into `GetTupleElement { param, indices }` using the pre-built index map.
///
/// Handles arbitrary nesting depth:
///   `(get (get p :l1) :W)` → `GetTupleElement { param: "p", indices: [0, 0] }`
///
/// `param_name`: the name of the parameter that holds the dict (e.g. `"p"`)
/// `index_map`: output of `build_index_map()`
pub fn lower_get_calls(
    expr: &CompiledExpr,
    param_name: &str,
    index_map: &BTreeMap<Vec<String>, Vec<usize>>,
) -> CompiledExpr {
    // Try to match a (get ...) chain rooted at `param_name`.
    // Returns Some((key_path, leaf)) if fully matched, None otherwise.
    if let Some(indices) = try_extract_get_chain(expr, param_name, index_map) {
        return CompiledExpr::GetTupleElement {
            param: param_name.to_string(),
            indices,
        };
    }

    // Recurse into sub-expressions
    match expr {
        CompiledExpr::FunctionCall { name, args } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| lower_get_calls(a, param_name, index_map))
                .collect(),
        },
        CompiledExpr::Let { bindings, body } => CompiledExpr::Let {
            bindings: bindings
                .iter()
                .map(|(k, v)| (k.clone(), lower_get_calls(v, param_name, index_map)))
                .collect(),
            body: Box::new(lower_get_calls(body, param_name, index_map)),
        },
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs
                .iter()
                .map(|e| lower_get_calls(e, param_name, index_map))
                .collect(),
        ),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(lower_get_calls(condition, param_name, index_map)),
            then_branch: Box::new(lower_get_calls(then_branch, param_name, index_map)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(lower_get_calls(e, param_name, index_map))),
        },
        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(lower_get_calls(body, param_name, index_map)),
        },
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(lower_get_calls(callee, param_name, index_map)),
            args: args
                .iter()
                .map(|a| lower_get_calls(a, param_name, index_map))
                .collect(),
        },
        CompiledExpr::Repeat {
            index_var,
            count,
            acc_var,
            acc_init,
            body,
        } => CompiledExpr::Repeat {
            index_var: index_var.clone(),
            count: Box::new(lower_get_calls(count, param_name, index_map)),
            acc_var: acc_var.clone(),
            acc_init: Box::new(lower_get_calls(acc_init, param_name, index_map)),
            body: Box::new(lower_get_calls(body, param_name, index_map)),
        },
        // Leaf nodes: unchanged
        other => other.clone(),
    }
}

/// Try to extract the tuple indices for a `(get ... :key)` chain rooted at `param_name`.
///
/// Returns `Some(indices)` if the chain resolves to a known path in `index_map`,
/// `None` otherwise.
fn try_extract_get_chain(
    expr: &CompiledExpr,
    param_name: &str,
    index_map: &BTreeMap<Vec<String>, Vec<usize>>,
) -> Option<Vec<usize>> {
    let path = extract_key_path(expr, param_name)?;
    index_map.get(&path).cloned()
}

/// Walk a `(get (get ... :k1) :k2)` chain and return the key path `["k1", "k2", ...]`.
/// Returns `None` if the root is not `Symbol(param_name)`.
fn extract_key_path(expr: &CompiledExpr, param_name: &str) -> Option<Vec<String>> {
    match expr {
        CompiledExpr::Symbol(name) if name == param_name => Some(vec![]),
        CompiledExpr::FunctionCall { name, args } if name == "get" && args.len() == 2 => {
            // args[1] must be a Keyword
            let key = match &args[1] {
                CompiledExpr::Keyword(k) => k.clone(),
                _ => return None,
            };
            // args[0] is the receiver — recurse
            let mut path = extract_key_path(&args[0], param_name)?;
            path.push(key);
            Some(path)
        }
        _ => None,
    }
}
