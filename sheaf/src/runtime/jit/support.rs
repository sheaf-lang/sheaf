// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Shared JIT support routines.

use super::*;

#[cfg(iree_runtime)]
pub(super) fn inject_tuple_shapes(
    param_name: &str,
    ty: &StableHLOType,
    indices: &[usize],
    shapes: &mut HashMap<String, Vec<i64>>,
) {
    match ty {
        StableHLOType::Tuple(elems, keys) => {
            for (i, elem_ty) in elems.iter().enumerate() {
                let mut child_indices = indices.to_vec();
                child_indices.push(i);
                if let Some(key_names) = keys
                    && let Some(key) = key_names.get(i) {
                        shapes.insert(key.clone(), elem_ty.shape().to_vec());
                        inject_tuple_shapes(key, elem_ty, &child_indices, shapes);
                        continue;
                    }
                inject_tuple_shapes(param_name, elem_ty, &child_indices, shapes);
            }
        }
        _ => {
            let shape = ty.shape();
            if !shape.is_empty() && !indices.is_empty() {
                let key = format!("{}@{:?}", param_name, indices);
                shapes.insert(key, shape.to_vec());
            }
        }
    }
}

#[cfg(iree_runtime)]
pub(super) fn log_remaining_reduces(expr: &CompiledExpr, label: &str) {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } if name == "reduce" => {
            let coll_desc = if let Some(coll) = args.get(2) {
                format!("{:?}", coll)
            } else {
                "missing".to_string()
            };
            sheaf_msg!("jit: [{}] reduce with coll={}", label, coll_desc);
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                log_remaining_reduces(a, label);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                log_remaining_reduces(v, label);
            }
            log_remaining_reduces(body, label);
        }
        CompiledExpr::Lambda { body, .. } => log_remaining_reduces(body, label),
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                log_remaining_reduces(e, label);
            }
        }
        _ => {}
    }
}

#[cfg(iree_runtime)]
pub(super) fn log_unresolved_shapes(expr: &CompiledExpr, label: &str) {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } if name == "reshape" && args.len() == 2 => {
            match &args[1] {
                CompiledExpr::Vector(elems) => {
                    let unresolved: Vec<_> = elems
                        .iter()
                        .filter(|e| !matches!(e, CompiledExpr::Integer(_)))
                        .map(|e| format!("{:?}", e))
                        .collect();
                    if !unresolved.is_empty() {
                        sheaf_msg!(
                            "jit: [{}] reshape with unresolved shape elements: {:?}",
                            label,
                            unresolved
                        );
                    }
                }
                other => {
                    sheaf_msg!(
                        "jit: [{}] reshape with non-vector shape: {:?}",
                        label,
                        other
                    );
                }
            }
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                log_unresolved_shapes(a, label);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                log_unresolved_shapes(v, label);
            }
            log_unresolved_shapes(body, label);
        }
        CompiledExpr::Lambda { body, .. } => log_unresolved_shapes(body, label),
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                log_unresolved_shapes(e, label);
            }
        }
        _ => {}
    }
}

/// Extract the outermost function call name from an expression (for cache naming).
pub(super) fn outermost_call_name(expr: &CompiledExpr) -> Option<String> {
    match expr {
        CompiledExpr::FunctionCall { name, .. } => Some(name.clone()),
        CompiledExpr::Let { body, .. } => outermost_call_name(body),
        CompiledExpr::Do(exprs) => exprs.last().and_then(outermost_call_name),
        _ => None,
    }
}

/// Read the manifest and check if the stored hash matches.
pub(super) fn manifest_hash_matches(
    cache_dir: &std::path::Path,
    name: &str,
    expected_hash: &str,
) -> bool {
    let manifest_path = cache_dir.join("manifest.json");
    let data = match std::fs::read_to_string(&manifest_path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let manifest: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return false,
    };
    manifest.get(name).and_then(|v| v.as_str()) == Some(expected_hash)
}

/// Update the manifest with a new hash entry.
pub(super) fn update_manifest(cache_dir: &std::path::Path, name: &str, hash: &str) -> std::io::Result<()> {
    let manifest_path = cache_dir.join("manifest.json");
    let mut manifest: serde_json::Map<String, serde_json::Value> =
        std::fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|d| serde_json::from_str(&d).ok())
            .unwrap_or_default();
    manifest.insert(
        name.to_string(),
        serde_json::Value::String(hash.to_string()),
    );
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)
}
