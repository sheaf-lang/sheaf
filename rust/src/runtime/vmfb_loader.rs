// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Shared VMFB loading logic: manifest hash validation, IREE session creation,
//! function tagging, and signature tracing.
//! Used by both `eval_source_with_path` (sheaf run) and the `(use)` form.

use std::path::Path;
use std::sync::Arc;

use crate::compiler::effects::has_side_effects;
use crate::core::compiler::CompilerContext;
use crate::runtime::iree_session::IreeSession;

/// Try to load a companion VMFB for a Sheaf source file.
///
/// Looks for `{shf_path}.vmfb` and validates freshness using a manifest
/// (`{shf_path}.vmfb.manifest.json`) if present, falling back to timestamp
/// comparison otherwise.
///
/// `candidate_fns`: function names to consider for IREE dispatch.
/// Only pure (side-effect-free) functions among these will be tagged.
///
/// Returns `true` if functions were successfully tagged for IREE dispatch.
pub fn try_load_vmfb(
    compiler: &mut CompilerContext,
    shf_path: &Path,
    candidate_fns: &[String],
) -> bool {
    let vmfb_path = shf_path.with_extension("vmfb");
    if !vmfb_path.exists() {
        return false;
    }

    // Filter to pure functions only
    let pure_fns: Vec<String> = candidate_fns
        .iter()
        .filter(|name| {
            compiler
                .registry
                .get(*name)
                .and_then(|fd| fd.body_compiled.as_ref())
                .map(|body| !has_side_effects(body))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    if pure_fns.is_empty() {
        return false;
    }

    // Try manifest-based validation first, then timestamp fallback
    let manifest_path = vmfb_path.with_extension("vmfb.manifest.json");
    let valid_fns = match std::fs::read_to_string(&manifest_path) {
        Ok(manifest_str) => match validate_manifest(&manifest_str, &vmfb_path, compiler, &pure_fns) {
            Some(fns) => fns,
            None => return false, // stale or invalid
        },
        Err(_) => {
            // No manifest — fall back to timestamp check
            let is_fresh = match (
                std::fs::metadata(shf_path).and_then(|m| m.modified()),
                std::fs::metadata(&vmfb_path).and_then(|m| m.modified()),
            ) {
                (Ok(shf_time), Ok(vmfb_time)) => vmfb_time >= shf_time,
                _ => false,
            };
            if !is_fresh {
                return false;
            }
            // Timestamp says fresh — use all pure functions
            pure_fns
        }
    };

    if valid_fns.is_empty() {
        return false;
    }

    // Load the VMFB into an IREE session
    let vmfb_data = match std::fs::read(&vmfb_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("warning: cannot read '{}': {}", vmfb_path.display(), e);
            return false;
        }
    };
    let mut session = match IreeSession::new() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("warning: IREE init failed: {}", e);
            return false;
        }
    };
    if let Err(e) = session.load_vmfb(vmfb_data) {
        eprintln!("warning: failed to load '{}': {}", vmfb_path.display(), e);
        return false;
    }

    let session_idx = compiler.vmfb_sessions.len();
    compiler.vmfb_sessions.push(Arc::new(session));

    // Tag functions for IREE dispatch
    for fn_name in &valid_fns {
        if let Some(fd) = compiler.registry.get_mut(fn_name) {
            fd.vmfb_session_idx = Some(session_idx);
        }
    }

    // Trace signatures for accurate return-type reconstruction
    for fn_name in &valid_fns {
        if let Some(fd) = compiler.registry.get(fn_name).cloned() {
            if let Ok(traced_sig) = crate::core::trace::trace_function_signature(compiler, &fd) {
                if let Some(fd_mut) = compiler.registry.get_mut(fn_name) {
                    fd_mut.signature = Some(traced_sig);
                }
            }
        }
    }

    eprintln!(
        "info: dispatching {} function(s) via IREE from '{}'",
        valid_fns.len(),
        vmfb_path.display()
    );
    true
}

/// Validate a manifest JSON against the current compiler registry.
/// Returns the list of valid function names, or None if stale/invalid.
fn validate_manifest(
    manifest_str: &str,
    vmfb_path: &Path,
    compiler: &CompilerContext,
    candidate_fns: &[String],
) -> Option<Vec<String>> {
    let manifest: serde_json::Value = match serde_json::from_str(manifest_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "warning: invalid manifest for '{}': {}",
                vmfb_path.display(),
                e
            );
            return None;
        }
    };

    let functions = manifest.get("functions").and_then(|f| f.as_object())?;

    // Validate hashes for every function in the manifest
    for (name, entry) in functions {
        let expected_hash = match entry.get("hash").and_then(|h| h.as_str()) {
            Some(h) => h,
            None => continue,
        };
        match compiler.registry.get(name) {
            Some(fd) => {
                if fd.body_hash() != expected_hash {
                    eprintln!(
                        "warning: '{}' is stale ('{}' changed), run 'sheaf build' to recompile",
                        vmfb_path.display(),
                        name
                    );
                    return None;
                }
            }
            None => {
                eprintln!(
                    "warning: '{}' is stale ('{}' not found), run 'sheaf build' to recompile",
                    vmfb_path.display(),
                    name
                );
                return None;
            }
        }
    }

    // Return only candidate functions that are both pure AND in the manifest
    let manifest_names: std::collections::HashSet<&str> =
        functions.keys().map(|s| s.as_str()).collect();
    let valid: Vec<String> = candidate_fns
        .iter()
        .filter(|name| manifest_names.contains(name.as_str()))
        .cloned()
        .collect();

    Some(valid)
}
