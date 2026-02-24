// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Utility special forms: get, dict, last, use, quote

use crate::ast::SheafValue;
use crate::compiler::effects::has_side_effects;
use crate::core::compiler::{CompiledExpr, CompilerContext};
use crate::core::error::{SheafError, SheafResult, SourceLocation};
use crate::forms::base::{SpecialForm, check_arity};

/// quote - Prevent evaluation: (quote expr) or 'expr
pub struct QuoteForm;

impl SpecialForm for QuoteForm {
    fn name(&self) -> &'static str {
        "quote"
    }

    fn compile(
        &self,
        _compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        check_arity("quote", args, 1, loc)?;
        Ok(CompiledExpr::Quoted(Box::new(args[0].clone())))
    }
}

/// get - Map/vector access: (get coll key) or (get coll key default)
pub struct GetForm;

impl SpecialForm for GetForm {
    fn name(&self) -> &'static str {
        "get"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        _loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        let compiled: SheafResult<Vec<CompiledExpr>> =
            args.iter().map(|a| compiler.compile(a)).collect();
        Ok(CompiledExpr::FunctionCall { name: "get".to_string(), args: compiled? })
    }
}

/// get-in - Nested access: (get-in coll [path] [default])
pub struct GetInForm;

impl SpecialForm for GetInForm {
    fn name(&self) -> &'static str {
        "get-in"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        _loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        let compiled: SheafResult<Vec<CompiledExpr>> =
            args.iter().map(|a| compiler.compile(a)).collect();
        Ok(CompiledExpr::FunctionCall { name: "get-in".to_string(), args: compiled? })
    }
}

/// dict - Dictionary construction: (dict :a 1 :b 2)
pub struct DictForm;

impl SpecialForm for DictForm {
    fn name(&self) -> &'static str {
        "dict"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        _loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        let compiled: SheafResult<Vec<CompiledExpr>> =
            args.iter().map(|a| compiler.compile(a)).collect();
        Ok(CompiledExpr::FunctionCall { name: "dict".to_string(), args: compiled? })
    }
}

/// assoc - Map association: (assoc map :key val)
pub struct AssocForm;

impl SpecialForm for AssocForm {
    fn name(&self) -> &'static str {
        "assoc"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        _loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        let compiled: SheafResult<Vec<CompiledExpr>> =
            args.iter().map(|a| compiler.compile(a)).collect();
        Ok(CompiledExpr::FunctionCall { name: "assoc".to_string(), args: compiled? })
    }
}

/// last - Get last element: (last coll)
pub struct LastForm;

impl SpecialForm for LastForm {
    fn name(&self) -> &'static str {
        "last"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        _loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        let compiled: SheafResult<Vec<CompiledExpr>> =
            args.iter().map(|a| compiler.compile(a)).collect();
        Ok(CompiledExpr::FunctionCall { name: "last".to_string(), args: compiled? })
    }
}

/// use - Module imports: (use module) or (use ./path/to/module.shf)
///
/// Loads and compiles a Sheaf module into the current compiler context.
/// Resolution order:
///   1. Paths with '/' treated as relative to current_dir (or cwd)
///   2. Bare names searched in load_path (stdlib dirs, then cwd)
/// Both `(use nn)` and `(use nn.shf)` are accepted.
/// Already-loaded modules (by absolute path) are silently skipped.
pub struct UseForm;

impl SpecialForm for UseForm {
    fn name(&self) -> &'static str {
        "use"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        check_arity("use", args, 1, loc)?;

        // Accept bare symbol `(use nn)` or string `(use "nn")`
        let raw = match &args[0] {
            SheafValue::Symbol(s, _) => s.clone(),
            SheafValue::String(s, _) => s.clone(),
            other => {
                return Err(SheafError::Compile {
                    message: format!("use: expected module name, got {}", other),
                    location: loc.clone(),
                });
            }
        };

        let resolved = resolve_module_path(compiler, &raw, loc)?;

        // Deduplicate: if already loaded, skip silently
        if compiler.loaded_modules.contains(&resolved) {
            return Ok(CompiledExpr::Nil);
        }
        compiler.loaded_modules.insert(resolved.clone());

        // Read the source
        let source = std::fs::read_to_string(&resolved).map_err(|e| SheafError::Compile {
            message: format!("use: cannot read '{}': {}", resolved.display(), e),
            location: loc.clone(),
        })?;

        // Save and update current_dir for nested (use ...) in the module
        let prev_dir = compiler.current_dir.clone();
        if let Some(parent) = resolved.parent() {
            compiler.current_dir = Some(parent.to_path_buf());
        }

        // Parse and compile all expressions into current context
        let exprs = crate::core::parse(&source, resolved.to_str().unwrap_or("<use>"))
            .map_err(|e| SheafError::Compile {
                message: format!("use: parse error in '{}': {}", resolved.display(), e),
                location: loc.clone(),
            })?;

        // Track which functions exist before compiling the module
        let pre_fns: std::collections::HashSet<String> =
            compiler.registry.keys().cloned().collect();

        for expr in &exprs {
            compiler.compile(expr)?;
        }

        #[cfg(iree_runtime)]
        try_load_companion_vmfb(compiler, &resolved, &pre_fns, loc)?;

        // Restore previous current_dir
        compiler.current_dir = prev_dir;

        Ok(CompiledExpr::Nil)
    }
}

/// Load a companion VMFB if present and fresh, tag pure functions for IREE dispatch,
/// and trace them to discover accurate return types.
#[cfg(iree_runtime)]
fn try_load_companion_vmfb(
    compiler: &mut CompilerContext,
    shf_path: &std::path::Path,
    pre_fns: &std::collections::HashSet<String>,
    loc: &SourceLocation,
) -> SheafResult<()> {
    use std::sync::Arc;
    use crate::runtime::iree_session::IreeSession;

    let vmfb_path = shf_path.with_extension("vmfb");
    if !vmfb_path.exists() {
        return Ok(());
    }

    let is_fresh = match (
        std::fs::metadata(shf_path).and_then(|m| m.modified()),
        std::fs::metadata(&vmfb_path).and_then(|m| m.modified()),
    ) {
        (Ok(shf_time), Ok(vmfb_time)) => vmfb_time >= shf_time,
        _ => false,
    };
    if !is_fresh {
        return Ok(());
    }

    let new_fns: Vec<String> = compiler
        .registry
        .keys()
        .filter(|k| !pre_fns.contains(*k))
        .cloned()
        .collect();

    let pure_fns: Vec<String> = new_fns
        .into_iter()
        .filter(|name| {
            compiler
                .registry
                .get(name)
                .and_then(|fd| fd.body_compiled.as_ref())
                .map(|body| !has_side_effects(body))
                .unwrap_or(false)
        })
        .collect();

    if pure_fns.is_empty() {
        return Ok(());
    }

    // Load the VMFB into an IREE session
    let vmfb_data = std::fs::read(&vmfb_path).map_err(|e| SheafError::Compile {
        message: format!("use: cannot read VMFB '{}': {}", vmfb_path.display(), e),
        location: loc.clone(),
    })?;
    let mut session = IreeSession::new().map_err(|e| SheafError::Compile {
        message: format!("use: IREE init failed: {}", e),
        location: loc.clone(),
    })?;
    session.load_vmfb(vmfb_data).map_err(|e| SheafError::Compile {
        message: format!("use: failed to load VMFB '{}': {}", vmfb_path.display(), e),
        location: loc.clone(),
    })?;

    let session_idx = compiler.vmfb_sessions.len();
    compiler.vmfb_sessions.push(Arc::new(session));

    for fn_name in &pure_fns {
        if let Some(fd) = compiler.registry.get_mut(fn_name) {
            fd.vmfb_session_idx = Some(session_idx);
        }
    }

    // Trace pure functions to discover accurate return types.
    // Static inference can't see through value-and-grad, tree-map, etc.
    for fn_name in &pure_fns {
        if let Some(fd) = compiler.registry.get(fn_name).cloned() {
            if let Ok(traced_sig) = crate::core::trace::trace_function_signature(compiler, &fd) {
                if let Some(fd_mut) = compiler.registry.get_mut(fn_name) {
                    fd_mut.signature = Some(traced_sig);
                }
            }
        }
    }

    Ok(())
}

/// Resolve a module name to an absolute path.
fn resolve_module_path(
    compiler: &CompilerContext,
    raw: &str,
    loc: &SourceLocation,
) -> SheafResult<std::path::PathBuf> {
    use std::path::Path;

    // Strip .shf extension for searching, or keep it if explicitly given
    let has_slash = raw.contains('/');

    // Candidate file names to try (with and without .shf)
    let candidates: Vec<String> = if raw.ends_with(".shf") {
        vec![raw.to_string()]
    } else {
        vec![raw.to_string(), format!("{}.shf", raw)]
    };

    if has_slash {
        // Relative or absolute path: resolve relative to current_dir, then cwd
        let base = compiler
            .current_dir
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_default();

        for name in &candidates {
            let p = base.join(name);
            if p.exists() {
                return Ok(p.canonicalize().unwrap_or(p));
            }
        }
    } else {
        // Bare name: search load_path
        let mut search_roots: Vec<std::path::PathBuf> = compiler.load_path.clone();
        // Also search current_dir for local modules (e.g. (use mlp) in same dir as run.shf)
        if let Some(dir) = &compiler.current_dir {
            if !search_roots.contains(dir) {
                search_roots.push(dir.clone());
            }
        }

        for root in &search_roots {
            for name in &candidates {
                let p = root.join(name);
                if p.exists() {
                    return Ok(p.canonicalize().unwrap_or(p));
                }
            }
        }
    }

    Err(SheafError::Compile {
        message: format!(
            "use: module '{}' not found\n  searched: {:?}",
            raw,
            compiler.load_path
        ),
        location: loc.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_form_names() {
        assert_eq!(QuoteForm.name(), "quote");
        assert_eq!(GetForm.name(), "get");
        assert_eq!(GetInForm.name(), "get-in");
        assert_eq!(DictForm.name(), "dict");
        assert_eq!(AssocForm.name(), "assoc");
        assert_eq!(LastForm.name(), "last");
        assert_eq!(UseForm.name(), "use");
    }
}
