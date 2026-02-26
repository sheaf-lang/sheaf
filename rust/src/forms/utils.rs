// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Utility special forms: get, dict, last, use, quote

use crate::ast::SheafValue;
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

        // Try to load companion VMFB for the imported module
        #[cfg(iree_runtime)]
        {
            let new_fns: Vec<String> = compiler
                .registry
                .keys()
                .filter(|k| !pre_fns.contains(*k))
                .cloned()
                .collect();
            crate::runtime::vmfb_loader::try_load_vmfb(compiler, &resolved, &new_fns);
        }

        // Restore previous current_dir
        compiler.current_dir = prev_dir;

        Ok(CompiledExpr::Nil)
    }
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
