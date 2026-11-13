// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Utility special forms: get, dict, last, use, quote

use crate::core::ast::SheafValue;
use crate::core::expr::{CompiledExpr, CompilerContext};
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

/// get - Map/vector access: (get coll key) or (get coll :k1 :k2)
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
        Ok(CompiledExpr::FunctionCall { name: "get".to_string(), args: compiled?, loc: None })
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
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // Desugar (get-in m [:k1 :k2 :k3]) -> (get (get (get m :k1) :k2) :k3)
        if args.len() == 2
            && let SheafValue::Vector(keys, _) = &args[1]
        {
            let mut result = compiler.compile(&args[0])?;
            for key in keys {
                let compiled_key = compiler.compile(key)?;
                result = CompiledExpr::FunctionCall {
                    name: "get".to_string(),
                    args: vec![result, compiled_key],
                    loc: Some(loc.clone()),
                };
            }
            return Ok(result);
        }
        // Fallback: compile as function call
        let compiled: SheafResult<Vec<CompiledExpr>> =
            args.iter().map(|a| compiler.compile(a)).collect();
        Ok(CompiledExpr::FunctionCall { name: "get-in".to_string(), args: compiled?, loc: Some(loc.clone()) })
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
        Ok(CompiledExpr::FunctionCall { name: "dict".to_string(), args: compiled?, loc: None })
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
        Ok(CompiledExpr::FunctionCall { name: "assoc".to_string(), args: compiled?, loc: None })
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
        Ok(CompiledExpr::FunctionCall { name: "last".to_string(), args: compiled?, loc: None })
    }
}

/// use - Module imports: (use module) or (use ./path/to/module.shf)
///
/// Loads and compiles a Sheaf module into the current compiler context.
/// Resolution order:
///   1. Paths with '/' treated as relative to current_dir (or cwd)
///   2. Bare names searched in load_path (stdlib dirs, then cwd)
///
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

        // Warn the user if a local file with the same name than a prelude module exists.
        let bare = raw.strip_suffix(".shf").unwrap_or(&raw);
        if compiler.prelude_modules.contains(bare) {
            if resolve_module_path(compiler, &raw, loc).is_ok() {
                crate::sheaf_msg!(
                    "warning: '{}' is a built-in module (auto-loaded). Local file '{}.shf' is ignored.\n         Rename your file to avoid this conflict.",
                    bare, bare
                );
            }
            return Ok(CompiledExpr::Nil);
        }

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
        crate::core::error_format::register_source(
            resolved.to_str().unwrap_or("<use>"),
            &source,
        );

        // Save and update current_dir for nested (use ...) in the module
        let prev_dir = compiler.current_dir.clone();
        if let Some(parent) = resolved.parent() {
            compiler.current_dir = Some(parent.to_path_buf());
        }

        // Parse and compile all expressions into current context
        let exprs = crate::core::parse(&source, resolved.to_str().unwrap_or("<use>"))
            .map_err(|e| match e {
                SheafError::Parse { .. } => e,
                other => SheafError::Compile {
                    message: format!("use: error in '{}': {}", resolved.display(), other),
                    location: loc.clone(),
                },
            })?;

        // Keep the scopes and control flow surrounding imported initializers.
        fn has_initializer(expr: &CompiledExpr) -> bool {
            if matches!(expr, CompiledExpr::Def { .. }) { return true; }
            let mut found = false;
            expr.for_each_child(|child| found |= has_initializer(child));
            found
        }
        let mut defs = Vec::new();
        for expr in &exprs {
            let compiled = compiler.compile(expr)?;
            if has_initializer(&compiled) { defs.push(compiled); }
        }

        // Restore previous current_dir
        compiler.current_dir = prev_dir;

        if defs.is_empty() {
            Ok(CompiledExpr::Nil)
        } else {
            Ok(CompiledExpr::Do(defs))
        }
    }
}

/// stop-gradient - Block gradient flow: (stop-gradient x) returns x unchanged
/// in the forward pass, but its derivative is zero (like jax.lax.stop_gradient).
pub struct StopGradientForm;

impl SpecialForm for StopGradientForm {
    fn name(&self) -> &'static str {
        "stop-gradient"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        check_arity("stop-gradient", args, 1, loc)?;
        let compiled = compiler.compile(&args[0])?;
        Ok(CompiledExpr::FunctionCall {
            name: "stop-gradient".to_string(),
            args: vec![compiled],
            loc: None,
        })
    }
}

/// Resolve a module name to an absolute path.
fn resolve_module_path(
    compiler: &CompilerContext,
    raw: &str,
    loc: &SourceLocation,
) -> SheafResult<std::path::PathBuf> {
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
        // Bare name: search current_dir first, then load_path
        let mut search_roots: Vec<std::path::PathBuf> = Vec::new();
        // File's own directory takes priority (e.g. (use hydra) in scan/run.shf finds scan/hydra.shf)
        if let Some(dir) = &compiler.current_dir {
            search_roots.push(dir.clone());
        }
        for p in &compiler.load_path {
            if !search_roots.contains(p) {
                search_roots.push(p.clone());
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
        assert_eq!(StopGradientForm.name(), "stop-gradient");
    }
}
