// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Top-level evaluation entry points for the Sheaf interpreter.
//!
//! Provides stateless (`eval_source`) and stateful (`Interpreter`) interfaces,
//! the latter being used by the REPL to persist bindings across inputs.

use crate::compiler::effects::has_side_effects;
use crate::core::compiler::{CompiledExpr, CompilerContext};
use crate::core::error::SheafError;
use crate::interpreter::builtins::register_builtins;
use crate::interpreter::env::Env;
use crate::interpreter::value::Value;
use crate::interpreter;

/// Evaluate a complete Sheaf source string and return the last value.
/// Each call is fully independent (no shared state).
/// `file_path` is used to resolve relative `(use ...)` paths; pass `None` for inline expressions.
pub fn eval_source(source: &str) -> Result<Value, SheafError> {
    eval_source_with_path(source, None)
}

pub fn eval_source_with_path(
    source: &str,
    file_path: Option<&std::path::Path>,
) -> Result<Value, SheafError> {
    let filename = file_path
        .and_then(|p| p.to_str())
        .unwrap_or("<eval>");
    let exprs = crate::core::parse(source, filename)?;
    let mut compiler = CompilerContext::new();
    // Set current_dir so (use module) resolves relative to the file being evaluated
    if let Some(path) = file_path {
        if let Some(dir) = path.parent() {
            compiler.set_current_dir(
                dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()),
            );
        }
    }
    let mut compiled = Vec::new();
    for expr in &exprs {
        compiled.push(compiler.compile(expr)?);
    }
    // Hint: if all defn bodies defined directly in this file are side-effect-free,
    // suggest compilation. Imported functions (from `use`) are excluded.
    if file_path.is_some() {
        let local_defns: Vec<&str> = exprs
            .iter()
            .filter_map(|e| {
                e.as_list()
                    .and_then(|l| l.first())
                    .and_then(|h| h.as_symbol())
                    .filter(|&s| s == "defn")
                    .and_then(|_| {
                        e.as_list()
                            .and_then(|l| l.get(1))
                            .and_then(|n| n.as_symbol())
                    })
            })
            .collect();
        let has_any_defn = !local_defns.is_empty();
        let all_pure = has_any_defn
            && local_defns.iter().all(|name| {
                compiler
                    .registry
                    .get(*name)
                    .and_then(|f| f.body_compiled.as_ref())
                    .map(|b| !has_side_effects(b))
                    .unwrap_or(true)
            });
        if all_pure {
            eprintln!(
                "hint: '{}' has no side effects — consider compiling with `sheaf build`",
                filename
            );
        }
    }

    let mut env = Env::with_registry(compiler.registry.clone());
    env.vmfb_sessions = compiler.vmfb_sessions.clone();
    register_builtins(&mut env);
    let mut last = Value::Nil;
    for c in &compiled {
        if !matches!(c, CompiledExpr::Nil) {
            last = interpreter::eval(c, &mut env)?;
        }
    }
    Ok(last)
}

/// Stateful interpreter: accumulates definitions and bindings across calls.
/// Used by the REPL so that `(defn f ...)` in one line is visible in the next.
pub struct Interpreter {
    compiler: CompilerContext,
    env: Env,
}

impl Interpreter {
    pub fn new() -> Self {
        let compiler = CompilerContext::new();
        let mut env = Env::with_registry(compiler.registry.clone());
        register_builtins(&mut env);
        Self { compiler, env }
    }

    /// Evaluate one input (expression or definition). Returns the resulting value.
    pub fn eval(&mut self, source: &str) -> Result<Value, SheafError> {
        let exprs = crate::core::parse(source, "<repl>")?;
        let mut last = Value::Nil;
        for expr in &exprs {
            let compiled = self.compiler.compile(expr)?;
            // Sync any newly registered functions and VMFB sessions into env
            self.env.registry = self.compiler.registry.clone();
            self.env.vmfb_sessions = self.compiler.vmfb_sessions.clone();
            if !matches!(compiled, CompiledExpr::Nil) {
                last = interpreter::eval(&compiled, &mut self.env)?;
            }
        }
        Ok(last)
    }
}
