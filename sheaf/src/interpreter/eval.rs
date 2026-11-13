// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Stateless source evaluation and the stateful REPL interpreter.

use crate::core::expr::{CompiledExpr, CompilerContext};
use crate::core::error::SheafError;
use crate::interpreter::builtins::register_builtins;
use crate::interpreter::env::Env;
use crate::interpreter::tracer::{Tracer, TracerConfig};
use crate::interpreter::value::Value;
use crate::interpreter;

/// Evaluates an isolated source string and returns its last value.
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
    if let Some(path) = file_path
        && let Some(dir) = path.parent()
    {
        compiler.set_current_dir(
            dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()),
        );
    }
    let mut compiled = Vec::new();
    for expr in &exprs {
        compiled.push(compiler.compile(expr)?);
    }
    let mut env = Env::with_registry(compiler.registry.clone());
    register_builtins(&mut env);
    let mut last = Value::Nil;
    for c in &compiled {
        if !matches!(c, CompiledExpr::Nil) {
            last = interpreter::eval(c, &mut env)?;
        }
    }
    Ok(last)
}

/// Evaluates source with tracing and CLI guards.
pub fn eval_source_with_tracing(
    source: &str,
    file_path: Option<&std::path::Path>,
    config: TracerConfig,
) -> Result<Value, SheafError> {
    let filename = file_path
        .and_then(|p| p.to_str())
        .unwrap_or("<eval>");
    let exprs = crate::core::parse(source, filename)?;
    let mut compiler = CompilerContext::new();
    if let Some(path) = file_path
        && let Some(dir) = path.parent()
    {
        compiler.set_current_dir(
            dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()),
        );
    }
    let mut compiled = Vec::new();
    for expr in &exprs {
        compiled.push(compiler.compile(expr)?);
    }

    let mut env = Env::with_registry(compiler.registry.clone());
    env.tracer = Some(Tracer::from_config(config));
    register_builtins(&mut env);
    let mut last = Value::Nil;
    for c in &compiled {
        if !matches!(c, CompiledExpr::Nil) {
            last = interpreter::eval(c, &mut env)?;
        }
    }
    Ok(last)
}

/// Evaluates source with call profiling and optional tracing.
pub fn eval_source_with_blame(
    source: &str,
    file_path: Option<&std::path::Path>,
    tracer_config: Option<TracerConfig>,
) -> Result<Value, SheafError> {
    let (val, _) = eval_source_with_blame_internal(source, file_path, tracer_config, true, false)?;
    Ok(val)
}

/// Evaluates source with call and memory profiling.
pub fn eval_source_with_blame_mem(
    source: &str,
    file_path: Option<&std::path::Path>,
    tracer_config: Option<TracerConfig>,
) -> Result<(Value, String), SheafError> {
    let (val, report) = eval_source_with_blame_internal(source, file_path, tracer_config, true, true)?;
    let report_str = report.unwrap_or_default();
    Ok((val, report_str))
}

/// Evaluates source with memory profiling but without call profiling.
pub fn eval_source_with_mem(
    source: &str,
    file_path: Option<&std::path::Path>,
) -> Result<(Value, String), SheafError> {
    let (val, report) = eval_source_with_blame_internal(source, file_path, None, false, true)?;
    let report_str = report.unwrap_or_default();
    Ok((val, report_str))
}

fn eval_source_with_blame_internal(
    source: &str,
    file_path: Option<&std::path::Path>,
    tracer_config: Option<TracerConfig>,
    blame_report: bool,
    mem_profile: bool,
) -> Result<(Value, Option<String>), SheafError> {
    let filename = file_path
        .and_then(|p| p.to_str())
        .unwrap_or("<eval>");
    let exprs = crate::core::parse(source, filename)?;
    let mut compiler = CompilerContext::new();
    if let Some(path) = file_path
        && let Some(dir) = path.parent()
    {
        compiler.set_current_dir(
            dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()),
        );
    }
    let mut compiled = Vec::new();
    for expr in &exprs {
        compiled.push(compiler.compile(expr)?);
    }

    let mut env = Env::with_registry(compiler.registry.clone());
    register_builtins(&mut env);
    if blame_report {
        env.profiler = Some(crate::interpreter::profiler::Profiler::new());
    }
    if let Some(config) = tracer_config {
        env.tracer = Some(Tracer::from_config(config));
    }
    if mem_profile {
        env.mem_profiler = Some(crate::interpreter::mem_profile::MemProfiler::new());
        if let Some(ref mut mp) = env.mem_profiler {
            mp.sample("after init");
        }
    }
    let mut last = Value::Nil;
    for c in &compiled {
        if !matches!(c, CompiledExpr::Nil) {
            last = interpreter::eval(c, &mut env)?;
        }
    }
    if mem_profile
        && let Some(ref mut mp) = env.mem_profiler
    {
        mp.sample("after eval");
    }
    #[cfg(iree_runtime)]
    if mem_profile
        && let Some(ref mut mp) = env.mem_profiler
        && let Some(session) = crate::runtime::iree_session::initialized_shared_session()
    {
        mp.sample_iree(&session);
    }
    let tracing_mode = env.tracer.as_ref().is_some_and(|tracer| tracer.enabled);
    if blame_report
        && let Some(ref profiler) = env.profiler
    {
        profiler.report(tracing_mode);
    }

    // Release interpreter-owned values before collecting the final memory report.
    let mp = env.mem_profiler.take();
    drop(env);

    let mem_report = if mem_profile {
        mp.as_ref().map(|m| m.report())
    } else {
        None
    };
    Ok((last, mem_report))
}

/// Interpreter that preserves definitions and bindings between REPL inputs.
pub struct Interpreter {
    compiler: CompilerContext,
    env: Env,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    pub fn new() -> Self {
        let compiler = CompilerContext::new();
        let mut env = Env::with_registry(compiler.registry.clone());
        register_builtins(&mut env);
        Self { compiler, env }
    }

    /// Evaluates one REPL input and returns its last value.
    pub fn eval(&mut self, source: &str) -> Result<Value, SheafError> {
        let exprs = crate::core::parse(source, "<repl>")?;
        let mut last = Value::Nil;
        for expr in &exprs {
            let compiled = self.compiler.compile(expr)?;
            // Compilation may add function definitions needed during evaluation.
            self.env.registry = self.compiler.registry.clone();
            if !matches!(compiled, CompiledExpr::Nil) {
                last = interpreter::eval(&compiled, &mut self.env)?;
            }
        }
        Ok(last)
    }

    pub fn registry_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.compiler.registry.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    /// Returns a stdlib or user-defined function for REPL source display.
    pub fn registry_get(&self, name: &str) -> Option<&crate::core::expr::FunctionDef> {
        self.compiler.registry.get(name)
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn env_mut(&mut self) -> &mut Env {
        &mut self.env
    }

    pub fn load_path(&self) -> &[std::path::PathBuf] {
        &self.compiler.load_path
    }
}
