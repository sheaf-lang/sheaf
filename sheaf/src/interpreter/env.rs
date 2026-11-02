// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Environment for the Sheaf interpreter: scoped variable bindings.

use crate::core::expr::FunctionDef;
use crate::core::error::SheafError;
use crate::interpreter::value::{BuiltinFnPtr, Value};
use std::collections::HashMap;

pub fn runtime_error(message: impl Into<String>) -> SheafError {
    SheafError::Runtime {
        message: message.into(),
        location: None,
    }
}

pub fn arity_error(name: &str, expected: usize, got: usize) -> SheafError {
    let hint = if got > expected {
        let arg_list: Vec<String> = (1..=expected).map(|i| format!("arg{}", i)).collect();
        format!(
            "\n  = hint: Did you mean ({} {})? A closing parenthesis is probably missing before the {} argument.",
            name, arg_list.join(" "), ordinal(expected + 1)
        )
    } else {
        "\n  = hint: A closing parenthesis may be too early, cutting off arguments.".to_string()
    };
    runtime_error(format!(
        "{} requires {} argument{}, got {}{}",
        name,
        expected,
        if expected == 1 { "" } else { "s" },
        got,
        hint,
    ))
}

fn ordinal(n: usize) -> String {
    match n {
        1 => "1st".to_string(),
        2 => "2nd".to_string(),
        3 => "3rd".to_string(),
        n => format!("{}th", n),
    }
}

/// A recorded function call: argument values observed during tracing.
#[derive(Clone, Debug)]
pub struct CallRecord {
    pub arg_values: Vec<Value>,
}

pub struct Env {
    scopes: Vec<HashMap<String, Value>>,
    function_frames: Vec<usize>,
    pub registry: HashMap<String, FunctionDef>,
    /// When set, records the first call to each registry function.
    /// Used by `sheaf build --trace-with` to discover concrete param shapes.
    pub call_records: Option<HashMap<String, CallRecord>>,
    /// Runtime tracer for function call logging and CLI guards.
    pub tracer: Option<crate::interpreter::tracer::Tracer>,
    /// Aggregated profiler for --blame mode.
    pub profiler: Option<crate::interpreter::profiler::Profiler>,
    /// Memory profiler for --mem-profile mode.
    pub mem_profiler: Option<crate::interpreter::mem_profile::MemProfiler>,
    /// Functions for which an IREE structure mismatch warning was already emitted.
    pub iree_mismatch_warned: std::collections::HashSet<String>,
    /// Optional wall-clock deadline for evaluation (safety net).
    pub eval_deadline: Option<std::time::Instant>,
    /// Target functions for tracing. When all have been observed in
    /// call_records, the interpreter aborts early with "trace complete".
    pub trace_targets: Option<std::collections::HashSet<String>>,
    /// Counts consecutive registry function calls without a new recording.
    /// When this exceeds a threshold, auto-trace stops (no new shapes to learn).
    pub trace_stale_calls: usize,
    /// JIT compiler for transparent on-demand compilation of pure functions.
    #[cfg(iree_runtime)]
    pub jit_compiler: Option<crate::runtime::jit::JitCompiler>,
    #[cfg(iree_runtime)]
    pub(super) preparation_cache: std::cell::RefCell<std::collections::VecDeque<super::specialize::CachedPreparation>>,
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Env {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            function_frames: Vec::new(),
            registry: HashMap::new(),
            call_records: None,
            tracer: None,
            profiler: None,
            iree_mismatch_warned: std::collections::HashSet::new(),
            eval_deadline: None,
            trace_targets: None,
            trace_stale_calls: 0,
            mem_profiler: None,
            #[cfg(iree_runtime)]
            jit_compiler: Some(crate::runtime::jit::JitCompiler::new()),
            #[cfg(iree_runtime)]
            preparation_cache: Default::default(),
        }
    }

    pub fn with_registry(registry: HashMap<String, FunctionDef>) -> Self {
        Self { registry, ..Self::new() }
    }

    pub fn get(&self, name: &str) -> Result<Value, SheafError> {
        let start = self.function_frames.last().copied().unwrap_or(0);
        for scope in self.scopes[start..].iter().rev() {
            if let Some(val) = scope.get(name) {
                return Ok(val.clone());
            }
        }
        Err(runtime_error(format!("Undefined symbol: {}", name)))
    }

    pub(crate) fn get_global(&self, name: &str) -> Option<Value> {
        self.scopes.first().and_then(|scope| scope.get(name)).cloned()
    }

    pub(crate) fn module_value(&self, name: &str) -> Result<Value, SheafError> {
        if let Some(value) = self.get_global(name) {
            return Ok(value);
        }
        if let Some(function) = self.registry.get(name)
            && !function.requires_initialization
            && let Some(body) = &function.body_compiled {
            return Ok(Value::Function {
                kind: super::value::FunctionKind::User,
                name: Some(name.to_string()), params: function.params.clone(),
                body: body.clone(), closure: Vec::new(),
            });
        }
        Err(runtime_error(format!("Undefined symbol: {}", name)))
    }

    pub fn set(&mut self, name: &str, val: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), val);
        }
    }

    pub fn set_global(&mut self, name: &str, val: Value) {
        if let Some(scope) = self.scopes.first_mut() {
            scope.insert(name.to_string(), val);
        }
    }

    pub fn set_builtin(&mut self, name: &str, func: BuiltinFnPtr) {
        self.set(name, Value::BuiltinFn {
            name: name.to_string(),
            func,
        });
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn push_function_scope(&mut self) {
        self.push_scope();
        self.function_frames.push(self.scopes.len() - 1);
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            if self.function_frames.last() == Some(&(self.scopes.len() - 1)) {
                self.function_frames.pop();
            }
            self.scopes.pop();
        }
    }

    pub fn all_names(&self) -> Vec<String> {
        let mut names = std::collections::HashSet::new();
        for scope in &self.scopes {
            names.extend(scope.keys().cloned());
        }
        let mut sorted: Vec<String> = names.into_iter().collect();
        sorted.sort();
        sorted
    }
}
