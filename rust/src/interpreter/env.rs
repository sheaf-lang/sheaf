// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Environment for the Sheaf interpreter — scoped variable bindings.

use crate::core::compiler::{FunctionDef, VmfbSession};
use crate::core::error::SheafError;
use crate::interpreter::value::{BuiltinFnPtr, Value};
use std::collections::HashMap;

pub fn runtime_error(message: impl Into<String>) -> SheafError {
    SheafError::Runtime {
        message: message.into(),
        location: None,
    }
}

/// A recorded function call: argument values observed during tracing.
#[derive(Clone, Debug)]
pub struct CallRecord {
    pub arg_values: Vec<Value>,
}

#[derive(Clone)]
pub struct Env {
    scopes: Vec<HashMap<String, Value>>,
    pub registry: HashMap<String, FunctionDef>,
    pub vmfb_sessions: Vec<VmfbSession>,
    /// When set, records the first call to each registry function.
    /// Used by `sheaf build --trace-with` to discover concrete param shapes.
    pub call_records: Option<HashMap<String, CallRecord>>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            registry: HashMap::new(),
            vmfb_sessions: Vec::new(),
            call_records: None,
        }
    }

    pub fn with_registry(registry: HashMap<String, FunctionDef>) -> Self {
        Self {
            scopes: vec![HashMap::new()],
            registry,
            vmfb_sessions: Vec::new(),
            call_records: None,
        }
    }

    pub fn get(&self, name: &str) -> Result<Value, SheafError> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Ok(val.clone());
            }
        }
        Err(runtime_error(format!("Undefined symbol: {}", name)))
    }

    pub fn set(&mut self, name: &str, val: Value) {
        if let Some(scope) = self.scopes.last_mut() {
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

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }
}
