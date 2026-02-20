// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Environment for the Sheaf interpreter — scoped variable bindings.

use crate::core::compiler::FunctionDef;
use crate::core::error::SheafError;
use crate::interpreter::value::{BuiltinFnPtr, Value};
use std::collections::HashMap;

pub fn runtime_error(message: impl Into<String>) -> SheafError {
    SheafError::Runtime {
        message: message.into(),
        location: None,
    }
}

#[derive(Clone)]
pub struct Env {
    scopes: Vec<HashMap<String, Value>>,
    pub registry: HashMap<String, FunctionDef>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            registry: HashMap::new(),
        }
    }

    pub fn with_registry(registry: HashMap<String, FunctionDef>) -> Self {
        Self {
            scopes: vec![HashMap::new()],
            registry,
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
