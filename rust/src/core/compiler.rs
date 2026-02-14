// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf compiler - orchestrates compilation from AST to executable code
//!
//! Corresponds to Python sheaf/core/compiler.py

use crate::ast::SheafValue;
use crate::core::error::{SheafError, SheafResult};
use std::collections::HashMap;

/// Compilation context - tracks environment, registry, etc.
pub struct CompilerContext {
    /// Global environment (built-in functions, runtime ops)
    pub env: HashMap<String, SheafValue>,

    /// Function registry (user-defined functions)
    pub registry: HashMap<String, FunctionDef>,

    /// Local variables (for let bindings, function params)
    pub local_vars: HashMap<String, SheafValue>,
}

/// Function definition
#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: SheafValue,
}

impl CompilerContext {
    pub fn new() -> Self {
        Self {
            env: Self::init_env(),
            registry: HashMap::new(),
            local_vars: HashMap::new(),
        }
    }

    /// Initialize global environment with built-in operations
    fn init_env() -> HashMap<String, SheafValue> {
        let env = HashMap::new();

        // Built-in constants
        // env.insert("true".to_string(), SheafValue::Boolean(true, SourceLocation::unknown()));
        // env.insert("false".to_string(), SheafValue::Boolean(false, SourceLocation::unknown()));
        // env.insert("nil".to_string(), SheafValue::Nil(SourceLocation::unknown()));

        // Built-in functions will be added as we port runtime ops

        env
    }

    /// Compile a Sheaf expression
    pub fn compile(&mut self, exp: &SheafValue) -> SheafResult<CompiledExpr> {
        match exp {
            // --- Literals ---
            SheafValue::Integer(n, _) => Ok(CompiledExpr::Integer(*n)),
            SheafValue::Float(x, _) => Ok(CompiledExpr::Float(*x)),
            SheafValue::Boolean(b, _) => Ok(CompiledExpr::Boolean(*b)),
            SheafValue::Nil(_) => Ok(CompiledExpr::Nil),
            SheafValue::String(s, _) => Ok(CompiledExpr::String(s.clone())),

            // --- Symbols ---
            SheafValue::Symbol(name, loc) => self.resolve_symbol(name, loc),

            // --- Keywords ---
            SheafValue::Keyword(k, _) => Ok(CompiledExpr::Keyword(k.clone())),

            // --- Lists (function calls, special forms) ---
            SheafValue::List(elements, loc) => {
                if elements.is_empty() {
                    return Err(SheafError::Compile {
                        message: "Cannot compile empty list".to_string(),
                        location: loc.clone(),
                    });
                }

                // Check for special forms
                if let Some(op) = elements[0].as_symbol() {
                    // Try to compile as special form, fall back to function call
                    match self.try_compile_special_form(op, &elements[1..], loc) {
                        Some(result) => result,
                        None => self.compile_function_call(elements, loc),
                    }
                } else {
                    self.compile_function_call(elements, loc)
                }
            }

            // --- Vectors ---
            SheafValue::Vector(elements, _) => {
                // Compile each element
                let compiled: SheafResult<Vec<CompiledExpr>> =
                    elements.iter().map(|e| self.compile(e)).collect();
                Ok(CompiledExpr::Vector(compiled?))
            }

            // --- Dicts ---
            SheafValue::Dict(pairs, _) => {
                let compiled: SheafResult<Vec<(CompiledExpr, CompiledExpr)>> = pairs
                    .iter()
                    .map(|(k, v)| Ok((self.compile(k)?, self.compile(v)?)))
                    .collect();
                Ok(CompiledExpr::Dict(compiled?))
            }

            // --- Quotes ---
            SheafValue::Quote(inner, _) => {
                // Quote prevents evaluation
                Ok(CompiledExpr::Quoted(inner.clone()))
            }

            _ => Err(SheafError::Compile {
                message: format!("Unsupported expression type: {}", exp),
                location: exp.location().clone(),
            }),
        }
    }

    /// Resolve a symbol to its value
    fn resolve_symbol(
        &mut self,
        name: &str,
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // Check local variables first - return as Symbol for let bindings
        if self.local_vars.contains_key(name) {
            return Ok(CompiledExpr::Symbol(name.to_string()));
        }

        // Check environment
        if let Some(value) = self.env.get(name).cloned() {
            return self.compile(&value);
        }

        // Check registry (user-defined functions)
        if self.registry.contains_key(name) {
            return Ok(CompiledExpr::FunctionRef(name.to_string()));
        }

        Err(SheafError::Compile {
            message: format!("Undefined symbol: {}", name),
            location: loc.clone(),
        })
    }

    // Special form compilation methods have been moved to src/forms/
    // See: binding.rs (defn, let, fn), control.rs (if, do), utils.rs (quote)

    /// Try to compile as a special form, return None if not a special form
    fn try_compile_special_form(
        &mut self,
        op: &str,
        args: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> Option<SheafResult<CompiledExpr>> {
        // Static dispatch to special forms
        use crate::forms::*;

        let result = match op {
            "defn" => DefnForm.compile(self, args, loc),
            "let" => LetForm.compile(self, args, loc),
            "fn" => FnForm.compile(self, args, loc),
            "if" => IfForm.compile(self, args, loc),
            "do" => DoForm.compile(self, args, loc),
            "quote" => QuoteForm.compile(self, args, loc),
            "case" => CaseForm.compile(self, args, loc),
            "while" => WhileForm.compile(self, args, loc),
            "repeat" => RepeatForm.compile(self, args, loc),
            "guard" => GuardForm.compile(self, args, loc),
            "->" => ThreadFirstForm.compile(self, args, loc),
            "as->" => ThreadAsForm.compile(self, args, loc),
            "get" => GetForm.compile(self, args, loc),
            "get-in" => GetInForm.compile(self, args, loc),
            "dict" => DictForm.compile(self, args, loc),
            "assoc" => AssocForm.compile(self, args, loc),
            "last" => LastForm.compile(self, args, loc),
            "use" => UseForm.compile(self, args, loc),
            _ => return None, // Not a special form
        };

        Some(result)
    }

    /// Compile function call (op arg1 arg2 ...)
    fn compile_function_call(
        &mut self,
        elements: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        let op = &elements[0];
        let args = &elements[1..];

        // Get function name
        let func_name = op.as_symbol().ok_or_else(|| SheafError::Compile {
            message: format!("Function name must be a symbol, got: {}", op),
            location: loc.clone(),
        })?;

        // Compile arguments
        let compiled_args: SheafResult<Vec<CompiledExpr>> =
            args.iter().map(|arg| self.compile(arg)).collect();
        let compiled_args = compiled_args?;

        Ok(CompiledExpr::FunctionCall {
            name: func_name.to_string(),
            args: compiled_args,
        })
    }
}

/// Compiled expression - intermediate representation
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Nil,
    String(String),
    Keyword(String),
    Vector(Vec<CompiledExpr>),
    Dict(Vec<(CompiledExpr, CompiledExpr)>),
    Quoted(Box<SheafValue>),
    FunctionRef(String),
    FunctionCall {
        name: String,
        args: Vec<CompiledExpr>,
    },
    Let {
        bindings: Vec<(String, CompiledExpr)>,
        body: Box<CompiledExpr>,
    },
    If {
        condition: Box<CompiledExpr>,
        then_branch: Box<CompiledExpr>,
        else_branch: Option<Box<CompiledExpr>>,
    },
    Do(Vec<CompiledExpr>),
    Symbol(String),
}

impl Default for CompilerContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SourceLocation;

    fn make_int(n: i64) -> SheafValue {
        SheafValue::Integer(n, SourceLocation::unknown())
    }

    fn make_symbol(s: &str) -> SheafValue {
        SheafValue::Symbol(s.to_string(), SourceLocation::unknown())
    }

    fn make_list(elems: Vec<SheafValue>) -> SheafValue {
        SheafValue::List(elems, SourceLocation::unknown())
    }

    #[test]
    fn test_compile_literal() {
        let mut ctx = CompilerContext::new();
        let expr = make_int(42);
        let result = ctx.compile(&expr).unwrap();
        assert!(matches!(result, CompiledExpr::Integer(42)));
    }

    #[test]
    fn test_compile_function_call() {
        let mut ctx = CompilerContext::new();
        // (+ 1 2)
        let expr = make_list(vec![make_symbol("+"), make_int(1), make_int(2)]);
        let result = ctx.compile(&expr).unwrap();

        match result {
            CompiledExpr::FunctionCall { name, args } => {
                assert_eq!(name, "+");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_compile_let() {
        let mut ctx = CompilerContext::new();
        // (let [x 1 y 2] (+ x y))
        let expr = make_list(vec![
            make_symbol("let"),
            SheafValue::Vector(
                vec![make_symbol("x"), make_int(1), make_symbol("y"), make_int(2)],
                SourceLocation::unknown(),
            ),
            make_list(vec![make_symbol("+"), make_symbol("x"), make_symbol("y")]),
        ]);

        let result = ctx.compile(&expr).unwrap();

        match result {
            CompiledExpr::Let { bindings, body } => {
                assert_eq!(bindings.len(), 2);
                assert_eq!(bindings[0].0, "x");
                assert_eq!(bindings[1].0, "y");
                assert!(matches!(bindings[0].1, CompiledExpr::Integer(1)));
                assert!(matches!(bindings[1].1, CompiledExpr::Integer(2)));
                assert!(matches!(*body, CompiledExpr::FunctionCall { .. }));
            }
            _ => panic!("Expected let expression"),
        }
    }

    #[test]
    fn test_compile_defn() {
        let mut ctx = CompilerContext::new();
        // (defn add [x y] (+ x y))
        let expr = make_list(vec![
            make_symbol("defn"),
            make_symbol("add"),
            SheafValue::Vector(
                vec![make_symbol("x"), make_symbol("y")],
                SourceLocation::unknown(),
            ),
            make_list(vec![make_symbol("+"), make_symbol("x"), make_symbol("y")]),
        ]);

        let result = ctx.compile(&expr).unwrap();
        assert!(matches!(result, CompiledExpr::Nil));

        // Check function was registered
        assert!(ctx.registry.contains_key("add"));
        let func = &ctx.registry["add"];
        assert_eq!(func.params, vec!["x", "y"]);
    }
}
