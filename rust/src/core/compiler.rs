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
                    match op {
                        "defn" => self.compile_defn(elements, loc),
                        "let" => self.compile_let(elements, loc),
                        "fn" => self.compile_fn(elements, loc),
                        "if" => self.compile_if(elements, loc),
                        "do" => self.compile_do(elements, loc),
                        "quote" => self.compile_quote(elements, loc),
                        _ => self.compile_function_call(elements, loc),
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

    /// Compile (defn name [params] body)
    fn compile_defn(
        &mut self,
        elements: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        if elements.len() < 4 {
            return Err(SheafError::Compile {
                message: "defn requires: (defn name [params] body)".to_string(),
                location: loc.clone(),
            });
        }

        let name = elements[1].as_symbol().ok_or_else(|| SheafError::Compile {
            message: "defn name must be a symbol".to_string(),
            location: loc.clone(),
        })?;

        let params_vec = elements[2].as_vector().ok_or_else(|| SheafError::Compile {
            message: "defn params must be a vector".to_string(),
            location: loc.clone(),
        })?;

        let params: Result<Vec<String>, SheafError> = params_vec
            .iter()
            .map(|p| {
                p.as_symbol()
                    .map(|s| s.to_string())
                    .ok_or_else(|| SheafError::Compile {
                        message: "Parameter must be a symbol".to_string(),
                        location: loc.clone(),
                    })
            })
            .collect();

        let params = params?;
        let body = elements[3].clone();

        // Register the function
        self.registry.insert(
            name.to_string(),
            FunctionDef {
                name: name.to_string(),
                params,
                body,
            },
        );

        // defn returns nil
        Ok(CompiledExpr::Nil)
    }

    /// Compile (let [bindings] body)
    fn compile_let(
        &mut self,
        elements: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        if elements.len() < 3 {
            return Err(SheafError::Compile {
                message: "let requires: (let [bindings] body)".to_string(),
                location: loc.clone(),
            });
        }

        let bindings_vec = elements[1].as_vector().ok_or_else(|| SheafError::Compile {
            message: "let bindings must be a vector".to_string(),
            location: loc.clone(),
        })?;

        if bindings_vec.len() % 2 != 0 {
            return Err(SheafError::Compile {
                message: "let bindings must have even number of elements (name value pairs)"
                    .to_string(),
                location: loc.clone(),
            });
        }

        // Save current local_vars state
        let saved_locals = self.local_vars.clone();

        // Process bindings in pairs
        let mut compiled_bindings = Vec::new();
        for i in (0..bindings_vec.len()).step_by(2) {
            let name = bindings_vec[i]
                .as_symbol()
                .ok_or_else(|| SheafError::Compile {
                    message: "let binding name must be a symbol".to_string(),
                    location: loc.clone(),
                })?;

            let value = &bindings_vec[i + 1];
            let compiled_value = self.compile(value)?;

            // Add to local scope
            self.local_vars.insert(name.to_string(), value.clone());
            compiled_bindings.push((name.to_string(), compiled_value));
        }

        // Compile body with bindings in scope
        let body = &elements[2];
        let compiled_body = self.compile(body)?;

        // Restore local_vars
        self.local_vars = saved_locals;

        Ok(CompiledExpr::Let {
            bindings: compiled_bindings,
            body: Box::new(compiled_body),
        })
    }

    /// Compile (fn [params] body)
    fn compile_fn(
        &mut self,
        _elements: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // TODO: Implement anonymous functions
        Err(SheafError::Compile {
            message: "fn not yet implemented".to_string(),
            location: loc.clone(),
        })
    }

    /// Compile (if condition then else)
    fn compile_if(
        &mut self,
        elements: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        if elements.len() < 3 || elements.len() > 4 {
            return Err(SheafError::Compile {
                message: "if requires: (if condition then) or (if condition then else)".to_string(),
                location: loc.clone(),
            });
        }

        let condition = self.compile(&elements[1])?;
        let then_branch = self.compile(&elements[2])?;
        let else_branch = if elements.len() == 4 {
            Some(Box::new(self.compile(&elements[3])?))
        } else {
            None
        };

        Ok(CompiledExpr::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch,
        })
    }

    /// Compile (do expr1 expr2 ...)
    fn compile_do(
        &mut self,
        elements: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        if elements.len() < 2 {
            return Err(SheafError::Compile {
                message: "do requires at least one expression".to_string(),
                location: loc.clone(),
            });
        }

        let exprs: SheafResult<Vec<CompiledExpr>> =
            elements[1..].iter().map(|e| self.compile(e)).collect();

        Ok(CompiledExpr::Do(exprs?))
    }

    /// Compile (quote expr)
    fn compile_quote(
        &mut self,
        elements: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        if elements.len() != 2 {
            return Err(SheafError::Compile {
                message: "quote requires exactly one argument".to_string(),
                location: loc.clone(),
            });
        }
        Ok(CompiledExpr::Quoted(Box::new(elements[1].clone())))
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
