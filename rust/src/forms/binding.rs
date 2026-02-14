// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Binding special forms: defn, let, fn

use crate::ast::SheafValue;
use crate::core::compiler::{CompiledExpr, CompilerContext, FunctionDef};
use crate::core::error::{SheafResult, SourceLocation};
use crate::forms::base::{SpecialForm, check_min_arity, expect_symbol, expect_vector};

/// defn - Function definition: (defn name [params] body)
pub struct DefnForm;

impl SpecialForm for DefnForm {
    fn name(&self) -> &'static str {
        "defn"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // (defn name [params] body)
        check_min_arity("defn", args, 3, loc)?;

        let name = expect_symbol(&args[0], "defn name", loc)?;
        let params_vec = expect_vector(&args[1], "defn parameters", loc)?;

        // Extract parameter names
        let params: SheafResult<Vec<String>> = params_vec
            .iter()
            .map(|p| expect_symbol(p, "parameter name", loc).map(|s| s.to_string()))
            .collect();
        let params = params?;

        // Body is the third argument
        let body = args[2].clone();

        // Register the function in the compiler
        compiler.registry.insert(
            name.to_string(),
            FunctionDef {
                name: name.to_string(),
                params,
                body,
            },
        );

        // defn returns nil (side effect: registers function)
        Ok(CompiledExpr::Nil)
    }
}

/// let - Local bindings: (let [var1 val1 var2 val2] body)
pub struct LetForm;

impl SpecialForm for LetForm {
    fn name(&self) -> &'static str {
        "let"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // (let [bindings] body)
        check_min_arity("let", args, 2, loc)?;

        let bindings_vec = expect_vector(&args[0], "let bindings", loc)?;

        if bindings_vec.len() % 2 != 0 {
            return Err(crate::core::error::SheafError::Compile {
                message: "let bindings must have even number of elements (name value pairs)"
                    .to_string(),
                location: loc.clone(),
            });
        }

        // Save current local_vars state
        let saved_locals = compiler.local_vars.clone();

        // Process bindings in pairs
        let mut compiled_bindings = Vec::new();
        for i in (0..bindings_vec.len()).step_by(2) {
            let name = expect_symbol(&bindings_vec[i], "let binding name", loc)?;
            let value = &bindings_vec[i + 1];
            let compiled_value = compiler.compile(value)?;

            // Add to local scope (for subsequent bindings in same let)
            compiler.local_vars.insert(name.to_string(), value.clone());
            compiled_bindings.push((name.to_string(), compiled_value));
        }

        // Compile body with bindings in scope
        let body = &args[1];
        let compiled_body = compiler.compile(body)?;

        // Restore local_vars (let is scoped)
        compiler.local_vars = saved_locals;

        Ok(CompiledExpr::Let {
            bindings: compiled_bindings,
            body: Box::new(compiled_body),
        })
    }
}

/// fn - Anonymous function: (fn [params] body)
pub struct FnForm;

impl SpecialForm for FnForm {
    fn name(&self) -> &'static str {
        "fn"
    }

    fn compile(
        &self,
        _compiler: &mut CompilerContext,
        _args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // TODO: Implement anonymous functions
        // Will need to capture closure, compile body, etc.
        Err(crate::core::error::SheafError::Compile {
            message: "fn (anonymous functions) not yet implemented".to_string(),
            location: loc.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SourceLocation;

    #[test]
    fn test_defn_form_name() {
        let form = DefnForm;
        assert_eq!(form.name(), "defn");
    }

    #[test]
    fn test_let_form_name() {
        let form = LetForm;
        assert_eq!(form.name(), "let");
    }
}
