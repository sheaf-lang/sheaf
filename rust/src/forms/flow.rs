// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Threading and composition special forms: ->, as->

use crate::ast::SheafValue;
use crate::core::compiler::{CompiledExpr, CompilerContext};
use crate::core::error::{SheafError, SheafResult, SourceLocation};
use crate::forms::base::SpecialForm;

/// -> Thread-first macro: (-> x (f1) (f2 a)) becomes (f2 (f1 x) a)
///
/// Rewrites the AST then compiles the result. If a step is a bare symbol,
/// it is wrapped into a single-element call: (-> x f) becomes (f x).
pub struct ThreadFirstForm;

impl SpecialForm for ThreadFirstForm {
    fn name(&self) -> &'static str {
        "->"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        if args.is_empty() {
            return Err(SheafError::Compile {
                message: "-> requires at least one argument".to_string(),
                location: loc.clone(),
            });
        }

        // Start with the initial value
        let mut acc = args[0].clone();

        // Thread through each step
        for step in &args[1..] {
            acc = match step {
                // (-> x (f a b)) => (f x a b)
                SheafValue::List(elems, sloc) => {
                    let mut new_elems = Vec::with_capacity(elems.len() + 1);
                    new_elems.push(elems[0].clone()); // function
                    new_elems.push(acc);              // threaded value as first arg
                    new_elems.extend_from_slice(&elems[1..]); // remaining args
                    SheafValue::List(new_elems, sloc.clone())
                }
                // (-> x f) => (f x)
                SheafValue::Symbol(_, sloc) => {
                    SheafValue::List(vec![step.clone(), acc], sloc.clone())
                }
                _ => {
                    return Err(SheafError::Compile {
                        message: format!("-> step must be a list or symbol, got {:?}", step),
                        location: loc.clone(),
                    });
                }
            };
        }

        compiler.compile(&acc)
    }
}

/// as-> Thread-as macro: (as-> init name step1 step2) binds value at each step
///
/// Rewrites to nested let:
///   (as-> x h (f h) (g h 1)) => (let [h x h (f h)] (g h 1))
pub struct ThreadAsForm;

impl SpecialForm for ThreadAsForm {
    fn name(&self) -> &'static str {
        "as->"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        if args.len() < 3 {
            return Err(SheafError::Compile {
                message: "as-> requires init, name, and at least one step".to_string(),
                location: loc.clone(),
            });
        }

        let init = &args[0];
        let name = &args[1];

        // Validate that name is a symbol
        if name.as_symbol().is_none() {
            return Err(SheafError::Compile {
                message: format!("as-> binding name must be a symbol, got {:?}", name),
                location: loc.clone(),
            });
        }

        let steps = &args[2..];

        // Build let bindings: [name init, name step1, name step2, ...]
        // The last step becomes the body of the let.
        let mut bindings = Vec::new();
        bindings.push(name.clone());
        bindings.push(init.clone());

        for step in &steps[..steps.len() - 1] {
            bindings.push(name.clone());
            bindings.push(step.clone());
        }

        let binding_vec = SheafValue::Vector(bindings, loc.clone());
        let body = steps.last().unwrap().clone();

        // Rewrite as (let [name init name step1 ...] last-step)
        let let_expr = SheafValue::List(
            vec![
                SheafValue::Symbol("let".to_string(), loc.clone()),
                binding_vec,
                body,
            ],
            loc.clone(),
        );

        compiler.compile(&let_expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_form_names() {
        assert_eq!(ThreadFirstForm.name(), "->");
        assert_eq!(ThreadAsForm.name(), "as->");
    }
}
