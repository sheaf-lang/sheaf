// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Base trait and utilities for special forms

use crate::ast::SheafValue;
use crate::core::compiler::{CompiledExpr, CompilerContext};
use crate::core::error::{SheafError, SheafResult, SourceLocation};

/// Special form trait - defines compilation behavior for language constructs
///
/// Each special form (defn, let, if, etc.) implements this trait to define
/// how it compiles from AST to CompiledExpr.
pub trait SpecialForm: Send + Sync {
    /// Return the name of this special form (e.g., "defn", "let")
    fn name(&self) -> &'static str;

    /// Compile this special form
    ///
    /// # Arguments
    /// - `compiler`: Mutable reference to compiler context
    /// - `args`: Arguments to the special form (elements[1..], excluding the operator)
    /// - `loc`: Source location for error reporting
    ///
    /// # Returns
    /// Compiled expression or compilation error
    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr>;
}

/// Utility: Check if a value is a vector (for binding syntax validation)
///
/// In Sheaf, bindings use [] not ():
/// - Correct: `(let [x 10] x)`
/// - Wrong: `(let (x 10) x)`
pub fn expect_vector<'a>(
    value: &'a SheafValue,
    context: &str,
    loc: &SourceLocation,
) -> SheafResult<&'a [SheafValue]> {
    value.as_vector().ok_or_else(|| SheafError::Compile {
        message: format!("{} must be a vector (use [] not ())", context),
        location: loc.clone(),
    })
}

/// Utility: Extract symbol from value
pub fn expect_symbol<'a>(
    value: &'a SheafValue,
    context: &str,
    loc: &SourceLocation,
) -> SheafResult<&'a str> {
    value.as_symbol().ok_or_else(|| SheafError::Compile {
        message: format!("{} must be a symbol", context),
        location: loc.clone(),
    })
}

/// Utility: Check argument count
pub fn check_arity(
    form_name: &str,
    args: &[SheafValue],
    expected: usize,
    loc: &SourceLocation,
) -> SheafResult<()> {
    if args.len() != expected {
        return Err(SheafError::Compile {
            message: format!(
                "{} expects {} arguments, got {}",
                form_name,
                expected,
                args.len()
            ),
            location: loc.clone(),
        });
    }
    Ok(())
}

/// Utility: Check minimum argument count
pub fn check_min_arity(
    form_name: &str,
    args: &[SheafValue],
    min: usize,
    loc: &SourceLocation,
) -> SheafResult<()> {
    if args.len() < min {
        return Err(SheafError::Compile {
            message: format!(
                "{} expects at least {} arguments, got {}",
                form_name,
                min,
                args.len()
            ),
            location: loc.clone(),
        });
    }
    Ok(())
}
