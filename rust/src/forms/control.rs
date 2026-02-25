// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Control flow special forms: if, do, case, while, repeat, guard

use crate::ast::SheafValue;
use crate::core::compiler::{CompiledExpr, CompilerContext};
use crate::core::error::{SheafError, SheafResult, SourceLocation};
use crate::forms::base::{SpecialForm, check_arity, check_min_arity, expect_symbol, expect_vector};

/// if - Conditional: (if condition then else)
pub struct IfForm;

impl SpecialForm for IfForm {
    fn name(&self) -> &'static str {
        "if"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // (if condition then) or (if condition then else)
        if args.len() < 2 || args.len() > 3 {
            return Err(SheafError::Compile {
                message: "if requires 2 or 3 arguments: (if condition then [else])".to_string(),
                location: loc.clone(),
            });
        }

        let condition = compiler.compile(&args[0])?;
        let then_branch = compiler.compile(&args[1])?;
        let else_branch = if args.len() == 3 {
            Some(Box::new(compiler.compile(&args[2])?))
        } else {
            None
        };

        Ok(CompiledExpr::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch,
        })
    }
}

/// do - Sequential evaluation: (do expr1 expr2 ... exprN) → returns exprN
pub struct DoForm;

impl SpecialForm for DoForm {
    fn name(&self) -> &'static str {
        "do"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        check_min_arity("do", args, 1, loc)?;

        let compiled: SheafResult<Vec<CompiledExpr>> =
            args.iter().map(|expr| compiler.compile(expr)).collect();

        Ok(CompiledExpr::Do(compiled?))
    }
}

/// case - Pattern matching: (case target val1 result1 val2 result2 ... default)
///
/// Compiles to nested if/else: (if (= target val1) result1 (if (= target val2) result2 ... default))
pub struct CaseForm;

impl SpecialForm for CaseForm {
    fn name(&self) -> &'static str {
        "case"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        check_min_arity("case", args, 2, loc)?;
        let target = compiler.compile(&args[0])?;
        let pairs = &args[1..];
        if pairs.len() % 2 == 0 {
            // Even: all key-value pairs, no default
            build_case_chain(compiler, &target, pairs, CompiledExpr::Nil)
        } else {
            // Odd: pairs + trailing default
            let default = compiler.compile(pairs.last().unwrap())?;
            build_case_chain(compiler, &target, &pairs[..pairs.len() - 1], default)
        }
    }
}

fn build_case_chain(
    compiler: &mut CompilerContext,
    target: &CompiledExpr,
    pairs: &[SheafValue],
    default: CompiledExpr,
) -> SheafResult<CompiledExpr> {
    if pairs.is_empty() {
        return Ok(default);
    }
    let val = compiler.compile(&pairs[0])?;
    let result = compiler.compile(&pairs[1])?;
    let rest = build_case_chain(compiler, target, &pairs[2..], default)?;
    Ok(CompiledExpr::If {
        condition: Box::new(CompiledExpr::FunctionCall {
            name: "=".to_string(),
            args: vec![target.clone(), val],
        }),
        then_branch: Box::new(result),
        else_branch: Some(Box::new(rest)),
    })
}

/// while - While loop: (while cond [acc init] body)
///
/// Loops while `cond` is truthy. The accumulator variable is visible
/// in both the condition and body expressions.
pub struct WhileForm;

impl SpecialForm for WhileForm {
    fn name(&self) -> &'static str {
        "while"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // (while cond [acc init] body)
        check_arity("while", args, 3, loc)?;

        let acc_vec = expect_vector(&args[1], "while acc binding [acc init]", loc)?;
        if acc_vec.len() != 2 {
            return Err(SheafError::Compile {
                message: "while: accumulator binding must be [acc init]".to_string(),
                location: loc.clone(),
            });
        }
        let acc_var = expect_symbol(&acc_vec[0], "while acc var", loc)?.to_string();

        // Register acc_var as a local so condition and body can reference it
        let saved_locals = compiler.local_vars.clone();
        compiler.local_vars.insert(
            acc_var.clone(),
            SheafValue::Symbol(acc_var.clone(), loc.clone()),
        );

        let acc_init = compiler.compile(&acc_vec[1])?;
        let condition = compiler.compile(&args[0])?;
        let body = compiler.compile(&args[2])?;

        compiler.local_vars = saved_locals;

        Ok(CompiledExpr::While {
            condition: Box::new(condition),
            acc_var,
            acc_init: Box::new(acc_init),
            body: Box::new(body),
        })
    }
}

/// repeat - Counted loop: (repeat [i n] [acc init] body)
///
/// Runs body `n` times, binding `i` to 0..n-1 and `acc` to the accumulator.
/// Returns the final accumulator value.
pub struct RepeatForm;

impl SpecialForm for RepeatForm {
    fn name(&self) -> &'static str {
        "repeat"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // (repeat [i n] [acc init] body)
        check_arity("repeat", args, 3, loc)?;

        let loop_vec = expect_vector(&args[0], "repeat loop binding [i n]", loc)?;
        if loop_vec.len() != 2 {
            return Err(SheafError::Compile {
                message: "repeat: first binding must be [index count]".to_string(),
                location: loc.clone(),
            });
        }
        let index_var = expect_symbol(&loop_vec[0], "repeat index var", loc)?.to_string();
        let count = compiler.compile(&loop_vec[1])?;

        let acc_vec = expect_vector(&args[1], "repeat acc binding [acc init]", loc)?;
        if acc_vec.len() != 2 {
            return Err(SheafError::Compile {
                message: "repeat: second binding must be [acc init]".to_string(),
                location: loc.clone(),
            });
        }
        let acc_var = expect_symbol(&acc_vec[0], "repeat acc var", loc)?.to_string();

        let saved_locals = compiler.local_vars.clone();
        compiler.local_vars.insert(
            index_var.clone(),
            SheafValue::Symbol(index_var.clone(), loc.clone()),
        );
        compiler.local_vars.insert(
            acc_var.clone(),
            SheafValue::Symbol(acc_var.clone(), loc.clone()),
        );

        let acc_init = compiler.compile(&acc_vec[1])?;
        let body = compiler.compile(&args[2])?;

        compiler.local_vars = saved_locals;

        Ok(CompiledExpr::Repeat {
            index_var,
            count: Box::new(count),
            acc_var,
            acc_init: Box::new(acc_init),
            body: Box::new(body),
        })
    }
}

/// guard - Runtime assertions: (guard :no-nan x) or (guard :shape [64 256] x)
///
/// Not yet implemented - placeholder
pub struct GuardForm;

impl SpecialForm for GuardForm {
    fn name(&self) -> &'static str {
        "guard"
    }

    fn compile(
        &self,
        _compiler: &mut CompilerContext,
        _args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        Err(SheafError::Compile {
            message: "guard not yet implemented".to_string(),
            location: loc.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_form_names() {
        assert_eq!(IfForm.name(), "if");
        assert_eq!(DoForm.name(), "do");
        assert_eq!(CaseForm.name(), "case");
        assert_eq!(WhileForm.name(), "while");
        assert_eq!(RepeatForm.name(), "repeat");
        assert_eq!(GuardForm.name(), "guard");
    }
}
