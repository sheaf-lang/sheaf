// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Math operations runtime module
//!
//! Contains:
//! - Arithmetic operations: +, -, *, /, @, **, //, mod (%)
//! - Comparison operations: =, ==, !=, <, <=, >, >=
//! - Boolean operations: and, or, not
//! - Math functions: sqrt, exp, log, abs
//! - Min/max operations: min, max
//!
//! This module provides runtime emission helpers for StableHLO math operations.

use crate::compiler::stablehlo::{Register, StableHLOEmitter, StableHLOType};

/// Emit arithmetic binary operation: +, -, *, /
pub fn emit_arithmetic_binop(
    emitter: &mut StableHLOEmitter,
    op: &str,
    lhs: &Register,
    rhs: &Register,
    lhs_ty: &StableHLOType,
    rhs_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_binop(op, lhs, rhs, lhs_ty, rhs_ty)
}

/// Emit matrix multiply: @
pub fn emit_matmul(
    emitter: &mut StableHLOEmitter,
    lhs: &Register,
    rhs: &Register,
    lhs_ty: &StableHLOType,
    rhs_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_matmul(lhs, rhs, lhs_ty, rhs_ty)
}

/// Emit comparison operation: =, ==, !=, <, <=, >, >=
pub fn emit_comparison(
    emitter: &mut StableHLOEmitter,
    op: &str,
    lhs: &Register,
    rhs: &Register,
    lhs_ty: &StableHLOType,
    rhs_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_compare(op, lhs, rhs, lhs_ty, rhs_ty)
}

/// Emit boolean binary operation: and, or
pub fn emit_boolean_binop(
    emitter: &mut StableHLOEmitter,
    op: &str,
    lhs: &Register,
    rhs: &Register,
    lhs_ty: &StableHLOType,
    rhs_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_bool_binop(op, lhs, rhs, lhs_ty, rhs_ty)
}

/// Emit boolean unary operation: not
pub fn emit_not(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> Register {
    emitter.emit_unary("not", operand, ty)
}

/// Emit math unary operations: sqrt, exp, log, abs
pub fn emit_math_unary(
    emitter: &mut StableHLOEmitter,
    op: &str,
    operand: &Register,
    ty: &StableHLOType,
) -> Register {
    match op {
        "sqrt" | "exp" | "log" | "abs" => emitter.emit_unary(op, operand, ty),
        _ => panic!("Unsupported math unary operation: {}", op),
    }
}

/// Emit min/max binary operations
pub fn emit_minmax(
    emitter: &mut StableHLOEmitter,
    op: &str,
    lhs: &Register,
    rhs: &Register,
    lhs_ty: &StableHLOType,
    rhs_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_binop(op, lhs, rhs, lhs_ty, rhs_ty)
}

/// Emit power, floor division, and modulo operations
pub fn emit_extended_arithmetic(
    emitter: &mut StableHLOEmitter,
    op: &str,
    lhs: &Register,
    rhs: &Register,
    lhs_ty: &StableHLOType,
    rhs_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_binop(op, lhs, rhs, lhs_ty, rhs_ty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_ops_exist() {
        // Just verify the module compiles
        assert!(true);
    }
}
