// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Tensor operations runtime module
//!
//! Contains:
//! - Tensor creation: zeros, ones, random-normal, etc.
//! - Tensor manipulation: reshape, transpose, slice, concat, etc.
//! - Tensor reduction: sum, mean, min, max, etc.
//!
//! This module provides runtime emission helpers for StableHLO tensor operations.

use crate::compiler::stablehlo::{Register, StableHLOEmitter, StableHLOType};

/// Emit zeros tensor: (zeros [M N]) -> tensor<MxNxf32>
pub fn emit_zeros(emitter: &mut StableHLOEmitter, shape: &[i64]) -> (Register, StableHLOType) {
    emitter.emit_zeros(shape)
}

/// Emit random-normal tensor: (random-normal key [M N])
/// For now, we emit a constant with small values (placeholder)
/// TODO: Proper RNG with seed/key
pub fn emit_random_normal(
    emitter: &mut StableHLOEmitter,
    shape: &[i64],
) -> (Register, StableHLOType) {
    emitter.emit_random_normal(shape)
}

/// Emit ones tensor: (ones [M N]) -> tensor<MxNxf32>
pub fn emit_ones(emitter: &mut StableHLOEmitter, shape: &[i64]) -> (Register, StableHLOType) {
    emitter.emit_ones(shape)
}

/// Emit reshape: (reshape tensor [M N]) -> tensor<MxNxf32>
pub fn emit_reshape(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    new_shape: &[i64],
) -> (Register, StableHLOType) {
    emitter.emit_reshape(operand, operand_ty, new_shape)
}

/// Emit transpose: (transpose tensor [1 0]) -> permutes dimensions
pub fn emit_transpose(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    permutation: &[i64],
) -> (Register, StableHLOType) {
    emitter.emit_transpose(operand, operand_ty, permutation)
}

/// Emit iota (arange): (arange [N]) -> tensor<Nxf32> with values [0, 1, 2, ..., N-1]
pub fn emit_arange(
    emitter: &mut StableHLOEmitter,
    shape: &[i64],
    dimension: i64,
) -> (Register, StableHLOType) {
    emitter.emit_iota(shape, dimension)
}

/// Emit concatenate: (concat [tensor1 tensor2 ...] axis)
pub fn emit_concatenate(
    emitter: &mut StableHLOEmitter,
    operands: &[Register],
    operand_types: &[StableHLOType],
    dimension: i64,
) -> (Register, StableHLOType) {
    emitter.emit_concatenate(operands, operand_types, dimension)
}

// TODO: Add more tensor operations
// - slice
// - sum, mean (reductions - require reduce with body)
// - argmax, argmin
// - where, einsum
// - broadcast (already internal in stablehlo.rs)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_ops_exist() {
        // Just verify the module compiles
        assert!(true);
    }
}
