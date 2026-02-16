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

// TODO: Add more tensor operations
// - reshape
// - transpose
// - slice
// - concat
// - sum, mean, min, max (reductions)
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
