// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Neural network operations runtime module
//!
//! Contains:
//! - Activation functions: relu, sigmoid, tanh, softmax, etc.
//! - Layer operations: conv, pool, batch-norm, layer-norm, etc.
//! - Loss functions: mse, cross-entropy, etc.
//!
//! This module provides runtime emission helpers for StableHLO neural network operations.

use crate::compiler::stablehlo::{Register, StableHLOEmitter, StableHLOType};

/// Emit ReLU activation: max(x, 0)
pub fn emit_relu(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> Register {
    emitter.emit_unary("relu", operand, ty)
}

/// Emit sigmoid activation: 1 / (1 + exp(-x))
pub fn emit_sigmoid(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> Register {
    emitter.emit_unary("sigmoid", operand, ty)
}

/// Emit tanh activation
pub fn emit_tanh(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> Register {
    emitter.emit_unary("tanh", operand, ty)
}

// TODO: Add more NN operations
// - softmax
// - layer-norm
// - batch-norm
// - conv2d
// - max-pool, avg-pool
// - dropout
// - etc.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn_ops_exist() {
        // Just verify the module compiles
        assert!(true);
    }
}
