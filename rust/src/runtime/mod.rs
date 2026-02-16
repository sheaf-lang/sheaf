// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Sheaf V2 Runtime Operations
//!
//! This module contains runtime operation emitters organized by category,
//! mirroring the structure of Sheaf V1 Python implementation.
//!
//! Structure:
//! - math_ops: Arithmetic, comparisons, booleans, math functions
//! - tensor_ops: Tensor operations (reshape, transpose, slice, etc.) [TODO]
//! - nn_ops: Neural network operations (relu, sigmoid, conv, etc.) [TODO]
//! - core_ops: Core operations (print, error, type, etc.) [TODO]
//! - io_ops: I/O operations (read, write, etc.) [TODO]
//! - string_ops: String operations (concat, split, etc.) [TODO]

pub mod math_ops;
// TODO: Add other runtime modules
// pub mod tensor_ops;
// pub mod nn_ops;
// pub mod core_ops;
// pub mod io_ops;
// pub mod string_ops;
