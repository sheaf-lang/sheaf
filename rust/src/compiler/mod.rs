// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf compiler - transforms AST to StableHLO

pub mod codegen;
pub mod stablehlo;

pub use codegen::CodeGenerator;
pub use stablehlo::{StableHLOEmitter, StableHLOType};
