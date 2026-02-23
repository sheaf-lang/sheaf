// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf compiler - transforms AST to StableHLO

pub mod codegen;
pub mod config;
pub mod effects;
pub mod stablehlo;

pub use codegen::CodeGenerator;
pub use config::{build_index_map, json_to_stablehlo_type, lower_get_calls};
pub use effects::{collect_effects, format_effects, has_side_effects};
pub use stablehlo::{StableHLOEmitter, StableHLOType};
