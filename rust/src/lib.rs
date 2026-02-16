// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf V2 - Rust implementation
//!
//! A functional language for differentiable programming,
//! compiling directly to StableHLO and running on IREE.

pub mod ast;
pub mod compiler;
pub mod core;
pub mod forms;
pub mod runtime;

// Re-export main types
pub use ast::SheafValue;
pub use compiler::{CodeGenerator, StableHLOEmitter, StableHLOType};
pub use core::compiler::{CompiledExpr, CompilerContext};
pub use core::error::{SheafError, SheafResult, SourceLocation};
pub use core::parse;
