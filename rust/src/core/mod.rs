// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Core compiler components

pub mod compiler;
pub mod error;
pub mod parser;

pub use compiler::{CompiledExpr, CompilerContext, FunctionDef};
pub use error::{SheafError, SheafResult};
pub use parser::parse;
