// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Error types for Sheaf compiler

use std::fmt;

/// Source location for error reporting
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
    pub filename: String,
}

impl SourceLocation {
    pub fn new(line: usize, column: usize, filename: impl Into<String>) -> Self {
        Self {
            line,
            column,
            filename: filename.into(),
        }
    }

    pub fn unknown() -> Self {
        Self {
            line: 0,
            column: 0,
            filename: "<unknown>".to_string(),
        }
    }
}

/// Sheaf error types
#[derive(Debug, Clone)]
pub enum SheafError {
    /// Parse error
    Parse {
        message: String,
        location: SourceLocation,
    },
    /// Compilation error
    Compile {
        message: String,
        location: SourceLocation,
    },
    /// Runtime error
    Runtime {
        message: String,
        location: Option<SourceLocation>,
    },
    /// IO error
    Io(String),
}

impl fmt::Display for SheafError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SheafError::Parse { message, location } => {
                write!(
                    f,
                    "{}:{}: parse error: {}",
                    location.filename, location.line, message
                )
            }
            SheafError::Compile { message, location } => {
                write!(
                    f,
                    "{}:{}: compile error: {}",
                    location.filename, location.line, message
                )
            }
            SheafError::Runtime {
                message,
                location: Some(loc),
            } => {
                write!(
                    f,
                    "{}:{}: runtime error: {}",
                    loc.filename, loc.line, message
                )
            }
            SheafError::Runtime {
                message,
                location: None,
            } => {
                write!(f, "runtime error: {}", message)
            }
            SheafError::Io(msg) => write!(f, "io error: {}", msg),
        }
    }
}

impl std::error::Error for SheafError {}

/// Result type for Sheaf operations
pub type SheafResult<T> = Result<T, SheafError>;
