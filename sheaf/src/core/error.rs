// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Error types for Sheaf compiler

use std::borrow::Cow;
use std::fmt;
use std::rc::Rc;

/// Source location for error reporting
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
    pub filename: Rc<str>,
}

impl SourceLocation {
    pub fn new(line: usize, column: usize, filename: Rc<str>) -> Self {
        Self {
            line,
            column,
            filename,
        }
    }

    pub fn unknown() -> Self {
        Self {
            line: 0,
            column: 0,
            filename: Rc::from("<unknown>"),
        }
    }
}

/// Sheaf error types
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Autodiff has no differentiation rule for an operation that affects the result.
    AutodiffMissingRule {
        operation: String,
        location: Option<SourceLocation>,
    },
    /// Reverse-mode AD did not provide the required output for a differentiated symbol.
    AutodiffMissingGradientOutput {
        symbol: String,
    },
    /// IO error
    Io(String),
    /// Invalid source forms cannot be retained as AST-only macro helpers.
    InvalidSourceForm {
        message: String,
        location: SourceLocation,
    },
    /// A shape argument has a non-integer source type.
    InvalidShapeDimension {
        message: String,
        location: SourceLocation,
    },
    CompilerUnsupported {
        construct: super::jit_diagnostic::Construct,
        message: String,
        location: SourceLocation,
    },
    CompilerInvariant {
        message: String,
        location: SourceLocation,
    },
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
            SheafError::Compile { message, location }
            | SheafError::CompilerUnsupported { message, location, .. }
            | SheafError::CompilerInvariant { message, location }
            | SheafError::InvalidSourceForm { message, location }
            | SheafError::InvalidShapeDimension { message, location } => {
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
            SheafError::AutodiffMissingRule {
                operation,
                location: Some(loc),
            } => {
                write!(
                    f,
                    "{}:{}: autodiff error: no differentiation rule for operation '{}'",
                    loc.filename, loc.line, operation
                )
            }
            SheafError::AutodiffMissingRule {
                operation,
                location: None,
            } => write!(
                f,
                "autodiff error: no differentiation rule for operation '{}'",
                operation
            ),
            SheafError::AutodiffMissingGradientOutput { symbol } => write!(
                f,
                "autodiff error: missing gradient output for symbol '{}'",
                symbol
            ),
            SheafError::Io(msg) => write!(f, "io error: {}", msg),
        }
    }
}

impl SheafError {
    pub fn with_location(self, loc: &SourceLocation) -> Self {
        match self {
            SheafError::Compile { message, .. } => SheafError::Compile {
                message,
                location: loc.clone(),
            },
            SheafError::CompilerUnsupported { construct, message, .. } => SheafError::CompilerUnsupported {
                construct, message, location: loc.clone(),
            },
            SheafError::CompilerInvariant { message, .. } => SheafError::CompilerInvariant {
                message, location: loc.clone(),
            },
            SheafError::InvalidSourceForm { message, .. } => SheafError::InvalidSourceForm {
                message,
                location: loc.clone(),
            },
            SheafError::InvalidShapeDimension { message, .. } => SheafError::InvalidShapeDimension {
                message,
                location: loc.clone(),
            },
            SheafError::Runtime { message, .. } => SheafError::Runtime {
                message,
                location: Some(loc.clone()),
            },
            SheafError::AutodiffMissingRule { operation, .. } => SheafError::AutodiffMissingRule {
                operation,
                location: Some(loc.clone()),
            },
            other => other,
        }
    }

    /// Keep the original location when an outer expression adds context.
    pub fn with_fallback_location(self, loc: &SourceLocation) -> Self {
        let existing = match &self {
            SheafError::Compile { location, .. }
            | SheafError::CompilerUnsupported { location, .. }
            | SheafError::CompilerInvariant { location, .. }
            | SheafError::InvalidSourceForm { location, .. }
            | SheafError::InvalidShapeDimension { location, .. } => Some(location),
            SheafError::Runtime { location, .. }
            | SheafError::AutodiffMissingRule { location, .. } => location.as_ref(),
            _ => return self,
        };
        if existing.is_none_or(|location| location.line == 0) {
            self.with_location(loc)
        } else {
            self
        }
    }

    /// Return just the message, without location or error-kind prefix.
    pub fn short_message(&self) -> Cow<'_, str> {
        match self {
            SheafError::Parse { message, .. }
            | SheafError::Compile { message, .. }
            | SheafError::Runtime { message, .. }
            | SheafError::CompilerUnsupported { message, .. }
            | SheafError::CompilerInvariant { message, .. }
            | SheafError::InvalidSourceForm { message, .. }
            | SheafError::InvalidShapeDimension { message, .. } => Cow::Borrowed(message),
            SheafError::AutodiffMissingRule { operation, .. } => Cow::Owned(format!(
                "no differentiation rule for operation '{}'",
                operation
            )),
            SheafError::AutodiffMissingGradientOutput { symbol } => Cow::Owned(format!(
                "missing gradient output for symbol '{}'",
                symbol
            )),
            SheafError::Io(msg) => Cow::Borrowed(msg),
        }
    }
}

impl std::error::Error for SheafError {}

/// Result type for Sheaf operations
pub type SheafResult<T> = Result<T, SheafError>;

#[cfg(test)]
mod tests {
    use super::{SheafError, SourceLocation};
    use crate::core::error_format::format_error;
    use std::rc::Rc;

    #[test]
    fn fallback_location_keeps_the_innermost_known_source() {
        let inner = SourceLocation::new(3, 5, Rc::from("model.shf"));
        let outer = SourceLocation::new(12, 1, Rc::from("train.shf"));
        let error = SheafError::InvalidShapeDimension {
            message: "reshape: invalid shape dimension".into(),
            location: SourceLocation::unknown(),
        }.with_fallback_location(&inner).with_fallback_location(&outer);
        assert!(matches!(error, SheafError::InvalidShapeDimension { location, .. } if location == inner));
        let error = SheafError::Runtime { message: "invalid argument".into(), location: None }
            .with_fallback_location(&outer);
        assert!(matches!(error, SheafError::Runtime { location: Some(location), .. } if location == outer));
    }

    #[test]
    fn autodiff_missing_rule_preserves_its_diagnostic_contract() {
        let location = SourceLocation::new(4, 7, Rc::from("loss.shf"));
        let error = SheafError::AutodiffMissingRule {
            operation: "flip".to_string(),
            location: None,
        }.with_location(&location);

        assert!(matches!(
            error,
            SheafError::AutodiffMissingRule {
                ref operation,
                location: Some(ref actual),
            } if operation == "flip" && actual == &location
        ));
        assert_eq!(
            error.short_message(),
            "no differentiation rule for operation 'flip'"
        );
        assert_eq!(
            error.to_string(),
            "loss.shf:4: autodiff error: no differentiation rule for operation 'flip'"
        );
        assert!(format_error(&error).contains(
            "error: no differentiation rule for operation 'flip'"
        ));
    }
}
