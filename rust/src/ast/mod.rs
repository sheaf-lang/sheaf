// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Abstract Syntax Tree types for Sheaf

pub use crate::core::error::SourceLocation;
use std::fmt;

/// Sheaf value - the primary AST node type
#[derive(Debug, Clone, PartialEq)]
pub enum SheafValue {
    /// Symbol: foo, defn, +, etc.
    Symbol(String, SourceLocation),

    /// Keyword: :foo, :name, etc.
    Keyword(String, SourceLocation),

    /// Integer literal
    Integer(i64, SourceLocation),

    /// Float literal
    Float(f64, SourceLocation),

    /// String literal
    String(String, SourceLocation),

    /// Boolean: true/false
    Boolean(bool, SourceLocation),

    /// Nil
    Nil(SourceLocation),

    /// List: (foo bar baz)
    List(Vec<SheafValue>, SourceLocation),

    /// Vector: [1 2 3]
    Vector(Vec<SheafValue>, SourceLocation),

    /// Dict: {:a 1 :b 2}
    Dict(Vec<(SheafValue, SheafValue)>, SourceLocation),

    /// Quote: 'expr
    Quote(Box<SheafValue>, SourceLocation),

    /// Quasiquote: `expr
    Quasiquote(Box<SheafValue>, SourceLocation),

    /// Unquote: ~expr
    Unquote(Box<SheafValue>, SourceLocation),

    /// Unquote-splicing: ~@expr
    UnquoteSplicing(Box<SheafValue>, SourceLocation),
}

impl SheafValue {
    /// Get the source location of this value
    pub fn location(&self) -> &SourceLocation {
        match self {
            SheafValue::Symbol(_, loc) => loc,
            SheafValue::Keyword(_, loc) => loc,
            SheafValue::Integer(_, loc) => loc,
            SheafValue::Float(_, loc) => loc,
            SheafValue::String(_, loc) => loc,
            SheafValue::Boolean(_, loc) => loc,
            SheafValue::Nil(loc) => loc,
            SheafValue::List(_, loc) => loc,
            SheafValue::Vector(_, loc) => loc,
            SheafValue::Dict(_, loc) => loc,
            SheafValue::Quote(_, loc) => loc,
            SheafValue::Quasiquote(_, loc) => loc,
            SheafValue::Unquote(_, loc) => loc,
            SheafValue::UnquoteSplicing(_, loc) => loc,
        }
    }

    /// Check if this is a symbol with a specific name
    pub fn is_symbol(&self, name: &str) -> bool {
        matches!(self, SheafValue::Symbol(s, _) if s == name)
    }

    /// Check if this is a list
    pub fn is_list(&self) -> bool {
        matches!(self, SheafValue::List(_, _))
    }

    /// Check if this is a vector
    pub fn is_vector(&self) -> bool {
        matches!(self, SheafValue::Vector(_, _))
    }

    /// Try to extract a symbol name
    pub fn as_symbol(&self) -> Option<&str> {
        match self {
            SheafValue::Symbol(s, _) => Some(s),
            _ => None,
        }
    }

    /// Try to extract list elements
    pub fn as_list(&self) -> Option<&[SheafValue]> {
        match self {
            SheafValue::List(elems, _) => Some(elems),
            _ => None,
        }
    }

    /// Try to extract vector elements
    pub fn as_vector(&self) -> Option<&[SheafValue]> {
        match self {
            SheafValue::Vector(elems, _) => Some(elems),
            _ => None,
        }
    }
}

impl fmt::Display for SheafValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SheafValue::Symbol(s, _) => write!(f, "{}", s),
            SheafValue::Keyword(k, _) => write!(f, ":{}", k),
            SheafValue::Integer(n, _) => write!(f, "{}", n),
            SheafValue::Float(x, _) => write!(f, "{}", x),
            SheafValue::String(s, _) => write!(f, "\"{}\"", s),
            SheafValue::Boolean(b, _) => write!(f, "{}", b),
            SheafValue::Nil(_) => write!(f, "nil"),
            SheafValue::List(elems, _) => {
                write!(f, "(")?;
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, ")")
            }
            SheafValue::Vector(elems, _) => {
                write!(f, "[")?;
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            SheafValue::Dict(pairs, _) => {
                write!(f, "{{")?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{} {}", k, v)?;
                }
                write!(f, "}}")
            }
            SheafValue::Quote(expr, _) => write!(f, "'{}", expr),
            SheafValue::Quasiquote(expr, _) => write!(f, "`{}", expr),
            SheafValue::Unquote(expr, _) => write!(f, "~{}", expr),
            SheafValue::UnquoteSplicing(expr, _) => write!(f, "~@{}", expr),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol() {
        let loc = SourceLocation::unknown();
        let sym = SheafValue::Symbol("foo".to_string(), loc);
        assert!(sym.is_symbol("foo"));
        assert!(!sym.is_symbol("bar"));
        assert_eq!(sym.as_symbol(), Some("foo"));
    }

    #[test]
    fn test_list() {
        let loc = SourceLocation::unknown();
        let list = SheafValue::List(
            vec![
                SheafValue::Symbol("+".to_string(), loc.clone()),
                SheafValue::Integer(1, loc.clone()),
                SheafValue::Integer(2, loc.clone()),
            ],
            loc,
        );
        assert!(list.is_list());
        assert_eq!(list.as_list().map(|l| l.len()), Some(3));
    }

    #[test]
    fn test_display() {
        let loc = SourceLocation::unknown();
        let expr = SheafValue::List(
            vec![
                SheafValue::Symbol("+".to_string(), loc.clone()),
                SheafValue::Integer(1, loc.clone()),
                SheafValue::Integer(2, loc.clone()),
            ],
            loc,
        );
        assert_eq!(format!("{}", expr), "(+ 1 2)");
    }
}
