// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Runtime values for the Sheaf interpreter.

use crate::core::compiler::CompiledExpr;
use ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dtype {
    F32,
    I32,
    Bool,
}

pub type BuiltinFnPtr = fn(&[Value], &BTreeMap<String, Value>) -> Result<Value, crate::core::error::SheafError>;

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Nil,
    String(String),
    Keyword(String),
    Tensor { data: ArrayD<f64>, dtype: Dtype },
    List(Vec<Value>),
    Dict(BTreeMap<String, Value>),
    Function {
        params: Vec<String>,
        body: CompiledExpr,
        closure: Vec<(String, Value)>,
    },
    BuiltinFn {
        name: String,
        func: BuiltinFnPtr,
    },
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Nil => false,
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            _ => true,
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Value::Int(n) => Some(*n as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn to_tensor(&self) -> Option<ArrayD<f64>> {
        match self {
            Value::Int(n) => Some(ArrayD::from_elem(vec![], *n as f64)),
            Value::Float(f) => Some(ArrayD::from_elem(vec![], *f)),
            Value::Tensor { data, .. } => Some(data.clone()),
            _ => None,
        }
    }

    pub fn tensor_f32(data: ArrayD<f64>) -> Self {
        Value::Tensor { data, dtype: Dtype::F32 }
    }

    pub fn tensor_i32(data: ArrayD<f64>) -> Self {
        Value::Tensor { data, dtype: Dtype::I32 }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Bool(_) => "bool",
            Value::Nil => "nil",
            Value::String(_) => "string",
            Value::Keyword(_) => "keyword",
            Value::Tensor { .. } => "tensor",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Function { .. } => "function",
            Value::BuiltinFn { .. } => "builtin",
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "Int({})", n),
            Value::Float(x) => write!(f, "Float({})", x),
            Value::Bool(b) => write!(f, "Bool({})", b),
            Value::Nil => write!(f, "Nil"),
            Value::String(s) => write!(f, "String({:?})", s),
            Value::Keyword(k) => write!(f, "Keyword(:{})", k),
            Value::Tensor { data, dtype } => write!(f, "Tensor({:?}, {:?})", data, dtype),
            Value::List(items) => write!(f, "List({:?})", items),
            Value::Dict(map) => write!(f, "Dict({:?})", map),
            Value::Function { params, .. } => write!(f, "Function({:?})", params),
            Value::BuiltinFn { name, .. } => write!(f, "BuiltinFn({})", name),
        }
    }
}

fn format_scalar_f64(x: f64) -> String {
    if x == x.floor() && x.abs() < 1e15 {
        format!("{}.0", x as i64)
    } else {
        format!("{}", x)
    }
}

fn format_tensor_f64(x: f64) -> String {
    if x == x.floor() && x.abs() < 1e15 {
        format!("{}.", x as i64)
    } else {
        format!("{}", x)
    }
}

fn format_element(x: f64, dtype: Dtype) -> String {
    match dtype {
        Dtype::I32 => format!("{}", x as i64),
        Dtype::F32 => format_tensor_f64(x),
        Dtype::Bool => if x != 0.0 { " True".to_string() } else { "False".to_string() },
    }
}

fn format_tensor_1d(data: &[f64], dtype: Dtype) -> String {
    let formatted: Vec<String> = data.iter().map(|&x| format_element(x, dtype)).collect();
    let max_width = formatted.iter().map(|s| s.len()).max().unwrap_or(0);
    let padded: Vec<String> = formatted.iter().map(|s| {
        format!("{:>width$}", s, width = max_width)
    }).collect();
    format!("[{}]", padded.join(" "))
}

fn format_tensor_nd(arr: &ArrayD<f64>, dtype: Dtype) -> String {
    let shape = arr.shape();
    match shape.len() {
        0 => {
            let x = arr.first().copied().unwrap_or(0.0);
            match dtype {
                Dtype::I32 => format!("{}", x as i64),
                Dtype::F32 => format_tensor_f64(x),
                Dtype::Bool => if x != 0.0 { "True".to_string() } else { "False".to_string() },
            }
        }
        1 => format_tensor_1d(arr.as_slice().unwrap(), dtype),
        2 => format_tensor_2d(arr, dtype),
        _ => {
            let rows: Vec<String> = (0..shape[0]).map(|i| {
                let sub = arr.index_axis(ndarray::Axis(0), i).to_owned();
                format_tensor_nd(&sub, dtype)
            }).collect();
            format!("[{}]", rows.join("\n "))
        }
    }
}

fn format_tensor_2d(arr: &ArrayD<f64>, dtype: Dtype) -> String {
    let shape = arr.shape();
    let (nrows, ncols) = (shape[0], shape[1]);
    let all_formatted: Vec<Vec<String>> = (0..nrows).map(|r| {
        (0..ncols).map(|c| format_element(arr[IxDyn(&[r, c])], dtype)).collect()
    }).collect();
    let col_widths: Vec<usize> = (0..ncols).map(|c| {
        all_formatted.iter().map(|row| row[c].len()).max().unwrap_or(0)
    }).collect();
    let rows: Vec<String> = all_formatted.iter().map(|row| {
        let padded: Vec<String> = row.iter().enumerate().map(|(c, s)| {
            format!("{:>width$}", s, width = col_widths[c])
        }).collect();
        format!("[{}]", padded.join(" "))
    }).collect();
    format!("[{}]", rows.join("\n "))
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(x) => write!(f, "{}", format_scalar_f64(*x)),
            Value::Bool(true) => write!(f, "True"),
            Value::Bool(false) => write!(f, "False"),
            Value::Nil => write!(f, "nil"),
            Value::String(s) => write!(f, "{}", s),
            Value::Keyword(k) => write!(f, ":{}", k),
            Value::Tensor { data, dtype } => write!(f, "{}", format_tensor_nd(data, *dtype)),
            Value::List(items) => {
                let formatted: Vec<String> = items.iter().map(|v| {
                    match v {
                        Value::String(s) => format!("'{}'", s),
                        _ => format!("{}", v),
                    }
                }).collect();
                write!(f, "({})", formatted.join(", "))
            }
            Value::Dict(map) => {
                let pairs: Vec<String> = map.iter().map(|(k, v)| {
                    format!("'{}': {}", k, v)
                }).collect();
                write!(f, "{{{}}}", pairs.join(", "))
            }
            Value::Function { .. } => write!(f, "<function>"),
            Value::BuiltinFn { name, .. } => write!(f, "<builtin:{}>", name),
        }
    }
}
