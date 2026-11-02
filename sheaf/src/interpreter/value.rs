// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Runtime values for the Sheaf interpreter.

use crate::core::expr::CompiledExpr;
use crate::core::error::SheafError;
pub use crate::core::dtype::ElementType as Dtype;
use crate::runtime::iree_session::DeviceBufferInner;
use ndarray::{ArrayD, IxDyn};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;
use unicode_width::UnicodeWidthStr;

pub type BuiltinFnPtr = fn(&[Value], &BTreeMap<String, Value>) -> Result<Value, crate::core::error::SheafError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionKind {
    User,
    ValueAndGrad,
    Vmap,
}

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Float(f32),
    Bool(bool),
    Nil,
    String(String),
    Keyword(String),
    Tensor { data: Arc<ArrayD<f32>>, dtype: Dtype },
    List(Vec<Value>),
    /// Fixed-size heterogeneous tuple. Output of VMFB calls and value-and-grad.
    /// Destructured with `let [[a b] expr]`.
    Tuple(Vec<Value>),
    Dict(BTreeMap<String, Value>),
    Function {
        kind: FunctionKind,
        name: Option<String>,
        params: Vec<String>,
        body: Rc<CompiledExpr>,
        closure: Vec<(String, Value)>,
    },
    BuiltinFn {
        name: String,
        func: BuiltinFnPtr,
    },
    /// Tensor data living on IREE device. Materializes to host lazily.
    DeviceBuffer(Arc<DeviceBufferInner>),
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

    /// Type prefix for REPL display. Returns e.g. "tensor f32[2x3]", "tensor i32[]".
    /// Returns None for non-tensor types (printed without prefix).
    pub fn repl_type_prefix(&self) -> Option<String> {
        match self {
            Value::Tensor { data, dtype } => {
                let shape = data.shape();
                Some(format!("tensor {}{}", dtype.name(), format_shape(shape)))
            }
            Value::DeviceBuffer(db) => {
                Some(format!("tensor {}{}", db.dtype.name(), format_shape(&db.shape)))
            }
            _ => None,
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Value::Int(n) => Some(*n as f64),
            Value::Float(f) => Some(*f as f64),
            Value::Tensor { data, .. } if data.ndim() == 0 => data.first().map(|&x| x as f64),
            Value::DeviceBuffer(db) if db.shape.is_empty() => {
                db.to_host().ok().and_then(|a| a.first().map(|&x| x as f64))
            }
            _ => None,
        }
    }

    pub fn to_f32(&self) -> Option<f32> {
        match self {
            Value::Int(n) => Some(*n as f32),
            Value::Float(f) => Some(*f),
            Value::Tensor { data, .. } if data.ndim() == 0 => data.first().copied(),
            Value::DeviceBuffer(db) if db.shape.is_empty() => {
                db.to_host().ok().and_then(|a| a.first().copied())
            }
            _ => None,
        }
    }

    pub fn to_tensor(&self) -> Option<ArrayD<f32>> {
        match self {
            Value::Int(n) => Some(ArrayD::from_elem(vec![], *n as f32)),
            Value::Float(f) => Some(ArrayD::from_elem(vec![], *f)),
            Value::Tensor { data, .. } => Some((**data).clone()),
            Value::DeviceBuffer(db) => db.to_host().ok(),
            _ => None,
        }
    }

    /// Materialize a DeviceBuffer to a host Tensor, or borrow self if already host.
    pub fn ensure_host_cow(&self) -> Result<Cow<'_, Value>, SheafError> {
        match self {
            Value::DeviceBuffer(db) => {
                let data = db.to_host()?;
                Ok(Cow::Owned(Value::Tensor { data: Arc::new(data), dtype: db.dtype }))
            }
            other => Ok(Cow::Borrowed(other)),
        }
    }

    /// Materialize a DeviceBuffer to a host Tensor, or clone self if already host.
    pub fn ensure_host(&self) -> Result<Value, SheafError> {
        match self {
            Value::DeviceBuffer(db) => {
                let data = db.to_host()?;
                Ok(Value::Tensor { data: Arc::new(data), dtype: db.dtype })
            }
            other => Ok(other.clone()),
        }
    }

    /// Shape accessor that works for both Tensor and DeviceBuffer.
    pub fn tensor_shape(&self) -> Option<&[usize]> {
        match self {
            Value::Tensor { data, .. } => Some(data.shape()),
            Value::DeviceBuffer(db) => Some(&db.shape),
            _ => None,
        }
    }

    pub(crate) fn numeric_list_elements(&self) -> Option<&[Value]> {
        match self {
            Value::List(values) if !values.is_empty()
                && values.iter().all(|value| matches!(value, Value::Int(_) | Value::Float(_))) => Some(values),
            _ => None,
        }
    }

    pub(crate) fn tensor_element(value: f32, dtype: Dtype) -> Self {
        match dtype {
            Dtype::I32 | Dtype::I64 => Value::Int(value as i64),
            Dtype::Bool => Value::Bool(value != 0.0),
            _ => Value::Float(value),
        }
    }

    pub fn tensor(data: ArrayD<f32>, dtype: Dtype) -> Self {
        Value::Tensor { data: Arc::new(data), dtype }
    }

    pub fn tensor_f32(data: ArrayD<f32>) -> Self {
        Value::Tensor { data: Arc::new(data), dtype: Dtype::F32 }
    }

    pub fn tensor_bf16(data: ArrayD<f32>) -> Self {
        Value::Tensor { data: Arc::new(data), dtype: Dtype::BF16 }
    }

    pub fn tensor_i32(data: ArrayD<f32>) -> Self {
        Value::Tensor { data: Arc::new(data), dtype: Dtype::I32 }
    }

    pub fn short_desc(&self) -> String {
        match self {
            Value::Int(n) => format!("{}", n),
            Value::Float(f) => format!("{}", f),
            Value::Bool(b) => format!("{}", b),
            Value::Nil => "nil".to_string(),
            Value::String(s) => format!("\"{}\"", s),
            Value::Keyword(k) => format!(":{}", k),
            Value::Tensor { data, .. } => {
                let shape: Vec<String> = data.shape().iter().map(|d| d.to_string()).collect();
                format!("f32[{}]", shape.join("x"))
            }
            Value::DeviceBuffer(db) => {
                let shape: Vec<String> = db.shape.iter().map(|d| d.to_string()).collect();
                format!("f32[{}]", shape.join("x"))
            }
            Value::List(items) => format!("list({})", items.len()),
            Value::Dict(map) => format!("dict({})", map.len()),
            _ => self.type_name().to_string(),
        }
    }

    pub fn contains_tensors(&self) -> bool {
        match self {
            Value::Tensor { .. } | Value::DeviceBuffer(_) => true,
            Value::List(items) | Value::Tuple(items) => items.iter().any(|v| v.contains_tensors()),
            Value::Dict(map) => map.values().any(|v| v.contains_tensors()),
            _ => false,
        }
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
            Value::Tuple(_) => "tuple",
            Value::Dict(_) => "dict",
            Value::Function { .. } => "function",
            Value::BuiltinFn { .. } => "builtin",
            Value::DeviceBuffer(_) => "tensor",
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
            Value::Tuple(items) => write!(f, "Tuple({:?})", items),
            Value::Dict(map) => write!(f, "Dict({:?})", map),
            Value::Function { params, .. } => write!(f, "Function({:?})", params),
            Value::BuiltinFn { name, .. } => write!(f, "BuiltinFn({})", name),
            Value::DeviceBuffer(db) => {
                write!(f, "DeviceBuffer({:?}, {:?})", db.shape, db.dtype)
            }
        }
    }
}

fn format_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        "[]".to_string()
    } else {
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("[{}]", dims.join("x"))
    }
}

fn format_scalar_f32(x: f32) -> String {
    if x == x.floor() && x.abs() < 1e15 {
        format!("{}.0", x as i64)
    } else {
        format!("{}", x)
    }
}

fn format_tensor_f32(x: f32) -> String {
    if x == x.floor() && x.abs() < 1e15 {
        format!("{}.", x as i64)
    } else {
        format!("{}", x)
    }
}

fn format_element(x: f32, dtype: Dtype) -> String {
    match dtype {
        Dtype::I32 => format!("{}", x as i32),
        Dtype::I64 => format!("{}", x as i64),
        Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64 => format_tensor_f32(x),
        Dtype::Bool => if x != 0.0 { "true".to_string() } else { "false".to_string() },
    }
}

/// Number of elements shown at each edge of a truncated dimension.
const EDGE_ITEMS: usize = 3;
/// Truncate a dimension in display output when it exceeds this size.
const TRUNC_THRESHOLD: usize = 10;

/// Indices to display for a dimension of size `n`. When `n` exceeds
/// `TRUNC_THRESHOLD`, only the first and last `EDGE_ITEMS` indices are
/// returned and the caller inserts an ellipsis between them.
fn trunc_indices(n: usize) -> (Vec<usize>, bool) {
    if n > TRUNC_THRESHOLD {
        let mut v: Vec<usize> = (0..EDGE_ITEMS).collect();
        v.extend((n - EDGE_ITEMS)..n);
        (v, true)
    } else {
        ((0..n).collect(), false)
    }
}

/// Format one row of a 2D tensor, inserting a column ellipsis when needed.
fn format_row(tokens: &[String], col_widths: &[usize], ellipsis_cols: bool, dtype: Dtype) -> String {
    if dtype == Dtype::Bool {
        let mut t: Vec<String> = tokens.to_vec();
        if ellipsis_cols { t.insert(EDGE_ITEMS, "...".to_string()); }
        return format!("[{}]", t.join(" "));
    }
    let mut padded: Vec<String> = tokens.iter().enumerate().map(|(c, s)| {
        format!("{:>width$}", s, width = col_widths[c])
    }).collect();
    if ellipsis_cols {
        padded.insert(EDGE_ITEMS, "...".to_string());
    }
    format!("[{}]", padded.join(" "))
}

fn format_tensor_1d(data: &[f32], dtype: Dtype) -> String {
    let n = data.len();
    if n == 0 {
        return "[]".to_string();
    }
    let (indices, ellipsis) = trunc_indices(n);
    let formatted: Vec<String> = indices.iter().map(|&i| format_element(data[i], dtype)).collect();
    if dtype == Dtype::Bool {
        let mut tokens = formatted;
        if ellipsis { tokens.insert(EDGE_ITEMS, "...".to_string()); }
        return format!("[{}]", tokens.join(" "));
    }
    let max_width = formatted.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut padded: Vec<String> = formatted.iter().map(|s| {
        format!("{:>width$}", s, width = max_width)
    }).collect();
    if ellipsis {
        padded.insert(EDGE_ITEMS, "...".to_string());
    }
    format!("[{}]", padded.join(" "))
}

fn format_tensor_nd(arr: &ArrayD<f32>, dtype: Dtype) -> String {
    let shape = arr.shape();
    match shape.len() {
        0 => {
            let x = arr.first().copied().unwrap_or(0.0);
            match dtype {
                Dtype::I32 => format!("{}", x as i32),
                Dtype::I64 => format!("{}", x as i64),
                Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64 => format_tensor_f32(x),
                Dtype::Bool => if x != 0.0 { "true".to_string() } else { "false".to_string() },
            }
        }
        1 => format_tensor_1d(arr.as_slice().unwrap(), dtype),
        2 => format_tensor_2d(arr, dtype),
        _ => {
            let n = shape[0];
            let (indices, ellipsis) = trunc_indices(n);
            let rows: Vec<String> = indices.iter().map(|&i| {
                let sub = arr.index_axis(ndarray::Axis(0), i).to_owned();
                format_tensor_nd(&sub, dtype)
            }).collect();
            let mut all = rows;
            if ellipsis {
                all.insert(EDGE_ITEMS, "...".to_string());
            }
            format!("[{}]", all.join("\n "))
        }
    }
}

fn format_tensor_2d(arr: &ArrayD<f32>, dtype: Dtype) -> String {
    let shape = arr.shape();
    let (nrows, ncols) = (shape[0], shape[1]);
    let (row_idx, ellipsis_rows) = trunc_indices(nrows);
    let (col_idx, ellipsis_cols) = trunc_indices(ncols);

    let all_formatted: Vec<Vec<String>> = row_idx.iter().map(|&r| {
        col_idx.iter().map(|&c| format_element(arr[IxDyn(&[r, c])], dtype)).collect()
    }).collect();
    // When an ellipsis row is shown, every column holds "...", so pad to >= 3.
    let min_w = if ellipsis_rows { 3 } else { 0 };
    let col_widths: Vec<usize> = (0..col_idx.len()).map(|c| {
        all_formatted.iter().map(|row| row[c].len()).max().unwrap_or(0).max(min_w)
    }).collect();

    let mut rows: Vec<String> = all_formatted.iter()
        .map(|row| format_row(row, &col_widths, ellipsis_cols, dtype))
        .collect();
    if ellipsis_rows {
        let ellipsis_tokens: Vec<String> = col_idx.iter().map(|_| "...".to_string()).collect();
        rows.insert(EDGE_ITEMS, format_row(&ellipsis_tokens, &col_widths, ellipsis_cols, dtype));
    }
    format!("[{}]", rows.join("\n "))
}

fn indent_continuation_lines(text: &str, columns: usize) -> String {
    if !text.contains('\n') || columns == 0 {
        return text.to_string();
    }
    let indentation = " ".repeat(columns);
    text.replace('\n', &format!("\n{}", indentation))
}

fn format_sequence(items: &[Value], open: char, close: char, quote_strings: bool) -> String {
    let formatted: Vec<String> = items.iter().map(|value| {
        if quote_strings && let Value::String(text) = value {
            format!("'{}'", text)
        } else {
            format_value(value)
        }
    }).collect();

    if !formatted.iter().any(|item| item.contains('\n')) {
        return format!("{}{}{}", open, formatted.join(", "), close);
    }

    let mut result = String::new();
    result.push(open);
    for (index, item) in formatted.iter().enumerate() {
        if index > 0 {
            result.push_str(",\n ");
        }
        result.push_str(&indent_continuation_lines(item, 1));
    }
    result.push(close);
    result
}

fn format_dict(map: &BTreeMap<String, Value>) -> String {
    let entries: Vec<(&str, String)> = map.iter()
        .map(|(key, value)| (key.as_str(), format_value(value)))
        .collect();

    if !entries.iter().any(|(_, value)| value.contains('\n')) {
        let pairs: Vec<String> = entries.iter()
            .map(|(key, value)| format!(":{} {}", key, value))
            .collect();
        return format!("{{{}}}", pairs.join(", "));
    }

    let mut result = String::from("{");
    for (index, (key, value)) in entries.iter().enumerate() {
        if index > 0 {
            result.push_str(",\n ");
        }
        let prefix = format!(":{} ", key);
        result.push_str(&prefix);
        result.push_str(&indent_continuation_lines(value, 1 + UnicodeWidthStr::width(prefix.as_str())));
    }
    result.push('}');
    result
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Int(n) => format!("{}", n),
        Value::Float(x) => format_scalar_f32(*x),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Nil => "nil".to_string(),
        Value::String(s) => s.clone(),
        Value::Keyword(k) => format!(":{}", k),
        Value::Tensor { data, dtype } => format_tensor_nd(data, *dtype),
        Value::List(items) => format_sequence(items, '[', ']', true),
        Value::Tuple(items) => format_sequence(items, '(', ')', false),
        Value::Dict(map) => format_dict(map),
        Value::Function { name: Some(n), .. } => format!("<fn:{}>", n),
        Value::Function { name: None, .. } => "<function>".to_string(),
        Value::BuiltinFn { name, .. } => format!("<builtin:{}>", name),
        Value::DeviceBuffer(db) => match db.to_host() {
            Ok(data) => format_tensor_nd(&data, db.dtype),
            Err(_) => {
                let dims: Vec<String> = db.shape.iter().map(|d| d.to_string()).collect();
                format!("<tensor {:?} [{}] on device>", db.dtype, dims.join("x"))
            }
        },
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_value(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn tensor_f32(shape: Vec<usize>, fill: f32) -> Value {
        Value::Tensor {
            data: Arc::new(ArrayD::from_elem(shape, fill)),
            dtype: Dtype::F32,
        }
    }

    #[test]
    fn short_1d_not_truncated() {
        let t = tensor_f32(vec![5], 1.0);
        let s = format!("{}", t);
        assert!(!s.contains("..."), "small tensor should not truncate: {s}");
    }

    #[test]
    fn long_1d_truncated() {
        let t = tensor_f32(vec![50], 1.0);
        let s = format!("{}", t);
        assert!(s.contains("..."), "expected ellipsis: {s}");
        // Only 6 elements shown (3 head + 3 tail), not all 50.
        let count = s.matches("1.").count();
        assert_eq!(count, 6, "expected 6 elements, got: {s}");
    }

    #[test]
    fn large_2d_truncated_both_axes() {
        let t = tensor_f32(vec![100, 100], 1.0);
        let s = format!("{}", t);
        assert!(s.contains("..."), "expected ellipsis: {s}");
        // 3 head rows + 1 ellipsis row + 3 tail rows = 7 lines.
        assert!(s.lines().count() <= 8, "too many lines: {s}");
    }

    #[test]
    fn tall_2d_truncates_rows_only() {
        let t = tensor_f32(vec![100, 2], 1.0);
        let s = format!("{}", t);
        // Columns fit (2 <= threshold), so no column ellipsis in a data row.
        let first = s.lines().next().unwrap();
        assert!(!first.contains("..."), "no col ellipsis expected: {first}");
        // But rows are truncated.
        assert!(s.lines().count() <= 8, "too many lines: {s}");
    }

    #[test]
    fn list_aligns_multiline_elements() {
        let value = Value::List(vec![
            tensor_f32(vec![2, 2], 1.0),
            tensor_f32(vec![2, 2], 2.0),
        ]);

        assert_eq!(
            value.to_string(),
            "[[[1. 1.]\n  [1. 1.]],\n [[2. 2.]\n  [2. 2.]]]",
        );
    }

    #[test]
    fn tuple_aligns_multiline_elements() {
        let value = Value::Tuple(vec![
            tensor_f32(vec![2, 2], 1.0),
            tensor_f32(vec![2, 2], 2.0),
        ]);

        assert_eq!(
            value.to_string(),
            "([[1. 1.]\n  [1. 1.]],\n [[2. 2.]\n  [2. 2.]])",
        );
    }

    #[test]
    fn dict_aligns_values_beneath_their_keys() {
        let mut map = BTreeMap::new();
        map.insert("a".to_string(), tensor_f32(vec![2, 2], 1.0));
        map.insert("b".to_string(), tensor_f32(vec![2, 2], 2.0));
        let value = Value::Dict(map);

        assert_eq!(
            value.to_string(),
            "{:a [[1. 1.]\n     [1. 1.]],\n :b [[2. 2.]\n     [2. 2.]]}",
        );
    }
}
