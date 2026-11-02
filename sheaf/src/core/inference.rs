// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Type inference.

use crate::lowering::stablehlo::StableHLOType;
use crate::core::dtype::{
    DtypeOperand, ElementType, resolve_arithmetic_dtype,
};
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::error::{SheafError, SheafResult};

/// Layout of a flattened value.
#[derive(Debug, Clone, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ValueLayout {
    Dict(Vec<(String, ValueLayout)>),
    List(Vec<ValueLayout>),
    Tuple(Vec<ValueLayout>),
    Leaf,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FunctionSignature {
    pub param_types: Vec<StableHLOType>,
    pub return_type: StableHLOType,
    pub return_dict_keys: Option<Vec<String>>,
    #[serde(default)]
    pub return_layout: Option<ValueLayout>,
    /// Layouts of structured arguments.
    pub arg_type_layouts: Vec<(StableHLOType, ValueLayout)>,
    #[serde(with = "captured_scalar_map")]
    pub captured_scalars: std::collections::HashMap<(String, Vec<usize>), f64>,
}

mod captured_scalar_map {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    type CapturedScalars = HashMap<(String, Vec<usize>), f64>;

    pub fn serialize<S>(value: &CapturedScalars, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut entries: Vec<(&(String, Vec<usize>), &f64)> = value.iter().collect();
        entries.sort_by_key(|(key, _)| *key);
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<CapturedScalars, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries = Vec::<((String, Vec<usize>), f64)>::deserialize(deserializer)?;
        Ok(entries.into_iter().collect())
    }
}

impl ValueLayout {
    #[cfg(not(sheaf_frontend))]
    pub fn from_value(val: &crate::interpreter::value::Value) -> Self {
        use crate::interpreter::value::Value;
        match val {
            Value::Dict(map) => {
                let mut entries: Vec<(String, ValueLayout)> = map
                    .iter()
                    .map(|(k, v)| (k.clone(), ValueLayout::from_value(v)))
                    .collect();
                entries.sort_by(|(a, _), (b, _)| a.cmp(b));
                ValueLayout::Dict(entries)
            }
            Value::List(items) => {
                ValueLayout::List(items.iter().map(ValueLayout::from_value).collect())
            }
            Value::Tuple(items) => {
                ValueLayout::Tuple(items.iter().map(ValueLayout::from_value).collect())
            }
            _ => ValueLayout::Leaf,
        }
    }

    /// Restores a flattened value.
    #[cfg(not(sheaf_frontend))]
    pub fn reconstruct(&self, val: crate::interpreter::value::Value) -> crate::interpreter::value::Value {
        use crate::interpreter::value::Value;
        match (self, val) {
            (ValueLayout::Dict(entries), Value::Tuple(elems)) if entries.len() == elems.len() => {
                let map = entries
                    .iter()
                    .zip(elems)
                    .map(|((key, sub_layout), elem)| (key.clone(), sub_layout.reconstruct(elem)))
                    .collect();
                Value::Dict(map)
            }
            (ValueLayout::Dict(entries), Value::Dict(map)) if entries.len() == map.len() => {
                let mut result = std::collections::BTreeMap::new();
                for (key, sub_layout) in entries {
                    if let Some(sub_val) = map.get(key) {
                        result.insert(key.clone(), sub_layout.reconstruct(sub_val.clone()));
                    }
                }
                Value::Dict(result)
            }
            (ValueLayout::List(items), Value::Tuple(elems)) if items.len() == elems.len() => {
                let list = items
                    .iter()
                    .zip(elems)
                    .map(|(sub_layout, elem)| sub_layout.reconstruct(elem))
                    .collect();
                Value::List(list)
            }
            (ValueLayout::Tuple(items), Value::Tuple(elems)) if items.len() == elems.len() => {
                let tuple = items
                    .iter()
                    .zip(elems)
                    .map(|(sub_layout, elem)| sub_layout.reconstruct(elem))
                    .collect();
                Value::Tuple(tuple)
            }
            (_, v) => v,
        }
    }
}

/// Restores a flattened JIT result.
#[cfg(not(sheaf_frontend))]
pub fn reconstruct_jit_result(
    val: crate::interpreter::value::Value,
    ty: &StableHLOType,
    layouts: &[(StableHLOType, ValueLayout)],
) -> crate::interpreter::value::Value {
    use crate::interpreter::value::Value;
    if let Some((_, layout)) = layouts.iter().find(|(t, _)| t == ty) {
        return layout.reconstruct(val);
    }
    match (ty, val) {
        (StableHLOType::Tuple(elem_tys, _), Value::Tuple(elems)) if elem_tys.len() == elems.len() => {
            let reconstructed: Vec<Value> = elem_tys
                .iter()
                .zip(elems)
                .map(|(et, ev)| reconstruct_jit_result(ev, et, layouts))
                .collect();
            Value::Tuple(reconstructed)
        }
        (StableHLOType::Tuple(elem_tys, _), Value::Dict(map)) if elem_tys.len() == map.len() => {
            let mut result = std::collections::BTreeMap::new();
            for (i, (key, sub_val)) in map.into_iter().enumerate() {
                let sub_ty = elem_tys.get(i).cloned().unwrap_or(StableHLOType::scalar_f32());
                result.insert(key, reconstruct_jit_result(sub_val, &sub_ty, layouts));
            }
            Value::Dict(result)
        }
        (_, v) => v,
    }
}

pub fn infer_function_signature(
    params: &[String],
    body_expr: &CompiledExpr,
) -> SheafResult<FunctionSignature> {
    infer_function_signature_with_known(params, body_expr, &[])
}

/// Infers a signature with known parameter types.
pub fn infer_function_signature_with_known(
    params: &[String],
    body_expr: &CompiledExpr,
    known: &[(String, StableHLOType)],
) -> SheafResult<FunctionSignature> {
    let mut symbol_types = std::collections::HashMap::new();

    for (name, ty) in known {
        symbol_types.insert(name.clone(), ty.clone());
    }

    infer_symbol_types(body_expr, &mut symbol_types)?;

    let param_types: Vec<StableHLOType> = params
        .iter()
        .map(|p| {
            symbol_types
                .get(p)
                .cloned()
                .unwrap_or(StableHLOType::scalar_f32())
        })
        .collect();

    let return_type = infer_type_with_context(body_expr, &symbol_types)?;
    let return_dict_keys = find_return_dict_keys(body_expr);

    Ok(FunctionSignature {
        param_types,
        return_type,
        return_dict_keys,
        return_layout: None,
        arg_type_layouts: vec![],
        captured_scalars: std::collections::HashMap::new(),
    })
}

pub(crate) fn find_return_dict_keys(expr: &CompiledExpr) -> Option<Vec<String>> {
    match expr {
        CompiledExpr::Dict(pairs) => {
            let mut keys: Vec<String> = pairs
                .iter()
                .map(|(key, _)| match key {
                    CompiledExpr::Keyword(name) | CompiledExpr::String(name) => {
                        Some(name.clone())
                    }
                    _ => None,
                })
                .collect::<Option<_>>()?;
            keys.sort();
            Some(keys)
        }
        CompiledExpr::Let { body, .. } => find_return_dict_keys(body),
        CompiledExpr::Do(exprs) => exprs.last().and_then(find_return_dict_keys),
        _ => None,
    }
}

fn resolve_tuple_index(ty: &StableHLOType, indices: &[usize]) -> Option<StableHLOType> {
    let mut current = ty.clone();
    for &idx in indices {
        match current {
            StableHLOType::Tuple(elems, _) => {
                current = elems.into_iter().nth(idx)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

fn infer_symbol_types(
    expr: &CompiledExpr,
    symbol_types: &mut std::collections::HashMap<String, StableHLOType>,
) -> SheafResult<()> {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } => {
            if name == "@" && args.len() == 2 {
                let rhs_ty = infer_type_with_context(&args[1], symbol_types).ok();
                let lhs_ty = infer_type_with_context(&args[0], symbol_types).ok();

                if let CompiledExpr::Symbol(sym) = &args[0] {
                    let ty = if let Some(rhs) = &rhs_ty {
                        let rhs_shape = rhs.shape();
                        if rhs_shape.len() == 2 {
                            StableHLOType::f32_tensor(vec![1, rhs_shape[0]])
                        } else {
                            StableHLOType::f32_tensor(vec![1, 1])
                        }
                    } else {
                        StableHLOType::f32_tensor(vec![1, 1])
                    };
                    symbol_types.entry(sym.clone()).or_insert(ty);
                }
                if let CompiledExpr::Symbol(sym) = &args[1] {
                    let ty = if let Some(lhs) = &lhs_ty {
                        let lhs_shape = lhs.shape();
                        if lhs_shape.len() == 2 {
                            StableHLOType::f32_tensor(vec![lhs_shape[1], 1])
                        } else {
                            StableHLOType::f32_tensor(vec![1, 1])
                        }
                    } else {
                        StableHLOType::f32_tensor(vec![1, 1])
                    };
                    symbol_types.entry(sym.clone()).or_insert(ty);
                }
            }

            if (name == "sum" || name == "mean")
                && !args.is_empty()
                && let CompiledExpr::Symbol(sym) = &args[0]
            {
                symbol_types
                    .entry(sym.clone())
                    .or_insert(StableHLOType::f32_tensor(vec![1, 1]));
            }

            for arg in args {
                infer_symbol_types(arg, symbol_types)?;
            }
        }

        CompiledExpr::Let { bindings, body } => {
            for (_, value) in bindings {
                infer_symbol_types(value, symbol_types)?;
            }
            infer_symbol_types(body, symbol_types)?;
        }

        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            infer_symbol_types(condition, symbol_types)?;
            infer_symbol_types(then_branch, symbol_types)?;
            if let Some(else_expr) = else_branch {
                infer_symbol_types(else_expr, symbol_types)?;
            }
        }

        CompiledExpr::Do(exprs) => {
            for e in exprs {
                infer_symbol_types(e, symbol_types)?;
            }
        }

        CompiledExpr::Vector(elems) => {
            for e in elems {
                infer_symbol_types(e, symbol_types)?;
            }
        }

        CompiledExpr::Symbol(name) => {
            symbol_types
                .entry(name.clone())
                .or_insert(StableHLOType::scalar_f32());
        }

        _ => {}
    }

    Ok(())
}

pub(crate) fn infer_type_with_context(
    expr: &CompiledExpr,
    symbol_types: &std::collections::HashMap<String, StableHLOType>,
) -> SheafResult<StableHLOType> {
    infer_type_with_weak_scalars(
        expr,
        symbol_types,
        &std::collections::HashSet::new(),
    )
}

fn infer_type_with_weak_scalars(
    expr: &CompiledExpr,
    symbol_types: &std::collections::HashMap<String, StableHLOType>,
    weak_scalars: &std::collections::HashSet<String>,
) -> SheafResult<StableHLOType> {
    match expr {
        CompiledExpr::Integer(_) | CompiledExpr::Float(_) | CompiledExpr::Boolean(_) => {
            Ok(StableHLOType::scalar_f32())
        }
        CompiledExpr::Vector(elements) => {
            Ok(StableHLOType::f32_tensor(infer_vector_shape(elements)))
        }
        CompiledExpr::Symbol(name) => Ok(symbol_types
            .get(name)
            .cloned()
            .unwrap_or(StableHLOType::scalar_f32())),
        CompiledExpr::GetTupleElement { param, indices } => Ok(symbol_types
            .get(param)
            .and_then(|ty| resolve_tuple_index(ty, indices))
            .unwrap_or(StableHLOType::scalar_f32())),
        CompiledExpr::FunctionCall { name, args, .. } => {
            infer_function_call_type(name, args, symbol_types, weak_scalars)
        }
        CompiledExpr::Let { bindings, body } => {
            let mut extended = symbol_types.clone();
            let mut extended_weak = weak_scalars.clone();
            for (pattern, value) in bindings {
                let ty = infer_type_with_weak_scalars(value, &extended, &extended_weak)?;
                if let BindingPattern::Simple(name) = pattern {
                    extended.insert(name.clone(), ty);
                    if is_weak_scalar(value, &extended_weak) {
                        extended_weak.insert(name.clone());
                    } else {
                        extended_weak.remove(name);
                    }
                }
            }
            infer_type_with_weak_scalars(body, &extended, &extended_weak)
        }
        CompiledExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let then_ty = infer_type_with_weak_scalars(
                then_branch,
                symbol_types,
                weak_scalars,
            )?;
            if let Some(else_expr) = else_branch {
                infer_type_with_weak_scalars(else_expr, symbol_types, weak_scalars)?;
            }
            Ok(then_ty)
        }
        CompiledExpr::Do(exprs) => exprs
            .last()
            .map(|expr| infer_type_with_weak_scalars(expr, symbol_types, weak_scalars))
            .unwrap_or_else(|| Ok(StableHLOType::scalar_f32())),
        CompiledExpr::Dict(pairs) => {
            let mut sorted: Vec<_> = pairs.iter().collect();
            sorted.sort_by(|(lhs, _), (rhs, _)| {
                let lhs = match lhs {
                    CompiledExpr::Keyword(key) => key.clone(),
                    _ => format!("{:?}", lhs),
                };
                let rhs = match rhs {
                    CompiledExpr::Keyword(key) => key.clone(),
                    _ => format!("{:?}", rhs),
                };
                lhs.cmp(&rhs)
            });
            let keys = sorted
                .iter()
                .filter_map(|(key, _)| match key {
                    CompiledExpr::Keyword(key) => Some(key.clone()),
                    _ => None,
                })
                .collect();
            let elements = sorted
                .iter()
                .map(|(_, value)| {
                    infer_type_with_weak_scalars(value, symbol_types, weak_scalars)
                })
                .collect::<SheafResult<Vec<_>>>()?;
            Ok(StableHLOType::Tuple(elements, Some(keys)))
        }
        _ => Ok(StableHLOType::scalar_f32()),
    }
}

/// Returns the shape of a vector literal.
fn infer_vector_shape(elements: &[CompiledExpr]) -> Vec<i64> {
    if elements.is_empty() {
        return vec![0];
    }
    match &elements[0] {
        CompiledExpr::Vector(inner) => {
            let mut shape = vec![elements.len() as i64];
            shape.extend(infer_vector_shape(inner));
            shape
        }
        _ => vec![elements.len() as i64],
    }
}

fn infer_function_call_type(
    name: &str,
    args: &[CompiledExpr],
    symbol_types: &std::collections::HashMap<String, StableHLOType>,
    weak_scalars: &std::collections::HashSet<String>,
) -> SheafResult<StableHLOType> {
    match name {
        "+" | "-" | "*" | "/" | "**" | "//" | "%" | "mod" | "minimum" | "maximum" => {
            infer_arithmetic_type(name, args, symbol_types, weak_scalars)
        },
        "min" | "max"
            if args.len() >= 2
                && !matches!(args.get(1), Some(CompiledExpr::Keyword(_))) =>
        {
            infer_arithmetic_type(name, args, symbol_types, weak_scalars)
        },
        "@" => infer_matmul_type(args, symbol_types, weak_scalars),
        "__cast-like" if args.len() == 2 => {
            let source = infer_type_with_weak_scalars(&args[0], symbol_types, weak_scalars)?;
            let target = infer_type_with_weak_scalars(&args[1], symbol_types, weak_scalars)?;
            let dtype = target.element_type().ok_or_else(|| SheafError::Compile {
                message: "__cast-like expects a tensor target".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })?;
            source.with_element_type(dtype).ok_or_else(|| SheafError::Compile {
                message: "__cast-like expects a tensor source".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        },
        "where" if args.len() == 3 => infer_arithmetic_type(
            "where",
            &args[1..],
            symbol_types,
            weak_scalars,
        ),
        "relu" | "gelu" | "sigmoid" | "tanh" | "sqrt" | "exp" | "log" | "softmax"
        | "abs" | "sin" | "cos" | "tan" | "round" | "ceil" | "floor"
        | "stop-gradient" => args
            .first()
            .map(|arg| infer_type_with_weak_scalars(arg, symbol_types, weak_scalars))
            .unwrap_or_else(|| Ok(StableHLOType::scalar_f32())),
        "zeros" => Ok(args
            .first()
            .and_then(literal_shape)
            .map(StableHLOType::f32_tensor)
            .unwrap_or_else(StableHLOType::scalar_f32)),
        "random-normal" => Ok(args
            .get(1)
            .and_then(literal_shape)
            .map(StableHLOType::f32_tensor)
            .unwrap_or_else(StableHLOType::scalar_f32)),
        "sum" | "mean" | "var" | "product" | "min" | "max" => {
            infer_reduction_type(args, symbol_types, weak_scalars)
        },
        _ => Ok(StableHLOType::scalar_f32()),
    }
}

fn infer_matmul_type(
    args: &[CompiledExpr],
    symbol_types: &std::collections::HashMap<String, StableHLOType>,
    weak_scalars: &std::collections::HashSet<String>,
) -> SheafResult<StableHLOType> {
    if args.len() != 2 {
        return Ok(StableHLOType::scalar_f32());
    }
    let lhs = infer_type_with_weak_scalars(&args[0], symbol_types, weak_scalars)?;
    let rhs = infer_type_with_weak_scalars(&args[1], symbol_types, weak_scalars)?;
    let dtype = resolve_arithmetic_dtype(
        if is_weak_scalar(&args[0], weak_scalars) {
            DtypeOperand::weak(lhs.element_type().unwrap_or(ElementType::F32))
        } else {
            DtypeOperand::strong(lhs.element_type().unwrap_or(ElementType::F32))
        },
        if is_weak_scalar(&args[1], weak_scalars) {
            DtypeOperand::weak(rhs.element_type().unwrap_or(ElementType::F32))
        } else {
            DtypeOperand::strong(rhs.element_type().unwrap_or(ElementType::F32))
        },
    )
    .map_err(|error| SheafError::Compile {
        message: format!("@: {}", error),
        location: crate::core::error::SourceLocation::unknown(),
    })?;
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let shape = match (lhs_shape.len(), rhs_shape.len()) {
        (0, _) | (_, 0) => crate::core::shape::broadcast_shapes(lhs_shape, rhs_shape)
            .map_err(|error| SheafError::Compile {
                message: format!(
                    "@: cannot broadcast dimensions {} and {}",
                    error.lhs, error.rhs,
                ),
                location: crate::core::error::SourceLocation::unknown(),
            })?,
        (1, 1) => Vec::new(),
        (1, _) => {
            let mut shape = rhs_shape[..rhs_shape.len() - 2].to_vec();
            shape.push(rhs_shape[rhs_shape.len() - 1]);
            shape
        }
        (_, 1) => lhs_shape[..lhs_shape.len() - 1].to_vec(),
        _ => {
            let mut shape = crate::core::shape::broadcast_shapes(
                &lhs_shape[..lhs_shape.len() - 2],
                &rhs_shape[..rhs_shape.len() - 2],
            )
            .map_err(|error| SheafError::Compile {
                message: format!(
                    "@: cannot broadcast dimensions {} and {}",
                    error.lhs, error.rhs,
                ),
                location: crate::core::error::SourceLocation::unknown(),
            })?;
            shape.push(lhs_shape[lhs_shape.len() - 2]);
            shape.push(rhs_shape[rhs_shape.len() - 1]);
            shape
        }
    };
    Ok(StableHLOType::tensor(shape, dtype))
}

fn infer_arithmetic_type(
    name: &str,
    args: &[CompiledExpr],
    symbol_types: &std::collections::HashMap<String, StableHLOType>,
    weak_scalars: &std::collections::HashSet<String>,
) -> SheafResult<StableHLOType> {
    let Some(first) = args.first() else {
        return Ok(StableHLOType::scalar_f32());
    };
    let mut result = infer_type_with_weak_scalars(first, symbol_types, weak_scalars)?;
    let mut result_is_weak = is_weak_scalar(first, weak_scalars);

    for arg in &args[1..] {
        let rhs = infer_type_with_weak_scalars(arg, symbol_types, weak_scalars)?;
        let rhs_is_weak = is_weak_scalar(arg, weak_scalars);
        let lhs_dtype = result.element_type().unwrap_or(ElementType::F32);
        let rhs_dtype = rhs.element_type().unwrap_or(ElementType::F32);
        let dtype = resolve_arithmetic_dtype(
            if result_is_weak {
                DtypeOperand::weak(lhs_dtype)
            } else {
                DtypeOperand::strong(lhs_dtype)
            },
            if rhs_is_weak {
                DtypeOperand::weak(rhs_dtype)
            } else {
                DtypeOperand::strong(rhs_dtype)
            },
        )
        .map_err(|error| SheafError::Compile {
            message: format!("{}: {}", name, error),
            location: crate::core::error::SourceLocation::unknown(),
        })?;
        let shape = crate::core::shape::broadcast_shapes(result.shape(), rhs.shape())
            .map_err(|error| SheafError::Compile {
                message: format!(
                    "{}: cannot broadcast dimensions {} and {}",
                    name,
                    error.lhs,
                    error.rhs,
                ),
                location: crate::core::error::SourceLocation::unknown(),
            })?;
        result = if shape.is_empty()
            && (matches!(result, StableHLOType::Tensor { .. })
                || matches!(rhs, StableHLOType::Tensor { .. }))
        {
            StableHLOType::tensor(shape, dtype)
        } else {
            inferred_tensor_type(shape, dtype)
        };
        result_is_weak &= rhs_is_weak;
    }
    Ok(result)
}

fn is_weak_scalar(
    expr: &CompiledExpr,
    weak_scalars: &std::collections::HashSet<String>,
) -> bool {
    match expr {
        CompiledExpr::Integer(_) | CompiledExpr::Float(_) => true,
        CompiledExpr::Symbol(name) => weak_scalars.contains(name),
        CompiledExpr::FunctionCall { name, args, .. }
            if matches!(
                name.as_str(),
                "+" | "-" | "*" | "/" | "**" | "//" | "%" | "mod" | "min" | "max"
                    | "minimum" | "maximum"
            ) =>
        {
            args.iter().all(|arg| is_weak_scalar(arg, weak_scalars))
        }
        CompiledExpr::FunctionCall { name, args, .. }
            if matches!(
                name.as_str(),
                "abs" | "relu" | "gelu" | "sigmoid" | "sqrt" | "exp" | "log" | "sin"
                    | "cos" | "tan" | "tanh" | "stop-gradient"
            ) =>
        {
            args.first()
                .is_some_and(|arg| is_weak_scalar(arg, weak_scalars))
        }
        CompiledExpr::Do(exprs) => exprs
            .last()
            .is_some_and(|expr| is_weak_scalar(expr, weak_scalars)),
        _ => false,
    }
}

fn literal_shape(expr: &CompiledExpr) -> Option<Vec<i64>> {
    let CompiledExpr::Vector(elements) = expr else {
        return None;
    };
    elements
        .iter()
        .map(|element| match element {
            CompiledExpr::Integer(dimension) => Some(*dimension),
            _ => None,
        })
        .collect()
}

fn infer_reduction_type(
    args: &[CompiledExpr],
    symbol_types: &std::collections::HashMap<String, StableHLOType>,
    weak_scalars: &std::collections::HashSet<String>,
) -> SheafResult<StableHLOType> {
    let Some(input) = args.first() else {
        return Ok(StableHLOType::scalar_f32());
    };
    let input_ty = infer_type_with_weak_scalars(input, symbol_types, weak_scalars)?;
    let shape = input_ty.shape();
    if shape.is_empty() {
        return Ok(input_ty);
    }

    let mut axis = -1;
    let mut keepdims = false;
    let mut index = 1;
    while index < args.len() {
        match &args[index] {
            CompiledExpr::Keyword(keyword) if keyword == "axis" => {
                if let CompiledExpr::Integer(value) = &args[index + 1] {
                    axis = *value;
                }
                index += 2;
            }
            CompiledExpr::Keyword(keyword) if keyword == "keepdims" => {
                keepdims = args
                    .get(index + 1)
                    .and_then(|value| match value {
                        CompiledExpr::Boolean(value) => Some(*value),
                        _ => None,
                    })
                    .unwrap_or(true);
                index += if matches!(args.get(index + 1), Some(CompiledExpr::Boolean(_))) {
                    2
                } else {
                    1
                };
            }
            _ => index += 1,
        }
    }

    let rank = shape.len();
    let axis = if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    }
    .min(rank - 1);
    let mut result_shape = shape.to_vec();
    if keepdims {
        result_shape[axis] = 1;
    } else {
        result_shape.remove(axis);
    }
    let dtype = input_ty.element_type().unwrap_or(ElementType::F32);
    if matches!(input_ty, StableHLOType::Tensor { .. }) {
        Ok(StableHLOType::tensor(result_shape, dtype))
    } else {
        Ok(inferred_tensor_type(result_shape, dtype))
    }
}

fn inferred_tensor_type(shape: Vec<i64>, dtype: ElementType) -> StableHLOType {
    if !shape.is_empty() {
        return StableHLOType::tensor(shape, dtype);
    }
    match dtype {
        ElementType::F16 => StableHLOType::ScalarF16,
        ElementType::BF16 => StableHLOType::ScalarBF16,
        ElementType::F32 => StableHLOType::ScalarF32,
        ElementType::F64 => StableHLOType::ScalarF64,
        ElementType::I64 => StableHLOType::ScalarI64,
        ElementType::Bool => StableHLOType::ScalarI1,
        ElementType::I32 => StableHLOType::tensor(shape, dtype),
    }
}

pub fn expr_is_tensor(
    expr: &CompiledExpr,
    symbol_types: &std::collections::HashMap<String, StableHLOType>,
) -> bool {
    infer_type_with_context(expr, symbol_types)
        .is_ok_and(|ty| matches!(ty, StableHLOType::Tuple(..)) || !ty.shape().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn infer_type(expr: &CompiledExpr) -> SheafResult<StableHLOType> {
        infer_type_with_context(expr, &std::collections::HashMap::new())
    }

    fn make_compiled_float(x: f64) -> CompiledExpr {
        CompiledExpr::Float(x)
    }

    fn make_compiled_vector(elems: Vec<CompiledExpr>) -> CompiledExpr {
        CompiledExpr::Vector(elems)
    }

    fn make_compiled_call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: name.to_string(),
            args,
            loc: None,
        }
    }

    #[cfg(not(sheaf_frontend))]
    #[test]
    fn tuple_layouts_preserve_nested_containers() {
        use crate::interpreter::value::Value;
        use ValueLayout::{Dict, Leaf, List, Tuple};
        let value = Value::Tuple(vec![
            Value::Dict(std::collections::BTreeMap::from([
                ("x".to_string(), Value::List(vec![Value::Float(1.0)])),
            ])),
            Value::Tuple(vec![Value::Float(2.0)]),
            Value::Tuple(vec![]),
        ]);
        let expected = Tuple(vec![
            Dict(vec![("x".to_string(), List(vec![Leaf]))]), Tuple(vec![Leaf]), Tuple(vec![]),
        ]);
        let layout = ValueLayout::from_value(&value);
        assert_eq!(layout, expected);
        let flat = Value::Tuple(vec![
            Value::Tuple(vec![Value::Tuple(vec![Value::Float(1.0)])]),
            Value::Tuple(vec![Value::Float(2.0)]), Value::Tuple(vec![]),
        ]);
        let Value::Tuple(restored) = layout.reconstruct(flat) else { panic!("expected tuple") };
        let Value::Dict(fields) = &restored[0] else { panic!("expected nested dict") };
        let Value::List(items) = &fields["x"] else { panic!("expected nested list") };
        assert!(matches!(items.as_slice(), [Value::Float(1.0)]));
        assert!(matches!(&restored[1], Value::Tuple(items) if matches!(items.as_slice(), [Value::Float(2.0)])));
        assert!(matches!(&restored[2], Value::Tuple(items) if items.is_empty()));
    }

    #[test]
    fn return_dict_keys_include_strings_and_keywords() {
        let expr = CompiledExpr::Dict(vec![
            (
                CompiledExpr::String("p".to_string()),
                CompiledExpr::Float(1.0),
            ),
            (
                CompiledExpr::Keyword("loss".to_string()),
                CompiledExpr::Float(2.0),
            ),
        ]);

        assert_eq!(
            find_return_dict_keys(&expr),
            Some(vec!["loss".to_string(), "p".to_string()]),
        );
    }

    #[test]
    fn test_infer_scalar() {
        let expr = make_compiled_float(42.0);
        let ty = infer_type(&expr).unwrap();
        assert_eq!(ty, StableHLOType::scalar_f32());
    }

    #[test]
    fn test_infer_add() {
        let expr = make_compiled_call(
            "+",
            vec![make_compiled_float(1.0), make_compiled_float(2.0)],
        );
        let ty = infer_type(&expr).unwrap();
        assert_eq!(ty, StableHLOType::scalar_f32());
    }

    #[test]
    fn context_flows_through_let_bindings() {
        let mut symbols = std::collections::HashMap::new();
        symbols.insert(
            "x".to_string(),
            StableHLOType::bf16_tensor(vec![2, 3]),
        );
        let expr = CompiledExpr::Let {
            bindings: vec![(
                BindingPattern::Simple("y".to_string()),
                CompiledExpr::Symbol("x".to_string()),
            )],
            body: Box::new(CompiledExpr::Symbol("y".to_string())),
        };
        assert_eq!(
            infer_type_with_context(&expr, &symbols).unwrap(),
            StableHLOType::bf16_tensor(vec![2, 3]),
        );
    }

    #[test]
    fn tuple_elements_resolve_from_parameter_types() {
        let mut symbols = std::collections::HashMap::new();
        symbols.insert(
            "params".to_string(),
            StableHLOType::Tuple(
                vec![StableHLOType::f16_tensor(vec![3, 4])],
                Some(vec!["weight".to_string()]),
            ),
        );
        let expr = CompiledExpr::GetTupleElement {
            param: "params".to_string(),
            indices: vec![0],
        };
        assert_eq!(
            infer_type_with_context(&expr, &symbols).unwrap(),
            StableHLOType::f16_tensor(vec![3, 4]),
        );
    }

    #[test]
    fn arithmetic_uses_tensor_dtype_for_weak_scalars() {
        let mut symbols = std::collections::HashMap::new();
        symbols.insert(
            "x".to_string(),
            StableHLOType::f16_tensor(vec![2]),
        );
        for name in ["+", "-", "*", "/"] {
            let expr = CompiledExpr::Let {
                bindings: vec![(
                    BindingPattern::Simple("one".to_string()),
                    make_compiled_float(1.0),
                )],
                body: Box::new(make_compiled_call(
                    name,
                    vec![
                        CompiledExpr::Symbol("x".to_string()),
                        CompiledExpr::Symbol("one".to_string()),
                    ],
                )),
            };
            assert_eq!(
                infer_type_with_context(&expr, &symbols).unwrap(),
                StableHLOType::f16_tensor(vec![2]),
            );
        }
    }

    #[test]
    fn arithmetic_rejects_strong_dtype_mismatches() {
        let mut symbols = std::collections::HashMap::new();
        symbols.insert(
            "x".to_string(),
            StableHLOType::f16_tensor(vec![2]),
        );
        symbols.insert(
            "y".to_string(),
            StableHLOType::bf16_tensor(vec![2]),
        );
        for name in ["+", "-", "*", "/"] {
            let expr = make_compiled_call(
                name,
                vec![
                    CompiledExpr::Symbol("x".to_string()),
                    CompiledExpr::Symbol("y".to_string()),
                ],
            );
            assert!(infer_type_with_context(&expr, &symbols).is_err());
        }
    }

    #[test]
    fn unary_operations_preserve_element_types() {
        let mut symbols = std::collections::HashMap::new();
        symbols.insert(
            "x".to_string(),
            StableHLOType::f16_tensor(vec![2]),
        );
        for name in ["relu", "gelu", "sigmoid", "tanh", "stop-gradient"] {
            let expr = make_compiled_call(
                name,
                vec![CompiledExpr::Symbol("x".to_string())],
            );
            assert_eq!(
                infer_type_with_context(&expr, &symbols).unwrap(),
                StableHLOType::f16_tensor(vec![2]),
            );
        }
    }

    #[test]
    fn reductions_preserve_element_types() {
        let mut symbols = std::collections::HashMap::new();
        symbols.insert(
            "x".to_string(),
            StableHLOType::f16_tensor(vec![2, 3]),
        );
        let expr = make_compiled_call(
            "sum",
            vec![
                CompiledExpr::Symbol("x".to_string()),
                CompiledExpr::Keyword("axis".to_string()),
                CompiledExpr::Integer(1),
            ],
        );
        assert_eq!(
            infer_type_with_context(&expr, &symbols).unwrap(),
            StableHLOType::f16_tensor(vec![2]),
        );
    }

    #[test]
    fn test_infer_matrix() {
        let expr = make_compiled_vector(vec![
            make_compiled_vector(vec![make_compiled_float(1.0), make_compiled_float(2.0)]),
            make_compiled_vector(vec![make_compiled_float(3.0), make_compiled_float(4.0)]),
        ]);
        let ty = infer_type(&expr).unwrap();
        assert_eq!(ty, StableHLOType::f32_tensor(vec![2, 2]));
    }

    #[test]
    fn test_infer_matmul() {
        let lhs = make_compiled_vector(vec![make_compiled_vector(vec![
            make_compiled_float(1.0),
            make_compiled_float(2.0),
        ])]);
        let rhs = make_compiled_vector(vec![
            make_compiled_vector(vec![make_compiled_float(3.0)]),
            make_compiled_vector(vec![make_compiled_float(4.0)]),
        ]);
        let expr = make_compiled_call("@", vec![lhs, rhs]);

        let ty = infer_type(&expr).unwrap();
        assert_eq!(ty, StableHLOType::f32_tensor(vec![1, 1]));
    }

    #[test]
    fn tuple_expressions_are_non_scalar() {
        let expr = CompiledExpr::Dict(vec![(
            CompiledExpr::Keyword("x".to_string()),
            make_compiled_float(1.0),
        )]);
        assert!(expr_is_tensor(&expr, &std::collections::HashMap::new()));
    }

    #[test]
    fn test_infer_signature() {
        let params = vec!["x".to_string(), "y".to_string()];

        let body = make_compiled_call(
            "+",
            vec![
                CompiledExpr::Symbol("x".to_string()),
                CompiledExpr::Symbol("y".to_string()),
            ],
        );

        let sig = infer_function_signature(&params, &body).unwrap();

        assert_eq!(sig.param_types.len(), 2);
        assert_eq!(sig.param_types[0], StableHLOType::scalar_f32());
        assert_eq!(sig.param_types[1], StableHLOType::scalar_f32());
        assert_eq!(sig.return_type, StableHLOType::scalar_f32());
    }

    #[test]
    fn test_infer_matmul_signature() {
        let params = vec!["A".to_string(), "B".to_string()];

        let a_matrix = make_compiled_vector(vec![
            make_compiled_vector(vec![
                make_compiled_float(1.0),
                make_compiled_float(2.0),
                make_compiled_float(3.0),
            ]),
            make_compiled_vector(vec![
                make_compiled_float(4.0),
                make_compiled_float(5.0),
                make_compiled_float(6.0),
            ]),
        ]);
        let b_matrix = make_compiled_vector(vec![
            make_compiled_vector(vec![
                make_compiled_float(1.0),
                make_compiled_float(2.0),
                make_compiled_float(3.0),
                make_compiled_float(4.0),
            ]),
            make_compiled_vector(vec![
                make_compiled_float(5.0),
                make_compiled_float(6.0),
                make_compiled_float(7.0),
                make_compiled_float(8.0),
            ]),
            make_compiled_vector(vec![
                make_compiled_float(9.0),
                make_compiled_float(10.0),
                make_compiled_float(11.0),
                make_compiled_float(12.0),
            ]),
        ]);

        let body = make_compiled_call("@", vec![a_matrix, b_matrix]);
        let sig = infer_function_signature(&params, &body).unwrap();

        assert_eq!(sig.return_type, StableHLOType::f32_tensor(vec![2, 4]));
    }
}
