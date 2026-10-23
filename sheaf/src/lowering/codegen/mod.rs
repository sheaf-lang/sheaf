// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! StableHLO code generation from `CompiledExpr`.

mod autodiff;
mod builtins;
mod collection_builtins;
mod control_flow;
mod helpers;
mod math_builtins;
mod reduction_builtins;
mod tensor_builtins;
#[cfg(test)]
mod tests;

use crate::lowering::stablehlo::{Register, StableHLOEmitter, StableHLOType};
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::inference::ValueLayout;
use crate::core::error::{SheafError, SheafResult, SourceLocation};
pub(crate) use helpers::{
    try_flatten_to_constant, TupleLeaf, collect_tuple_references, collect_tuple_type_leaves,
    expand_tuple_to_symbols,
};
use std::collections::{HashMap, HashSet};

pub fn flatten_param_types(param_types: &[StableHLOType]) -> Vec<StableHLOType> {
    let mut flat = Vec::new();
    for ty in param_types {
        flatten_type(ty, &mut flat);
    }
    flat
}

fn flatten_type(ty: &StableHLOType, out: &mut Vec<StableHLOType>) {
    match ty {
        StableHLOType::Tuple(elems, _) => {
            for elem in elems {
                flatten_type(elem, out);
            }
        }
        other => out.push(other.clone()),
    }
}

pub struct CodeGenerator<'a> {
    emitter: StableHLOEmitter,
    bindings: HashMap<String, (Register, StableHLOType)>,
    weak_scalars: HashSet<Register>,
    lambda_bindings: HashMap<String, CompiledExpr>,
    function_registry: Option<&'a HashMap<String, crate::core::expr::FunctionDef>>,
    tuple_key_layouts: HashMap<String, std::collections::BTreeMap<String, usize>>,
    idx_to_key: HashMap<(String, usize), String>,
    layout_key_map: HashMap<Register, String>,
    static_tuple_indices: HashSet<Register>,
    list_regs: HashSet<Register>,
}

impl<'a> CodeGenerator<'a> {
    pub fn new() -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
            weak_scalars: HashSet::new(),
            lambda_bindings: HashMap::new(),
            function_registry: None,
            tuple_key_layouts: HashMap::new(),
            idx_to_key: HashMap::new(),
            layout_key_map: HashMap::new(),
            static_tuple_indices: HashSet::new(),
            list_regs: HashSet::new(),
        }
    }

    pub fn with_registry(
        registry: &'a HashMap<String, crate::core::expr::FunctionDef>,
    ) -> Self {
        Self {
            emitter: StableHLOEmitter::new(),
            bindings: HashMap::new(),
            weak_scalars: HashSet::new(),
            lambda_bindings: HashMap::new(),
            function_registry: Some(registry),
            tuple_key_layouts: HashMap::new(),
            idx_to_key: HashMap::new(),
            layout_key_map: HashMap::new(),
            static_tuple_indices: HashSet::new(),
            list_regs: HashSet::new(),
        }
    }

    pub fn with_function_params(
        registry: &'a HashMap<String, crate::core::expr::FunctionDef>,
        param_names: &[String],
        param_types: &[StableHLOType],
    ) -> Self {
        let mut emitter = StableHLOEmitter::new();
        let mut bindings = HashMap::new();
        let mut flat_idx: usize = 0;

        for (name, ty) in param_names.iter().zip(param_types.iter()) {
            let reg = emitter.register_virtual_param(ty, &mut flat_idx);
            bindings.insert(name.clone(), (reg, ty.clone()));
        }

        Self {
            emitter,
            bindings,
            weak_scalars: HashSet::new(),
            lambda_bindings: HashMap::new(),
            function_registry: Some(registry),
            tuple_key_layouts: HashMap::new(),
            idx_to_key: HashMap::new(),
            layout_key_map: HashMap::new(),
            static_tuple_indices: HashSet::new(),
            list_regs: HashSet::new(),
        }
    }

    pub fn set_tuple_key_layouts(
        &mut self,
        layouts: HashMap<String, std::collections::BTreeMap<String, usize>>,
    ) {
        self.tuple_key_layouts = layouts;
    }

    pub fn set_idx_to_key(&mut self, map: HashMap<(String, usize), String>) {
        self.idx_to_key = map;
    }

    fn mark_list_layout(&mut self, reg: Register, layout: &ValueLayout) {
        if let ValueLayout::List(_) = layout {
            self.list_regs.insert(reg);
        }
        let children = match layout {
            ValueLayout::Dict(entries) => entries.iter().map(|(_, layout)| layout).collect::<Vec<_>>(),
            ValueLayout::List(items) | ValueLayout::Tuple(items) => items.iter().collect(),
            ValueLayout::Leaf => return,
        };
        let Some(regs) = self.emitter.virtual_tuple_elements(&reg) else { return };
        let regs: Vec<_> = regs.iter().map(|(reg, _)| *reg).collect();
        for (child, layout) in regs.into_iter().zip(children) {
            self.mark_list_layout(child, layout);
        }
    }

    pub fn set_parameter_value_layouts(&mut self, names: &[String], layouts: &[ValueLayout]) {
        for (name, layout) in names.iter().zip(layouts) {
            if let Some(&(reg, _)) = self.bindings.get(name) {
                self.mark_list_layout(reg, layout);
            }
        }
    }

    pub fn specialize_parameter_leaf(
        &mut self, name: &str, indices: &[usize], value: &CompiledExpr,
    ) -> SheafResult<()> {
        let Some(&(root, _)) = self.bindings.get(name) else {
            return Err(crate::SheafError::Compile {
                message: format!("static input references unknown parameter '{name}'"),
                location: SourceLocation::unknown(),
            });
        };
        let (reg, ty) = self.generate(value)?;
        if indices.is_empty() {
            self.bind_symbol(name, reg, ty);
        } else if !self.emitter.replace_virtual_leaf(root, indices, reg, ty) {
            return Err(crate::SheafError::Compile {
                message: format!("invalid static input projection for parameter '{name}'"),
                location: SourceLocation::unknown(),
            });
        }
        Ok(())
    }

    pub fn bind_symbol(&mut self, name: &str, reg: Register, ty: StableHLOType) {
        self.bindings.insert(name.to_string(), (reg, ty));
    }

    pub fn binding_shapes(&self) -> std::collections::HashMap<String, Vec<i64>> {
        self.bindings
            .iter()
            .map(|(name, (_, ty))| (name.clone(), ty.shape().to_vec()))
            .collect()
    }

    pub fn generate_binding(&mut self, name: &str, value_expr: &CompiledExpr) -> SheafResult<()> {
        if matches!(value_expr, CompiledExpr::Lambda { .. }) {
            self.lambda_bindings.insert(name.to_string(), value_expr.clone());
        } else if name.starts_with('[') && name.ends_with(']') {
            let names: Vec<&str> = name[1..name.len() - 1].split_whitespace().collect();
            let (tuple_reg, tuple_ty) = self.generate(value_expr)?;
            let element_types = match &tuple_ty {
                StableHLOType::Tuple(tys, _) => tys.clone(),
                other => {
                    return Err(SheafError::Compile {
                        message: format!("Let destructuring requires a tuple, got: {}", other.to_mlir()),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            };
            for (i, n) in names.iter().enumerate() {
                let elem_reg = self.emitter.emit_get_tuple_element(
                    &tuple_reg, &tuple_ty, i, &element_types[i],
                );
                self.bindings.insert(n.to_string(), (elem_reg, element_types[i].clone()));
            }
    } else {
        let (reg, ty) = self.generate(value_expr)?;
        if matches!(&ty, StableHLOType::Tuple(..)) {
            if let CompiledExpr::FunctionCall { name: fn_name, args: fn_args, .. } = value_expr
                && fn_name == "get"
                && fn_args.len() >= 2
                && let Some(CompiledExpr::Keyword(k) | CompiledExpr::String(k)) = fn_args.last()
                && let Some(sub_layout) = self.tuple_key_layouts.get(k).cloned()
            {
                self.tuple_key_layouts.insert(name.to_string(), sub_layout);
            } else if let CompiledExpr::Symbol(src) = value_expr
                && let Some(layout) = self.tuple_key_layouts.get(src).cloned()
            {
                self.tuple_key_layouts.insert(name.to_string(), layout);
            }
        }
            self.bindings.insert(name.to_string(), (reg, ty));
            }
        Ok(())
    }

    pub fn emit_get_tuple_element(
        &mut self,
        tuple_reg: &Register,
        tuple_ty: &StableHLOType,
        index: usize,
        elem_ty: &StableHLOType,
    ) -> Register {
        self.emitter
            .emit_get_tuple_element(tuple_reg, tuple_ty, index, elem_ty)
    }

    fn try_const_eval(&self, expr: &CompiledExpr) -> Option<f64> {
        match expr {
            CompiledExpr::Integer(n) => Some(*n as f64),
            CompiledExpr::Float(f) => Some(*f),
            CompiledExpr::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            CompiledExpr::FunctionCall { name, args, .. } => {
                match name.as_str() {
                    "ndim" if args.len() == 1 => {
                        if let CompiledExpr::Symbol(sym) = &args[0] {
                            let (_, ty) = self.bindings.get(sym.as_str())?;
                            Some(ty.shape().len() as f64)
                        } else {
                            None
                        }
                    }
                    "len" | "length" if args.len() == 1 => {
                        if let CompiledExpr::FunctionCall { name: inner_name, args: inner_args, .. } = &args[0]
                            && inner_name == "shape"
                            && inner_args.len() == 1
                            && let CompiledExpr::Symbol(sym) = &inner_args[0]
                        {
                            let (_, ty) = self.bindings.get(sym.as_str())?;
                            return Some(ty.shape().len() as f64);
                        }
                        None
                    }
                    "==" | "!=" | "<" | ">" | "<=" | ">=" if args.len() == 2 => {
                        let a = self.try_const_eval(&args[0])?;
                        let b = self.try_const_eval(&args[1])?;
                        let result = match name.as_str() {
                            "==" => (a - b).abs() < 1e-10,
                            "!=" => (a - b).abs() >= 1e-10,
                            "<" => a < b,
                            ">" => a > b,
                            "<=" => a <= b,
                            ">=" => a >= b,
                            _ => return None,
                        };
                        Some(if result { 1.0 } else { 0.0 })
                    }
                    "-" if args.len() == 1 => {
                        let a = self.try_const_eval(&args[0])?;
                        Some(-a)
                    }
                    "+" | "-" | "*" | "/" if args.len() == 2 => {
                        let a = self.try_const_eval(&args[0])?;
                        let b = self.try_const_eval(&args[1])?;
                        Some(match name.as_str() {
                            "+" => a + b,
                            "-" => a - b,
                            "*" => a * b,
                            "/" => a / b,
                            _ => return None,
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub fn generate(&mut self, expr: &CompiledExpr) -> SheafResult<(Register, StableHLOType)> {
        match expr {
            CompiledExpr::Integer(n) => {
                let reg = self.emitter.emit_constant_f32(*n as f64);
                self.weak_scalars.insert(reg);
                Ok((reg, StableHLOType::scalar_f32()))
            }

            CompiledExpr::Float(x) => {
                let reg = self.emitter.emit_constant_f32(*x);
                self.weak_scalars.insert(reg);
                Ok((reg, StableHLOType::scalar_f32()))
            }

            CompiledExpr::Boolean(value) => {
                let reg = self.emitter.emit_constant_f32(if *value { 1.0 } else { 0.0 });
                Ok((reg, StableHLOType::scalar_f32()))
            }

            CompiledExpr::Vector(elements) => {
                match try_flatten_to_constant(elements) {
                    Some((data, shape)) => {
                        let (reg, ty) = self.emitter.emit_nd_tensor_constant(&data, &shape);
                        Ok((reg, ty))
                    }
                    None => {
                        let mut regs = Vec::new();
                        let mut tys = Vec::new();
                        for elem in elements {
                            let (reg, ty) = self.generate(elem)?;
                            regs.push(reg);
                            tys.push(ty);
                        }
                        let result = self.emitter.emit_tuple(&regs, &tys);
                        self.list_regs.insert(result.0);
                        Ok(result)
                    }
                }
            }

            CompiledExpr::Tuple(elements) => {
                let mut regs = Vec::new();
                let mut tys = Vec::new();
                for elem in elements {
                    let (reg, ty) = self.generate(elem)?;
                    regs.push(reg);
                    tys.push(ty);
                }
                let result = self.emitter.emit_tuple(&regs, &tys);
                self.list_regs.insert(result.0);
                Ok(result)
            }

            CompiledExpr::Symbol(name) => {
                if let Some(&(reg, ref ty)) = self.bindings.get(name) {
                    Ok((reg, ty.clone()))
                } else {
                    Err(SheafError::Compile {
                        message: format!("Undefined symbol in codegen: {}", name),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            }

            CompiledExpr::GetTupleElement { param, indices } => {
                let (param_reg, param_ty) =
                    self.bindings
                        .get(param)
                        .cloned()
                        .ok_or_else(|| SheafError::Compile {
                            message: format!(
                                "GetTupleElement: parameter '{}' not found in bindings",
                                param
                            ),
                            location: crate::core::error::SourceLocation::unknown(),
                        })?;

                let mut current_reg = param_reg;
                let mut current_ty = param_ty;

                for &idx in indices {
                    let element_ty = match &current_ty {
                        StableHLOType::Tuple(elems, _) => elems.get(idx).cloned().ok_or_else(|| {
                            SheafError::Compile {
                                message: format!(
                                    "GetTupleElement: index {} out of range for tuple with {} elements",
                                    idx,
                                    elems.len()
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            }
                        })?,
                        other => {
                            return Err(SheafError::Compile {
                                message: format!(
                                    "GetTupleElement: expected tuple type, got {}",
                                    other.to_mlir()
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }
                    };
                    let result_reg = self.emitter.emit_get_tuple_element(
                        &current_reg,
                        &current_ty,
                        idx,
                        &element_ty,
                    );
                    current_reg = result_reg;
                    current_ty = element_ty;
                }

                self.track_layout_key(param, indices, current_reg);

                Ok((current_reg, current_ty))
            }

            CompiledExpr::FunctionCall { name, args, loc } => self.generate_function_call(name, args)
                .map_err(|error| match loc {
                    Some(location) => error.with_fallback_location(location),
                    None => error,
                }),

    CompiledExpr::Let { bindings, body } => {
        let saved_bindings = self.bindings.clone();
        let saved_lambda_bindings = self.lambda_bindings.clone();
        for (pattern, value_expr) in bindings {
            if matches!(value_expr, CompiledExpr::Lambda { .. }) {
                if let BindingPattern::Simple(name) = pattern {
                    self.lambda_bindings.insert(name.clone(), value_expr.clone());
                }
            } else if let BindingPattern::Destructure(_) = pattern {
                return Err(SheafError::Compile {
                    message: "Destructuring patterns should have been desugared".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            } else if let BindingPattern::Simple(name) = pattern {
                let (reg, ty) = self.generate(value_expr)?;
                if matches!(&ty, StableHLOType::Tuple(..)) {
                    if let CompiledExpr::FunctionCall { name: fn_name, args: fn_args, .. } = value_expr {
                        if fn_name == "get"
                            && fn_args.len() >= 2
                            && let Some(CompiledExpr::Keyword(k) | CompiledExpr::String(k)) = fn_args.last()
                            && let Some(sub_layout) = self.tuple_key_layouts.get(k).cloned()
                        {
                            self.tuple_key_layouts.insert(name.clone(), sub_layout);
                        }
                    }
                    // Follow virtual tuple aliases introduced by inlining.
                    else if let CompiledExpr::GetTupleElement { param, indices } = value_expr
                        && let Some(layout_key) = self.resolve_layout_key(param, indices)
                    {
                        self.layout_key_map.insert(reg, layout_key.clone());
                        if let Some(sub_layout) = self.tuple_key_layouts.get(&layout_key).cloned() {
                            self.tuple_key_layouts.insert(name.clone(), sub_layout);
                        }
                    }
                    else if let CompiledExpr::Symbol(src) = value_expr
                        && let Some(layout) = self.tuple_key_layouts.get(src).cloned()
                    {
                        self.tuple_key_layouts.insert(name.clone(), layout);
                    }
                }
                self.bindings.insert(name.clone(), (reg, ty));
            }
        }
        let result = self.generate(body)?;
        self.bindings = saved_bindings;
        self.lambda_bindings = saved_lambda_bindings;
        Ok(result)
            }

            CompiledExpr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                // Branches may have incompatible static shapes.
                if let Some(const_val) = self.try_const_eval(condition) {
                    let is_true = const_val.abs() > 1e-10;
                    if is_true {
                        return self.generate(then_branch);
                    } else if let Some(else_expr) = else_branch {
                        return self.generate(else_expr);
                    } else {
                        let reg = self.emitter.emit_constant_f32(0.0);
                        return Ok((reg, StableHLOType::ScalarF32));
                    }
                }

                let (cond_reg, cond_ty) = self.generate(condition)?;
                let (then_reg, then_ty) = self.generate(then_branch)?;

                if let Some(else_expr) = else_branch {
                    let (else_reg, else_ty) = self.generate(else_expr)?;
                    let (result_reg, result_ty) = self.emitter.emit_select(
                        &cond_reg, &then_reg, &else_reg, &cond_ty, &then_ty, &else_ty,
                    );
                    Ok((result_reg, result_ty))
                } else {
                    // An omitted else branch is only valid for a true condition.
                    Ok((then_reg, then_ty))
                }
            }

            CompiledExpr::Do(exprs) => {
                let mut last_result = None;
                for expr in exprs {
                    last_result = Some(self.generate(expr)?);
                }
                last_result.ok_or_else(|| SheafError::Compile {
                    message: "do requires at least one expression".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }

            CompiledExpr::FunctionRef(name) => Err(SheafError::Compile {
                message: format!("Cannot generate code for bare function reference: {}", name),
                location: crate::core::error::SourceLocation::unknown(),
            }),

            CompiledExpr::Lambda { .. } => Err(SheafError::Compile {
                message: "Cannot emit a lambda without a call site".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),

            CompiledExpr::LambdaCall { callee, args } => {
                let callee = callee.clone();
                let args = args.clone();
                self.inline_lambda_call(&callee, &args)
            }

            CompiledExpr::ValueAndGrad { fn_name, .. } => Err(SheafError::Compile {
                message: format!(
                    "ValueAndGrad '{}' is a module-level form, not an inline expression",
                    fn_name
                ),
                location: crate::core::error::SourceLocation::unknown(),
            }),


            CompiledExpr::Dict(pairs) => {
                let mut sorted = Vec::with_capacity(pairs.len());
                for (key, value) in pairs {
                    let name = match key {
                        CompiledExpr::Keyword(name) | CompiledExpr::String(name) => name.clone(),
                        _ => {
                            return Err(SheafError::Compile {
                                message: format!("dictionary key must be static, got {:?}", key),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }
                    };
                    sorted.push((name, value));
                }
                sorted.sort_by(|(left, _), (right, _)| left.cmp(right));

                let mut regs = Vec::new();
                let mut tys = Vec::new();
                let mut keys = Vec::new();
                for (key, value) in sorted {
                    let (reg, ty) = self.generate(value)?;
                    regs.push(reg);
                    tys.push(ty);
                    keys.push(key);
                }
                let (reg, _) = self.emitter.emit_tuple(&regs, &tys);
                Ok((reg, StableHLOType::Tuple(tys, Some(keys))))
            }

            CompiledExpr::Quoted(val) => {
                use crate::core::ast::SheafValue;
                match val.as_ref() {
                    SheafValue::Vector(elems, _) => {
                        let converted: Vec<CompiledExpr> = elems.iter().map(|e| match e {
                            SheafValue::Integer(n, _) => CompiledExpr::Integer(*n),
                            SheafValue::Float(f, _) => CompiledExpr::Float(*f),
                            _ => CompiledExpr::Integer(0),
                        }).collect();
                        self.generate(&CompiledExpr::Vector(converted))
                    }
                    _ => Err(SheafError::Compile {
                        message: format!("Code generation not yet implemented for: {:?}", expr),
                        location: crate::core::error::SourceLocation::unknown(),
                    }),
                }
            }

            _ => Err(SheafError::Compile {
                message: format!("Code generation not yet implemented for: {:?}", expr),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }
    }

    fn parse_shape_vec(&self, operation: &str, elems: &[CompiledExpr]) -> SheafResult<Vec<i64>> {
        elems.iter().map(|e| match e {
            CompiledExpr::Integer(n) => Ok(*n),
            CompiledExpr::Float(f) if *f == f.floor() => Ok(*f as i64),
            CompiledExpr::Float(value) => Err(SheafError::InvalidShapeDimension {
                message: format!("{operation}: shape dimension is float ({value:?}), expected an integer"),
                location: SourceLocation::unknown(),
            }),
            CompiledExpr::Boolean(value) => Err(SheafError::InvalidShapeDimension {
                message: format!("{operation}: shape dimension is bool ({value}), expected an integer"),
                location: SourceLocation::unknown(),
            }),
            CompiledExpr::Symbol(name) => {
                if let Some(&(reg, _)) = self.bindings.get(name.as_str()) {
                    if let Some(v) = self.emitter.known_scalar_value(&reg) {
                        Ok(v as i64)
                    } else {
                        Err(SheafError::Compile {
                            message: format!(
                                "shape element must be a constant integer, got: Symbol({:?}) (not a known constant)",
                                name
                            ),
                            location: crate::core::error::SourceLocation::unknown(),
                        })
                    }
                } else {
                    Err(SheafError::Compile {
                        message: format!(
                            "shape element must be a constant integer, got: Symbol({:?}) (not bound)",
                            name
                        ),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            }
            other => Err(SheafError::Compile {
                message: format!(
                    "shape element must be a constant integer, got: {:?}",
                    other
                ),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }).collect()
    }

    fn resolve_layout_key(&self, param: &str, indices: &[usize]) -> Option<String> {
        let mut current = self.bindings
            .get(param)
            .and_then(|(reg, _)| self.layout_key_map.get(reg))
            .cloned()
            .unwrap_or_else(|| param.to_string());
        for &index in indices {
            current = self.idx_to_key.get(&(current, index))?.clone();
        }
        Some(current)
    }

    fn track_layout_key(&mut self, param: &str, indices: &[usize], reg: Register) {
        if let Some(layout_key) = self.resolve_layout_key(param, indices) {
            self.layout_key_map.insert(reg, layout_key);
        }
    }

    pub fn emit_function(mut self, name: &str, expr: &CompiledExpr) -> SheafResult<String> {
        let (result_reg, result_ty) = self.generate(expr)?;
        self.emitter.emit_return(&result_reg, &result_ty);

        Ok(self.emitter.emit_function_body(name, &result_ty))
    }

    fn result_layout(&self, reg: Register, ty: &StableHLOType) -> Option<ValueLayout> {
        let StableHLOType::Tuple(types, keys) = ty else { return None };
        let elements = self.emitter.virtual_tuple_elements(&reg)?;
        if elements.len() != types.len() {
            return None;
        }
        let layouts: Vec<_> = elements.iter()
            .zip(types)
            .map(|((child, _), ty)| self.result_layout(*child, ty))
            .collect();
        if !self.list_regs.contains(&reg) && layouts.iter().all(Option::is_none) {
            return None;
        }
        let layouts: Vec<_> = layouts.into_iter()
            .map(|layout| layout.unwrap_or(ValueLayout::Leaf))
            .collect();
        match keys {
            Some(keys) => Some(ValueLayout::Dict(keys.iter().cloned().zip(layouts).collect())),
            None if self.list_regs.contains(&reg) => Some(ValueLayout::List(layouts)),
            None => Some(ValueLayout::Tuple(layouts)),
        }
    }

    pub fn emit_func_declaration(
        self,
        name: &str,
        expr: &CompiledExpr,
        param_types: &[StableHLOType],
        return_type: &StableHLOType,
    ) -> SheafResult<(String, StableHLOType)> {
        self.emit_func_declaration_with_layout(name, expr, param_types, return_type)
            .map(|(decl, ty, _)| (decl, ty))
    }

    pub fn emit_func_declaration_with_layout(
        mut self,
        name: &str,
        expr: &CompiledExpr,
        param_types: &[StableHLOType],
        _return_type: &StableHLOType,
    ) -> SheafResult<(String, StableHLOType, Option<ValueLayout>)> {
        let (result_reg, result_ty) = self.generate(expr)?;
        let result_layout = self.result_layout(result_reg, &result_ty);
        let flat_params = flatten_param_types(param_types);
        let leaves = self.emitter.collect_virtual_leaves(result_reg, &result_ty);
        let (leaf_regs, leaf_tys): (Vec<_>, Vec<_>) = leaves.into_iter().unzip();

    if leaf_regs.len() == 1 {
        self.emitter.emit_return(&leaf_regs[0], &leaf_tys[0]);
        let body = self.emitter.body.clone();
            let decl = self.emitter
                .emit_func_declaration(name, &flat_params, &leaf_tys[0], &body);
            Ok((decl, result_ty, result_layout))
    } else {
        self.emitter.emit_return_multi(&leaf_regs, &leaf_tys);
        let body = self.emitter.body.clone();
            let decl = self.emitter
                .emit_func_declaration_multi(name, &flat_params, &leaf_tys, &body);
            Ok((decl, result_ty, result_layout))
        }
    }

    pub fn finish_multi(
        mut self,
        name: &str,
        param_types: &[StableHLOType],
        result_regs: &[crate::lowering::stablehlo::Register],
        result_types: &[StableHLOType],
    ) -> String {
        let flat_params = flatten_param_types(param_types);
        let mut all_regs = Vec::new();
        let mut all_tys = Vec::new();
        for (r, t) in result_regs.iter().zip(result_types.iter()) {
            for (leaf_r, leaf_t) in self.emitter.collect_virtual_leaves(*r, t) {
                all_regs.push(leaf_r);
                all_tys.push(leaf_t);
            }
        }
    self.emitter.emit_return_multi(&all_regs, &all_tys);
    let body = self.emitter.body.clone();
        self.emitter
            .emit_func_declaration_multi(name, &flat_params, &all_tys, &body)
    }
}

impl<'a> Default for CodeGenerator<'a> {
    fn default() -> Self {
        Self::new()
    }
}

