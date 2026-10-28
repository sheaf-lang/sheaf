// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Lambda inlining and code generation for tree operations, reduce, and scan.

use std::collections::{BTreeMap, HashMap, HashSet};
use crate::autodiff::reverse::{ReverseGradResult, reverse_grad, to_anf};
use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::lowering::transforms::try_infer_shape;
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::error::{SheafError, SheafResult};
use super::CodeGenerator;

/// Builds paths from a root layout key to nested tuple indices.
pub(super) fn build_deep_index_map(
    root_key: &str,
    tuple_key_layouts: &HashMap<String, BTreeMap<String, usize>>,
) -> BTreeMap<Vec<String>, Vec<usize>> {
    let mut map = BTreeMap::new();
    build_deep_index_map_rec(root_key, &[], &[], tuple_key_layouts, &mut map);
    map
}

pub(super) fn build_deep_index_map_rec(
    key: &str,
    path: &[String],
    indices: &[usize],
    layouts: &HashMap<String, BTreeMap<String, usize>>,
    map: &mut BTreeMap<Vec<String>, Vec<usize>>,
) {
    if !path.is_empty() {
        map.insert(path.to_vec(), indices.to_vec());
    }
    if let Some(child_layout) = layouts.get(key) {
        for (field_name, &field_idx) in child_layout {
            let mut child_path = path.to_vec();
            child_path.push(field_name.clone());
            let mut child_indices = indices.to_vec();
            child_indices.push(field_idx);
            build_deep_index_map_rec(field_name, &child_path, &child_indices, layouts, map);
        }
    }
}

/// How a reduce or scan obtains each collection element.
enum ElemKind {
    VecTuple(Vec<StableHLOType>),
    StackedDict(Vec<StableHLOType>),
    PlainTensor,
}

fn fresh_scan_field_symbol(
    bindings: &[(BindingPattern, CompiledExpr)],
    carry_param: &str,
    elem_param: &str,
    field_idx: usize,
) -> String {
    let mut used: HashSet<&str> = bindings
        .iter()
        .filter_map(|(pattern, _)| match pattern {
            BindingPattern::Simple(name) => Some(name.as_str()),
            BindingPattern::Destructure(_) => None,
        })
        .collect();
    used.insert(carry_param);
    used.insert(elem_param);

    let base = format!("__scan_elem_field_{}", field_idx);
    let mut suffix = 0;
    loop {
        let candidate = if suffix == 0 {
            base.clone()
        } else {
            format!("{}_{}", base, suffix)
        };
        if !used.contains(candidate.as_str()) {
            return candidate;
        }
        suffix += 1;
    }
}

impl<'a> CodeGenerator<'a> {
    /// Emits a lambda body after binding its parameters.
    pub(super) fn inline_lambda_call(
        &mut self,
        callee: &CompiledExpr,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        if let CompiledExpr::FunctionCall { name, args: inner_args, .. } = callee
            && name == "__value-and-grad-hof__"
            && inner_args.len() == 1
        {
            return self.generate_vag_inline(&inner_args[0], args);
        }

        let lambda = match callee {
            CompiledExpr::Lambda { .. } => callee.clone(),
            CompiledExpr::Symbol(name) => {
                self.lambda_bindings
                    .get(name)
                    .cloned()
                    .ok_or_else(|| SheafError::Compile {
                        message: format!("Undefined lambda: {}", name),
                        location: crate::core::error::SourceLocation::unknown(),
                    })?
            }
            other => {
                return Err(SheafError::Compile {
                    message: format!("Expected lambda at call site, got: {:?}", other),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }
        };

        let (params, body) = match lambda {
            CompiledExpr::Lambda { params, body } => (params, *body),
            _ => unreachable!(),
        };

        let mut arg_regs = Vec::new();
        let mut arg_tys = Vec::new();
        for arg in args {
            let (reg, ty) = self.generate(arg)?;
            arg_regs.push(reg);
            arg_tys.push(ty);
        }

        let saved = self.bindings.clone();
        for (param, (reg, ty)) in params.iter().zip(arg_regs.iter().zip(arg_tys.iter())) {
            self.bindings
                .insert(param.clone(), (*reg, ty.clone()));
        }
        let result = self.generate(&body);
        self.bindings = saved;
        result
    }

    /// Applies a lambda to matching leaves of statically known tuple trees.
    pub(super) fn generate_tree_map(
        &mut self,
        lambda: &CompiledExpr,
        tree_regs: &[Register],
        tree_tys: &[StableHLOType],
    ) -> SheafResult<(Register, StableHLOType)> {
        match &tree_tys[0] {
            StableHLOType::Tuple(first_elem_tys, keys) => {
                let mut result_regs = Vec::new();
                let mut result_tys = Vec::new();
                for (idx, _) in first_elem_tys.iter().enumerate() {
                    let mut sub_regs = Vec::new();
                    let mut sub_tys = Vec::new();
                    for (tree_reg, tree_ty) in tree_regs.iter().zip(tree_tys.iter()) {
                        let elem_ty = match tree_ty {
                            StableHLOType::Tuple(elems, _) => &elems[idx],
                            other => other,
                        };
                        let elem_reg = self.emitter.emit_get_tuple_element(
                            tree_reg,
                            tree_ty,
                            idx,
                            elem_ty,
                        );
                        sub_regs.push(elem_reg);
                        sub_tys.push(elem_ty.clone());
                    }
                    let (r, t) = self.generate_tree_map(lambda, &sub_regs, &sub_tys)?;
                    result_regs.push(r);
                    result_tys.push(t);
                }
                let (reg, _) = self.emitter.emit_tuple_with_keys(&result_regs, &result_tys, keys);
                if self.list_regs.contains(&tree_regs[0]) {
                    self.list_regs.insert(reg);
                }
                Ok((reg, StableHLOType::Tuple(result_tys, keys.clone())))
            }
            _ => {
                let (params, body) = match lambda {
                    CompiledExpr::Lambda { params, body } => (params, body),
                    _ => {
                        return Err(SheafError::Compile {
                            message: "tree-map: first argument must be a lambda".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        })
                    }
                };
                if params.len() != tree_regs.len() {
                    return Err(SheafError::Compile {
                        message: format!(
                            "tree-map: lambda has {} params but {} trees provided",
                            params.len(),
                            tree_regs.len()
                        ),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
                let saved = self.bindings.clone();
                for (param, (reg, ty)) in params.iter().zip(tree_regs.iter().zip(tree_tys.iter())) {
                    self.bindings
                        .insert(param.clone(), (*reg, ty.clone()));
                }
                let result = self.generate(body);
                self.bindings = saved;
                result
            }
        }
    }

    /// Folds a lambda over the leaves of a statically known tuple tree.
    pub(super) fn generate_tree_reduce(
        &mut self,
        lambda: &CompiledExpr,
        tree_reg: Register,
        tree_ty: &StableHLOType,
        acc_reg: Register,
        acc_ty: &StableHLOType,
    ) -> SheafResult<(Register, StableHLOType)> {
        match tree_ty {
            StableHLOType::Tuple(elem_tys, _) => {
                let mut cur_acc = acc_reg;
                let mut cur_ty = acc_ty.clone();
                for (idx, elem_ty) in elem_tys.iter().enumerate() {
                    let elem_reg = self.emitter.emit_get_tuple_element(
                        &tree_reg, tree_ty, idx, elem_ty,
                    );
                    (cur_acc, cur_ty) = self.generate_tree_reduce(
                        lambda, elem_reg, elem_ty, cur_acc, &cur_ty,
                    )?;
                }
                Ok((cur_acc, cur_ty))
            }
            _ => {
                match lambda {
                    CompiledExpr::Lambda { params, body } => {
                        if params.len() != 2 {
                            return Err(SheafError::Compile {
                                message: format!(
                                    "tree-reduce: lambda must have 2 params (acc, leaf), got {}",
                                    params.len()
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }
                        let saved = self.bindings.clone();
                        self.bindings.insert(params[0].clone(), (acc_reg, acc_ty.clone()));
                        self.bindings.insert(params[1].clone(), (tree_reg, tree_ty.clone()));
                        let result = self.generate(body);
                        self.bindings = saved;
                        result
                    }
                    CompiledExpr::Symbol(name) => {
                        let call = CompiledExpr::FunctionCall {
                            name: name.clone(),
                            args: vec![
                                CompiledExpr::Symbol("__tree_reduce_acc".to_string()),
                                CompiledExpr::Symbol("__tree_reduce_leaf".to_string()),
                            ],
                            loc: None,
                        };
                        let saved = self.bindings.clone();
                        self.bindings.insert("__tree_reduce_acc".to_string(), (acc_reg, acc_ty.clone()));
                        self.bindings.insert("__tree_reduce_leaf".to_string(), (tree_reg, tree_ty.clone()));
                        let result = self.generate(&call);
                        self.bindings = saved;
                        result
                    }
                    _ => Err(SheafError::Compile {
                        message: "tree-reduce: first argument must be a lambda or builtin symbol".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    }),
                }
            }
        }
    }

    /// Unrolls a reduce or scan whose collection type is known at compile time.
    ///
    /// The element layout is bound to the lambda parameter so nested `get` calls
    /// still resolve after unrolling.
    pub(super) fn generate_reduce_scan(
        &mut self,
        lambda: &CompiledExpr,
        init: &CompiledExpr,
        coll: &CompiledExpr,
        is_scan: bool,
    ) -> SheafResult<(Register, StableHLOType)> {
        let coll_sym = if let CompiledExpr::Symbol(s) = coll { Some(s.clone()) } else { None };
        let (coll_reg, coll_ty) = self.generate(coll)?;
        let range_start = match coll {
            CompiledExpr::FunctionCall { name, args, .. }
                if (name == "range" || name == "arange") && args.len() == 1 => Some(0),
            CompiledExpr::FunctionCall { name, args, .. }
                if (name == "range" || name == "arange") && args.len() == 2 =>
            {
                self.static_range_bound(&args[0])
            }
            _ => None,
        };
        let (mut carry_reg, mut carry_ty) = self.generate(init)?;

        let lambda_resolved = match lambda {
            CompiledExpr::Lambda { .. } => lambda.clone(),
            CompiledExpr::Symbol(sym_name) => {
                self.lambda_bindings.get(sym_name).cloned().ok_or_else(|| SheafError::Compile {
                    message: format!("reduce/scan: lambda '{}' not found", sym_name),
                    location: crate::core::error::SourceLocation::unknown(),
                })?
            }
            _ => return Err(SheafError::Compile {
                message: "reduce/scan: first argument must be a lambda".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };
        let (carry_param, elem_param, body) = match lambda_resolved {
            CompiledExpr::Lambda { params, body } if params.len() == 2 => {
                (params[0].clone(), params[1].clone(), *body)
            }
            _ => return Err(SheafError::Compile {
                message: "reduce/scan: lambda must have exactly 2 parameters (carry, elem)".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };

        let (n, kind) = match &coll_ty {
            StableHLOType::Tuple(types, _) if !types.is_empty() => {
                if types.iter().all(|t| matches!(t, StableHLOType::Tuple(..))) {
                    (types.len(), ElemKind::VecTuple(types.clone()))
                } else if types.iter().all(|t| !t.shape().is_empty()) {
                    let n = types[0].shape()[0] as usize;
                    if n == 0 {
                        return Err(SheafError::Compile {
                            message: "reduce/scan: stacked dict has zero elements".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        });
                    }
                    (n, ElemKind::StackedDict(types.clone()))
                } else {
                    return Err(SheafError::Compile {
                        message: "reduce/scan: mixed Tuple element types not supported".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            }
            t if !t.shape().is_empty() => (t.shape()[0] as usize, ElemKind::PlainTensor),
            _ => return Err(SheafError::Compile {
                message: "reduce/scan: cannot determine iteration count from collection type".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };

        // A VecTuple layout describes its elements under numeric keys.
        let resolve_elem_layout = |list_layout: &std::collections::BTreeMap<String, usize>| -> Option<std::collections::BTreeMap<String, usize>> {
            let first_key = list_layout.keys().next()?;
            if first_key.parse::<usize>().is_ok() {
                self.tuple_key_layouts.get(first_key).cloned()
            } else {
                Some(list_layout.clone())
            }
        };

        // The collection may be named directly, renamed by ANF, or selected from a tuple.
        let elem_layout = coll_sym
            .as_deref()
            .and_then(|s| self.tuple_key_layouts.get(s))
            .and_then(&resolve_elem_layout)
            .or_else(|| {
                let layout_key = self.layout_key_map.get(&coll_reg)?;
                let layout = self.tuple_key_layouts.get(layout_key)?;
                resolve_elem_layout(layout)
            })
            .or_else(|| {
                if let CompiledExpr::GetTupleElement { param, indices } = coll {
                    let mut cur = param.clone();
                    for &idx in indices {
                        cur = self.idx_to_key.get(&(cur, idx))?.clone();
                    }
                    let list_layout = self.tuple_key_layouts.get(&cur)?;
                    resolve_elem_layout(list_layout)
                } else {
                    None
                }
            });

        let mut last_scan_result: Option<(Register, StableHLOType)> = None;

        for i in 0..n {
            let (elem_reg, elem_ty) = match &kind {
                ElemKind::VecTuple(types) => {
                    let elem_ty = types[i].clone();
                    let reg = self.emitter.emit_get_tuple_element(&coll_reg, &coll_ty, i, &elem_ty);
                    (reg, elem_ty)
                }
                ElemKind::StackedDict(comp_types) => {
                    let mut parts = Vec::new();
                    let mut part_types = Vec::new();
                    for (k, comp_ty) in comp_types.iter().enumerate() {
                        let comp_reg = self.emitter.emit_get_tuple_element(
                            &coll_reg, &coll_ty, k, comp_ty,
                        );
                        let (slice_reg, slice_ty) =
                            self.emitter.emit_index_axis0(&comp_reg, comp_ty, i as i64);
                        parts.push(slice_reg);
                        part_types.push(slice_ty);
                    }
                    self.emitter.emit_tuple(&parts, &part_types)
                }
                ElemKind::PlainTensor => {
                    self.emitter.emit_index_axis0(&coll_reg, &coll_ty, i as i64)
                }
            };
            if matches!(&kind, ElemKind::PlainTensor)
                && let Some(value) = range_start
                    .and_then(|start| i64::try_from(i).ok()?.checked_add(start))
            {
                self.emitter.set_known_scalar(elem_reg, value as f64);
                self.known_integers.insert(elem_reg, value);
                self.static_tuple_indices.insert(elem_reg);
            }

            if let Some(ref layout) = elem_layout {
                self.tuple_key_layouts.insert(elem_param.clone(), layout.clone());
            }

            let saved_bindings = self.bindings.clone();
            self.bindings.insert(carry_param.clone(), (carry_reg, carry_ty.clone()));
            self.bindings.insert(elem_param.clone(), (elem_reg, elem_ty));
            let result = self.generate(&body);
            self.bindings = saved_bindings;

            if elem_layout.is_some() {
                self.tuple_key_layouts.remove(&elem_param);
            }

            let (step_reg, step_ty) = result?;
            if is_scan {
                match &step_ty {
                    StableHLOType::Tuple(elems, _) if elems.len() == 2 => {
                        let carry_elem_ty = elems[0].clone();
                        carry_reg = self.emitter.emit_get_tuple_element(
                            &step_reg, &step_ty, 0, &carry_elem_ty,
                        );
                        carry_ty = carry_elem_ty;
                        last_scan_result = Some((step_reg, step_ty));
                    }
                    _ => {
                        carry_reg = step_reg;
                        carry_ty = step_ty.clone();
                        let unit_ty = StableHLOType::Tuple(vec![], None);
                        let (unit_reg, _) = self.emitter.emit_tuple(&[], &[]);
                        let (wrapped_reg, wrapped_ty) = self.emitter.emit_tuple(
                            &[carry_reg, unit_reg],
                            &[carry_ty.clone(), unit_ty],
                        );
                        last_scan_result = Some((wrapped_reg, wrapped_ty));
                    }
                }
            } else {
                carry_reg = step_reg;
                carry_ty = step_ty;
            }
        }

        // `first(scan(...))` needs the full carry/output pair.
        if is_scan && let Some(result) = last_scan_result {
            return Ok(result);
        }
        Ok((carry_reg, carry_ty))
    }

    /// Returns the collection length and element extraction strategy.
    fn determine_scan_params(coll_ty: &StableHLOType) -> SheafResult<(usize, ElemKind)> {
        match coll_ty {
            StableHLOType::Tuple(types, _) if !types.is_empty() => {
                if types.iter().all(|t| matches!(t, StableHLOType::Tuple(..))) {
                    Ok((types.len(), ElemKind::VecTuple(types.clone())))
                } else if types.iter().all(|t| !t.shape().is_empty()) {
                    let n = types[0].shape()[0] as usize;
                    if n == 0 {
                        return Err(SheafError::Compile {
                            message: "scan VJP: stacked dict has zero elements".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        });
                    }
                    Ok((n, ElemKind::StackedDict(types.clone())))
                } else {
                    Err(SheafError::Compile {
                        message: "scan VJP: mixed Tuple element types not supported".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            }
            t if !t.shape().is_empty() => Ok((t.shape()[0] as usize, ElemKind::PlainTensor)),
            _ => Err(SheafError::Compile {
                message: "scan VJP: cannot determine iteration count from collection type".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }
    }

    /// Emits the `i`th collection element.
    fn extract_scan_elem(
        &mut self,
        i: usize,
        coll_reg: &Register,
        coll_ty: &StableHLOType,
        kind: &ElemKind,
    ) -> (Register, StableHLOType) {
        match kind {
            ElemKind::VecTuple(types) => {
                let elem_ty = types[i].clone();
                let reg = self.emitter.emit_get_tuple_element(coll_reg, coll_ty, i, &elem_ty);
                (reg, elem_ty)
            }
            ElemKind::StackedDict(comp_types) => {
                let mut parts = Vec::new();
                let mut part_types = Vec::new();
                for (k, comp_ty) in comp_types.iter().enumerate() {
                    let comp_reg = self.emitter.emit_get_tuple_element(
                        coll_reg, coll_ty, k, comp_ty,
                    );
                    let (slice_reg, slice_ty) =
                        self.emitter.emit_index_axis0(&comp_reg, comp_ty, i as i64);
                    parts.push(slice_reg);
                    part_types.push(slice_ty);
                }
                self.emitter.emit_tuple(&parts, &part_types)
            }
            ElemKind::PlainTensor => {
                self.emitter.emit_index_axis0(coll_reg, coll_ty, i as i64)
            }
        }
    }

    /// Differentiates a scan by replaying its carries, then walking them backward.
    pub(super) fn generate_scan_vjp(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let lambda_expr = &args[0];
        let init_expr = &args[1];
        let coll_expr = &args[2];
        let adj_expr = &args[3];

        let lambda = match lambda_expr {
            CompiledExpr::Lambda { .. } => lambda_expr.clone(),
            CompiledExpr::Symbol(s) => {
                self.lambda_bindings.get(s).cloned().ok_or_else(|| SheafError::Compile {
                    message: format!("scan VJP: lambda '{}' not found", s),
                    location: crate::core::error::SourceLocation::unknown(),
                })?
            }
            _ => return Err(SheafError::Compile {
                message: "scan VJP: first argument must be a lambda".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };
        let (carry_param, elem_param, body) = match lambda {
            CompiledExpr::Lambda { params, body } if params.len() == 2 => {
                (params[0].clone(), params[1].clone(), *body)
            }
            _ => return Err(SheafError::Compile {
                message: "scan VJP: lambda must have exactly 2 parameters".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };

        let (init_reg, init_ty) = self.generate(init_expr)?;
        let (coll_reg, coll_ty) = self.generate(coll_expr)?;
        let (adj_reg, adj_ty) = self.generate(adj_expr)?;

        let (n, kind) = Self::determine_scan_params(&coll_ty)?;

        let coll_sym = if let CompiledExpr::Symbol(s) = coll_expr { Some(s.clone()) } else { None };
        let elem_layout = coll_sym
            .as_deref()
            .and_then(|s| self.tuple_key_layouts.get(s))
            .cloned()
            .or_else(|| {
                let layout_key = self.layout_key_map.get(&coll_reg)?;
                self.tuple_key_layouts.get(layout_key).cloned()
            })
            .or_else(|| {
                if let CompiledExpr::GetTupleElement { param, indices } = coll_expr {
                    let mut cur = param.clone();
                    for &idx in indices {
                        cur = self.idx_to_key.get(&(cur, idx))?.clone();
                    }
                    let list_layout = self.tuple_key_layouts.get(&cur)?;
                    let first_child_key = list_layout.values().next()
                        .and_then(|_| list_layout.keys().next())?;
                    self.tuple_key_layouts.get(first_child_key).cloned()
                } else {
                    None
                }
            });

        // The backward pass needs every forward carry.
        let mut carries: Vec<(Register, StableHLOType)> = Vec::new();
        carries.push((init_reg, init_ty.clone()));
        let mut fwd_carry_reg = init_reg;
        let mut fwd_carry_ty = init_ty.clone();

        for i in 0..n {
            let (elem_reg, elem_ty) = self.extract_scan_elem(i, &coll_reg, &coll_ty, &kind);

            if let Some(ref layout) = elem_layout {
                self.tuple_key_layouts.insert(elem_param.clone(), layout.clone());
            }

            let saved = self.bindings.clone();
            self.bindings.insert(carry_param.clone(), (fwd_carry_reg, fwd_carry_ty.clone()));
            self.bindings.insert(elem_param.clone(), (elem_reg, elem_ty));
            let (result_reg, result_ty) = self.generate(&body)?;
            self.bindings = saved;

            if elem_layout.is_some() {
                self.tuple_key_layouts.remove(&elem_param);
            }

            match &result_ty {
                StableHLOType::Tuple(elems, _) if elems.len() >= 2 => {
                    let carry_ty = elems[0].clone();
                    let carry_reg = self.emitter.emit_get_tuple_element(
                        &result_reg, &result_ty, 0, &carry_ty,
                    );
                    carries.push((carry_reg, carry_ty.clone()));
                    fwd_carry_reg = carry_reg;
                    fwd_carry_ty = carry_ty;
                }
                _ => {
                    carries.push((result_reg, result_ty.clone()));
                    fwd_carry_reg = result_reg;
                    fwd_carry_ty = result_ty;
                }
            }
        }


        // Scan-body field access is lowered after the element layout is known.
        let lowered_body = if let Some(ref layout) = elem_layout {
            let mut index_map = std::collections::BTreeMap::new();
            for (key, &idx) in layout.iter() {
                index_map.insert(vec![key.clone()], vec![idx]);
            }
            for key in layout.keys() {
                let deep = build_deep_index_map(key, &self.tuple_key_layouts);
                index_map.extend(deep);
            }
            crate::lowering::config::lower_get_calls(&body, &elem_param, &index_map)
        } else {
            body.clone()
        };

        let body_anf = to_anf(&lowered_body);
        let (mut body_bindings, body_result) = match &body_anf {
            CompiledExpr::Let { bindings, body } => (bindings.clone(), body.as_ref().clone()),
            other => (vec![], other.clone()),
        };

        // The scan body's first result is the next carry.
        let carry_result = match &body_result {
            CompiledExpr::Vector(elems) if !elems.is_empty() => elems[0].clone(),
            _ => body_result.clone(),
        };

        let mut body_shapes: HashMap<String, Vec<i64>> = HashMap::new();
        body_shapes.insert(carry_param.clone(), fwd_carry_ty.shape().to_vec());

        let sample_elem_ty = {
            let (_, ty) = self.extract_scan_elem(0, &coll_reg, &coll_ty, &kind);
            ty
        };

        match &sample_elem_ty {
            StableHLOType::Tuple(elem_tys, _) => {
                for (idx, ety) in elem_tys.iter().enumerate() {
                    let key = format!("{}@[{}]", elem_param, idx);
                    body_shapes.insert(key, ety.shape().to_vec());
                }
            }
            _ => {
                body_shapes.insert(elem_param.clone(), sample_elem_ty.shape().to_vec());
            }
        }

        for (name, val) in &body_bindings {
            if let BindingPattern::Simple(name_str) = name
                && let Some(sh) = try_infer_shape(val, &body_shapes)
            {
                body_shapes.insert(name_str.clone(), sh);
            }
        }

        let mut wrt = vec![carry_param.clone()];
        let elem_gte_names: Vec<(String, usize)> = if let StableHLOType::Tuple(elem_tys, _) = &sample_elem_ty {
            let mut names_by_field: Vec<Option<String>> = vec![None; elem_tys.len()];
            for (name, val) in &body_bindings {
                if let CompiledExpr::GetTupleElement { param, indices } = val
                    && param == &elem_param
                    && indices.len() == 1
                    && let BindingPattern::Simple(name_str) = name
                    && let Some(field_name) = names_by_field.get_mut(indices[0])
                {
                    *field_name = Some(name_str.clone());
                }
            }
            for (field_idx, field_name) in names_by_field.iter_mut().enumerate() {
                if field_name.is_none() {
                    let name = fresh_scan_field_symbol(
                        &body_bindings,
                        &carry_param,
                        &elem_param,
                        field_idx,
                    );
                    body_bindings.push((
                        BindingPattern::Simple(name.clone()),
                        CompiledExpr::GetTupleElement {
                            param: elem_param.clone(),
                            indices: vec![field_idx],
                        },
                    ));
                    body_shapes.insert(name.clone(), elem_tys[field_idx].shape().to_vec());
                    *field_name = Some(name);
                }
            }
            names_by_field
                .into_iter()
                .enumerate()
                .map(|(field_idx, name)| {
                    name.map(|name| (name, field_idx)).ok_or_else(|| {
                        SheafError::AutodiffMissingGradientOutput {
                            symbol: format!("scan gradient field {}", field_idx),
                        }
                    })
                })
                .collect::<SheafResult<_>>()?
        } else {
            wrt.push(elem_param.clone());
            vec![]
        };
        for (name, _) in &elem_gte_names {
            wrt.push(name.clone());
        }

        let body_bindings_str: Vec<(String, CompiledExpr)> = body_bindings
            .iter()
            .filter_map(|(pat, expr)| {
                if let BindingPattern::Simple(s) = pat {
                    Some((s.clone(), expr.clone()))
                } else {
                    None
                }
            })
            .collect();

        let ReverseGradResult {
            backward_bindings: mut body_bwd,
            gradients: body_grad_map,
        } = reverse_grad(&body_bindings_str, &carry_result, &wrt, &body_shapes)?;

        // Each backward step starts from the adjoint produced by the next step.
        if !body_bwd.is_empty() {
            body_bwd[0].1 = CompiledExpr::Symbol("__scan_adj_carry".to_string());
        }

        let mut adj_carry_reg = adj_reg;
        let mut adj_carry_ty = adj_ty.clone();
        // Element adjoints are collected in reverse iteration order.
        let mut adj_elem_parts: Vec<Vec<(Register, StableHLOType)>> = Vec::new();

        for i in (0..n).rev() {
            let (elem_reg, elem_ty) = self.extract_scan_elem(i, &coll_reg, &coll_ty, &kind);

            let saved_bindings = self.bindings.clone();
            let saved_layouts = self.tuple_key_layouts.clone();
            self.bindings.insert(carry_param.clone(), carries[i].clone());
            self.bindings.insert(elem_param.clone(), (elem_reg, elem_ty.clone()));
            if let Some(ref layout) = elem_layout {
                self.tuple_key_layouts.insert(elem_param.clone(), layout.clone());
            }

            for (name, val) in &body_bindings {
                if let BindingPattern::Simple(name_str) = name {
                    self.generate_binding(name_str, val)?;
                }
            }

            self.bindings.insert("__scan_adj_carry".to_string(), (adj_carry_reg, adj_carry_ty.clone()));

            for (name, val) in &body_bwd {
                self.generate_binding(name, val)?;
            }

            match crate::autodiff::reverse::gradient_output(&body_grad_map, &carry_param)? {
                crate::autodiff::reverse::GradientOutput::Computed(grad_sym) => {
                    let &(reg, ref ty) = self.bindings.get(grad_sym).ok_or_else(|| {
                        SheafError::AutodiffMissingGradientOutput {
                            symbol: grad_sym.clone(),
                        }
                    })?;
                    adj_carry_reg = reg;
                    adj_carry_ty = ty.clone();
                }
                crate::autodiff::reverse::GradientOutput::ProvenZero => {
                    let (reg, ty) = self.emitter.emit_zeros(fwd_carry_ty.shape());
                    adj_carry_reg = reg;
                    adj_carry_ty = ty;
                }
            }

            if matches!(&sample_elem_ty, StableHLOType::Tuple(..)) {
                let n_fields = match &sample_elem_ty {
                    StableHLOType::Tuple(tys, _) => tys.len(),
                    _ => 0,
                };
                let mut parts = vec![None; n_fields];
                for (gte_name, field_idx) in &elem_gte_names {
                    let output = crate::autodiff::reverse::gradient_output(&body_grad_map, gte_name)?;
                    let field_ty = match &sample_elem_ty {
                        StableHLOType::Tuple(tys, _) => tys.get(*field_idx).ok_or_else(|| {
                            SheafError::AutodiffMissingGradientOutput {
                                symbol: gte_name.clone(),
                            }
                        })?,
                        _ => return Err(SheafError::AutodiffMissingGradientOutput {
                            symbol: gte_name.clone(),
                        }),
                    };
                    let part = match output {
                        crate::autodiff::reverse::GradientOutput::Computed(grad_sym) => {
                            let &(reg, ref ty) = self.bindings.get(grad_sym).ok_or_else(|| {
                                SheafError::AutodiffMissingGradientOutput {
                                    symbol: grad_sym.clone(),
                                }
                            })?;
                            (reg, ty.clone())
                        }
                        crate::autodiff::reverse::GradientOutput::ProvenZero => {
                            self.emitter.emit_zeros(field_ty.shape())
                        }
                    };
                    parts[*field_idx] = Some(part);
                }
                // Unused fields receive a zero adjoint.
                let resolved_parts: Vec<(Register, StableHLOType)> = parts.into_iter()
                    .enumerate()
                    .map(|(idx, opt)| opt.ok_or_else(|| {
                        SheafError::AutodiffMissingGradientOutput {
                            symbol: format!("scan gradient field {}", idx),
                        }
                    }))
                    .collect::<SheafResult<_>>()?;
                adj_elem_parts.push(resolved_parts);
            } else {
                let part = match crate::autodiff::reverse::gradient_output(&body_grad_map, &elem_param)? {
                    crate::autodiff::reverse::GradientOutput::Computed(grad_sym) => {
                        let &(reg, ref ty) = self.bindings.get(grad_sym).ok_or_else(|| {
                            SheafError::AutodiffMissingGradientOutput {
                                symbol: grad_sym.clone(),
                            }
                        })?;
                        (reg, ty.clone())
                    }
                    crate::autodiff::reverse::GradientOutput::ProvenZero => {
                        self.emitter.emit_zeros(sample_elem_ty.shape())
                    }
                };
                adj_elem_parts.push(vec![part]);
            }

            self.bindings = saved_bindings;
            self.tuple_key_layouts = saved_layouts;
        }

        adj_elem_parts.reverse();

        let (adj_coll_reg, adj_coll_ty) = match &kind {
            ElemKind::StackedDict(comp_types) => {
                let n_fields = comp_types.len();
                let mut stacked_fields = Vec::new();
                let mut stacked_types = Vec::new();
                for field_idx in 0..n_fields {
                    let field_regs: Vec<Register> = adj_elem_parts.iter()
                        .map(|parts| parts[field_idx].0)
                        .collect();
                    let field_ty = &adj_elem_parts[0][field_idx].1;

                    let old_shape = field_ty.shape();
                    let mut expanded_shape = vec![1i64];
                    expanded_shape.extend_from_slice(old_shape);

                    let mut reshaped_regs = Vec::new();
                    let mut reshaped_types = Vec::new();
                    for &reg in &field_regs {
                        let (r, t) = self.emitter.emit_reshape(&reg, field_ty, &expanded_shape);
                        reshaped_regs.push(r);
                        reshaped_types.push(t);
                    }

                    let (concat_reg, concat_ty) = self.emitter.emit_concatenate(
                        &reshaped_regs, &reshaped_types, 0,
                    );
                    stacked_fields.push(concat_reg);
                    stacked_types.push(concat_ty);
                }
                self.emitter.emit_tuple(&stacked_fields, &stacked_types)
            }
            ElemKind::PlainTensor => {
                let field_regs: Vec<Register> = adj_elem_parts.iter()
                    .map(|parts| parts[0].0)
                    .collect();
                let field_ty = &adj_elem_parts[0][0].1;

                let old_shape = field_ty.shape();
                let mut expanded_shape = vec![1i64];
                expanded_shape.extend_from_slice(old_shape);

                let mut reshaped_regs = Vec::new();
                let mut reshaped_types = Vec::new();
                for &reg in &field_regs {
                    let (r, t) = self.emitter.emit_reshape(&reg, field_ty, &expanded_shape);
                    reshaped_regs.push(r);
                    reshaped_types.push(t);
                }

                self.emitter.emit_concatenate(&reshaped_regs, &reshaped_types, 0)
            }
            ElemKind::VecTuple(_types) => {
                let mut elem_regs = Vec::new();
                let mut elem_tys = Vec::new();
                for parts in &adj_elem_parts {
                    let regs: Vec<Register> = parts.iter().map(|(r, _)| *r).collect();
                    let tys: Vec<StableHLOType> = parts.iter().map(|(_, t)| t.clone()).collect();
                    let (r, t) = self.emitter.emit_tuple(&regs, &tys);
                    elem_regs.push(r);
                    elem_tys.push(t);
                }
                self.emitter.emit_tuple(&elem_regs, &elem_tys)
            }
        };

        let adj_init_reg = adj_carry_reg;
        let adj_init_ty = adj_carry_ty;
        Ok(self.emitter.emit_tuple(
            &[adj_init_reg, adj_coll_reg],
            &[adj_init_ty, adj_coll_ty],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::{GradientOutput, reverse_grad};

    #[test]
    fn scan_field_symbol_avoids_existing_bindings_and_parameters() {
        let bindings = vec![(
            BindingPattern::Simple("__scan_elem_field_1".to_string()),
            CompiledExpr::Float(0.0),
        )];
        let symbol = fresh_scan_field_symbol(&bindings, "carry", "elem", 1);
        assert_eq!(symbol, "__scan_elem_field_1_1");

        let parameter_collision = fresh_scan_field_symbol(
            &[],
            "__scan_elem_field_1",
            "elem",
            1,
        );
        assert_eq!(parameter_collision, "__scan_elem_field_1_1");
    }

    #[test]
    fn scan_tuple_unused_field_is_a_proven_zero_output() {
        let bindings = vec![
            (
                "active".to_string(),
                CompiledExpr::GetTupleElement {
                    param: "elem".to_string(),
                    indices: vec![0],
                },
            ),
            (
                "result".to_string(),
                CompiledExpr::FunctionCall {
                    name: "+".to_string(),
                    args: vec![
                        CompiledExpr::Symbol("carry".to_string()),
                        CompiledExpr::Symbol("active".to_string()),
                    ],
                    loc: None,
                },
            ),
        ];
        let wrt = vec![
            "carry".to_string(),
            "active".to_string(),
            "__scan_elem_field_1".to_string(),
        ];
        let result = reverse_grad(
            &bindings,
            &CompiledExpr::Symbol("result".to_string()),
            &wrt,
            &HashMap::new(),
        ).unwrap();

        assert!(matches!(
            result.gradient_output("active"),
            Ok(GradientOutput::Computed(_))
        ));
        assert!(matches!(
            result.gradient_output("__scan_elem_field_1"),
            Ok(GradientOutput::ProvenZero)
        ));
    }
}
