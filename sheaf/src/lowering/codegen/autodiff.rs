// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Autodiff-related codegen: tuple gradient assembly and broadcast reduction.
//! Used by the VAG (value-and-grad) trace-based path in jit.rs.

use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::error::{SheafError, SheafResult};
use crate::lowering::config::lower_get_calls;
use crate::lowering::transforms::{lower_inlined_gets, unroll_reduces};
use super::helpers::TupleLeaf;
use super::{CodeGenerator, collect_tuple_references, collect_tuple_type_leaves, expand_tuple_to_symbols};
use crate::autodiff::reverse::GradientOutput;
use super::control_flow::build_deep_index_map;
use std::collections::{BTreeMap, HashMap};

fn vag_argument_layout_key(
    call_arg: &CompiledExpr,
    wrt_reg: &Register,
    register_layouts: &HashMap<Register, String>,
    tuple_layouts: &HashMap<String, BTreeMap<String, usize>>,
) -> Option<String> {
    register_layouts.get(wrt_reg).cloned().or_else(|| {
        if let CompiledExpr::Symbol(name) = call_arg
            && tuple_layouts.contains_key(name)
        {
            Some(name.clone())
        } else {
            None
        }
    })
}

fn rewrite_gte_param(expr: &CompiledExpr, aliases: &[String], canonical: &str) -> CompiledExpr {
    match expr {
        CompiledExpr::GetTupleElement { param, indices } if aliases.contains(param) => {
            CompiledExpr::GetTupleElement { param: canonical.to_string(), indices: indices.clone() }
        }
        other => other.map_children(|e| rewrite_gte_param(e, aliases, canonical)),
    }
}

fn zero_gradient_expr(param_ty: &StableHLOType, indices: &[usize]) -> SheafResult<CompiledExpr> {
    let mut leaf_ty = param_ty;
    for &index in indices {
        let StableHLOType::Tuple(elems, _) = leaf_ty else {
            return Err(SheafError::AutodiffMissingGradientOutput {
                symbol: format!("gradient leaf at {:?}", indices),
            });
        };
        let Some(next) = elems.get(index) else {
            return Err(SheafError::AutodiffMissingGradientOutput {
                symbol: format!("gradient leaf at {:?}", indices),
            });
        };
        leaf_ty = next;
    }
    let shape = leaf_ty.shape();
    if shape.is_empty() {
        Ok(CompiledExpr::Float(0.0))
    } else {
        Ok(CompiledExpr::FunctionCall {
            name: "zeros".to_string(),
            args: vec![CompiledExpr::Vector(
                shape.iter().map(|&d| CompiledExpr::Integer(d)).collect(),
            )],
            loc: None,
        })
    }
}

fn collect_param_aliases(expr: &CompiledExpr, param_name: &str, aliases: &mut Vec<String>) {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            for (name, value) in bindings {
                match value {
                    CompiledExpr::Symbol(s) if s == param_name => {
                        if let BindingPattern::Simple(name_str) = &name
                            && !aliases.contains(name_str) {
                                aliases.push(name_str.clone());
                            }
                    }
                    CompiledExpr::GetTupleElement { param, .. } if param == param_name => {
                        if let BindingPattern::Simple(name_str) = &name
                            && !aliases.contains(name_str) {
                                aliases.push(name_str.clone());
                            }
                    }
                    _ => {}
                }
                collect_param_aliases(value, param_name, aliases);
            }
            collect_param_aliases(body, param_name, aliases);
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args { collect_param_aliases(a, param_name, aliases); }
        }
        CompiledExpr::LambdaCall { callee, args } => {
            collect_param_aliases(callee, param_name, aliases);
            for a in args { collect_param_aliases(a, param_name, aliases); }
        }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            collect_param_aliases(condition, param_name, aliases);
            collect_param_aliases(then_branch, param_name, aliases);
            if let Some(e) = else_branch { collect_param_aliases(e, param_name, aliases); }
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs { collect_param_aliases(e, param_name, aliases); }
        }
        _ => {}
    }
}

impl<'a> CodeGenerator<'a> {
    pub(crate) fn build_grad_tuple_from_map(
        &mut self,
        leaves: &[TupleLeaf],
        param_ty: &StableHLOType,
        grad_map: &std::collections::HashMap<String, CompiledExpr>,
    ) -> SheafResult<(Register, StableHLOType)> {
        self.build_grad_tuple_from_map_rec(leaves, param_ty, grad_map, &[])
    }

    fn build_grad_tuple_from_map_rec(
        &mut self,
        leaves: &[TupleLeaf],
        ty: &StableHLOType,
        grad_map: &std::collections::HashMap<String, CompiledExpr>,
        prefix: &[usize],
    ) -> SheafResult<(Register, StableHLOType)> {
        match ty {
            StableHLOType::Tuple(elems, _) => {
                let mut sub_regs = Vec::new();
                let mut sub_tys = Vec::new();
                for (i, elem_ty) in elems.iter().enumerate() {
                    let mut child_prefix = prefix.to_vec();
                    child_prefix.push(i);
                    let (r, t) = self.build_grad_tuple_from_map_rec(
                        leaves, elem_ty, grad_map, &child_prefix,
                    )?;
                    sub_regs.push(r);
                    sub_tys.push(t);
                }
                Ok(self.emitter.emit_tuple(&sub_regs, &sub_tys))
            }
            _leaf_ty => {
                if let Some(leaf) = leaves.iter().find(|l| l.indices == prefix) {
                    let grad_expr = grad_map.get(&leaf.symbol).cloned().ok_or_else(|| {
                        SheafError::AutodiffMissingGradientOutput {
                            symbol: leaf.symbol.clone(),
                        }
                    })?;
                    let (grad_reg, grad_ty) = self.generate(&grad_expr)?;
                    self.reduce_broadcast_grad(grad_reg, &grad_ty, ty)
                } else {
                    Err(SheafError::AutodiffMissingGradientOutput {
                        symbol: format!("gradient leaf at {:?}", prefix),
                    })
                }
            }
        }
    }

    pub(crate) fn reduce_broadcast_grad(
        &mut self,
        grad_reg: Register,
        grad_ty: &StableHLOType,
        param_ty: &StableHLOType,
    ) -> SheafResult<(Register, StableHLOType)> {
        let grad_shape = grad_ty.shape();
        let param_shape = param_ty.shape();

        if grad_shape == param_shape {
            return Ok((grad_reg, grad_ty.clone()));
        }

        // Scalar gradient broadcast up to a tensor parameter shape.
        // Arises for fully stop-gradient'd parameters: no adjoint is accumulated
        // so the gradient defaults to scalar 0.0.
        if grad_shape.is_empty() && !param_shape.is_empty() {
            let bcast_reg = self.emitter.emit_broadcast(&grad_reg, grad_ty, param_ty);
            return Ok((bcast_reg, param_ty.clone()));
        }

        let grad_elems: i64 = grad_shape.iter().product();
        let param_elems: i64 = param_shape.iter().product();
        if grad_elems == param_elems && grad_elems > 0 && grad_shape != param_shape {
            let (r, t) = self.emitter.emit_reshape(&grad_reg, grad_ty, param_shape);
            return Ok((r, t));
        }

        let extra = grad_shape.len().saturating_sub(param_shape.len());
        if extra == 0 {
            return Ok((grad_reg, grad_ty.clone()));
        }

        let trailing = &grad_shape[extra..];
        if trailing != param_shape {
            return Ok((grad_reg, grad_ty.clone()));
        }

        let mut cur_reg = grad_reg;
        let mut cur_ty = grad_ty.clone();
        for _ in 0..extra {
            let (r, t) = self.emitter.emit_reduce_sum(&cur_reg, &cur_ty, 0, false);
            cur_reg = r;
            cur_ty = t;
        }
        Ok((cur_reg, cur_ty))
    }

    pub(super) fn generate_vag_inline(
        &mut self,
        loss_fn_expr: &CompiledExpr,
        call_args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
    let (fn_params, fn_body) = match loss_fn_expr {
        CompiledExpr::Lambda { params, body, .. } => {
    let b = body.as_ref().clone();
    (params.clone(), b)
        }
            other => {
                return Err(SheafError::Compile {
                    message: format!("value-and-grad: expected lambda, got {:?}", other),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }
        };

        if call_args.len() != 1 {
            return Err(SheafError::Compile {
                message: format!("value-and-grad inline: expected 1 arg, got {}", call_args.len()),
                location: crate::core::error::SourceLocation::unknown(),
            });
        }

        let (wrt_reg, wrt_ty) = self.generate(&call_args[0])?;
        let param_name = fn_params.first()
            .ok_or_else(|| SheafError::Compile {
                message: "value-and-grad: lambda must have at least 1 parameter".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })?;

        let saved_bindings = self.bindings.clone();
        let saved_lambda_bindings = self.lambda_bindings.clone();

        self.bindings
            .insert(param_name.clone(), (wrt_reg, wrt_ty.clone()));

        let layout_key = vag_argument_layout_key(
            &call_args[0],
            &wrt_reg,
            &self.layout_key_map,
            &self.tuple_key_layouts,
        );
        if let Some(layout_key) = layout_key {
            if let Some(layout) = self.tuple_key_layouts.get(&layout_key).cloned() {
                self.tuple_key_layouts.insert(param_name.clone(), layout);
            }
            let idx_entries: Vec<_> = self
                .idx_to_key
                .iter()
                .filter(|((name, _), _)| name == &layout_key)
                .map(|((_, idx), key)| (*idx, key.clone()))
                .collect();
            for (idx, key) in idx_entries {
                self.idx_to_key.insert((param_name.clone(), idx), key);
            }
        }

        let fn_body = if self.tuple_key_layouts.contains_key(param_name) {
            let index_map = build_deep_index_map(param_name, &self.tuple_key_layouts);
            let mut param_aliases = vec![param_name.clone()];
            collect_param_aliases(&fn_body, param_name, &mut param_aliases);
            let mut body = fn_body;
            for alias in &param_aliases {
                body = lower_get_calls(&body, alias, &index_map);
            }
            if param_aliases.len() > 1 {
                body = rewrite_gte_param(&body, &param_aliases[1..], param_name);
            }

            body = unroll_reduces(&body, &[(param_name.clone(), wrt_ty.clone())]);
            body = lower_get_calls(&body, param_name, &index_map);
            lower_inlined_gets(&body, &[(param_name.clone(), index_map)])
        } else {
            fn_body
        };

        let (leaves, structural_references, expanded_body) = match &wrt_ty {
            StableHLOType::Tuple(..) => {
                let leaves = collect_tuple_type_leaves(param_name, &wrt_ty);
                let references = collect_tuple_references(&fn_body, param_name);
                let expanded = expand_tuple_to_symbols(&fn_body, param_name, &wrt_ty);
                (leaves, references, expanded)
            }
            _ => (
                Vec::<TupleLeaf>::new(),
                Vec::<TupleLeaf>::new(),
                fn_body.clone(),
            ),
        };

        let mut all_wrt_symbols: Vec<String> = Vec::new();
        if !leaves.is_empty() {
            for leaf in &leaves {
                let gte = CompiledExpr::GetTupleElement {
                    param: param_name.clone(),
                    indices: leaf.indices.clone(),
                };
                let (reg, ty) = self.generate(&gte)?;
                self.bindings.insert(leaf.symbol.clone(), (reg, ty));
                all_wrt_symbols.push(leaf.symbol.clone());
            }
            for reference in &structural_references {
                if leaves.iter().any(|leaf| leaf.indices == reference.indices) {
                    continue;
                }
                let gte = CompiledExpr::GetTupleElement {
                    param: param_name.clone(),
                    indices: reference.indices.clone(),
                };
                let (reg, ty) = self.generate(&gte)?;
                self.bindings.insert(reference.symbol.clone(), (reg, ty));
            }
        } else {
            all_wrt_symbols.push(param_name.clone());
        }

    let anf_expr = crate::autodiff::reverse::to_anf(&expanded_body);
    let (anf_bindings, anf_body) = match &anf_expr {
        CompiledExpr::Let { bindings, body } => (bindings.clone(), body.as_ref().clone()),
        other => (vec![], other.clone()),
    };

        for (name, value_expr) in &anf_bindings {
            if let BindingPattern::Simple(name_str) = name {
                self.generate_binding(name_str, value_expr)?;
            }
        }

        let anf_bindings_str: Vec<(String, CompiledExpr)> = anf_bindings
            .iter()
            .filter_map(|(pat, expr)| {
                if let BindingPattern::Simple(s) = pat {
                    Some((s.clone(), expr.clone()))
                } else {
                    None
                }
            })
            .collect();

        let (loss_reg, loss_ty) = self.generate(&anf_body)?;

        let shape_map: HashMap<String, Vec<i64>> = self.binding_shapes();

        let crate::autodiff::reverse::ReverseGradResult {
            backward_bindings,
            gradients: grad_sym_map,
        } = crate::autodiff::reverse::reverse_grad(
            &anf_bindings_str,
            &anf_body,
            &all_wrt_symbols,
            &shape_map,
        )?;

    let backward_bindings: Vec<(String, CompiledExpr)> = backward_bindings
            .into_iter()
            .map(|(name, expr)| {
                use crate::autodiff::simplify;
                (name, simplify(expr))
            })
            .collect();

        for (name, value_expr) in &backward_bindings {
            let (reg, ty) = self.generate(value_expr)?;
            self.bindings.insert(name.clone(), (reg, ty));
        }

    let (grad_reg, grad_ty) = if !leaves.is_empty() {
        let leaf_grad_map: HashMap<String, CompiledExpr> = leaves.iter()
            .map(|leaf| {
                let grad_expr = match crate::autodiff::reverse::gradient_output(&grad_sym_map, &leaf.symbol)? {
                    GradientOutput::Computed(sym_name) => CompiledExpr::Symbol(sym_name.clone()),
                    GradientOutput::ProvenZero => zero_gradient_expr(&wrt_ty, &leaf.indices)?,
                };
                Ok((leaf.symbol.clone(), grad_expr))
            })
            .collect::<SheafResult<_>>()?;
        self.build_grad_tuple_from_map(&leaves, &wrt_ty, &leaf_grad_map)?
    } else {
        let grad_expr = match crate::autodiff::reverse::gradient_output(&grad_sym_map, param_name)? {
            GradientOutput::Computed(sym_name) => CompiledExpr::Symbol(sym_name.clone()),
            GradientOutput::ProvenZero => zero_gradient_expr(&wrt_ty, &[])?
        };
        let (grad_reg, grad_ty) = self.generate(&grad_expr)?;
        self.reduce_broadcast_grad(grad_reg, &grad_ty, &wrt_ty)?
    };

    self.lambda_bindings = saved_lambda_bindings;

        let (tuple_reg, tuple_ty) = self.emitter.emit_tuple(
            &[loss_reg, grad_reg],
            &[loss_ty, grad_ty],
        );

        self.bindings = saved_bindings;

        Ok((tuple_reg, tuple_ty))
    }
}

#[cfg(test)]
mod tests {
    use super::vag_argument_layout_key;
    use crate::core::expr::CompiledExpr;
    use crate::lowering::stablehlo::Register;
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn vag_layout_falls_back_to_the_argument_symbol() {
        let argument = CompiledExpr::Symbol("params".to_string());
        let register = Register::Reg(7);
        let tuple_layouts = HashMap::from([(
            "params".to_string(),
            BTreeMap::from([("transformer".to_string(), 0)]),
        )]);

        assert_eq!(
            vag_argument_layout_key(
                &argument,
                &register,
                &HashMap::new(),
                &tuple_layouts,
            ),
            Some("params".to_string()),
        );
    }
}
