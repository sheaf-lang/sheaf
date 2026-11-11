// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Collection and indexing builtin codegen: get, first, second, last, assoc,
//! shape, ndim, count/len, top_k, int, float, get-in.

use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::core::expr::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use super::CodeGenerator;

impl<'a> CodeGenerator<'a> {
    pub(super) fn generate_collection_builtin(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> Option<SheafResult<(Register, StableHLOType)>> {
        match name {
            "shape" if !args.is_empty() => Some(self.gen_shape(args)),
            "ndim" if args.len() == 1 => Some(self.gen_ndim(args)),
            "count" | "len" if args.len() == 1 => Some(self.gen_count_len(name, args)),
            "first" if args.len() == 1 => Some(self.gen_first(args)),
            "second" if args.len() == 1 => Some(self.gen_second(args)),
            "last" if args.len() == 1 => Some(self.gen_last(args)),
            "get" if args.len() >= 2 => Some(self.gen_get(args)),
            "get-in" if args.len() >= 2 => Some(self.gen_get_in()),
            "assoc" if args.len() >= 3 && args.len() % 2 == 1 => Some(self.gen_assoc(args)),
            "append" if args.len() == 2 => Some(self.gen_append(args)),
            "top_k" if args.len() == 2 => Some(self.gen_top_k(args)),
            "int" if args.len() == 1 => Some(self.generate(&args[0])),
            "float" if args.len() == 1 => Some(self.generate(&args[0])),
            _ => None,
        }
    }

    fn gen_append(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (list_reg, list_ty) = self.generate(&args[0])?;
        if !self.list_regs.contains(&list_reg) {
            return Err(SheafError::Compile {
                message: "append: first argument must be a list".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            });
        }
        let StableHLOType::Tuple(elems, None) = &list_ty else {
            return Err(SheafError::Compile {
                message: "append: first argument must be a list".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            });
        };
        let mut regs = Vec::with_capacity(elems.len() + 1);
        let mut types = elems.clone();
        for (index, ty) in elems.iter().enumerate() {
            regs.push(self.emitter.emit_get_tuple_element(&list_reg, &list_ty, index, ty));
        }
        let (value_reg, value_ty) = self.generate(&args[1])?;
        regs.push(value_reg);
        types.push(value_ty);
        let result = self.emitter.emit_tuple(&regs, &types);
        self.list_regs.insert(result.0);
        Ok(result)
    }

    fn gen_shape(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (_, operand_ty) = self.generate(&args[0])?;
        let dims = operand_ty.shape();
        if dims.is_empty() {
            return Err(SheafError::Compile {
                message: "shape: cannot query shape of scalar or tuple".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            });
        }
        if args.len() >= 2 {
            let ax_val = match &args[1] {
                CompiledExpr::Integer(n) => Some(*n),
                CompiledExpr::Float(f) => Some(*f as i64),
                _ => {
                    // Try resolving via known scalar constants
                    if let Ok((reg, _)) = self.generate(&args[1]) {
                        self.emitter.known_scalar_value(&reg).map(|v| v as i64)
                    } else {
                        None
                    }
                }
            };
            if let Some(ax) = ax_val {
                let idx = if ax < 0 { (dims.len() as i64 + ax) as usize } else { ax as usize };
                let dim_val = dims[idx] as f64;
                let reg = self.emitter.emit_constant_f32(dim_val);
                self.emitter.set_known_scalar(reg, dim_val);
                self.known_integers.insert(reg, dims[idx]);
                Ok((reg, StableHLOType::ScalarF32))
            } else {
                Err(SheafError::Compile {
                    message: "shape: axis must be a constant integer".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        } else {
        let data: Vec<f64> = dims.iter().map(|&d| d as f64).collect();
        let (reg, ty) = self.emitter.emit_integer_vector_constant(dims);
        self.emitter.set_known_tensor(reg, data);
        self.known_integer_vectors.insert(reg, dims.to_vec());
        Ok((reg, ty))
        }
    }

    fn gen_ndim(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (_, operand_ty) = self.generate(&args[0])?;
    let ndim = operand_ty.shape().len() as f64;
    let reg = self.emitter.emit_constant_f32(ndim);
    self.emitter.set_known_scalar(reg, ndim);
    self.known_integers.insert(reg, operand_ty.shape().len() as i64);
    Ok((reg, StableHLOType::ScalarF32))
    }

    fn gen_count_len(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (_, operand_ty) = self.generate(&args[0])?;
        let len = match &operand_ty {
            StableHLOType::Tuple(elems, _) => elems.len() as f64,
            ty if !ty.shape().is_empty() => ty.shape()[0] as f64,
            _ => {
                return Err(SheafError::Compile {
                    message: format!("{}: cannot get length of scalar", name),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }
        };
        let reg = self.emitter.emit_constant_f32(len);
        self.emitter.set_known_scalar(reg, len);
        self.known_integers.insert(reg, len as i64);
        Ok((reg, StableHLOType::ScalarF32))
    }

    fn gen_first(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        match &operand_ty {
            StableHLOType::Tuple(elems, _) => {
                let elem_ty = elems[0].clone();
                let reg = self.emitter.emit_get_tuple_element(&operand_reg, &operand_ty, 0, &elem_ty);
                Ok((reg, elem_ty))
            }
            _ if operand_ty.shape().is_empty() => {
                Ok((operand_reg, operand_ty))
            }
            _ => {
        if let Some(vals) = self.emitter.known_tensor_values(&operand_reg)
            && !vals.is_empty()
        {
            let val = vals[0];
            let reg = self.emitter.emit_constant_f32(val);
            self.emitter.set_known_scalar(reg, val);
            if let Some(&value) = self.known_integer_vectors.get(&operand_reg)
                .and_then(|values| values.first()) {
                self.known_integers.insert(reg, value);
            }
            return Ok((reg, StableHLOType::ScalarF32));
        }
                let (reg, ty) = self.emitter.emit_index_axis0(&operand_reg, &operand_ty, 0);
                Ok((reg, ty))
            }
        }
    }

    fn gen_second(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        match &operand_ty {
            StableHLOType::Tuple(elems, _) => {
                let elem_ty = elems[1].clone();
                let reg = self.emitter.emit_get_tuple_element(&operand_reg, &operand_ty, 1, &elem_ty);
                Ok((reg, elem_ty))
            }
            _ if operand_ty.shape().is_empty() => {
                Err(SheafError::Compile {
                    message: "second: cannot index a scalar".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
            _ => {
                let (reg, ty) = self.emitter.emit_index_axis0(&operand_reg, &operand_ty, 1);
                Ok((reg, ty))
            }
        }
    }

    fn gen_last(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let shape = operand_ty.shape();
        if shape.is_empty() {
            return Err(SheafError::Compile {
                message: "last: cannot index a scalar".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            });
        }
        if let Some(vals) = self.emitter.known_tensor_values(&operand_reg)
            && !vals.is_empty()
        {
            let val = vals[vals.len() - 1];
            let reg = self.emitter.emit_constant_f32(val);
            self.emitter.set_known_scalar(reg, val);
            if let Some(&value) = self.known_integer_vectors.get(&operand_reg)
                .and_then(|values| values.last()) {
                self.known_integers.insert(reg, value);
            }
            return Ok((reg, StableHLOType::ScalarF32));
        }
        let last_idx = shape[0] - 1;
        let (reg, ty) = self.emitter.emit_index_axis0(&operand_reg, &operand_ty, last_idx);
        if let Some(vals) = self.emitter.known_tensor_values(&operand_reg)
            && (last_idx as usize) < vals.len()
        {
            self.emitter.set_known_scalar(reg, vals[last_idx as usize]);
        }
        Ok((reg, ty))
    }

    fn gen_get(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let sym_name = if let CompiledExpr::Symbol(s) = &args[0] { Some(s.clone()) } else { None };
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        match &operand_ty {
            StableHLOType::Tuple(elements, _) => {
                let static_index = match &args[1] {
                    CompiledExpr::Integer(index) => Some(*index),
                    CompiledExpr::Symbol(_) => {
                        let (reg, _) = self.generate(&args[1])?;
                        if self.static_tuple_indices.contains(&reg) {
                            self.emitter.known_scalar_value(&reg).and_then(|value| {
                                let limit = elements.len() as f64;
                                (value.is_finite() && value.fract() == 0.0
                                    && value >= -limit && value < limit)
                                    .then_some(value as i64)
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(index) = static_index {
                    let index = if index < 0 {
                        elements.len() as i64 + index
                    } else {
                        index
                    };
                    let element_ty = elements.get(index as usize).ok_or_else(|| {
                        SheafError::Compile {
                            message: format!(
                                "get: tuple index {} out of bounds for {} elements",
                                index,
                                elements.len()
                            ),
                            location: crate::core::error::SourceLocation::unknown(),
                        }
                    })?;
                    let reg = self.emitter.emit_get_tuple_element(
                        &operand_reg,
                        &operand_ty,
                        index as usize,
                        element_ty,
                    );
                    let parent_key = self.layout_key_map.get(&operand_reg).cloned()
                        .or(sym_name.clone());
                    if let Some(parent_key) = parent_key
                        && let Some(child_key) = self.idx_to_key.get(&(parent_key, index as usize))
                    {
                        self.layout_key_map.insert(reg, child_key.clone());
                    }
                    return Ok((reg, element_ty.clone()));
                }
                let layout_key = self.layout_key_map.get(&operand_reg).cloned()
                    .or(sym_name.clone());
                let keywords: Vec<String> = args[1..].iter().filter_map(|a| match a {
                    CompiledExpr::Keyword(k) | CompiledExpr::String(k) => Some(k.clone()),
                    _ => None,
                }).collect();
                if keywords.len() == args.len() - 1 && !keywords.is_empty() {
                    let mut cur_reg = operand_reg;
                    let mut cur_ty = operand_ty.clone();
                    let mut cur_key = layout_key;
                    let mut ok = true;
                    for key in &keywords {
                        if let StableHLOType::Tuple(sub_types, type_keys) = &cur_ty {
                            let index = match type_keys {
                                Some(keys) => keys.iter().position(|k| k == key),
                                None => cur_key.as_ref()
                                    .and_then(|name| self.tuple_key_layouts.get(name))
                                    .and_then(|layout| layout.get(key).copied()),
                            };
                            if let Some((idx, sub_ty)) = index
                                .and_then(|idx| sub_types.get(idx).cloned().map(|ty| (idx, ty)))
                            {
                                cur_reg = self.emitter.emit_get_tuple_element(
                                    &cur_reg, &cur_ty, idx, &sub_ty,
                                );
                                cur_ty = sub_ty;
                                cur_key = Some(key.clone());
                                continue;
                            }
                        }
                        ok = false;
                        break;
                    }
                    if ok {
                        if let Some(key) = cur_key {
                            self.layout_key_map.insert(cur_reg, key);
                        }
                        return Ok((cur_reg, cur_ty));
                    }
                }
                Err(SheafError::Compile {
                    message: format!(
                        "get on dict/tuple requires type info (sym={:?} keys={:?})",
                        sym_name,
                        args[1..].iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>()
                    ),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
            _ if !operand_ty.shape().is_empty() => {
                match &args[1] {
                    CompiledExpr::Integer(idx) => {
                        let actual_idx = if *idx < 0 {
                            operand_ty.shape()[0] + *idx
                        } else {
                            *idx
                        };
                        let known_value = if operand_ty.shape().len() == 1 {
                            usize::try_from(actual_idx).ok().and_then(|index| {
                                self.emitter.known_tensor_values(&operand_reg)
                                    .and_then(|values| values.get(index).copied())
                            })
                        } else {
                            None
                        };
                        let (reg, ty) = self.emitter.emit_index_axis0(&operand_reg, &operand_ty, actual_idx);
                        if let Some(value) = known_value {
                            self.emitter.set_known_scalar(reg, value);
                        }
                        if let Some(value) = usize::try_from(actual_idx).ok()
                            .and_then(|index| self.known_integer_vectors.get(&operand_reg)
                                .and_then(|values| values.get(index))).copied() {
                            self.known_integers.insert(reg, value);
                        }
                        Ok((reg, ty))
                    }
                    CompiledExpr::Symbol(s) | CompiledExpr::Keyword(s) if s == "..." => {
                        if args.len() >= 3 {
                            if let CompiledExpr::Integer(idx) = &args[2] {
                                let shape = operand_ty.shape();
                                let ndim = shape.len();
                                let last_axis_size = shape[ndim - 1];
                                let actual_idx = if *idx < 0 { last_axis_size + *idx } else { *idx };
                                let (reg, ty) = self.emitter.emit_slice_last_axis(
                                    &operand_reg, &operand_ty, actual_idx, actual_idx + 1,
                                );
                                Ok((reg, ty))
                            } else if let CompiledExpr::FunctionCall { name: range_name, args: range_args, .. } = &args[2] {
                                if (range_name == "range" || range_name == "arange")
                                    && range_args.len() == 2
                                    && let (CompiledExpr::Integer(start), CompiledExpr::Integer(end)) =
                                        (&range_args[0], &range_args[1])
                                {
                                    let (reg, ty) = self.emitter.emit_slice_last_axis(
                                        &operand_reg, &operand_ty, *start, *end,
                                    );
                                    return Ok((reg, ty));
                                }
                                Err(SheafError::Compile {
                                    message: "get with ellipsis: index must be integer or (range start end)".to_string(),
                                    location: crate::core::error::SourceLocation::unknown(),
                                })
                            } else {
                                Err(SheafError::Compile {
                                    message: "get with ellipsis: index must be integer or (range start end)".to_string(),
                                    location: crate::core::error::SourceLocation::unknown(),
                                })
                            }
                        } else {
                            Err(SheafError::Compile {
                                message: "get with ellipsis: missing index after ...".to_string(),
                                location: crate::core::error::SourceLocation::unknown(),
                            })
                        }
                    }
                    _ => {
                        let (idx_reg, idx_ty) = self.generate(&args[1])?;
                        if !idx_ty.shape().is_empty() {
                            let (reg, ty) = self.emitter.emit_gather_axis0(
                                &operand_reg, &operand_ty,
                                &idx_reg, &idx_ty,
                            );
                            Ok((reg, ty))
                        } else {
                            let (idx_1d_reg, idx_1d_ty) = self.emitter.emit_reshape(
                                &idx_reg, &idx_ty, &[1],
                            );
                            let (gathered_reg, gathered_ty) = self.emitter.emit_gather_axis0(
                                &operand_reg, &operand_ty,
                                &idx_1d_reg, &idx_1d_ty,
                            );
                            let gathered_shape = gathered_ty.shape();
                            let squeezed_shape: Vec<i64> = gathered_shape[1..].to_vec();
                            let (squeezed_reg, squeezed_ty) = self.emitter.emit_reshape(
                                &gathered_reg, &gathered_ty, &squeezed_shape,
                            );
                            Ok((squeezed_reg, squeezed_ty))
                        }
                    }
                }
            }
            _ => {
                Err(SheafError::Compile {
                    message: format!("get: unsupported operand type {}", operand_ty.to_mlir()),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
    }

    fn gen_get_in(&self) -> SheafResult<(Register, StableHLOType)> {
        Err(SheafError::Compile {
            message: "get-in requires type info (use --trace-with or auto-trace)".to_string(),
            location: crate::core::error::SourceLocation::unknown(),
        })
    }

    fn gen_assoc(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let sym_name = if let CompiledExpr::Symbol(s) = &args[0] { Some(s.clone()) } else { None };
        let (base_reg, base_ty) = self.generate(&args[0])?;
        let (elems, keys) = match &base_ty {
            StableHLOType::Tuple(elems, keys) => (elems.clone(), keys.clone()),
            _ => {
                return Err(SheafError::Compile {
                    message: "assoc: base must be a tuple".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        };
        let layout_key = sym_name.clone().or_else(|| self.layout_key_map.get(&base_reg).cloned());
        let layout = keys.as_ref().map(|keys| keys.iter().enumerate()
            .map(|(index, key)| (key.clone(), index)).collect())
            .or_else(|| layout_key.as_ref().and_then(|k| self.tuple_key_layouts.get(k).cloned()));
        let layout = match layout {
            Some(l) => l,
            None => {
                return Err(SheafError::Compile {
                    message: "assoc: no layout for tuple".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        };
        let mut fields = std::collections::BTreeMap::new();
        for (key, index) in &layout {
            let ty = elems.get(*index).ok_or_else(|| SheafError::Compile {
                message: "assoc: tuple layout index out of bounds".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })?;
            let reg = self.emitter.emit_get_tuple_element(&base_reg, &base_ty, *index, ty);
            fields.insert(key.clone(), (reg, ty.clone()));
        }
        let mut i = 1;
        while i + 1 < args.len() {
            let key_name = match &args[i] {
                CompiledExpr::Keyword(k) | CompiledExpr::String(k) => k.clone(),
                _ => {
                    return Err(SheafError::Compile {
                        message: format!("assoc: expected keyword key, got {:?}", args[i]),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            };
            let (val_reg, val_ty) = self.generate(&args[i + 1])?;
            fields.insert(key_name, (val_reg, val_ty));
            i += 2;
        }
        let keys = fields.keys().cloned().collect();
        let (new_regs, new_tys): (Vec<_>, Vec<_>) = fields.into_values().unzip();
        let (reg, _) = self.emitter.emit_tuple(&new_regs, &new_tys);
        Ok((reg, StableHLOType::Tuple(new_tys, Some(keys))))
    }

    fn gen_top_k(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (input_reg, input_ty) = self.generate(&args[0])?;
        let k = match &args[1] {
            CompiledExpr::Integer(n) => *n,
            CompiledExpr::Float(f) => *f as i64,
            _ => {
                let (k_reg, _k_ty) = self.generate(&args[1])?;
                match self.emitter.known_scalar_value(&k_reg) {
                    Some(v) => v as i64,
                    None => {
                        return Err(SheafError::Compile {
                            message: "top_k: k must be a compile-time constant".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        });
                    }
                }
            }
        };
        Ok(self.emitter.emit_top_k(&input_reg, &input_ty, k))
    }
}
