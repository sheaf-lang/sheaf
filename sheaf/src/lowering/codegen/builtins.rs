// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Builtin call codegen.

use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::core::expr::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use super::CodeGenerator;

impl<'a> CodeGenerator<'a> {
    pub(super) fn generate_function_call(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let registry = self.function_registry;
        let signature = registry
            .and_then(|registry| registry.get(name))
            .and_then(|func_def| func_def.signature.clone());

        if let Some(_signature) = signature {
            let mut arg_registers = Vec::new();
            let mut arg_types = Vec::new();

            for arg in args {
                let (reg, ty) = self.generate(arg)?;
                arg_registers.push(reg);
                arg_types.push(ty);
            }

            let func_def = registry.and_then(|registry| registry.get(name));
            if let Some(func_def) = func_def
                && let Some(body) = &func_def.body_compiled {
                    let saved_bindings = self.bindings.clone();
                    let saved_layouts = self.tuple_key_layouts.clone();
                    let saved_idx_to_key = self.idx_to_key.clone();
                    for (param, (arg_expr, (reg, ty))) in func_def
                        .params
                        .iter()
                        .zip(args.iter().zip(arg_registers.iter().zip(arg_types.iter())))
                    {
                        self.bindings
                            .insert(param.clone(), (*reg, ty.clone()));

                        // Preserve virtual tuple layouts.
                        if let CompiledExpr::Symbol(arg_sym) = arg_expr {
                            if let Some(layout) = self.tuple_key_layouts.get(arg_sym).cloned() {
                                self.tuple_key_layouts.insert(param.clone(), layout);
                            }
                            let entries: Vec<_> = self.idx_to_key.iter()
                                .filter(|((name, _), _)| name == arg_sym)
                                .map(|((_, idx), key)| (*idx, key.clone()))
                                .collect();
                            for (idx, key) in entries {
                                self.idx_to_key.insert((param.clone(), idx), key);
                            }
                        }
                    }
                    let body = body.clone();
                    let result = self.generate(&body);
                    self.bindings = saved_bindings;
                    self.tuple_key_layouts = saved_layouts;
                    self.idx_to_key = saved_idx_to_key;
                    return result;
                }

            let sig = registry
                .and_then(|registry| registry.get(name))
                .and_then(|f| f.signature.clone())
                .unwrap();
            let result_reg =
                self.emitter
                    .emit_func_call(name, &arg_registers, &arg_types, &sig.return_type);
            return Ok((result_reg, sig.return_type.clone()));
        }

        if name == "stop-gradient" && args.len() == 1 {
            return self.generate(&args[0]);
        }

        if let Some(result) = self.generate_math_builtin(name, args) {
            return result;
        }
        if let Some(result) = self.generate_tensor_builtin(name, args) {
            return result;
        }
        if let Some(result) = self.generate_reduction_builtin(name, args) {
            return result;
        }
        if let Some(result) = self.generate_collection_builtin(name, args) {
            result
        }

        else if name == "einsum" && args.len() >= 3 {
            let spec = match &args[0] {
                CompiledExpr::String(s) => s.clone(),
                _ => {
                    return Err(SheafError::Compile {
                        message: "einsum: first argument must be a string spec".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let (lhs_reg, lhs_ty, rhs_reg, rhs_ty) =
                self.generate_binary_operands("einsum", &args[1], &args[2])?;
            self.emitter
                .emit_einsum(&lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty, &spec)
                .map_err(|msg| SheafError::Compile {
                    message: msg,
                    location: crate::core::error::SourceLocation::unknown(),
                })
        }
        else if name == "sum_to_shape" && args.len() == 2 {
            self.gen_sum_to_shape(args)
        }
        else if name == "slice_grad" && args.len() == 3 {
            self.gen_slice_grad(args)
        }
        else if name == "reduce" && args.len() == 3 {
            self.generate_reduce_scan(&args[0], &args[1], &args[2], false)
        }
        else if name == "scan" && args.len() == 3 {
            self.generate_reduce_scan(&args[0], &args[1], &args[2], true)
        }
        else if name == "tree-map" && args.len() >= 2 {
            let lambda = &args[0];
            let tree_args = &args[1..];

            let mut tree_regs = Vec::new();
            let mut tree_tys = Vec::new();
            for arg in tree_args {
                let (reg, ty) = self.generate(arg)?;
                tree_regs.push(reg);
                tree_tys.push(ty);
            }

            if !tree_tys.iter().all(|t| t.tuple_structure_matches(&tree_tys[0])) {
                                let ty_strs: Vec<String> = tree_tys.iter().map(|t| {
                                let mlir = t.to_mlir();
                                let len = if let StableHLOType::Tuple(e, _) = t { e.len() } else { 0 };
                                format!("(len={} {}...)", len, &mlir[..mlir.len().min(80)])
                            }).collect();
                            return Err(SheafError::Compile {
                                message: format!("tree-map: all tree arguments must have the same tuple structure (got: {:?})", ty_strs),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
            }

            self.generate_tree_map(lambda, &tree_regs, &tree_tys)
        }
        else if name == "__scan_vjp__" && args.len() == 4 {
            self.generate_scan_vjp(args)
        }
        else if name == "tree-reduce" && args.len() == 3 {
            let lambda = &args[0];
            let (tree_reg, tree_ty) = self.generate(&args[1])?;
            let (acc_reg, acc_ty) = self.generate(&args[2])?;
            self.generate_tree_reduce(lambda, tree_reg, &tree_ty, acc_reg, &acc_ty)
        }
        else {
            let message = format!("{name} is not supported by the compiler with {} arguments", args.len());
            let location = crate::core::error::SourceLocation::unknown();
            match crate::core::signature::OpId::resolve(name) {
                Some(op) => Err(SheafError::CompilerUnsupported {
                    construct: crate::core::jit_diagnostic::Construct::Operation(op), message, location,
                }),
                None => Err(SheafError::Compile { message, location }),
            }
        }
    }

    fn gen_sum_to_shape(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (mut reg, mut ty) = self.generate(&args[0])?;
        if let CompiledExpr::Vector(shape_elems) = &args[1] {
            let target_shape = self.parse_shape_vec("sum_to_shape", shape_elems)?;
            let from_shape = ty.shape().to_vec();

            if from_shape == target_shape {
                return Ok((reg, ty));
            }

            let axes = crate::core::shape::sum_to_shape_axes(&from_shape, &target_shape)
                .map_err(|_| SheafError::Compile {
                    message: format!("sum_to_shape: cannot reduce {from_shape:?} to {target_shape:?}"),
                    location: crate::core::error::SourceLocation::unknown(),
                })?;
            let extra = from_shape.len() - target_shape.len();
            for axis in axes.into_iter().rev() {
                let (reduced, reduced_type) =
                    self.emitter.emit_reduce_sum(&reg, &ty, axis as i64, axis >= extra);
                reg = reduced;
                ty = reduced_type;
            }

            Ok((reg, ty))
        } else {
            Err(SheafError::Compile {
                message: "sum_to_shape expects a vector shape argument".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_slice_grad(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (adj_reg, adj_ty) = self.generate(&args[0])?;
        if let (CompiledExpr::Vector(shape_elems), CompiledExpr::Integer(start)) =
            (&args[1], &args[2])
        {
            let target_shape = self.parse_shape_vec("slice_grad", shape_elems)?;
            let adj_shape = adj_ty.shape().to_vec();

            let axis = adj_shape
                .iter()
                .zip(target_shape.iter())
                .position(|(a, t)| a != t)
                .unwrap_or(0);

            let start_val = *start;
            let end_val = start_val + adj_shape[axis];
            let total = target_shape[axis];
            let pad_low = start_val;
            let pad_high = total - end_val;

            let dtype = adj_ty.element_type().ok_or_else(|| SheafError::Compile {
                message: "slice_grad expects a tensor adjoint".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })?;
            let (zero_reg, zero_ty) =
                self.emitter.emit_typed_splat_constant(0.0, &[], dtype);
            let result_ty = StableHLOType::tensor(target_shape, dtype);
            let result_reg = self.emitter.fresh_register();

            let ndim = adj_shape.len();
            let low: Vec<String> = (0..ndim)
                .map(|i| if i == axis { pad_low.to_string() } else { "0".to_string() })
                .collect();
            let high: Vec<String> = (0..ndim)
                .map(|i| if i == axis { pad_high.to_string() } else { "0".to_string() })
                .collect();
            let interior: Vec<String> = vec!["0".to_string(); ndim];

            self.emitter.body.push(format!(
                "    {} = stablehlo.pad {}, {}, low = [{}], high = [{}], interior = [{}] : ({}, {}) -> {}",
                result_reg.to_mlir(),
                adj_reg.to_mlir(),
                zero_reg.to_mlir(),
                low.join(", "),
                high.join(", "),
                interior.join(", "),
                adj_ty.to_mlir(),
                zero_ty.to_mlir(),
                result_ty.to_mlir(),
            ));

            Ok((result_reg, result_ty))
        } else {
            Err(SheafError::Compile {
                message: "slice_grad expects (adj, [shape], start)".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

}
