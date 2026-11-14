// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! JIT compilation of value-and-grad closures.

use super::support::{
    inject_tuple_shapes, log_remaining_reduces, log_unresolved_shapes, manifest_hash_matches,
    outermost_call_name, update_manifest,
};
use super::*;

fn leaf_type_at(
    param_ty: &StableHLOType,
    indices: &[usize],
) -> crate::core::error::SheafResult<StableHLOType> {
    let mut current = param_ty;
    for &index in indices {
        let StableHLOType::Tuple(elems, _) = current else {
            return Err(crate::core::error::SheafError::AutodiffMissingGradientOutput {
                symbol: format!("gradient leaf at {:?}", indices),
            });
        };
        current = elems.get(index).ok_or_else(|| {
            crate::core::error::SheafError::AutodiffMissingGradientOutput {
                symbol: format!("gradient leaf at {:?}", indices),
            }
        })?;
    }
    Ok(current.clone())
}

fn zero_gradient_expr(ty: &StableHLOType) -> CompiledExpr {
    let shape = ty.shape();
    if shape.is_empty() {
        CompiledExpr::Float(0.0)
    } else {
        CompiledExpr::FunctionCall {
            name: "zeros".to_string(),
            args: vec![CompiledExpr::Vector(
                shape.iter().map(|&d| CompiledExpr::Integer(d)).collect(),
            )],
            loc: None,
        }
    }
}

impl JitCompiler {
    /// JIT-compile a value-and-grad closure into a single VMFB.
    ///
    /// Floating-point captures remain runtime inputs; integers and booleans are specialized.
    ///
    /// Returns `(module_name, signature, param_order)` for dispatch.
    pub fn try_jit_value_and_grad(
        &mut self,
        func: &Value,
        wrt_arg: &Value,
        registry: &HashMap<String, FunctionDef>,
        shared_session: &std::sync::Arc<crate::runtime::iree_session::IreeSession>,
    ) -> Result<(String, FunctionSignature, Vec<String>), JitDiagnostic> {
        let iree_compile = self.iree_compile_path.clone().ok_or(JitDiagnostic::Environment {
            message: "Compiler toolchain unavailable".into(), hint: "Install curl and unzip, or set IREE_COMPILE.",
        })?;

        let (fn_params, body, closure) = match func {
            Value::Function {
                params,
                body,
                closure,
                ..
            } => (params, body, closure),
            _ => return Err(JitDiagnostic::UserError(crate::interpreter::env::runtime_error("value-and-grad expects a function"))),
        };

        // Include wrt_arg type so shape changes (e.g. after grow-hydra) cause a cache miss.
        let wrt_type_str = value_to_stablehlo_type(wrt_arg)
            .map(|t| t.to_mlir())
            .unwrap_or_default();
        let mut keyed_values = vec![wrt_arg.clone(); fn_params.len()];
        keyed_values.extend(closure.iter().map(|(_, value)| value.clone()));
        let static_values = static_inputs::collect(&keyed_values);
        let capture_types: Vec<_> = closure.iter().map(|(name, value)| {
            (name, value_to_stablehlo_type(value).map(|ty| ty.to_mlir()).ok(),
                crate::core::inference::ValueLayout::from_value(value))
        }).collect();
        let vag_key = format!("__vag_{:?}_{}_{:?}_{:?}", body, wrt_type_str, static_values, capture_types);
        if let Some(diagnostic) = self.failed_vag.get(&vag_key) {
            return Err(diagnostic.clone());
        }
        if let Some(cached) = self.vag_cache.get(&vag_key) {
            return Ok(cached.clone());
        }
        let result = self.compile_vag(func, wrt_arg, registry, shared_session, &iree_compile, &vag_key);
        if let Err(diagnostic) = &result {
            self.failed_vag.insert(vag_key, diagnostic.clone());
        }
        result
    }

    fn compile_vag(
        &mut self, func: &Value, wrt_arg: &Value, registry: &HashMap<String, FunctionDef>,
        shared_session: &std::sync::Arc<crate::runtime::iree_session::IreeSession>,
        iree_compile: &str, vag_key: &str,
    ) -> Result<(String, FunctionSignature, Vec<String>), JitDiagnostic> {
        let Value::Function { params: fn_params, body, closure, .. } = func else {
            return Err(JitDiagnostic::Internal("Prepared value-and-grad function is not a function".into()));
        };
        let vag_fn_name = outermost_call_name(body).unwrap_or("anonymous".into());
        // Skip impure or HOF-containing bodies
        let effects = collect_effects(body);
        if !effects.is_empty() {
            let constructs = effects.iter().map(|site| Construct::named(&site.name)
                .ok_or_else(|| JitDiagnostic::Internal(format!("Unknown effect '{}'", site.name))))
                .collect::<Result<Vec<_>, _>>()?;
            return Err(JitDiagnostic::Interpreted(InterpretedReason::Effects(constructs)));
        }
        if let Some(name) = collect_hof_calls(body).first() {
            let construct = Construct::named(name)
                .ok_or_else(|| JitDiagnostic::Internal(format!("Unknown higher-order operation '{name}'")))?;
            return Err(JitDiagnostic::from_hof(construct));
        }

        let mut all_param_names: Vec<String> = Vec::new();
        let mut all_arg_values: Vec<Value> = Vec::new();


        for p in fn_params {
            all_param_names.push(p.clone());
            all_arg_values.push(wrt_arg.clone());
        }
        let wrt_indices: Vec<usize> = (0..fn_params.len()).collect();

        let aug_registry = registry;

        for (cap_name, cap_val) in closure {
            if value_to_stablehlo_type(cap_val).is_ok() {
                all_param_names.push(cap_name.clone());
                all_arg_values.push(cap_val.clone());
            } else {
                return Err(JitDiagnostic::Interpreted(InterpretedReason::NonTensorArgument(cap_name.clone())));
            }
        }

        let mut body = body.as_ref().clone();

        let mut known_types: Vec<(String, StableHLOType)> = Vec::new();
        let mut param_index_maps = ParamIndexMaps::new();
        let mut constants: HashMap<(String, Vec<usize>), i64> = HashMap::new();

        for (param_name, arg_val) in all_param_names.iter().zip(all_arg_values.iter()) {
            let ty = match value_to_stablehlo_type(arg_val) {
                Ok(ty) => ty,
                Err(_) => return Err(JitDiagnostic::Interpreted(InterpretedReason::NonTensorArgument(param_name.clone()))),
            };

            if let Some(layout) = value_to_param_layout(param_name, arg_val) {
                let imap = layout_to_index_map(&layout);
                extract_scalar_constants(arg_val, param_name, &imap, &mut constants);
                body = lower_get_calls(&body, param_name, &imap);
                param_index_maps.push((param_name.clone(), imap));
            }

            known_types.push((param_name.clone(), ty));
        }

        if let Err(error) = super::validate_shape_positions(
            &all_param_names, &all_arg_values, crate::lowering::transforms::collect_shape_gtes(&body),
        ) {
            return Err(JitDiagnostic::UserError(error));
        }
        let all_static_inputs = static_inputs::collect(&all_arg_values);
        let capture_static_inputs: Vec<_> = all_static_inputs.iter()
            .filter(|input| input.argument >= fn_params.len()).cloned().collect();
        body = static_inputs::specialize(
            &body, &all_param_names[fn_params.len()..], &all_arg_values[fn_params.len()..],
        );
        let mut sig = match infer_function_signature_with_known(
            &all_param_names,
            &body,
            &known_types,
        ) {
            Ok(s) => s,
            Err(error) => return Err(JitDiagnostic::from_codegen(error)),
        };

        for (param_name, ty) in &known_types {
            if let Some(idx) = all_param_names.iter().position(|p| p == param_name) {
                sig.param_types[idx] = ty.clone();
            }
        }

        // -vv: log VAG signature and captures
        if crate::core::config::verbosity() >= 2 {
            for (pname, pty) in all_param_names.iter().zip(sig.param_types.iter()) {
                if let Some((_, imap)) = param_index_maps.iter().find(|(n, _)| n == pname) {
                    let mut top_fields: BTreeMap<usize, (String, String)> = BTreeMap::new();
                    for (path, indices) in imap.iter() {
                        if path.len() == 1 {
                            let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                if let Some(t) = elems.get(indices[0]) {
                                    t.to_mlir()
                                } else {
                                    "?".to_string()
                                }
                            } else {
                                "?".to_string()
                            };
                            top_fields
                                .entry(indices[0])
                                .or_insert((path[0].clone(), ty_str));
                        } else {
                            top_fields.entry(indices[0]).or_insert_with(|| {
                                let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                    if let Some(StableHLOType::Tuple(sub, _)) = elems.get(indices[0]) {
                                        format!("tuple<...> ({} fields)", sub.len())
                                    } else {
                                        "tuple<...>".to_string()
                                    }
                                } else {
                                    "tuple<...>".to_string()
                                };
                                (path[0].clone(), ty_str)
                            });
                        }
                    }
                    for (key, ty_str) in top_fields.values() {
                        sheaf_msg!("jit: value-and-grad | {}.{}: {}", pname, key, ty_str);
                    }
                } else {
                    sheaf_msg!("jit: value-and-grad | {}: {}", pname, pty.to_mlir());
                }
            }
            if !capture_static_inputs.is_empty() {
                sheaf_msg!("jit: value-and-grad | {} static captures", capture_static_inputs.len());
            }
        }

        body = crate::autodiff::inline_function_calls(&body, aug_registry);

        // Bail out if inlining produced a graph that is too large
        let node_count = expr_node_count(&body);
        if node_count > MAX_VAG_GRAPH_NODES {
            return Err(JitDiagnostic::Unsupported(UnsupportedReason::GraphLimit {
                phase: GraphPhase::Inlining, nodes: node_count, limit: MAX_VAG_GRAPH_NODES,
            }, None));
        }

        // Fold dict literal gets: (get {:gamma g :beta b} :gamma) -> g
        body = crate::autodiff::fold_dict_gets(&body);

        // Post-inline: re-lower dict access from inlined bodies
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);

        // Unroll reduces so reverse_grad can differentiate through them
        let known_types_vec: Vec<(String, StableHLOType)> = sig
            .param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| (all_param_names[i].clone(), ty.clone()))
            .collect();
        // Debug: log remaining reduces before unrolling
        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!(
                "jit: [vag] param_index_maps keys: {:?}",
                param_index_maps
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
            );
            log_remaining_reduces(&body, "before unroll");
        }
        body = unroll_reduces(&body, &known_types_vec);
        if crate::core::config::verbosity() >= 2 {
            log_remaining_reduces(&body, "after unroll");
        }

        // Bail out if unrolling produced a graph that is too large
        let node_count = expr_node_count(&body);
        if node_count > MAX_VAG_GRAPH_NODES {
            return Err(JitDiagnostic::Unsupported(UnsupportedReason::GraphLimit {
                phase: GraphPhase::Unrolling, nodes: node_count, limit: MAX_VAG_GRAPH_NODES,
            }, None));
        }

        // Re-lower dict access introduced by unrolling (get on GetTupleElement elements)
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);

        body = static_inputs::apply(&body, &all_param_names, &capture_static_inputs);
        // Filter constants: only bake scalars used in shape-bearing positions
        // (see the main JIT path above for rationale).
        let filtered_vag_constants = filter_constants_for_shape_positions(&constants, &body);

        let mut param_shapes: HashMap<String, Vec<i64>> = all_param_names
            .iter()
            .zip(sig.param_types.iter())
            .filter_map(|(p, ty)| {
                let shape = ty.shape();
                if shape.is_empty() {
                    None
                } else {
                    Some((p.clone(), shape.to_vec()))
                }
            })
            .collect();
        // Inject shapes of GetTupleElement leaves for shape inference
        for (param_name, param_ty) in all_param_names.iter().zip(sig.param_types.iter()) {
            inject_tuple_shapes(param_name, param_ty, &[], &mut param_shapes);
        }
        body = resolve_static_constants(&body, &filtered_vag_constants, &param_shapes, false);

        body = match crate::lowering::transforms::lower_tuples_and_destructuring(
            body,
            &param_shapes,
        ) {
            Ok(b) => b,
            Err(error) => return Err(JitDiagnostic::from_codegen(error)),
        };

        // Build key layouts for codegen
        let mut tuple_key_layouts: HashMap<String, BTreeMap<String, usize>> = HashMap::new();
        let mut idx_to_key: HashMap<(String, usize), String> = HashMap::new();
        for (param_name, index_map) in &param_index_maps {
            for (key_path, indices) in index_map {
                for depth in 0..key_path.len() {
                    let parent = if depth == 0 {
                        param_name.clone()
                    } else {
                        key_path[depth - 1].clone()
                    };
                    let child = &key_path[depth];
                    let idx = indices[depth];
                    tuple_key_layouts
                        .entry(parent.clone())
                        .or_default()
                        .entry(child.clone())
                        .or_insert(idx);
                    idx_to_key
                        .entry((parent, idx))
                        .or_insert_with(|| child.clone());
                }
            }
        }
        propagate_let_layouts(&body, &idx_to_key, &mut tuple_key_layouts);

        // Debug: log unresolved shapes
        if crate::core::config::verbosity() >= 2 {
            log_unresolved_shapes(&body, "before codegen");
        }

        // Emit for the backend selected by the shared session.
        if let Some(backend) = crate::runtime::iree_session::IreeSession::cached_target_backend() {
            self.target_backend = backend.to_string();
        }
        let backend = self.target_backend.clone();
        let codegen_result = {
            let param_names = all_param_names.clone();
            let param_types = sig.param_types.clone();
            let body_clone = body.clone();
            let wrt_idx = wrt_indices.clone();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut codegen =
                    CodeGenerator::with_function_params(registry, &param_names, &param_types);
                codegen.set_tuple_key_layouts(tuple_key_layouts);
                codegen.set_idx_to_key(idx_to_key);
                for input in &all_static_inputs {
                    codegen.specialize_parameter_leaf(
                        &param_names[input.argument], &input.indices, &input.value.expression(),
                    )?;
                }

                let inlined_body = body_clone;

                let mut expanded_body = inlined_body.clone();
                let mut all_leaves: Vec<(usize, Vec<crate::lowering::codegen::TupleLeaf>)> =
                    Vec::new();
                let mut all_wrt_symbols: Vec<String> = Vec::new();

                for &idx in &wrt_idx {
                    let param_name = &param_names[idx];
                    let param_ty = &param_types[idx];
                    match param_ty {
                        StableHLOType::Tuple(..) => {
                            let leaves = collect_tuple_type_leaves(param_name, param_ty);
                            if crate::core::config::verbosity() >= 2 {
                                sheaf_msg!(
                                    "jit: [vag] param '{}': {} tuple leaves",
                                    param_name,
                                    leaves.len()
                                );
                            }
                            expanded_body = expand_tuple_to_symbols(&expanded_body, param_name, param_ty);
                            for leaf in &leaves {
                                all_wrt_symbols.push(leaf.symbol.clone());
                            }
                            all_leaves.push((idx, leaves));
                        }
                        _ => {
                            all_wrt_symbols.push(param_name.clone());
                        }
                    }
                }

                let anf_expr = to_anf(&expanded_body);
                let (anf_bindings, anf_body) = match &anf_expr {
                    CompiledExpr::Let { bindings, body } => {
                        (bindings.clone(), body.as_ref().clone())
                    }
                    other => (vec![], other.clone()),
                };

                // Bind tuple leaves for codegen (use generate to handle both
                // leaf lookups and sub-tuple reconstruction from flat args)
                for &(idx, ref leaves) in &all_leaves {
                    let param_name = &param_names[idx];
                    for leaf in leaves {
                        let gte = CompiledExpr::GetTupleElement {
                            param: param_name.clone(),
                            indices: leaf.indices.clone(),
                        };
                        let (reg, ty) = codegen.generate(&gte)?;
                        codegen.bind_symbol(&leaf.symbol, reg, ty);
                    }
                    for reference in collect_tuple_references(&inlined_body, param_name) {
                        if leaves.iter().any(|leaf| leaf.indices == reference.indices) {
                            continue;
                        }
                        let gte = CompiledExpr::GetTupleElement {
                            param: param_name.clone(),
                            indices: reference.indices,
                        };
                        let (reg, ty) = codegen.generate(&gte)?;
                        codegen.bind_symbol(&reference.symbol, reg, ty);
                    }
                }

                // Use generate_binding to handle Lambda, destructuring, layouts.
                for (name, value_expr) in &anf_bindings {
                    let name_str = name
                        .as_simple()
                        .expect("Expected simple binding pattern in ANF");
                    codegen.generate_binding(name_str, value_expr)?;
                }

                let (loss_reg, loss_ty) = codegen.generate(&anf_body)?;

                let shape_map: HashMap<String, Vec<i64>> = codegen.binding_shapes();

                let anf_bindings_str: Vec<(String, CompiledExpr)> = anf_bindings
                    .iter()
                    .map(|(name, expr)| {
                        (
                            name.as_simple()
                                .expect("Expected simple binding pattern in ANF")
                                .to_string(),
                            expr.clone(),
                        )
                    })
                    .collect();
                let ReverseGradResult {
                    backward_bindings,
                    gradients: grad_sym_map,
                } = reverse_grad(&anf_bindings_str, &anf_body, &all_wrt_symbols, &shape_map)?;

                let backward_bindings: Vec<(String, CompiledExpr)> = backward_bindings
                    .into_iter()
                    .map(|(name, expr)| (name, simplify(expr)))
                    .collect();

                if crate::core::config::verbosity() >= 2 {
                    sheaf_msg!(
                        "jit: [vag] ANF: {} fwd bindings, {} bwd bindings, {} wrt symbols",
                        anf_bindings.len(),
                        backward_bindings.len(),
                        all_wrt_symbols.len()
                    );
                }

                if std::env::var("SHEAF_DEBUG_GRAD").is_ok() {
                    eprintln!("--- Forward ANF bindings ---");
                    for (name, val) in &anf_bindings_str {
                        eprintln!("  {} = {:?}  [shape: {:?}]", name, val, shape_map.get(name));
                    }
                    eprintln!("  body = {:?}", anf_body);
                    eprintln!("--- Backward bindings ---");
                    for (name, val) in &backward_bindings {
                        eprintln!("  {} = {:?}", name, val);
                    }
                    eprintln!("--- Grad map ---");
                    for (sym, grad_name) in &grad_sym_map {
                        eprintln!("  {} -> {:?}", sym, grad_name);
                    }
                    eprintln!("--- wrt symbols ---");
                    for s in &all_wrt_symbols {
                        eprintln!("  {}", s);
                    }
                }

                for (name, value_expr) in &backward_bindings {
                    let (reg, ty) = codegen.generate(value_expr)?;
                    codegen.bind_symbol(name, reg, ty);
                }

                let mut grad_regs: Vec<Register> = Vec::new();
                let mut grad_tys: Vec<StableHLOType> = Vec::new();

                for &idx in &wrt_idx {
                    let param_name = &param_names[idx];
                    let param_ty = &param_types[idx];
                    match param_ty {
                        StableHLOType::Tuple(..) => {
                            let leaves = all_leaves
                                .iter()
                                .find(|(i, _)| *i == idx)
                                .map(|(_, l)| l)
                                .ok_or_else(|| crate::core::error::SheafError::AutodiffMissingGradientOutput {
                                    symbol: param_name.clone(),
                                })?;

                            let leaf_grad_map: std::collections::HashMap<String, CompiledExpr> =
                                leaves
                                    .iter()
                                    .map(|leaf| {
                                        let grad_expr = match crate::autodiff::reverse::gradient_output(&grad_sym_map, &leaf.symbol)? {
                                            GradientOutput::Computed(sym_name) => {
                                                CompiledExpr::Symbol(sym_name.clone())
                                            }
                                            GradientOutput::ProvenZero => {
                                                let leaf_ty = leaf_type_at(param_ty, &leaf.indices)?;
                                                zero_gradient_expr(&leaf_ty)
                                            }
                                        };
                                        Ok((leaf.symbol.clone(), grad_expr))
                                    })
                                    .collect::<crate::core::error::SheafResult<_>>()?;

                            let (grad_reg, grad_ty) = codegen.build_grad_tuple_from_map(
                                leaves,
                                param_ty,
                                &leaf_grad_map,
                            )?;
                            grad_regs.push(grad_reg);
                            grad_tys.push(grad_ty);
                        }
                        _ => {
                            let grad_expr = match crate::autodiff::reverse::gradient_output(&grad_sym_map, param_name)? {
                                GradientOutput::Computed(sym_name) => {
                                    CompiledExpr::Symbol(sym_name.clone())
                                }
                                GradientOutput::ProvenZero => zero_gradient_expr(param_ty),
                            };
                            let (grad_reg, grad_ty) = codegen.generate(&grad_expr)?;
                            let (grad_reg, grad_ty) =
                                codegen.reduce_broadcast_grad(grad_reg, &grad_ty, param_ty)?;
                            grad_regs.push(grad_reg);
                            grad_tys.push(grad_ty);
                        }
                    }
                }

                // Pack result: (loss, grad0, grad1, ...)
                let mut all_regs = vec![loss_reg];
                all_regs.extend(grad_regs);
                let mut all_tys = vec![loss_ty.clone()];
                all_tys.extend(grad_tys.clone());

                let decl =
                    codegen.finish_multi("value_and_grad", &param_types, &all_regs, &all_tys);
                let return_type = StableHLOType::Tuple(all_tys, None);
                Ok::<_, crate::core::error::SheafError>((decl, return_type))
            }))
        };

        let (mlir_decl, return_type) = match codegen_result {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => return Err(JitDiagnostic::from_codegen(error)),
            Err(payload) => return Err(JitDiagnostic::codegen_panic(payload)),
        };

        sig.return_type = return_type;
        sig.return_dict_keys = None;

        // IREE module names must be unique within the shared session.
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        vag_key.hash(&mut h);
        let vag_module_name = format!("__vag_{:016x}", h.finish());
        let mlir = StableHLOEmitter::emit_module_named(Some(&vag_module_name), &[mlir_decl]);

        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: value-and-grad | MLIR {} lines", mlir.lines().count());
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        mlir.hash(&mut hasher);
        backend.hash(&mut hasher);
        IREE_COMPILER_VERSION.hash(&mut hasher);
        let content_hash = format!("{:016x}", hasher.finish());

        let cache_dir = PathBuf::from("__sheaf__");
        let vag_cache_name = format!("{}-vag", vag_fn_name);
        let backend_suffix = backend.replace('-', "_");
        let cached_vmfb = cache_dir.join(format!("{}.{}.vmfb", vag_cache_name, backend_suffix));

        let force_recompile = crate::core::config::verbosity() >= 2;
        let vmfb_data = if !force_recompile
            && cached_vmfb.exists()
            && manifest_hash_matches(&cache_dir, &vag_cache_name, &content_hash)
        {
            match std::fs::read(&cached_vmfb) {
                Ok(d) => {
                    if crate::core::config::verbosity() >= 2 {
                        sheaf_msg!(
                            "jit: value_and_grad (cached, {}KB, {})",
                            d.len() / 1024,
                            backend
                        );
                    } else if crate::core::config::verbosity() >= 1 {
                        sheaf_msg!("jit: value_and_grad (cached)");
                    }
                    d
                }
                Err(_) => self.run_iree_compile(iree_compile, "value_and_grad", &mlir, &backend)?,
            }
        } else {
            // Debug: save a copy for inspection
            if crate::core::config::verbosity() >= 2 {
                let debug_path = cache_dir.join(format!("{}-vag-debug.mlir", vag_fn_name));
                let _ = std::fs::write(&debug_path, &mlir);
                sheaf_msg!("jit: value-and-grad | saved {}", debug_path.display());
            }

            let data = self.run_iree_compile(iree_compile, "value_and_grad", &mlir, &backend)?;
            std::fs::create_dir_all(&cache_dir).map_err(|e| JitDiagnostic::io("Cannot create compilation cache", e))?;
            std::fs::write(&cached_vmfb, &data).map_err(|e| JitDiagnostic::io("Cannot write compilation cache", e))?;
            update_manifest(&cache_dir, &vag_cache_name, &content_hash)
                .map_err(|e| JitDiagnostic::io("Cannot write compilation manifest", e))?;

            data
        };

        // Load into the single shared IREE session (no per-compile session,
        // same rationale as `compile_function`).
        if let Err(e) = shared_session.load_vmfb(vmfb_data) {
            let _ = std::fs::remove_file(&cached_vmfb);
            return Err(JitDiagnostic::Internal(format!("Cannot load compiled module: {e}")));
        }

        let result = (vag_module_name, sig, all_param_names);
        self.vag_cache.insert(vag_key.into(), result.clone());
        Ok(result)
    }

    /// Run iree-compile on MLIR source for the given backend.
    /// Returns compiled VMFB bytes, or None if compilation fails.
    pub(super) fn run_iree_compile(
        &self,
        iree_compile: &str,
        name: &str,
        mlir: &str,
        backend: &str,
    ) -> Result<Vec<u8>, JitDiagnostic> {
        if crate::core::config::verbosity() >= 1 {
            sheaf_msg!("jit: compiling {} [{}]...", name, backend);
        }

        let safe_name = name.replace('?', "_q").replace('!', "_b");
        let stamp = std::process::id();
        let cache_dir = std::path::PathBuf::from("__sheaf__");
        std::fs::create_dir_all(&cache_dir).map_err(|e| JitDiagnostic::io("Cannot create compilation cache", e))?;
        let mlir_path = cache_dir.join(format!("{}-{}.mlir", safe_name, stamp));
        let vmfb_path = cache_dir.join(format!("{}-{}.vmfb", safe_name, stamp));

        std::fs::write(&mlir_path, mlir).map_err(|e| JitDiagnostic::io("Cannot write compiler input", e))?;

        let mut cmd = std::process::Command::new(iree_compile);
        cmd.arg(&mlir_path)
            .arg(format!("--iree-hal-target-backends={}", backend))
            .arg("-o")
            .arg(&vmfb_path)
            .stderr(std::process::Stdio::piped());
        if backend == "metal-spirv" {
            cmd.arg("--iree-metal-compile-to-metallib=false");
        }
        if backend == "cuda" {
            // IREE const-eval fails when a CUDA container has no visible CPU queue.
            cmd.arg("--iree-opt-const-eval=false");
            // Since IREE 3.12, the reduction vector distribution pipeline fails
            // on a sum of a scaled tensor ("failed to distribute").
            cmd.arg("--iree-codegen-llvmgpu-use-reduction-vector-distribution=false");
            if let Some(target) = detect_cuda_target() {
                cmd.arg(format!("--iree-cuda-target={}", target));
            }
        }
        if backend == "llvm-cpu" {
            cmd.arg("--iree-llvmcpu-target-cpu=host");
            cmd.arg("--iree-llvmcpu-enable-ukernels=all");
            cmd.arg("--iree-opt-data-tiling");
        }
        let output = cmd.output();

        if crate::core::config::verbosity() >= 2 {
            let debug_mlir = cache_dir.join(format!("{}-debug.mlir", safe_name));
            let _ = std::fs::rename(&mlir_path, &debug_mlir);
            sheaf_msg!("jit: {} | saved {}", name, debug_mlir.display());
        } else {
            let _ = std::fs::remove_file(&mlir_path);
        }

        let output = output.map_err(|e| JitDiagnostic::Environment {
            message: format!("Cannot execute iree-compile: {e}"),
            hint: "Check the toolchain installation or set IREE_COMPILE.",
        })?;
        let result = if output.status.success() {
            std::fs::read(&vmfb_path).map_err(|e| JitDiagnostic::io("Cannot read compiler output", e))
        } else {
            Err(JitDiagnostic::Internal(format!("iree-compile failed ({}): {}",
                output.status, String::from_utf8_lossy(&output.stderr).trim())))
        };
        let _ = std::fs::remove_file(&vmfb_path);
        result
    }

}
