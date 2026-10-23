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

fn preserves_structured_vag_error(error: &crate::core::error::SheafError) -> bool {
    matches!(
        error,
        crate::core::error::SheafError::AutodiffMissingGradientOutput { .. }
            | crate::core::error::SheafError::AutodiffMissingRule { .. }
            | crate::core::error::SheafError::InvalidShapeDimension { .. }
    )
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
    /// Tensor captures become MLIR parameters; scalar captures are constants.
    ///
    /// Returns `(module_name, signature, param_order)` for dispatch.
    pub fn try_jit_value_and_grad(
        &mut self,
        func: &Value,
        wrt_arg: &Value,
        registry: &HashMap<String, FunctionDef>,
        shared_session: &std::sync::Arc<crate::runtime::iree_session::IreeSession>,
    ) -> Option<(String, FunctionSignature, Vec<String>)> {
        self.last_vag_error = None;
        let iree_compile = self.iree_compile_path.clone()?;

        let (fn_params, body, closure) = match func {
            Value::Function {
                params,
                body,
                closure,
                ..
            } => (params, body, closure),
            _ => return None,
        };

        let vag_fn_name = outermost_call_name(body).unwrap_or("anonymous".to_string());

        // Include wrt_arg type so shape changes (e.g. after grow-hydra) cause a cache miss.
        let wrt_type_str = value_to_stablehlo_type(wrt_arg)
            .map(|t| t.to_mlir())
            .unwrap_or_default();
        // Same AST + different scalar values produces a different VMFB.
        let mut scalar_captures_key: Vec<(String, String)> = closure
            .iter()
            .filter(|(name, _)| !name.starts_with("__"))
            .filter_map(|(name, val)| match val {
                Value::Float(f) => Some((name.clone(), format!("f{}", f))),
                Value::Int(n) => Some((name.clone(), format!("i{}", n))),
                _ => None,
            })
            .collect();
        scalar_captures_key.sort();
        let scalars_suffix: String = scalar_captures_key
            .iter()
            .map(|(n, v)| format!("{}={}", n, v))
            .collect::<Vec<_>>()
            .join(",");
        let vag_key = format!("__vag_{:?}_{}_{}", body, wrt_type_str, scalars_suffix);
        if self.failed_vag.contains(&vag_key) {
            return None;
        }

        if let Some(cached) = self.vag_cache.get(&vag_key) {
            return Some(cached.clone());
        }

        // Skip impure or HOF-containing bodies
        if !collect_effects(body).is_empty() {
            return None;
        }
        if !collect_hof_calls(body).is_empty() {
            return None;
        }

        let mut all_param_names: Vec<String> = Vec::new();
        let mut all_arg_values: Vec<Value> = Vec::new();
        let mut scalar_substitutions: Vec<(String, CompiledExpr)> = Vec::new();

        for p in fn_params {
            all_param_names.push(p.clone());
            all_arg_values.push(wrt_arg.clone());
        }
        let wrt_indices: Vec<usize> = (0..fn_params.len()).collect();

        let mut aug_registry = registry.clone();

        for (cap_name, cap_val) in closure {
            if cap_name.starts_with("__") {
                continue;
            }
            match cap_val {
                Value::Float(f) => {
                    scalar_substitutions.push((cap_name.clone(), CompiledExpr::Float(*f as f64)));
                }
                Value::Int(n) => {
                    scalar_substitutions.push((cap_name.clone(), CompiledExpr::Integer(*n)));
                }
                Value::Function {
                    params: fp,
                    body: fb,
                    ..
                } => {
                    aug_registry.entry(cap_name.clone()).or_insert_with(|| FunctionDef {
                        name: cap_name.clone(),
                        params: fp.clone(),
                        body: crate::core::ast::SheafValue::Nil(
                            crate::core::error::SourceLocation::new(0, 0, "".into()),
                        ),
                        body_compiled: Some(fb.clone()),
                        signature: None,
                        vmfb_module_name: None,
                        known_param_types: Vec::new(),
                        compile_error: None,
                    });
                }
                _ => {
                    // Tensor, Dict, Tuple etc. -> promote to MLIR parameter
                    if value_to_stablehlo_type(cap_val).is_ok() {
                        all_param_names.push(cap_name.clone());
                        all_arg_values.push(cap_val.clone());
                    } else {
                        self.jit_fail(
                            &vag_key,
                            &format!("unsupported capture type for '{}'", cap_name),
                        );
                        return None;
                    }
                }
            }
        }

        let mut body = body.clone();
        for (name, val) in &scalar_substitutions {
            body = crate::core::bindings::replace_local(&body, name, val);
        }

        let mut known_types: Vec<(String, StableHLOType)> = Vec::new();
        let mut param_index_maps = ParamIndexMaps::new();
        let mut constants: HashMap<(String, Vec<usize>), i64> = HashMap::new();

        for (param_name, arg_val) in all_param_names.iter().zip(all_arg_values.iter()) {
            let ty = match value_to_stablehlo_type(arg_val) {
                Ok(ty) => ty,
                Err(e) => {
                    self.jit_fail(
                        &vag_key,
                        &format!("arg '{}' has {}", param_name, e.short_message()),
                    );
                    return None;
                }
            };

            if let Some(layout) = value_to_param_layout(param_name, arg_val) {
                let imap = layout_to_index_map(&layout);
                extract_scalar_constants(arg_val, param_name, &imap, &mut constants);
                body = lower_get_calls(&body, param_name, &imap);
                param_index_maps.push((param_name.clone(), imap));
            }

            known_types.push((param_name.clone(), ty));
        }

        let mut sig = match infer_function_signature_with_known(
            &all_param_names,
            &body,
            &known_types,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(&vag_key, &format!("signature inference: {}", e));
                return None;
            }
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
            if !scalar_substitutions.is_empty() {
                sheaf_msg!(
                    "jit: value-and-grad | {} scalar captures",
                    scalar_substitutions.len()
                );
            }
        }

        body = crate::autodiff::inline_function_calls(&body, &aug_registry);

        // Bail out if inlining produced a graph that is too large
        let node_count = expr_node_count(&body);
        if node_count > MAX_VAG_GRAPH_NODES {
            self.jit_fail(
                &vag_key,
                &format!(
                    "graph too large after inlining ({} nodes, limit {})",
                    node_count, MAX_VAG_GRAPH_NODES
                ),
            );
            return None;
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
            self.jit_fail(
                &vag_key,
                &format!(
                    "graph too large after unrolling ({} nodes, limit {})",
                    node_count, MAX_VAG_GRAPH_NODES
                ),
            );
            return None;
        }

        // Re-lower dict access introduced by unrolling (get on GetTupleElement elements)
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);

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
            Err(e) => {
                self.jit_fail("value_and_grad", &e.short_message());
                return None;
            }
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
                            expanded_body = expand_tuple_to_symbols(&expanded_body, param_name);
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
            Ok(Err(e)) if preserves_structured_vag_error(&e) => {
                self.last_vag_error = Some(e);
                return None;
            }
            Ok(Err(e)) => {
                self.jit_fail(&vag_key, &format!("codegen: {}", e));
                return None;
            }
            Err(_) => {
                self.jit_fail(&vag_key, "codegen: internal panic");
                return None;
            }
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
                Err(_) => {
                    self.jit_fail(&vag_key, "failed to read cached compilation");
                    return None;
                }
            }
        } else {
            // Debug: save a copy for inspection
            if crate::core::config::verbosity() >= 2 {
                let debug_path = cache_dir.join(format!("{}-vag-debug.mlir", vag_fn_name));
                let _ = std::fs::write(&debug_path, &mlir);
                sheaf_msg!("jit: value-and-grad | saved {}", debug_path.display());
            }

            let data = match self.run_iree_compile(&iree_compile, "value_and_grad", &mlir, &backend)
            {
                Some(d) => d,
                None => {
                    self.jit_fail(&vag_key, "compilation failed on all backends");
                    return None;
                }
            };

            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::write(&cached_vmfb, &data);
            update_manifest(&cache_dir, &vag_cache_name, &content_hash);

            data
        };

        // Load into the single shared IREE session (no per-compile session,
        // same rationale as `compile_function`).
        if let Err(e) = shared_session.load_vmfb(vmfb_data) {
            let _ = std::fs::remove_file(&cached_vmfb);
            self.jit_fail(&vag_key, &format!("JIT load: {}", e));
            return None;
        }

        let result = (vag_module_name, sig, all_param_names);
        self.vag_cache.insert(vag_key, result.clone());
        Some(result)
    }

    /// Run iree-compile on MLIR source for the given backend.
    /// Returns compiled VMFB bytes, or None if compilation fails.
    pub(super) fn run_iree_compile(
        &self,
        iree_compile: &str,
        name: &str,
        mlir: &str,
        backend: &str,
    ) -> Option<Vec<u8>> {
        if crate::core::config::verbosity() >= 1 {
            sheaf_msg!("jit: compiling {} [{}]...", name, backend);
        }

        let safe_name = name.replace('?', "_q").replace('!', "_b");
        let stamp = std::process::id();
        let cache_dir = std::path::PathBuf::from("__sheaf__");
        let _ = std::fs::create_dir_all(&cache_dir);
        let mlir_path = cache_dir.join(format!("{}-{}.mlir", safe_name, stamp));
        let vmfb_path = cache_dir.join(format!("{}-{}.vmfb", safe_name, stamp));

        if std::fs::write(&mlir_path, mlir).is_err() {
            sheaf_msg!("sheaf: failed to write temp MLIR, aborting");
            std::process::exit(1);
        }

        let stderr_cfg = if crate::core::config::verbosity() >= 2 {
            std::process::Stdio::inherit()
        } else {
            std::process::Stdio::null()
        };

        let mut cmd = std::process::Command::new(iree_compile);
        cmd.arg(&mlir_path)
            .arg(format!("--iree-hal-target-backends={}", backend))
            .arg("-o")
            .arg(&vmfb_path)
            .stderr(stderr_cfg);
        if backend == "metal-spirv" {
            cmd.arg("--iree-metal-compile-to-metallib=false");
        }
        if backend == "cuda" {
            // IREE const-eval fails when a CUDA container has no visible CPU queue.
            cmd.arg("--iree-opt-const-eval=false");
            if let Some(target) = detect_cuda_target() {
                cmd.arg(format!("--iree-cuda-target={}", target));
            }
        }
        if backend == "llvm-cpu" {
            cmd.arg("--iree-llvmcpu-target-cpu=host");
            cmd.arg("--iree-llvmcpu-enable-ukernels=all");
            cmd.arg("--iree-opt-data-tiling");
        }
        let status = cmd.status();

        if crate::core::config::verbosity() >= 2 {
            let debug_mlir = cache_dir.join(format!("{}-debug.mlir", safe_name));
            let _ = std::fs::rename(&mlir_path, &debug_mlir);
            sheaf_msg!("jit: {} | saved {}", name, debug_mlir.display());
        } else {
            let _ = std::fs::remove_file(&mlir_path);
        }

        let ok = match &status {
            Ok(s) => s.success(),
            Err(_) => false,
        };

        if ok && let Ok(data) = std::fs::read(&vmfb_path) {
            let _ = std::fs::remove_file(&vmfb_path);
            return Some(data);
        }
        let _ = std::fs::remove_file(&vmfb_path);
        None
    }

    pub(super) fn jit_fail(&mut self, name: &str, reason: &str) {
        if name.starts_with("__vag_") {
            self.failed_vag.insert(name.to_string());
            self.last_vag_fail_reason = Some(reason.to_string());
        } else {
            self.last_compile_failure = Some(reason.to_string());
        }
        let display_name = if name.starts_with("__vag_") {
            "value-and-grad"
        } else {
            name
        };
        if crate::core::config::verbosity() >= 1 {
            sheaf_msg!("jit: {} skipped ({})", display_name, reason);
        }
    }

    /// Classify the last VAG skip reason as legitimate (unsupported pattern)
    /// or a bug (JIT should handle but failed).
    pub fn classify_vag_skip(&self) -> JitVagOutcome {
        if let Some(error) = &self.last_vag_error {
            return JitVagOutcome::Success(Err(error.clone()));
        }
        match &self.last_vag_fail_reason {
            None => JitVagOutcome::Unsupported,
            Some(reason) => {
                let legit = reason.starts_with("has HOF calls")
                    || reason.starts_with("scalar-only")
                    || reason.starts_with("unsupported capture type")
                    || reason.starts_with("graph too large")
                    || reason.contains("Function call not yet supported");
                if legit {
                    JitVagOutcome::Unsupported
                } else {
                    JitVagOutcome::Bug(reason.clone())
                }
            }
        }
    }
}
