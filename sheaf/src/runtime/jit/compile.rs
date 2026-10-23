// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Standard JIT compilation pipeline.

use super::preprocess::VagPreprocessContext;
use super::support::{manifest_hash_matches, update_manifest};
use super::*;

struct CompileRequest<'a> {
    iree_compile: String,
    func_def: &'a FunctionDef,
    args: &'a [Value],
    registry: &'a HashMap<String, FunctionDef>,
    shared_session: &'a std::sync::Arc<crate::runtime::iree_session::IreeSession>,
    target_backend: &'a str,
    cache_key: &'a JitCacheKey,
}

impl JitCompiler {
    pub fn try_jit_compile(
        &mut self,
        func_def: &FunctionDef,
        args: &[Value],
        registry: &HashMap<String, FunctionDef>,
        shared_session: &std::sync::Arc<crate::runtime::iree_session::IreeSession>,
    ) -> JitCompileOutcome {
        let name = &func_def.name;
        // Cache-key construction performs shape analysis with call inlining.
        // Establish eligibility first, including the negative definition cache.
        if !self.preflight_jit_eligibility(func_def, registry) {
            let identity = function_definition_identity(func_def, registry);
            return JitCompileOutcome::Unsupported(self.failed_definitions[&identity].clone());
        }
        let Some(iree_compile) = self.iree_compile_path.clone() else {
            return JitCompileOutcome::Failed("compiler toolchain unavailable".to_string());
        };

        // The session's active driver is the source of truth for compilation.
        if let Some(backend) = crate::runtime::iree_session::IreeSession::cached_target_backend() {
            self.target_backend = backend.to_string();
        }

        // Success and failure caches use the same variant identity.
        let Some(cache_key) = cache_key_for_eligible_function(func_def, args, registry) else {
            return if func_def.body_compiled.is_none() {
                JitCompileOutcome::Unsupported("no compiled function body".to_string())
            } else {
                JitCompileOutcome::Failed("cannot compute JIT cache key".to_string())
            };
        };
        if let Some(reason) = self.failed_definitions.get(&cache_key.definition_hash) {
            return JitCompileOutcome::Unsupported(reason.clone());
        }

        // (1) Success cache hit: the module is already loaded into the shared
        // session; return its signature without recompiling.
        if let Some(info) = self.module_for_key(&cache_key) {
            JIT_CATALOG_HITS.fetch_add(1, Ordering::Relaxed);
            return JitCompileOutcome::Compiled(info.sig);
        }
        JIT_CATALOG_MISSES.fetch_add(1, Ordering::Relaxed);

        // (2) Failure cache hit: this exact (fn, shape) was tried before and
        // the compile failed. Don't retry.
        if let Some(reason) = self.failed_keys.get(&cache_key) {
            return JitCompileOutcome::Failed(reason.clone());
        }

        // Skip scalar-only functions (no benefit from IREE)
        let has_tensor = args.iter().any(|a| {
            matches!(
                a,
                Value::Tensor { .. } | Value::DeviceBuffer(_) | Value::Dict(_) | Value::Tuple(_)
            )
        });
        if !has_tensor {
            self.failed_definitions
                .insert(cache_key.definition_hash.clone(), "scalar-only args".to_string());
            self.jit_fail(name, "scalar-only args");
            return JitCompileOutcome::Unsupported("scalar-only args".to_string());
        }

        let module_name = module_name_for(name, &cache_key);
        if let Some(existing) = shared_module_catalog()
            .lock()
            .expect("JIT catalogue lock poisoned")
            .identities
            .get(&module_name)
            .cloned()
            && existing != cache_key {
                sheaf_msg!(
                    "ERROR jit: module fingerprint collision for {} (refusing to load)",
                    module_name
                );
                self.failed_keys.insert(cache_key, "module fingerprint collision".to_string());
                return JitCompileOutcome::Failed("module fingerprint collision".to_string());
            }

        // Reserve the variant without holding the catalogue lock during compilation.
        {
            let mut catalogue = shared_module_catalog()
                .lock()
                .expect("JIT catalogue lock poisoned");
            if !catalogue.compiling.insert(cache_key.clone()) {
                return JitCompileOutcome::InProgress;
            }
        }
        let mut reservation = CompilationReservation::new(cache_key.clone());

        let backend = self.target_backend.clone();
        let request = CompileRequest {
            iree_compile,
            func_def,
            args,
            registry,
            shared_session,
            target_backend: &backend,
            cache_key: &cache_key,
        };
        self.last_compile_failure = None;
        let sig = match self.compile_function(request) {
            Some(x) => x,
            None => {
                // The reservation guard removes the in-flight marker.
                let reason = self.last_compile_failure.take()
                    .unwrap_or_else(|| "JIT compilation returned no module".to_string());
                self.failed_keys.insert(cache_key, reason.clone());
                return JitCompileOutcome::Failed(reason);
            }
        };
        let info = CompiledModuleInfo {
            static_inputs: static_inputs::collect(args),
            function_name: name.to_string(),
            module_name: module_name.clone(),
            sig: sig.clone(),
            argument_layouts: args
                .iter()
                .map(crate::core::inference::ValueLayout::from_value)
                .collect(),
        };
        {
            let mut catalogue = shared_module_catalog()
                .lock()
                .expect("JIT catalogue lock poisoned");
            catalogue.identities.insert(module_name, cache_key.clone());
            if catalogue.modules.insert(cache_key.clone(), info).is_none() {
                catalogue
                    .variants
                    .entry(name.clone())
                    .or_default()
                    .push(cache_key.clone());
            }
            catalogue.compiling.remove(&cache_key);
        }
        self.definition_identities
            .insert(name.clone(), cache_key.definition_hash.clone());
        reservation.finish();
        // Module unloading is unavailable, so report excessive catalogue growth.
        let threshold = crate::core::config::jit_module_warning_threshold();
        let warning = {
            let mut catalogue = shared_module_catalog()
                .lock()
                .expect("JIT catalogue lock poisoned");
            let variants = catalogue
                .modules
                .values()
                .fold(BTreeMap::new(), |mut counts, info| {
                    *counts.entry(info.function_name.as_str()).or_default() += 1;
                    counts
                });
            let message = module_growth_warning(
                catalogue.modules.len(),
                threshold,
                catalogue.growth_warned,
                &variants,
            );
            if message.is_some() {
                catalogue.growth_warned = true;
            }
            message
        };
        if let Some(message) = warning {
            sheaf_msg!("{}", message);
        }

        JitCompileOutcome::Compiled(sig)
    }

    fn compile_function(
        &mut self,
        request: CompileRequest<'_>,
    ) -> Option<FunctionSignature> {
        let CompileRequest {
            iree_compile,
            func_def,
            args,
            registry,
            shared_session,
            target_backend,
            cache_key,
        } = request;
        let name = &func_def.name;
        let module_name = module_name_for(name, cache_key);

        let prepared = self.prepare(func_def, args)?;
        let lowered = self.lower(func_def, registry, prepared)?;
        let emitted = self.emit_stablehlo(
            func_def,
            args,
            registry,
            &module_name,
            lowered,
        )?;

        if !self.compile_and_load_module(
            &iree_compile,
            name,
            &emitted.mlir,
            target_backend,
            shared_session,
        ) {
            return None;
        }

        Some(emitted.signature)
    }

    /// Infers the signature and lowers runtime argument layouts.
    fn prepare(
        &mut self,
        func_def: &FunctionDef,
        args: &[Value],
    ) -> Option<PreparedFunction> {
        let name = &func_def.name;
        let mut body = func_def.body_compiled.clone()?;
        let mut known_types: Vec<(String, StableHLOType)> = Vec::new();
        let mut param_index_maps = ParamIndexMaps::new();
        let mut constants: HashMap<(String, Vec<usize>), i64> = HashMap::new();

        for (param_name, arg_val) in func_def.params.iter().zip(args) {
            let ty = match value_to_stablehlo_type(arg_val) {
                Ok(ty) => ty,
                Err(e) => {
                    self.jit_fail(
                        name,
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

        body = static_inputs::specialize(&body, &func_def.params, args);
        let mut signature = match infer_function_signature_with_known(
            &func_def.params,
            &body,
            &known_types,
        ) {
            Ok(signature) => signature,
            Err(e) => {
                self.jit_fail(name, &format!("signature inference: {}", e));
                return None;
            }
        };

        for (param_name, ty) in &known_types {
            if let Some(index) = func_def.params.iter().position(|param| param == param_name) {
                signature.param_types[index] = ty.clone();
            }
        }

        log_inferred_signature(name, &func_def.params, &signature, &param_index_maps);
        capture_argument_layouts(args, &mut signature);

        Some(PreparedFunction {
            body,
            signature,
            known_types,
            param_index_maps,
            constants,
            static_inputs: static_inputs::collect(args),
        })
    }

    /// Applies call inlining and all shape / tuple-dependent lowerings.
    fn lower(
        &mut self,
        func_def: &FunctionDef,
        registry: &HashMap<String, FunctionDef>,
        prepared: PreparedFunction,
    ) -> Option<LoweredFunction> {
        let PreparedFunction {
            mut body,
            signature,
            known_types,
            param_index_maps,
            constants,
            static_inputs,
        } = prepared;

        body = crate::autodiff::inline_function_calls(&body, registry);
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);
        body = static_inputs::apply(&body, &func_def.params, &static_inputs);

        let param_shapes: HashMap<String, Vec<i64>> = func_def
            .params
            .iter()
            .zip(signature.param_types.iter())
            .filter_map(|(param, ty)| {
                let shape = ty.shape();
                if shape.is_empty() {
                    None
                } else {
                    Some((param.clone(), shape.to_vec()))
                }
            })
            .collect();
        let filtered_constants = filter_constants_for_shape_positions(&constants, &body);

        let arity_err: std::cell::Cell<Option<SheafError>> = std::cell::Cell::new(None);
        let preprocess_context = VagPreprocessContext {
            registry,
            constants: &filtered_constants,
            param_shapes: &param_shapes,
            param_index_maps: &param_index_maps,
            known_types: &known_types,
            arity_err: &arity_err,
        };
        body = self.preprocess_vag_lambda(&body, &preprocess_context);
        if let Some(err) = arity_err.into_inner() {
            self.jit_fail(&func_def.name, &err.short_message());
            return None;
        }

        body = resolve_static_constants(&body, &filtered_constants, &param_shapes, false);
        body = match crate::lowering::transforms::lower_tuples_and_destructuring(
            body,
            &param_shapes,
        ) {
            Ok(body) => body,
            Err(e) => {
                self.jit_fail(&func_def.name, &e.short_message());
                return None;
            }
        };

        Some(LoweredFunction {
            body,
            signature,
            param_index_maps,
            filtered_constants,
        })
    }

    /// Emits one StableHLO declaration and wraps it in its namespaced module.
    fn emit_stablehlo(
        &mut self,
        func_def: &FunctionDef,
        args: &[Value],
        registry: &HashMap<String, FunctionDef>,
        module_name: &str,
        lowered: LoweredFunction,
    ) -> Option<EmittedFunction> {
        let LoweredFunction {
            body,
            mut signature,
            param_index_maps,
            filtered_constants,
        } = lowered;
        let metadata = prepare_codegen_metadata(&body, &param_index_maps);

        let codegen_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut codegen = CodeGenerator::with_function_params(
                registry,
                &func_def.params,
                &signature.param_types,
            );
            codegen.set_tuple_key_layouts(metadata.tuple_key_layouts);
            codegen.set_idx_to_key(metadata.idx_to_key);
            let layouts = args.iter()
                .map(crate::core::inference::ValueLayout::from_value)
                .collect::<Vec<_>>();
            codegen.set_parameter_value_layouts(&func_def.params, &layouts);
            for input in static_inputs::collect(args) {
                codegen.specialize_parameter_leaf(
                    &func_def.params[input.argument], &input.indices, &input.value.expression(),
                )?;
            }
            codegen.emit_func_declaration_with_layout(
                &func_def.name,
                &body,
                &signature.param_types,
                &signature.return_type,
            )
        }));

        let (declaration, actual_return_ty, return_layout) = match codegen_result {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                self.jit_fail(&func_def.name, &format!("codegen: {}", e));
                return None;
            }
            Err(_) => {
                self.jit_fail(&func_def.name, "codegen: internal panic");
                return None;
            }
        };
        signature.return_type = actual_return_ty;
        signature.return_layout = return_layout;

        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!(
                "jit: {} | codegen return: {}",
                func_def.name,
                signature.return_type.to_mlir()
            );
            sheaf_msg!(
                "jit: {} | return_dict_keys: {:?}",
                func_def.name,
                signature.return_dict_keys
            );
        }

        register_return_layout(&mut signature);
        let mlir = assemble_stablehlo_module(module_name, declaration);
        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!(
                "jit: {} | MLIR {} lines",
                func_def.name,
                mlir.lines().count()
            );
        }

        signature.captured_scalars = filtered_constants.into_iter()
            .map(|(path, value)| (path, value as f64)).collect();
        Some(EmittedFunction { signature, mlir })
    }

    /// Reuses or compiles a VMFB and appends it to the shared runtime session.
    fn compile_and_load_module(
        &mut self,
        iree_compile: &str,
        name: &str,
        mlir: &str,
        target_backend: &str,
        shared_session: &std::sync::Arc<crate::runtime::iree_session::IreeSession>,
    ) -> bool {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        mlir.hash(&mut hasher);
        target_backend.hash(&mut hasher);
        IREE_COMPILER_VERSION.hash(&mut hasher);
        let content_hash = format!("{:016x}", hasher.finish());

        let cache_dir = PathBuf::from("__sheaf__");
        let safe_name = name.replace('?', "_q").replace('!', "_b");
        let backend_suffix = target_backend.replace('-', "_");
        let cached_vmfb = cache_dir.join(format!("{}.{}.vmfb", safe_name, backend_suffix));
        let force_recompile = crate::core::config::verbosity() >= 2;
        let vmfb_data = if !force_recompile
            && cached_vmfb.exists()
            && manifest_hash_matches(&cache_dir, name, &content_hash)
        {
            match std::fs::read(&cached_vmfb) {
                Ok(data) => {
                    if crate::core::config::verbosity() >= 2 {
                        sheaf_msg!(
                            "jit: {} (cached, {}KB, {})",
                            name,
                            data.len() / 1024,
                            target_backend
                        );
                    } else if crate::core::config::verbosity() >= 1 {
                        sheaf_msg!("jit: {} (cached)", name);
                    }
                    data
                }
                Err(_) => {
                    self.jit_fail(name, "failed to read cached compilation");
                    return false;
                }
            }
        } else {
            JIT_EXTERNAL_COMPILATIONS.fetch_add(1, Ordering::Relaxed);
            let data = match self.run_iree_compile(iree_compile, name, mlir, target_backend) {
                Some(data) => data,
                None => {
                    self.jit_fail(name, "iree-compile failed on all backends");
                    return false;
                }
            };
            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::write(&cached_vmfb, &data);
            update_manifest(&cache_dir, name, &content_hash);
            data
        };

        JIT_MODULE_LOADS.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = shared_session.load_vmfb(vmfb_data) {
            let _ = std::fs::remove_file(&cached_vmfb);
            self.jit_fail(name, &format!("JIT load: {}", e));
            return false;
        }
        true
    }
}

struct PreparedFunction {
    body: CompiledExpr,
    signature: FunctionSignature,
    known_types: Vec<(String, StableHLOType)>,
    param_index_maps: ParamIndexMaps,
    constants: HashMap<(String, Vec<usize>), i64>,
    static_inputs: Vec<static_inputs::StaticInput>,
}

struct LoweredFunction {
    body: CompiledExpr,
    signature: FunctionSignature,
    param_index_maps: ParamIndexMaps,
    filtered_constants: HashMap<(String, Vec<usize>), i64>,
}

struct CodegenMetadata {
    tuple_key_layouts: HashMap<String, BTreeMap<String, usize>>,
    idx_to_key: HashMap<(String, usize), String>,
}

struct EmittedFunction {
    signature: FunctionSignature,
    mlir: String,
}

fn log_inferred_signature(
    name: &str,
    params: &[String],
    signature: &FunctionSignature,
    param_index_maps: &ParamIndexMaps,
) {
    if crate::core::config::verbosity() < 2 {
        return;
    }
    for (param_name, param_ty) in params.iter().zip(signature.param_types.iter()) {
        if let Some((_, index_map)) = param_index_maps.iter().find(|(name, _)| name == param_name) {
            let mut top_fields: BTreeMap<usize, (String, String)> = BTreeMap::new();
            for (path, indices) in index_map {
                if path.len() == 1 {
                    let ty = if let StableHLOType::Tuple(elements, _) = param_ty {
                        elements
                            .get(indices[0])
                            .map(StableHLOType::to_mlir)
                            .unwrap_or_else(|| "?".to_string())
                    } else {
                        "?".to_string()
                    };
                    top_fields.entry(indices[0]).or_insert((path[0].clone(), ty));
                } else {
                    top_fields.entry(indices[0]).or_insert_with(|| {
                        let ty = if let StableHLOType::Tuple(elements, _) = param_ty {
                            if let Some(StableHLOType::Tuple(sub, _)) = elements.get(indices[0]) {
                                format!("tuple<...> ({} fields)", sub.len())
                            } else {
                                "tuple<...>".to_string()
                            }
                        } else {
                            "tuple<...>".to_string()
                        };
                        (path[0].clone(), ty)
                    });
                }
            }
            for (key, ty) in top_fields.values() {
                sheaf_msg!("jit: {} | {}.{}: {}", name, param_name, key, ty);
            }
        } else {
            sheaf_msg!("jit: {} | {}: {}", name, param_name, param_ty.to_mlir());
        }
    }
    sheaf_msg!("jit: {} | return: {}", name, signature.return_type.to_mlir());
}

fn capture_argument_layouts(args: &[Value], signature: &mut FunctionSignature) {
    use crate::core::inference::ValueLayout;
    let mut seen_types = std::collections::HashSet::new();
    for (arg, ty) in args.iter().zip(signature.param_types.iter()) {
        let layout = ValueLayout::from_value(arg);
        if !matches!(layout, ValueLayout::Leaf) {
            let type_key = format!("{:?}", ty);
            if seen_types.insert(type_key) {
                signature.arg_type_layouts.push((ty.clone(), layout));
            }
        }
    }
}

fn prepare_codegen_metadata(
    body: &CompiledExpr,
    param_index_maps: &ParamIndexMaps,
) -> CodegenMetadata {
    let mut tuple_key_layouts: HashMap<String, BTreeMap<String, usize>> = HashMap::new();
    let mut idx_to_key: HashMap<(String, usize), String> = HashMap::new();
    for (param_name, index_map) in param_index_maps {
        for (key_path, indices) in index_map {
            for depth in 0..key_path.len() {
                let parent = if depth == 0 {
                    param_name.clone()
                } else {
                    key_path[depth - 1].clone()
                };
                let child = &key_path[depth];
                let index = indices[depth];
                tuple_key_layouts
                    .entry(parent.clone())
                    .or_default()
                    .entry(child.clone())
                    .or_insert(index);
                idx_to_key
                    .entry((parent, index))
                    .or_insert_with(|| child.clone());
            }
        }
    }
    propagate_let_layouts(body, &idx_to_key, &mut tuple_key_layouts);

    CodegenMetadata {
        tuple_key_layouts,
        idx_to_key,
    }
}

fn register_return_layout(signature: &mut FunctionSignature) {
    let StableHLOType::Tuple(return_elements, return_keys) = &signature.return_type else {
        return;
    };
    if return_keys.is_none()
        || signature
            .arg_type_layouts
            .iter()
            .any(|(ty, _)| ty == &signature.return_type)
    {
        return;
    }
    for (param_ty, layout) in signature.arg_type_layouts.clone() {
        if let StableHLOType::Tuple(param_elements, param_keys) = &param_ty
            && param_elements.len() == return_elements.len()
            && param_keys == return_keys
        {
            signature
                .arg_type_layouts
                .push((signature.return_type.clone(), layout));
            break;
        }
    }
}

fn assemble_stablehlo_module(module_name: &str, declaration: String) -> String {
    StableHLOEmitter::emit_module_named(Some(module_name), &[declaration])
}
