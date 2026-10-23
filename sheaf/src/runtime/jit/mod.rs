// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! JIT auto-compilation: transparently compile pure functions on first call.
//!
//! Eligible functions compile to VMFB on first use:
//! type inference -> dict lowering -> inlining -> StableHLO generation
//! -> iree-compile -> shared-session module loading.
//! Subsequent calls dispatch through the loaded module. Unsupported patterns
//! fall back to the interpreter; structured value-and-grad errors propagate to
//! the caller.

use crate::SheafError;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Cached CUDA target architecture (e.g. "sm_75"), detected once via NVML.
static CUDA_TARGET: OnceLock<Option<String>> = OnceLock::new();

/// Process-wide JIT events, primarily for deterministic integration tests.
pub static JIT_CATALOG_HITS: AtomicUsize = AtomicUsize::new(0);
pub static JIT_CATALOG_MISSES: AtomicUsize = AtomicUsize::new(0);
pub static JIT_MODULE_LOADS: AtomicUsize = AtomicUsize::new(0);
pub static JIT_EXTERNAL_COMPILATIONS: AtomicUsize = AtomicUsize::new(0);

pub fn detect_cuda_target() -> Option<String> {
    CUDA_TARGET
        .get_or_init(|| {
            let nvml = nvml_wrapper::Nvml::init().or_else(|_| {
                nvml_wrapper::Nvml::builder()
                    .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
                    .init()
            });
            let nvml = match nvml {
                Ok(n) => n,
                Err(e) => {
                    sheaf_msg!("nvml: init failed: {}", e);
                    return None;
                }
            };
            let device = match nvml.device_by_index(0) {
                Ok(d) => d,
                Err(e) => {
                    sheaf_msg!("nvml: device_by_index failed: {}", e);
                    return None;
                }
            };
            let cc = match device.cuda_compute_capability() {
                Ok(c) => c,
                Err(e) => {
                    sheaf_msg!("nvml: compute_capability failed: {}", e);
                    return None;
                }
            };
            let target = format!("sm_{}{}", cc.major, cc.minor);
            Some(target)
        })
        .clone()
}

use super::toolchain::{ensure_toolchain, find_iree_compile, IREE_COMPILER_VERSION};

use crate::autodiff::{
    reverse::{GradientOutput, ReverseGradResult, reverse_grad, to_anf},
    simplify,
};
use crate::core::expr::{CompiledExpr, FunctionDef};
use crate::lowering::codegen::{
    collect_tuple_references, collect_tuple_type_leaves, expand_tuple_to_symbols, CodeGenerator,
};
use crate::lowering::config::{
    DictIndexMap, ParamIndexMap, ParamIndexMaps, layout_to_index_map, lower_get_calls,
};
use crate::lowering::effects::{collect_effects, collect_hof_calls};
use crate::lowering::jit_eligibility;
use crate::lowering::stablehlo::{Register, StableHLOEmitter};
use crate::lowering::transforms::{
    extract_scalar_constants, filter_constants_for_shape_positions, lower_inlined_gets,
    propagate_let_layouts, resolve_static_constants, unroll_reduces,
};
use crate::sheaf_msg;

/// Identity of a compiled JIT variant.
///
/// The key contains the transitive function definition, argument types and
/// layouts, and runtime scalars consumed in shape expressions. Tensor contents
/// and non-constant scalar values are excluded. Unlike the disk content hash,
/// this key can be computed without emitting MLIR.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct JitCacheKey {
    /// Definition and transitive inlining identity, computed before lowering.
    definition_hash: String,
    function_name: String,
    argument_types: Vec<String>,
    argument_layouts: Vec<ArgumentLayout>,
    captured_scalars: Vec<(String, Vec<usize>, u64)>,
    static_inputs: Vec<static_inputs::StaticInput>,
}

type ArgumentLayout = (String, Vec<(Vec<String>, Vec<i64>, Vec<usize>)>);

fn runtime_argument_layouts(args: &[Value]) -> Vec<ArgumentLayout> {
    args.iter()
        .filter_map(|arg| value_to_param_layout("", arg))
        .map(|layout| {
            let fields = layout
                .fields
                .into_iter()
                .map(|field| (field.path, field.shape, field.tuple_index))
                .collect();
            (layout.name, fields)
        })
        .collect()
}

struct StableHasher(u64);
impl std::hash::Hasher for StableHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

fn function_definition_identity(
    func_def: &FunctionDef,
    registry: &HashMap<String, FunctionDef>,
) -> String {
    let mut pending = vec![func_def.name.clone()];
    let mut seen = HashSet::new();
    let mut identities = Vec::new();

    while let Some(name) = pending.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(definition) = registry.get(&name) else {
            continue;
        };
        identities.push((name, definition.body_hash()));
        if let Some(body) = &definition.body_compiled {
            let mut callees: Vec<_> = crate::lowering::call_graph::direct_callees(body)
                .into_iter()
                .filter(|callee| registry.contains_key(callee))
                .collect();
            callees.sort();
            pending.extend(callees);
        }
    }

    identities.sort();
    format!("{:?}", identities)
}

impl JitCacheKey {
    fn fingerprint(&self) -> String {
        use std::hash::{Hash, Hasher};
        let mut first = StableHasher(0xcbf29ce484222325);
        let mut second = StableHasher(0x84222325cbf29ce4);
        self.hash(&mut first);
        self.hash(&mut second);
        format!("{:016x}{:016x}", first.finish(), second.finish())
    }
}

#[cfg(test)]
fn cache_key_for(
    fn_name: &str,
    params: &[String],
    args: &[Value],
    body_compiled: &CompiledExpr,
) -> JitCacheKey {
    cache_key_for_with_identity(fn_name, params, args, body_compiled, String::new())
}

/// Builds the cache key used by lookup callers that have not checked eligibility.
pub fn cache_key_for_function(
    func_def: &FunctionDef,
    args: &[Value],
    registry: &HashMap<String, FunctionDef>,
) -> Option<JitCacheKey> {
    // Shape analysis inlines the call graph. It is undefined for recursive
    // functions, so enforce eligibility at this public boundary.
    jit_eligibility(&func_def.name, registry).ok()?;
    cache_key_for_eligible_function(func_def, args, registry)
}

/// Builds a cache key after the caller has established JIT eligibility.
pub(crate) fn cache_key_for_eligible_function(
    func_def: &FunctionDef,
    args: &[Value],
    registry: &HashMap<String, FunctionDef>,
) -> Option<JitCacheKey> {
    let body = func_def.body_compiled.as_ref()?;
    let mut key = cache_key_for_with_identity(
        &func_def.name,
        &func_def.params,
        args,
        body,
        function_definition_identity(func_def, registry),
    );
    key.captured_scalars = shape_bearing_scalars(func_def, args, registry)?;
    Some(key)
}

/// Returns shape-dependent scalar constants after call inlining and get lowering.
fn shape_bearing_scalars(
    func_def: &FunctionDef,
    args: &[Value],
    registry: &HashMap<String, FunctionDef>,
) -> Option<Vec<(String, Vec<usize>, u64)>> {
    let mut body = func_def.body_compiled.clone()?;
    let mut constants = HashMap::new();
    let mut index_maps = Vec::new();
    let structural_signature = args
        .iter()
        .map(|arg| {
            value_to_stablehlo_type(arg)
                .map(|ty| ty.to_mlir())
                .unwrap_or_else(|_| "<untypeable>".into())
        })
        .collect::<Vec<_>>();
    let analysis_key = (
        function_definition_identity(func_def, registry),
        structural_signature,
        runtime_argument_layouts(args),
    );

    for (index, arg) in args.iter().enumerate() {
        let param = func_def
            .params
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("__a{index}"));
        let index_map = if let Some(layout) = value_to_param_layout(&param, arg) {
            layout_to_index_map(&layout)
        } else {
            std::iter::once((Vec::new(), vec![])).collect()
        };
        extract_scalar_constants(arg, &param, &index_map, &mut constants);
        body = lower_get_calls(&body, &param, &index_map);
        index_maps.push((param, index_map));
    }

    let positions = if let Some(positions) = shape_analysis_cache()
        .lock()
        .expect("shape analysis cache lock poisoned")
        .get(&analysis_key)
        .cloned()
    {
        positions
    } else {
        body = crate::autodiff::inline_function_calls(&body, registry);
        for (param, index_map) in &index_maps {
            body = lower_get_calls(&body, param, index_map);
        }
        body = lower_inlined_gets(&body, &index_maps);
        let mut positions = crate::lowering::transforms::collect_shape_gtes(&body)
            .into_iter()
            .collect::<Vec<_>>();
        positions.sort();
        shape_analysis_cache()
            .lock()
            .expect("shape analysis cache lock poisoned")
            .insert(analysis_key, positions.clone());
        positions
    };

    Some(
        constants
            .into_iter()
            .filter(|(path, _)| positions.contains(path))
            .collect::<BTreeMap<_, _>>()
            .into_iter()
            .map(|((param, indices), value)| (param, indices, value as u64))
            .collect(),
    )
}

fn cache_key_for_with_identity(
    fn_name: &str,
    params: &[String],
    args: &[Value],
    body_compiled: &CompiledExpr,
    definition_hash: String,
) -> JitCacheKey {
    let argument_types = args
        .iter()
        .map(|arg| {
            value_to_stablehlo_type(arg)
                .map(|ty| ty.to_mlir())
                .unwrap_or_else(|_| "<untypeable>".to_string())
        })
        .collect();
    let argument_layouts = runtime_argument_layouts(args);

    let mut constants: HashMap<(String, Vec<usize>), i64> = HashMap::new();
    for (i, arg) in args.iter().enumerate() {
        let param_name = params
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("__a{}", i));
        if let Some(layout) = value_to_param_layout(&param_name, arg) {
            let imap = layout_to_index_map(&layout);
            extract_scalar_constants(arg, &param_name, &imap, &mut constants);
        } else {
            let fake_imap: DictIndexMap =
                std::iter::once((Vec::new(), vec![])).collect();
            extract_scalar_constants(arg, &param_name, &fake_imap, &mut constants);
        }
    }
    let captured_scalars = filter_constants_for_shape_positions(&constants, body_compiled)
        .into_iter()
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .map(|((param, indices), value)| (param, indices, value as u64))
        .collect();

    JitCacheKey {
        definition_hash,
        function_name: fn_name.to_string(),
        argument_types,
        argument_layouts,
        captured_scalars,
        static_inputs: static_inputs::collect(args),
    }
}

/// Derives a unique IREE module name for a compiled variant.
pub fn module_name_for(fn_name: &str, cache_key: &JitCacheKey) -> String {
    let safe = fn_name
        .replace('-', "_")
        .replace('?', "_q")
        .replace('!', "_b");
    format!("{}__{}", safe, cache_key.fingerprint())
}

/// Count AST nodes in an expression tree. Used to bail out of JIT
/// when inlining or unrolling produces a graph that is too large.
fn expr_node_count(expr: &CompiledExpr) -> usize {
    match expr {
        CompiledExpr::FunctionCall { args, .. } => {
            1 + args.iter().map(expr_node_count).sum::<usize>()
        }
        CompiledExpr::Let { bindings, body } => {
            1 + bindings
                .iter()
                .map(|(_, v)| expr_node_count(v))
                .sum::<usize>()
                + expr_node_count(body)
        }
        CompiledExpr::Do(exprs) => 1 + exprs.iter().map(expr_node_count).sum::<usize>(),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            1 + expr_node_count(condition)
                + expr_node_count(then_branch)
                + else_branch.as_ref().map_or(0, |e| expr_node_count(e))
        }
        CompiledExpr::Lambda { body, .. } => 1 + expr_node_count(body),
        CompiledExpr::LambdaCall { callee, args } => {
            1 + expr_node_count(callee) + args.iter().map(expr_node_count).sum::<usize>()
        }
        CompiledExpr::Vector(elems) => 1 + elems.iter().map(expr_node_count).sum::<usize>(),
        _ => 1,
    }
}

const MAX_VAG_GRAPH_NODES: usize = 10_000;
use crate::core::inference::{
    infer_function_signature_with_known, FunctionSignature, ValueLayout,
};
use crate::core::trace::{value_to_param_layout, value_to_stablehlo_type};
use crate::interpreter::value::Value;
use crate::StableHLOType;

/// Process-wide metadata for modules loaded in the shared IREE session.
/// The mutex must not be held during compilation, module loading, or dispatch.
#[derive(Default)]
struct SharedModuleCatalog {
    modules: HashMap<JitCacheKey, CompiledModuleInfo>,
    variants: HashMap<String, Vec<JitCacheKey>>,
    identities: HashMap<String, JitCacheKey>,
    compiling: HashSet<JitCacheKey>,
    growth_warned: bool,
}

static SHARED_MODULE_CATALOG: OnceLock<Mutex<SharedModuleCatalog>> = OnceLock::new();

struct CompilationReservation {
    key: Option<JitCacheKey>,
}

impl CompilationReservation {
    fn new(key: JitCacheKey) -> Self {
        Self { key: Some(key) }
    }

    fn finish(&mut self) {
        self.key = None;
    }
}

impl Drop for CompilationReservation {
    fn drop(&mut self) {
        if let Some(key) = self.key.take() {
            shared_module_catalog()
                .lock()
                .expect("JIT catalogue lock poisoned")
                .compiling
                .remove(&key);
        }
    }
}

/// Cached shape-position analyses. Scalar values are projected afterwards.
type ShapeAnalysisKey = (String, Vec<String>, Vec<ArgumentLayout>);
type ShapeAnalysis = Vec<(String, Vec<usize>)>;
type ShapeAnalysisCache = Mutex<HashMap<ShapeAnalysisKey, ShapeAnalysis>>;

static SHAPE_ANALYSIS_CACHE: OnceLock<ShapeAnalysisCache> = OnceLock::new();

fn shared_module_catalog() -> &'static Mutex<SharedModuleCatalog> {
    SHARED_MODULE_CATALOG.get_or_init(|| Mutex::new(SharedModuleCatalog::default()))
}

fn shape_analysis_cache() -> &'static ShapeAnalysisCache {
    SHAPE_ANALYSIS_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct JitCompiler {
    iree_compile_path: Option<String>,
    target_backend: String,
    /// Definitions rejected independently of runtime argument shapes.
    failed_definitions: HashMap<String, String>,
    failed_vag: HashSet<String>,
    /// Variants that failed compilation, with the original diagnostic.
    failed_keys: HashMap<JitCacheKey, String>,
    last_compile_failure: Option<String>,
    last_vag_fail_reason: Option<String>,
    last_vag_error: Option<SheafError>,
    /// Cache compiled VAG sessions: vag_key -> (session_idx, signature, param_names)
    vag_cache: HashMap<String, (String, FunctionSignature, Vec<String>)>,
    /// Definition identities already validated by the full compilation-key path.
    definition_identities: HashMap<String, String>,
}

/// Dispatch metadata for a module loaded in the shared IREE session.
#[derive(Debug, Clone)]
pub struct CompiledModuleInfo {
    pub(crate) static_inputs: Vec<static_inputs::StaticInput>,
    pub function_name: String,
    pub module_name: String,
    pub sig: FunctionSignature,
    pub(crate) argument_layouts: Vec<ValueLayout>,
}

fn runtime_args_match(
    args: &[Value],
    params: &[String],
    info: &CompiledModuleInfo,
) -> bool {
    if args.len() != info.sig.param_types.len()
        || args.len() != info.argument_layouts.len()
        || static_inputs::collect(args) != info.static_inputs
    {
        return false;
    }
    if !args
        .iter()
        .zip(&info.argument_layouts)
        .all(|(value, layout)| runtime_layout_matches(value, layout))
    {
        return false;
    }
    if !args
        .iter()
        .zip(&info.sig.param_types)
        .all(|(value, ty)| runtime_type_matches(value, ty))
    {
        return false;
    }
    info.sig.captured_scalars.iter().all(|((param, indices), expected)| {
        let Some(param_index) = params.iter().position(|name| name == param) else {
            return false;
        };
        runtime_scalar_at(&args[param_index], indices)
            .is_some_and(|actual| actual.to_bits() == expected.to_bits())
    })
}

fn runtime_layout_matches(value: &Value, layout: &ValueLayout) -> bool {
    match (value, layout) {
        (Value::Dict(values), ValueLayout::Dict(entries)) => {
            values.len() == entries.len()
                && values.iter().zip(entries).all(
                    |((actual_key, actual), (expected_key, expected))| {
                        actual_key == expected_key
                            && runtime_layout_matches(actual, expected)
                    },
                )
        }
        (Value::List(values), ValueLayout::List(entries)) => {
            values.len() == entries.len()
                && values
                    .iter()
                    .zip(entries)
                    .all(|(actual, expected)| runtime_layout_matches(actual, expected))
        }
        (Value::Dict(_) | Value::List(_), ValueLayout::Leaf) => false,
        (_, ValueLayout::Leaf) => true,
        _ => false,
    }
}

fn runtime_type_matches(value: &Value, expected: &StableHLOType) -> bool {
    use crate::core::dtype::ElementType;

    match (value, expected) {
        (
            Value::Float(_) | Value::Int(_) | Value::Bool(_) | Value::Nil,
            StableHLOType::ScalarF32,
        ) => true,
        (Value::Tensor { data, dtype }, StableHLOType::Tensor { shape, dtype: expected_dtype }) => {
            let actual_dtype = match dtype {
                ElementType::F16 | ElementType::BF16 | ElementType::F32 => *dtype,
                _ => ElementType::F32,
            };
            actual_dtype == *expected_dtype
                && data.shape().iter().map(|&dim| dim as i64).eq(shape.iter().copied())
        }
        (Value::DeviceBuffer(buffer), StableHLOType::Tensor { shape, dtype: expected_dtype }) => {
            let actual_dtype = match buffer.dtype {
                ElementType::F16 | ElementType::BF16 | ElementType::F32 => buffer.dtype,
                _ => ElementType::F32,
            };
            actual_dtype == *expected_dtype
                && buffer.shape.iter().map(|&dim| dim as i64).eq(shape.iter().copied())
        }
        (Value::List(values), StableHLOType::Tensor { shape, dtype }) => {
            *dtype == ElementType::F32
                && shape.as_slice() == [values.len() as i64]
                && values
                    .iter()
                    .all(|value| matches!(value, Value::Float(_) | Value::Int(_)))
        }
        (Value::Dict(values), StableHLOType::Tuple(types, Some(keys))) => {
            values.len() == types.len()
                && values.len() == keys.len()
                && values.iter().zip(keys).zip(types).all(
                    |(((actual_key, actual), expected_key), expected_type)| {
                        actual_key == expected_key
                            && runtime_type_matches(actual, expected_type)
                    },
                )
        }
        (Value::Tuple(values), StableHLOType::Tuple(types, None))
        | (Value::List(values), StableHLOType::Tuple(types, None)) => {
            values.len() == types.len()
                && values
                    .iter()
                    .zip(types)
                    .all(|(actual, expected)| runtime_type_matches(actual, expected))
        }
        _ => false,
    }
}

fn runtime_scalar_at(value: &Value, indices: &[usize]) -> Option<f64> {
    let mut current = value;
    for &index in indices {
        current = match current {
            Value::Dict(values) => values.values().nth(index)?,
            Value::List(values) | Value::Tuple(values) => values.get(index)?,
            _ => return None,
        };
    }
    match current {
        Value::Float(value) => Some(*value as f64),
        Value::Int(value) => Some(*value as f64),
        Value::Tensor { data, .. } if data.len() == 1 => {
            data.iter().next().map(|&value| value as f64)
        }
        _ => None,
    }
}

fn module_growth_warning(
    total: usize,
    threshold: usize,
    already_warned: bool,
    variants: &BTreeMap<&str, usize>,
) -> Option<String> {
    if threshold == 0 || already_warned || total <= threshold {
        return None;
    }
    let breakdown = variants
        .iter()
        .map(|(name, count)| format!("{}={}", name, count))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "WARN jit: {} distinct VMFB modules loaded (variants: {}).",
        total, breakdown
    ))
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl JitCompiler {
    pub fn new() -> Self {
        let target_backend = Self::detect_target_backend();
        let iree_compile_path = find_iree_compile().or_else(|| {
            match ensure_toolchain() {
                Ok(path) => Some(path),
                Err(e) => {
                    // Check if cached VMFBs exist, if so, we can still run
                    let has_cache = std::path::Path::new("__sheaf__/manifest.json").exists();
                    if has_cache {
                        sheaf_msg!("sheaf: compiler toolchain not available, using cached compilations");
                    } else {
                        sheaf_msg!("error: Could not download the compiler toolchain and no cached compilation found.");
                        sheaf_msg!("       Reason: {}", e);
                        sheaf_msg!("       Install curl and unzip, or set IREE_COMPILE to a local iree-compile binary.");
                        std::process::exit(1);
                    }
                    None
                }
            }
        });
        Self {
            iree_compile_path,
            target_backend,
            failed_definitions: HashMap::new(),
            failed_vag: HashSet::new(),
            failed_keys: HashMap::new(),
            last_compile_failure: None,
            last_vag_fail_reason: None,
            last_vag_error: None,
            vag_cache: HashMap::new(),
            definition_identities: HashMap::new(),
        }
    }

    fn detect_target_backend() -> String {
        if let Some(d) = crate::core::config::device_override() {
            return match d {
                "cpu" => "llvm-cpu",
                "metal" => "metal-spirv",
                "cuda" => "cuda",
                "vulkan" => "vulkan-spirv",
                _ => "llvm-cpu",
            }
            .to_string();
        }
        // Avoid creating a probe session solely for backend detection.
        if let Some(backend) = crate::runtime::iree_session::IreeSession::cached_target_backend() {
            return backend.to_string();
        }
        // The active session backend is refreshed before compilation.
        "llvm-cpu".to_string()
    }

    /// Number of distinct VMFB modules loaded into the shared session.
    pub fn loaded_module_count(&self) -> usize {
        shared_module_catalog()
            .lock()
            .expect("JIT catalogue lock poisoned")
            .modules
            .len()
    }

    /// Obtain shared module metadata without exposing the catalogue lock to
    /// dispatch callers.
    pub(crate) fn module_for_key(&mut self, key: &JitCacheKey) -> Option<CompiledModuleInfo> {
        let info = shared_module_catalog()
            .lock()
            .expect("JIT catalogue lock poisoned")
            .modules
            .get(key)
            .cloned();
        if info.is_some() {
            self.definition_identities
                .insert(key.function_name.clone(), key.definition_hash.clone());
        }
        info
    }

    /// Find an already compiled variant without rebuilding its compilation key.
    pub(crate) fn module_for_args(
        &self,
        func_def: &FunctionDef,
        args: &[Value],
    ) -> Option<CompiledModuleInfo> {
        let definition_identity = self.definition_identities.get(&func_def.name)?;
        let catalog = shared_module_catalog()
            .lock()
            .expect("JIT catalogue lock poisoned");
        catalog
            .variants
            .get(&func_def.name)?
            .iter()
            .find_map(|key| {
                let info = catalog.modules.get(key)?;
                if key.definition_hash == *definition_identity
                    && runtime_args_match(args, &func_def.params, info)
                {
                    Some(info.clone())
                } else {
                    None
                }
            })
    }

    /// Reject and memoize definitions that cannot enter the JIT pipeline.
    ///
    /// `function_definition_identity` covers the transitive direct-call graph
    /// and body hashes without inlining, so it is safe for recursive functions.
    pub(crate) fn preflight_jit_eligibility(
        &mut self,
        func_def: &FunctionDef,
        registry: &HashMap<String, FunctionDef>,
    ) -> bool {
        let identity = function_definition_identity(func_def, registry);
        if self.failed_definitions.contains_key(&identity) {
            return false;
        }
        if let Err(reason) = jit_eligibility(&func_def.name, registry) {
            self.failed_definitions.insert(identity, reason.clone());
            self.jit_fail(&func_def.name, &reason);
            return false;
        }
        true
    }
}

/// Outcome of a JIT attempt for an ordinary function.
#[derive(Debug)]
pub enum JitCompileOutcome {
    Compiled(FunctionSignature),
    /// The function is not eligible for JIT compilation.
    Unsupported(String),
    /// Compilation was attempted but failed.
    Failed(String),
    /// Another attempt is compiling this variant.
    InProgress,
}

/// Why the JIT skipped a value-and-grad call.
#[derive(Debug)]
pub enum JitVagOutcome {
    /// JIT compiled and executed successfully, or had a runtime error.
    Success(Result<Value, SheafError>),
    /// The pattern is unsupported by JIT; use the interpreter fallback.
    Unsupported,
    /// JIT failed for a supported pattern.
    Bug(String),
}

mod compile;
mod preprocess;
mod static_inputs;
mod support;
mod vag;

#[cfg(test)]
mod cache_key_tests {
    use super::{
        cache_key_for, cache_key_for_function, function_definition_identity,
        module_growth_warning, runtime_args_match, CompiledModuleInfo, JitCompileOutcome,
        JitCompiler, JitVagOutcome,
    };
    use crate::core::ast::SheafValue;
    use crate::core::error::{SheafError, SourceLocation};
    use crate::core::expr::{CompiledExpr, FunctionDef};
    use crate::core::inference::{FunctionSignature, ValueLayout};
    use crate::core::trace::value_to_stablehlo_type;
    use crate::interpreter::value::{Dtype, Value};
    use ndarray::ArrayD;
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::rc::Rc;
    use std::sync::Arc;

    fn tensor_f32(shape: Vec<usize>, fill: f32) -> Value {
        Value::Tensor {
            data: Arc::new(ArrayD::from_elem(shape, fill)),
            dtype: Dtype::F32,
        }
    }

    /// Body without shape-bearing scalar inputs.
    fn body_no_shape_scalar() -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: "first".to_string(),
            args: vec![CompiledExpr::Symbol("x".to_string())],
            loc: None,
        }
    }

    /// Body whose reshape dimensions read the scalar at `cfg[0]`.
    fn body_with_shape_scalar_on_cfg() -> CompiledExpr {
        let cfg_shape = CompiledExpr::Vector(vec![
            CompiledExpr::GetTupleElement {
                param: "cfg".into(),
                indices: vec![0],
            },
            CompiledExpr::GetTupleElement {
                param: "cfg".into(),
                indices: vec![0],
            },
        ]);
        CompiledExpr::FunctionCall {
            name: "reshape".to_string(),
            args: vec![CompiledExpr::Symbol("x".to_string()), cfg_shape],
            loc: None,
        }
    }

    fn function(name: &str, body: CompiledExpr) -> FunctionDef {
        function_with_params(name, &["x"], body)
    }

    fn function_with_params(name: &str, params: &[&str], body: CompiledExpr) -> FunctionDef {
        FunctionDef {
            name: name.into(),
            params: params.iter().map(|param| (*param).to_string()).collect(),
            body: SheafValue::Nil(SourceLocation::unknown()),
            body_compiled: Some(body),
            signature: None,
            vmfb_module_name: None,
            known_param_types: Vec::new(),
            compile_error: None,
        }
    }

    fn dict_cfg(n: f32) -> Value {
        let mut map = std::collections::BTreeMap::new();
        map.insert("n".to_string(), Value::Float(n));
        Value::Dict(map)
    }

    fn tensor_with_dtype(shape: Vec<usize>, dtype: Dtype) -> Value {
        Value::Tensor {
            data: Arc::new(ArrayD::zeros(shape)),
            dtype,
        }
    }

    fn compiled_module_for_args(
        args: &[Value],
        captured_scalars: HashMap<(String, Vec<usize>), f64>,
    ) -> CompiledModuleInfo {
        CompiledModuleInfo {
            static_inputs: super::static_inputs::collect(args),
            function_name: "f".to_string(),
            module_name: "f__variant".to_string(),
            sig: FunctionSignature {
                param_types: args
                    .iter()
                    .map(|value| value_to_stablehlo_type(value).unwrap())
                    .collect(),
                return_type: crate::StableHLOType::scalar_f32(),
                return_dict_keys: None,
                return_layout: None,
                arg_type_layouts: Vec::new(),
                captured_scalars,
            },
            argument_layouts: args.iter().map(ValueLayout::from_value).collect(),
        }
    }

    #[test]
    fn fast_variant_lookup_checks_shapes_layouts_and_captured_scalars() {
        let params = vec!["x".to_string(), "cfg".to_string()];
        let args = vec![tensor_f32(vec![2, 3], 0.0), dict_cfg(4.0)];
        let captured = HashMap::from([(("cfg".to_string(), vec![0]), 4.0)]);
        let info = compiled_module_for_args(&args, captured);

        assert!(runtime_args_match(&args, &params, &info));
        assert!(!runtime_args_match(
            &[tensor_f32(vec![3, 2], 0.0), dict_cfg(4.0)],
            &params,
            &info,
        ));
        assert!(!runtime_args_match(
            &[tensor_f32(vec![2, 3], 0.0), dict_cfg(5.0)],
            &params,
            &info,
        ));
    }

    #[test]
    fn fast_variant_lookup_ignores_uncaptured_scalar_values() {
        let params = vec!["x".to_string(), "cfg".to_string()];
        let original = vec![tensor_f32(vec![2], 0.0), dict_cfg(4.0)];
        let info = compiled_module_for_args(&original, HashMap::new());
        let changed = vec![tensor_f32(vec![2], 1.0), dict_cfg(5.0)];

        assert!(runtime_args_match(&changed, &params, &info));
    }

    #[test]
    fn fast_variant_lookup_rejects_dtype_and_container_changes() {
        let params = vec!["params".to_string()];
        let original = Value::Dict(std::collections::BTreeMap::from([(
            "layers".to_string(),
            Value::List(vec![tensor_f32(vec![2], 0.0)]),
        )]));
        let info = compiled_module_for_args(std::slice::from_ref(&original), HashMap::new());
        let changed_dtype = Value::Dict(std::collections::BTreeMap::from([(
            "layers".to_string(),
            Value::List(vec![tensor_with_dtype(vec![2], Dtype::F16)]),
        )]));
        let changed_container = Value::Dict(std::collections::BTreeMap::from([(
            "layers".to_string(),
            Value::Tuple(vec![tensor_f32(vec![2], 0.0)]),
        )]));

        assert!(!runtime_args_match(&[changed_dtype], &params, &info));
        assert!(!runtime_args_match(&[changed_container], &params, &info));
    }

    /// Distinct runtime shapes produce distinct variants.
    #[test]
    fn all_integer_and_boolean_inputs_are_exact_static_variant_facts() {
        let params = vec!["x".to_string(), "option".to_string()];
        let tensor = tensor_f32(vec![2], 0.0);
        let body = body_no_shape_scalar();
        let key = |option| cache_key_for("static-input", &params, &[tensor.clone(), option], &body);
        assert_ne!(key(Value::Int(1)), key(Value::Bool(true)));
        assert_ne!(key(Value::Int(1)), key(Value::Float(1.0)));
        assert_ne!(key(Value::Int(0)), key(Value::Bool(false)));
        assert_ne!(key(Value::Int(1)), key(Value::Int(2)));
        assert_ne!(key(Value::Int(9_007_199_254_740_992)), key(Value::Int(9_007_199_254_740_993)));
        assert_eq!(key(Value::Float(1.0)), key(Value::Float(2.0)));
        let original = vec![tensor.clone(), Value::Int(1)];
        let info = compiled_module_for_args(&original, HashMap::new());
        assert!(runtime_args_match(&original, &params, &info));
        for option in [Value::Int(2), Value::Bool(true), Value::Float(1.0)] {
            assert!(!runtime_args_match(&[tensor.clone(), option], &params, &info));
        }
        let original = vec![tensor.clone(), Value::Float(1.0)];
        let info = compiled_module_for_args(&original, HashMap::new());
        assert!(runtime_args_match(&[tensor.clone(), Value::Float(2.0)], &params, &info));
        assert!(!runtime_args_match(&[tensor, Value::Int(1)], &params, &info));
    }

    #[test]
    fn cache_key_for_distinguishes_consecutive_distinct_shapes() {
        let body = body_no_shape_scalar();
        let params = vec!["x".to_string()];
        let args_a = vec![tensor_f32(vec![1, 5], 0.0)];
        let args_b = vec![tensor_f32(vec![1, 5, 1], 0.0)];
        let key_a = cache_key_for("sigmoid", &params, &args_a, &body);
        let key_b = cache_key_for("sigmoid", &params, &args_b, &body);
        assert_ne!(
            key_a, key_b,
            "two distinct shapes must yield distinct cache keys (CLEVR pattern)"
        );
        assert_eq!(key_a, cache_key_for("sigmoid", &params, &args_a, &body));
        assert_eq!(key_b, cache_key_for("sigmoid", &params, &args_b, &body));
    }

    /// Tensor contents do not affect the cache key.
    #[test]
    fn cache_key_for_collision_free_across_data() {
        let body = body_no_shape_scalar();
        let params = vec!["x".to_string()];
        let args_a = vec![tensor_f32(vec![1, 5], 0.0)];
        let args_b = vec![tensor_f32(vec![1, 5], 1.0)];
        let key_a = cache_key_for("sigmoid", &params, &args_a, &body);
        let key_b = cache_key_for("sigmoid", &params, &args_b, &body);
        assert_eq!(
            key_a, key_b,
            "two tensors with same shape/dtype but different data must hash the same (no tensor bytes hashed)"
        );
    }

    /// Shape-bearing scalar values affect the cache key.
    #[test]
    fn cache_key_for_distinguishes_direct_shape_scalars() {
        let body = CompiledExpr::FunctionCall {
            name: "reshape".into(),
            args: vec![
                CompiledExpr::Symbol("x".into()),
                CompiledExpr::Vector(vec![CompiledExpr::Symbol("n".into())]),
            ],
            loc: None,
        };
        let params = vec!["x".to_string(), "n".to_string()];
        let a = cache_key_for(
            "reshape",
            &params,
            &[tensor_f32(vec![4], 0.0), Value::Int(2)],
            &body,
        );
        let b = cache_key_for(
            "reshape",
            &params,
            &[tensor_f32(vec![4], 0.0), Value::Int(4)],
            &body,
        );
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_for_distinguishes_captured_shape_scalars() {
        let body = body_with_shape_scalar_on_cfg();
        let params = vec!["x".to_string(), "cfg".to_string()];
        // Two cfg dicts with different `:n` -- the shape-bearing scalar of `cfg`.
        let cfg = |n| Value::Dict(BTreeMap::from([("n".to_string(), Value::Int(n))]));
        let args_a = vec![tensor_f32(vec![4, 4], 0.0), cfg(4)];
        let args_b = vec![tensor_f32(vec![4, 4], 0.0), cfg(8)];
        let key_a = cache_key_for("my_reshape", &params, &args_a, &body);
        let key_b = cache_key_for("my_reshape", &params, &args_b, &body);
        assert_ne!(
            key_a, key_b,
            "shape-dependent scalar values must produce distinct cache keys"
        );
    }

    #[test]
    fn cache_key_for_function_includes_definition_and_inlined_callees() {
        let args = vec![tensor_f32(vec![2, 3], 1.0)];
        let callee = function("callee", body_no_shape_scalar());
        let caller = function(
            "caller",
            CompiledExpr::FunctionCall {
                name: "callee".into(),
                args: vec![CompiledExpr::Symbol("x".into())],
                loc: None,
            },
        );
        let mut registry = std::collections::HashMap::from([
            ("callee".to_string(), callee),
            ("caller".to_string(), caller.clone()),
        ]);
        let first = cache_key_for_function(&caller, &args, &registry).unwrap();
        let callee = registry.get_mut("callee").unwrap();
        callee.body = SheafValue::String("(relu x)".to_string(), SourceLocation::unknown());
        callee.body_compiled = Some(CompiledExpr::FunctionCall {
            name: "relu".into(),
            args: vec![CompiledExpr::Symbol("x".into())],
            loc: None,
        });
        let second = cache_key_for_function(&caller, &args, &registry).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn recursive_function_has_no_jit_cache_key() {
        let recursive = function(
            "recursive",
            CompiledExpr::FunctionCall {
                name: "recursive".into(),
                args: vec![CompiledExpr::Symbol("x".into())],
                loc: None,
            },
        );
        let registry = HashMap::from([("recursive".to_string(), recursive.clone())]);

        assert!(
            cache_key_for_function(&recursive, &[tensor_f32(vec![2], 0.0)], &registry).is_none()
        );
    }

    #[test]
    fn mutually_recursive_function_has_no_jit_cache_key() {
        let a = function(
            "a",
            CompiledExpr::FunctionCall {
                name: "b".into(),
                args: vec![CompiledExpr::Symbol("x".into())],
                loc: None,
            },
        );
        let b = function(
            "b",
            CompiledExpr::FunctionCall {
                name: "a".into(),
                args: vec![CompiledExpr::Symbol("x".into())],
                loc: None,
            },
        );
        let registry = HashMap::from([("a".to_string(), a.clone()), ("b".to_string(), b)]);

        assert!(cache_key_for_function(&a, &[tensor_f32(vec![2], 0.0)], &registry).is_none());
    }

    #[test]
    fn structured_vag_errors_do_not_hide_compiler_failures() {
        let mut compiler = JitCompiler {
            iree_compile_path: None,
            target_backend: "llvm-cpu".to_string(),
            failed_definitions: HashMap::new(),
            failed_vag: HashSet::new(),
            failed_keys: HashMap::new(),
            last_compile_failure: None,
            last_vag_fail_reason: Some("codegen failed".to_string()),
            last_vag_error: Some(SheafError::AutodiffMissingGradientOutput {
                symbol: "p_0".to_string(),
            }),
            vag_cache: HashMap::new(),
            definition_identities: HashMap::new(),
        };

        assert!(matches!(
            compiler.classify_vag_skip(),
            JitVagOutcome::Success(Err(SheafError::AutodiffMissingGradientOutput { symbol }))
                if symbol == "p_0"
        ));
        compiler.last_vag_error = Some(SheafError::InvalidShapeDimension {
            message: "reshape: shape dimension is float (64.0), expected an integer".to_string(),
            location: SourceLocation::unknown(),
        });
        assert!(matches!(
            compiler.classify_vag_skip(),
            JitVagOutcome::Success(Err(SheafError::InvalidShapeDimension { .. }))
        ));
        compiler.last_vag_error = None;
        compiler.last_vag_fail_reason = Some("codegen: internal panic".to_string());
        assert!(matches!(
            compiler.classify_vag_skip(),
            JitVagOutcome::Bug(reason) if reason == "codegen: internal panic"
        ));
    }

    #[test]
    fn preserves_autodiff_missing_rule_in_vag_outcome() {
        let location = SourceLocation::new(4, 2, Rc::from("loss.shf"));
        let compiler = JitCompiler {
            iree_compile_path: None,
            target_backend: "llvm-cpu".to_string(),
            failed_definitions: HashMap::new(),
            failed_vag: HashSet::new(),
            failed_keys: HashMap::new(),
            last_compile_failure: None,
            last_vag_fail_reason: Some("codegen failed".to_string()),
            last_vag_error: Some(SheafError::AutodiffMissingRule {
                operation: "unknown".to_string(),
                location: Some(location.clone()),
            }),
            vag_cache: HashMap::new(),
            definition_identities: HashMap::new(),
        };

        assert!(matches!(
            compiler.classify_vag_skip(),
            JitVagOutcome::Success(Err(SheafError::AutodiffMissingRule {
                operation,
                location: Some(actual),
            })) if operation == "unknown" && actual == location
        ));
    }

    #[test]
    fn recursive_definition_is_memoized_before_cache_key_analysis() {
        let recursive = function(
            "recursive",
            CompiledExpr::FunctionCall {
                name: "recursive".into(),
                args: vec![CompiledExpr::Symbol("x".into())],
                loc: None,
            },
        );
        let registry = HashMap::from([("recursive".to_string(), recursive.clone())]);
        let mut compiler = JitCompiler {
            iree_compile_path: None,
            target_backend: "llvm-cpu".to_string(),
            failed_definitions: HashMap::new(),
            failed_vag: HashSet::new(),
            failed_keys: HashMap::new(),
            last_compile_failure: None,
            last_vag_fail_reason: None,
            last_vag_error: None,
            vag_cache: HashMap::new(),
            definition_identities: HashMap::new(),
        };
        let identity = function_definition_identity(&recursive, &registry);

        assert!(!compiler.preflight_jit_eligibility(&recursive, &registry));
        assert!(!compiler.preflight_jit_eligibility(&recursive, &registry));
        assert_eq!(compiler.failed_definitions.len(), 1);
        assert_eq!(
            compiler.failed_definitions.get(&identity).map(String::as_str),
            Some("recursive call through 'recursive'")
        );
    }

    #[test]
    fn cache_key_tracks_shape_scalars_used_by_inlined_callees() {
        let callee = function_with_params(
            "reshape-with-config",
            &["x", "cfg"],
            body_with_shape_scalar_on_cfg(),
        );
        let caller = function_with_params(
            "caller",
            &["x", "cfg"],
            CompiledExpr::FunctionCall {
                name: "reshape-with-config".into(),
                args: vec![
                    CompiledExpr::Symbol("x".into()),
                    CompiledExpr::Symbol("cfg".into()),
                ],
                loc: None,
            },
        );
        let registry = HashMap::from([
            (callee.name.clone(), callee),
            (caller.name.clone(), caller.clone()),
        ]);
        let tensor = tensor_f32(vec![4, 4], 0.0);
        let cfg = |n| Value::Dict(BTreeMap::from([("n".to_string(), Value::Int(n))]));
        let first = cache_key_for_function(&caller, &[tensor.clone(), cfg(4)], &registry).unwrap();
        let second = cache_key_for_function(&caller, &[tensor, cfg(8)], &registry).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn jit_outcome_preserves_cached_skip_and_failure_reasons() {
        let session = crate::runtime::iree_session::shared_session().unwrap();
        let mut compiler = JitCompiler::new();
        compiler.iree_compile_path = Some("unused".to_string());
        let recursive = function(
            "recursive-outcome",
            CompiledExpr::FunctionCall {
                name: "recursive-outcome".into(),
                args: vec![CompiledExpr::Symbol("x".into())],
                loc: None,
            },
        );
        let registry = HashMap::from([(recursive.name.clone(), recursive.clone())]);
        let args = [tensor_f32(vec![2], 0.0)];
        for _ in 0..2 {
            assert!(matches!(
                compiler.try_jit_compile(&recursive, &args, &registry, &session),
                JitCompileOutcome::Unsupported(reason)
                    if reason == "recursive call through 'recursive-outcome'"
            ));
        }

        let ordinary = function("failed-outcome", body_no_shape_scalar());
        let registry = HashMap::from([(ordinary.name.clone(), ordinary.clone())]);
        let key = cache_key_for_function(&ordinary, &args, &registry).unwrap();
        compiler.failed_keys.insert(key, "codegen: test failure".to_string());
        assert!(matches!(
            compiler.try_jit_compile(&ordinary, &args, &registry, &session),
            JitCompileOutcome::Failed(reason) if reason == "codegen: test failure"
        ));
    }

    fn compile_jit_case(source: &str, name: &str, args: &[Value]) -> JitCompileOutcome {
        let mut compiler_context = crate::CompilerContext::new();
        for form in crate::parse(source, "<jit-regression>").unwrap() {
            compiler_context.compile(&form).unwrap();
        }
        let func = compiler_context.registry.get(name).unwrap();
        assert!(func.body_compiled.is_some(), "{name} must have a compiled body");
        let session = crate::runtime::iree_session::shared_session().unwrap();
        JitCompiler::new().try_jit_compile(func, args, &compiler_context.registry, &session)
    }

    #[test]
    fn indexed_parameter_shape_compiles() {
        for index in [0, -2] {
            let source = format!(
                "(defn regression-indexed-shape [params] \
                   (let [w (get-in params [:head :W]) n (get (shape w) {index})] \
                     (zeros [n 3])))"
            );
            let params = Value::Dict(std::collections::BTreeMap::from([(
                "head".to_string(),
                Value::Dict(std::collections::BTreeMap::from([(
                    "W".to_string(), tensor_f32(vec![2, 1], 0.0),
                )])),
            )]));
            let outcome = compile_jit_case(&source, "regression-indexed-shape", &[params]);
            assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
        }
    }

    #[test]
    fn quoted_tensor_creation_shapes_compile() {
        for operation in ["zeros", "ones"] {
            let source = format!(
                "(defn regression-quoted-shape [x] (+ x ({operation} '[2 3])))"
            );
            let outcome = compile_jit_case(
                &source, "regression-quoted-shape", &[tensor_f32(vec![2, 3], 0.0)],
            );
            assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
        }
    }

    #[test]
    fn tensor_creation_cache_keys_include_configuration_dimensions() {
        let mut context = crate::CompilerContext::new();
        for form in crate::parse(
            "(defn regression-config-shape [key config] \
               (let [d (get config :d)] (random-normal key [2 d])))",
            "<jit-regression>",
        ).unwrap() {
            context.compile(&form).unwrap();
        }
        let func = context.registry.get("regression-config-shape").unwrap();
        let args = |d| vec![
            tensor_f32(vec![2], 0.0),
            Value::Dict(std::collections::BTreeMap::from([("d".to_string(), Value::Int(d))])),
        ];
        let first = cache_key_for_function(func, &args(3), &context.registry).unwrap();
        let second = cache_key_for_function(func, &args(5), &context.registry).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn hydra_growth_compiles_across_layer_counts_and_widths() {
        let source = "(defn regression-grow-hydra [params key config] \
            (let [D (get config :d_model) \
                  head-w (get-in params [:head :W]) \
                  in_dim (get (shape head-w) 0) \
                  keys (random-split key) k1 (get keys 0) k2 (get keys 1) \
                  new-layer {:W (random-normal k1 [in_dim D]) :b (zeros [D])} \
                  new-head {:W (random-normal k2 [D 1]) :b (zeros '[1])}] \
              {:head new-head :layers (append (get params :layers) new-layer)}))";
        let mut interpreter = crate::interpreter::eval::Interpreter::new();
        interpreter.eval(source).unwrap();
        interpreter.eval(
            "(def growth-key (random-key 42)) \
             (def growth-params-0 {:head {:W (zeros '[2 1]) :b (zeros '[1])} :layers '[]})"
        ).unwrap();
        let key = interpreter.eval("growth-key").unwrap();
        let mut input_width = 2;
        for (layer_count, width) in [3, 5, 5].into_iter().enumerate() {
            let previous = format!("growth-params-{layer_count}");
            let current = format!("growth-params-{}", layer_count + 1);
            let params = interpreter.eval(&previous).unwrap();
            let config = Value::Dict(std::collections::BTreeMap::from([
                ("d_model".to_string(), Value::Int(width)),
            ]));
            let outcome = compile_jit_case(
                source, "regression-grow-hydra", &[params, key.clone(), config],
            );
            assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
            interpreter.eval(&format!(
                "(def {current} (regression-grow-hydra {previous} growth-key {{:d_model {width}}}))"
            )).unwrap();
            let count = interpreter.eval(&format!("(len (get {current} :layers))")).unwrap();
            assert!(matches!(count, Value::Int(n) if n == layer_count as i64 + 1));
            for (path, expected) in [
                (format!("[:layers {layer_count} :W]"), vec![input_width, width as usize]),
                (format!("[:layers {layer_count} :b]"), vec![width as usize]),
                ("[:head :W]".to_string(), vec![width as usize, 1]),
                ("[:head :b]".to_string(), vec![1]),
            ] {
                let value = interpreter.eval(&format!("(get-in {current} {path})"))
                    .unwrap().ensure_host().unwrap();
                let Value::Tensor { data, .. } = value else {
                    panic!("expected tensor at {path}, got {value:?}");
                };
                assert_eq!(data.shape(), expected);
                assert!(data.iter().all(|value| value.is_finite()));
            }
            input_width = width as usize;
        }
    }

    #[test]
    fn append_to_list_compiles_and_returns_a_list() {
        let outcome = compile_jit_case(
            "(defn regression-append [x] (append [] x))",
            "regression-append",
            &[tensor_f32(vec![2], 1.0)],
        );
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
        let result = crate::interpreter::eval::eval_source(
            "(defn regression-append [x] (append [] x)) \
             (regression-append [1 2])",
        ).unwrap();
        let Value::List(items) = result else {
            panic!("append must return a list, got {result:?}");
        };
        assert_eq!(items.len(), 1);
        let value = items[0].ensure_host().unwrap();
        assert!(matches!(&value, Value::Tensor { data, .. }
            if data.iter().copied().collect::<Vec<_>>() == vec![1.0, 2.0]));
    }

    #[test]
    fn append_to_list_parameter_preserves_the_result_layout() {
        let items = Value::List(vec![tensor_f32(vec![2], 1.0)]);
        let outcome = compile_jit_case(
            "(defn regression-append-param [items x] (append items x))",
            "regression-append-param",
            &[items, tensor_f32(vec![2], 2.0)],
        );
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
        let result = crate::interpreter::eval::eval_source(
            "(defn regression-append-param [items x] (append items x)) \
             (regression-append-param [(tensor [1 1])] (tensor [2 2]))",
        ).unwrap();
        let Value::List(items) = result else {
            panic!("append must return a list, got {result:?}");
        };
        assert_eq!(items.len(), 2);
        for (item, expected) in items.iter().zip([1.0, 2.0]) {
            let value = item.ensure_host().unwrap();
            assert!(matches!(value, Value::Tensor { data, .. }
                if data.iter().all(|&n| n == expected)));
        }
    }

    #[test]
    fn reduce_append_preserves_nested_lists() {
        let source = "(defn regression-append-cache [x] \
            (reduce (fn [cache i] (append cache [(+ x i) (* x i)])) \
              [] (range 2)))";
        let outcome = compile_jit_case(
            source,
            "regression-append-cache",
            &[tensor_f32(vec![2], 1.0)],
        );
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
        let result = crate::interpreter::eval::eval_source(
            &format!("{source} (regression-append-cache [1 1])"),
        ).unwrap();
        let Value::List(items) = result else {
            panic!("reduce must return a list, got {result:?}");
        };
        assert_eq!(items.len(), 2);
        for (index, item) in items.iter().enumerate() {
            let Value::List(pair) = item else {
                panic!("cache entry {index} must be a list, got {item:?}");
            };
            assert_eq!(pair.len(), 2);
        }
    }

    #[test]
    fn append_rejects_a_tuple_parameter() {
        let outcome = compile_jit_case(
            "(defn regression-append-tuple [items x] (append items x))",
            "regression-append-tuple",
            &[
                Value::Tuple(vec![tensor_f32(vec![2], 0.0)]),
                tensor_f32(vec![2], 1.0),
            ],
        );
        assert!(matches!(
            outcome,
            JitCompileOutcome::Failed(reason)
                if reason.contains("append: first argument must be a list")
        ));
    }

    #[test]
    fn reduce_over_dict_blocks_compiles_with_element_keys() {
        let blocks = Value::List(vec![
            Value::Dict(std::collections::BTreeMap::from([(
                "w".to_string(), tensor_f32(vec![2], 1.0),
            )])),
            Value::Dict(std::collections::BTreeMap::from([(
                "w".to_string(), tensor_f32(vec![2], 2.0),
            )])),
        ]);
        let outcome = compile_jit_case(
            "(defn regression-reduce-layout [blocks x] \
               (reduce (fn [acc block] (+ acc (get block :w))) x blocks))",
            "regression-reduce-layout",
            &[blocks, tensor_f32(vec![2], 0.0)],
        );
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
    }

    #[test]
    fn range_of_collection_length_compiles() {
        let outcome = compile_jit_case(
            "(defn regression-range-len [xs] (range (len xs)))",
            "regression-range-len",
            &[tensor_f32(vec![3], 0.0)],
        );
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");

        let blocks = Value::List(vec![Value::Dict(std::collections::BTreeMap::from([(
            "w".to_string(), tensor_f32(vec![2], 1.0),
        )]))]);
        let outcome = compile_jit_case(
            "(defn regression-range-blocks [blocks x] (range (len blocks)))",
            "regression-range-blocks",
            &[blocks, tensor_f32(vec![2], 0.0)],
        );
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
    }

    #[test]
    fn reduce_over_range_keeps_static_tuple_indices() {
        let blocks = Value::List(vec![
            Value::Dict(std::collections::BTreeMap::from([(
                "w".to_string(), tensor_f32(vec![2], 1.0),
            )])),
            Value::Dict(std::collections::BTreeMap::from([(
                "w".to_string(), tensor_f32(vec![2], 2.0),
            )])),
        ]);
        let outcome = compile_jit_case(
            "(defn regression-reduce-index [blocks x] \
               (reduce (fn [acc i] (+ acc (get (get blocks i) :w))) \
                 x (range (len blocks))))",
            "regression-reduce-index",
            &[blocks, tensor_f32(vec![2], 0.0)],
        );
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
    }

    #[test]
    fn integer_tuple_indices_are_static_but_float_values_remain_dynamic() {
        let blocks = Value::List(vec![Value::Dict(std::collections::BTreeMap::from([(
            "w".to_string(), tensor_f32(vec![2], 1.0),
        )]))]);
        let outcome = compile_jit_case(
            "(defn regression-dynamic-index [blocks i x] (get blocks i))",
            "regression-dynamic-index",
            &[blocks.clone(), Value::Int(0), tensor_f32(vec![2], 0.0)],
        );
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
        let outcome = compile_jit_case(
            "(defn regression-float-index [blocks i x] (get blocks i))",
            "regression-float-index",
            &[blocks, Value::Float(0.0), tensor_f32(vec![2], 0.0)],
        );
        assert!(matches!(
            outcome,
            JitCompileOutcome::Failed(reason)
                if reason.contains("get on dict/tuple requires type info")
        ));
    }

    #[test]
    fn variant_failure_does_not_blacklist_a_definition() {
        let mut compiler = JitCompiler::new();
        compiler.jit_fail("polymorphic", "shape-specific codegen failure");
        assert_eq!(
            compiler.last_compile_failure.as_deref(),
            Some("shape-specific codegen failure")
        );
        assert!(compiler.failed_definitions.is_empty());
        assert!(compiler.failed_keys.is_empty());
    }

    #[test]
    fn cache_key_ignores_non_shape_bearing_scalar_values() {
        let body = body_no_shape_scalar();
        let params = vec!["x".to_string(), "scale".to_string()];
        let tensor = tensor_f32(vec![2, 3], 1.0);
        let a = cache_key_for(
            "scale",
            &params,
            &[tensor.clone(), Value::Float(2.0)],
            &body,
        );
        let b = cache_key_for("scale", &params, &[tensor, Value::Float(9.0)], &body);
        assert_eq!(a, b);
    }

    #[test]
    fn cache_key_distinguishes_structure_and_dtype() {
        let body = body_no_shape_scalar();
        let params = vec!["x".to_string()];
        let f32_key = cache_key_for(
            "typed",
            &params,
            &[tensor_with_dtype(vec![2], Dtype::F32)],
            &body,
        );
        let bf16_key = cache_key_for(
            "typed",
            &params,
            &[tensor_with_dtype(vec![2], Dtype::BF16)],
            &body,
        );
        let tuple_key = cache_key_for(
            "typed",
            &params,
            &[Value::Tuple(vec![tensor_with_dtype(vec![2], Dtype::F32)])],
            &body,
        );
        assert_ne!(f32_key, bf16_key);
        assert_ne!(f32_key, tuple_key);
    }

    #[test]
    fn module_warning_fires_once_after_crossing_configured_threshold() {
        let variants = std::collections::BTreeMap::from([("alpha", 2), ("beta", 1)]);
        assert!(module_growth_warning(2, 2, false, &variants).is_none());
        let warning = module_growth_warning(3, 2, false, &variants).unwrap();
        assert!(warning.contains("3 distinct VMFB modules"));
        assert!(warning.contains("alpha=2"));
        assert!(warning.contains("beta=1"));
        assert!(module_growth_warning(4, 2, true, &variants).is_none());
        assert!(module_growth_warning(3, 0, false, &variants).is_none());
    }
}
