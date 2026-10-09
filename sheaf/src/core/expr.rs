// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf intermediate representation: compiled expressions, function definitions,
//! and parameter layouts.
//! Generates the IR that the lowering stage translates to StableHLO.

use crate::core::ast::SheafValue;
use crate::core::error::{SheafError, SheafResult};
use crate::core::macro_engine::MacroEngine;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// A leaf field in a parameter layout: a named tensor with its shape
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParamField {
    /// Path of keys from the root: e.g. ["attn", "Wq"]
    pub path: Vec<String>,
    /// Shape of the tensor: e.g. [512, 512]
    pub shape: Vec<i64>,
    /// Tuple index at this level (set during layout resolution)
    pub tuple_index: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParamLayout {
    pub name: String,
    /// Flattened list of fields in declaration order
    pub fields: Vec<ParamField>,
}

impl ParamLayout {
    /// Look up a field by its path (e.g. ["attn", "Wq"]) and return its tuple indices
    pub fn resolve_path(&self, path: &[&str]) -> Option<&ParamField> {
        self.fields.iter().find(|f| {
            f.path.len() == path.len() && f.path.iter().zip(path.iter()).all(|(a, b)| a == b)
        })
    }

    /// Look up a top-level key and return all fields under it
    pub fn fields_under(&self, key: &str) -> Vec<&ParamField> {
        self.fields
            .iter()
            .filter(|f| f.path.first().map(|k| k == key).unwrap_or(false))
            .collect()
    }
}

/// Compilation context - tracks environment, registry, etc.
pub struct CompilerContext {
    /// Global environment (built-in functions, runtime ops)
    pub env: HashMap<String, SheafValue>,

    /// Function registry (user-defined functions)
    pub registry: HashMap<String, FunctionDef>,

    /// Local variables (for let bindings, function params)
    pub local_vars: HashMap<String, SheafValue>,

    /// Parameter type registry (structured dict layouts)
    pub param_types: HashMap<String, ParamLayout>,

    /// Param scope: maps symbol name -> (param_name, tuple_indices)
    /// Set by with-params to resolve symbols like W -> get_tuple_element(p, [0, 1])
    pub param_scope: HashMap<String, (String, Vec<usize>)>,

    /// Search paths for (use module): stdlib directories + current file's directory
    pub load_path: Vec<PathBuf>,

    /// Absolute paths of already-loaded modules (prevents duplicate loading)
    pub loaded_modules: HashSet<PathBuf>,

    /// Bare names of prelude modules (nn, optim, etc.) are immune to shadowing by local files
    pub prelude_modules: HashSet<String>,

    /// Directory of the file currently being compiled (for relative use paths)
    pub current_dir: Option<PathBuf>,

    /// When true, skip VMFB loading (used during sheaf build --trace-with)
    pub disable_vmfb: bool,

    /// Directories already checked for VMFB loading (prevents duplicate warnings)
    pub checked_vmfb_dirs: HashSet<PathBuf>,

    /// Macro expansion engine
    pub macro_engine: MacroEngine,
}

/// Function definition
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: SheafValue,
    pub body_compiled: Option<CompiledExpr>,
    pub signature: Option<crate::core::inference::FunctionSignature>,
    /// IREE module name used for qualified dispatch.
    pub vmfb_module_name: Option<String>,
    /// Known parameter types from annotations (shape annotations + traced layouts).
    /// Used by the tracing compiler to create dummy inputs.
    pub known_param_types: Vec<(String, crate::lowering::stablehlo::StableHLOType)>,
    /// If body compilation failed, store the error for deferred reporting.
    pub compile_error: Option<SheafError>,
}

impl FunctionDef {
    /// Compiler identity and original AST body for manifest validation.
    /// Source locations do not affect compiled output and are excluded.
    pub fn body_hash(&self) -> String {
        canonical_ast_hash(&self.body)
    }
}

fn canonical_ast_hash(value: &SheafValue) -> String {
    use std::fmt::Write;

    fn write_text(output: &mut String, tag: char, text: &str) {
        let _ = write!(output, "{tag}{}:", text.len());
        output.push_str(text);
    }

    fn write_value(output: &mut String, value: &SheafValue) {
        use std::fmt::Write;

        match value {
            SheafValue::Symbol(value, _) => write_text(output, 'S', value),
            SheafValue::Keyword(value, _) => write_text(output, 'K', value),
            SheafValue::Integer(value, _) => {
                let _ = write!(output, "I{value};");
            }
            SheafValue::Float(value, _) => {
                let _ = write!(output, "F{:016x};", value.to_bits());
            }
            SheafValue::String(value, _) => write_text(output, 'T', value),
            SheafValue::Boolean(value, _) => {
                output.push_str(if *value { "B1;" } else { "B0;" });
            }
            SheafValue::Nil(_) => output.push_str("N;"),
            SheafValue::List(values, _) => {
                let _ = write!(output, "L{}[", values.len());
                for value in values {
                    write_value(output, value);
                }
                output.push(']');
            }
            SheafValue::Vector(values, _) => {
                let _ = write!(output, "V{}[", values.len());
                for value in values {
                    write_value(output, value);
                }
                output.push(']');
            }
            SheafValue::Dict(entries, _) => {
                let _ = write!(output, "D{}[", entries.len());
                for (key, value) in entries {
                    write_value(output, key);
                    write_value(output, value);
                }
                output.push(']');
            }
            SheafValue::Quote(value, _) => {
                output.push_str("Q[");
                write_value(output, value);
                output.push(']');
            }
            SheafValue::Quasiquote(value, _) => {
                output.push_str("QQ[");
                write_value(output, value);
                output.push(']');
            }
            SheafValue::Unquote(value, _) => {
                output.push_str("U[");
                write_value(output, value);
                output.push(']');
            }
            SheafValue::UnquoteSplicing(value, _) => {
                output.push_str("US[");
                write_value(output, value);
                output.push(']');
            }
        }
    }

    let mut encoded = format!("{};", super::prng::VERSION);
    write_value(&mut encoded, value);

    fn fnv1a(text: &str, seed: u64) -> u64 {
        text.bytes().fold(seed, |mut hash, byte| {
            hash ^= u64::from(byte);
            hash.wrapping_mul(0x100000001b3)
        })
    }

    let first = fnv1a(&encoded, 0xcbf29ce484222325);
    let second = fnv1a(&encoded, 0x84222325cbf29ce4);
    format!("{first:016x}{second:016x}")
}

/// Lower a runtime quasiquote into plain AST.
/// `[-1 ~expr 1] becomes Vector([-1, expr, 1]),
/// ~expr becomes expr, literals pass through unchanged.
fn lower_quasiquote(node: &SheafValue) -> SheafValue {
    match node {
        SheafValue::Unquote(inner, _) => (**inner).clone(),
        SheafValue::Quasiquote(inner, _) => lower_quasiquote(inner),
        SheafValue::List(elems, loc) => {
            let mut result = Vec::new();
            for e in elems {
                if let SheafValue::UnquoteSplicing(inner, _) = e {
                    // ~@expr -- the inner expression should produce a list at runtime;
                    // for now just lower it as a single element (covers CLEVR usage)
                    result.push(lower_quasiquote(inner));
                } else {
                    result.push(lower_quasiquote(e));
                }
            }
            SheafValue::List(result, loc.clone())
        }
        SheafValue::Vector(elems, loc) => {
            let mut result = Vec::new();
            for e in elems {
                if let SheafValue::UnquoteSplicing(inner, _) = e {
                    result.push(lower_quasiquote(inner));
                } else {
                    result.push(lower_quasiquote(e));
                }
            }
            SheafValue::Vector(result, loc.clone())
        }
        _ => node.clone(),
    }
}

fn is_builtin_name(name: &str) -> bool {
    if crate::core::signature::OpId::resolve(name).is_some_and(|op| op.is_public()) {
        return true;
    }
    // These stdlib names are also accepted while compiling the prelude.
    matches!(name,
        "relu" | "leaky-relu" | "sigmoid" | "gelu" | "selu" | "celu" | "silu"
        | "softmax" | "log-softmax" | "mse-loss" | "mae-loss"
    )
}

const STDLIB: &[(&str, &str)] = &[
    ("macros.shf", include_str!("../../lib/macros.shf")),
    ("misc.shf", include_str!("../../lib/misc.shf")),
    ("nn.shf", include_str!("../../lib/nn.shf")),
    ("optim.shf", include_str!("../../lib/optim.shf")),
];

impl CompilerContext {
    pub fn new() -> Self {
        let mut context = Self::new_without_prelude();
        #[cfg(any(sheaf_frontend, cargo_source_prelude))]
        context.load_prelude_from_source()
            .expect("embedded Sheaf prelude must compile");
        #[cfg(not(any(sheaf_frontend, cargo_source_prelude)))]
        context.load_prelude_image()
            .expect("embedded Sheaf prelude image must be valid");
        context
    }

    /// Construct a compiler context without installing the embedded prelude.
    ///
    /// This is the bootstrap entry point used by the prelude compiler.
    pub fn new_without_prelude() -> Self {
        Self {
            env: Self::init_env(),
            registry: HashMap::new(),
            local_vars: HashMap::new(),
            param_types: HashMap::new(),
            param_scope: HashMap::new(),
            load_path: Self::default_load_path(),
            loaded_modules: HashSet::new(),
            prelude_modules: HashSet::new(),
            current_dir: None,
            disable_vmfb: false,
            checked_vmfb_dirs: HashSet::new(),
            macro_engine: MacroEngine::new(),
        }
    }

    /// Parse and compile one prelude module into this context.
    #[cfg(any(sheaf_frontend, cargo_source_prelude, test))]
    pub fn compile_prelude_module(&mut self, name: &str, source: &str) -> SheafResult<()> {
        crate::core::error_format::register_source(name, source);
        let exprs = crate::core::parse(source, name)?;
        for expr in &exprs {
            self.compile(expr)?;
        }
        self.prelude_modules.insert(
            name.strip_suffix(".shf").unwrap_or(name).to_string()
        );
        Ok(())
    }

    /// Compile the embedded standard library from source for bootstrap and tests.
    #[cfg(any(sheaf_frontend, cargo_source_prelude, test))]
    pub fn load_prelude_from_source(&mut self) -> SheafResult<()> {
        for (name, source) in STDLIB {
            self.compile_prelude_module(name, source)?;
        }
        Ok(())
    }

    #[cfg(not(any(sheaf_frontend, cargo_source_prelude)))]
    fn load_prelude_image(&mut self) -> Result<(), crate::core::prelude::PreludeError> {
        for (name, source) in STDLIB {
            crate::core::error_format::register_source(name, source);
        }
        let source_hash = crate::core::prelude::content_identity(
            STDLIB.iter().map(|(name, source)| (*name, source.as_bytes())),
        );
        crate::core::prelude::install_embedded(
            self,
            env!("SHEAF_COMPILER_VERSION"),
            &source_hash,
        )
    }



    /// Build the default load path: stdlib dir (relative to binary) + cwd.
    fn default_load_path() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        if let Ok(lib) = std::env::var("SHEAF_LIB") {
            paths.push(PathBuf::from(lib));
        }

        if let Ok(exe) = std::env::current_exe()
            && let Some(bin_dir) = exe.parent()
        {
            let candidate = bin_dir.join("../lib");
            if candidate.exists() {
                paths.push(candidate.canonicalize().unwrap_or(candidate));
            }
            // Also try <binary>/../../sheaf/lib/ for dev layout (rust/target/debug/sheaf)
            let dev_candidate = bin_dir.join("../../../sheaf/lib");
            if dev_candidate.exists()
            {
                paths.push(dev_candidate.canonicalize().unwrap_or(dev_candidate));
            }
        }

        if let Ok(cwd) = std::env::current_dir() {
            paths.push(cwd);
        }

        paths
    }

    /// Set the directory of the file being compiled (for relative `use` paths).
    pub fn set_current_dir(&mut self, dir: PathBuf) {
        self.current_dir = Some(dir);
    }

    /// Initialize global environment with built-in operations
    fn init_env() -> HashMap<String, SheafValue> {
        HashMap::new()
    }

    /// Compile a Sheaf expression
    pub fn compile(&mut self, exp: &SheafValue) -> SheafResult<CompiledExpr> {
        match exp {
            SheafValue::Integer(n, _) => Ok(CompiledExpr::Integer(*n)),
            SheafValue::Float(x, _) => Ok(CompiledExpr::Float(*x)),
            SheafValue::Boolean(b, _) => Ok(CompiledExpr::Boolean(*b)),
            SheafValue::Nil(_) => Ok(CompiledExpr::Nil),
            SheafValue::String(s, _) => Ok(CompiledExpr::String(s.clone())),

            SheafValue::Symbol(name, loc) => self.resolve_symbol(name, loc),

            SheafValue::Keyword(k, _) => Ok(CompiledExpr::Keyword(k.clone())),

            SheafValue::Quasiquote(inner, _) => {
                let lowered = lower_quasiquote(inner);
                self.compile(&lowered)
            }
            SheafValue::Unquote(_, loc) => Err(SheafError::Compile {
                message: "Unquote (~) outside of quasiquote".to_string(),
                location: loc.clone(),
            }),
            SheafValue::UnquoteSplicing(_, loc) => Err(SheafError::Compile {
                message: "Unquote-splicing (~@) outside of quasiquote".to_string(),
                location: loc.clone(),
            }),

            SheafValue::List(elements, loc) => {
                if elements.is_empty() {
                    return Err(SheafError::Compile {
                        message: "Cannot compile empty list".to_string(),
                        location: loc.clone(),
                    });
                }

                // Check for macro expansion before special forms
                if let Some(op_name) = elements[0].as_symbol()
                    && self.macro_engine.macros.contains_key(op_name)
                {
                    let expanded = self.macro_engine.expand(exp, &self.env, &self.registry)?;
                    return self.compile(&expanded).map_err(|e| e.with_location(loc));
                }

                // Check for special forms
                if let Some(op) = elements[0].as_symbol() {
                    // Try to compile as special form, fall back to function call
                    match self.try_compile_special_form(op, &elements[1..], loc) {
                        Some(result) => result,
                        None => self.compile_function_call(elements, loc),
                    }
                } else {
                    // Head is not a symbol -- compile it and treat as lambda call.
                    // e.g. ((fn [x] (+ x 1)) 10)
                    let callee = self.compile(&elements[0])?;
                    let call_args: SheafResult<Vec<CompiledExpr>> =
                        elements[1..].iter().map(|a| self.compile(a)).collect();
                    let call_args = call_args?;

                    Ok(CompiledExpr::LambdaCall {
                        callee: Box::new(callee),
                        args: call_args,
                    })
                }
            }

            SheafValue::Vector(elements, _) => {
                // Compile each element
                let compiled: SheafResult<Vec<CompiledExpr>> =
                    elements.iter().map(|e| self.compile(e)).collect();
                Ok(CompiledExpr::Vector(compiled?))
            }

            SheafValue::Dict(pairs, _) => {
                let compiled: SheafResult<Vec<(CompiledExpr, CompiledExpr)>> = pairs
                    .iter()
                    .map(|(k, v)| Ok((self.compile(k)?, self.compile(v)?)))
                    .collect();
                Ok(CompiledExpr::Dict(compiled?))
            }

            SheafValue::Quote(inner, _) => {
                // Quote prevents evaluation
                Ok(CompiledExpr::Quoted(inner.clone()))
            }
        }
    }

    /// Resolve a symbol to its value
    fn resolve_symbol(
        &mut self,
        name: &str,
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // Check param_scope first (with-params bindings)
        // e.g. W -> GetTupleElement { param: "p", indices: [0] }
        if let Some((param_name, indices)) = self.param_scope.get(name).cloned() {
            return Ok(CompiledExpr::GetTupleElement {
                param: param_name,
                indices,
            });
        }

        // Check local variables
        if let Some(val) = self.local_vars.get(name).cloned() {
            // Inline constants so JIT-traced functions can use `def` values.
            match &val {
                SheafValue::Integer(_, _)
                | SheafValue::Float(_, _)
                | SheafValue::String(_, _)
                | SheafValue::Vector(_, _) => return self.compile(&val),
                SheafValue::List(elems, _)
                    if elems.first().and_then(|e| e.as_symbol()) == Some("fn") =>
                {
                    return self.compile(&val);
                }
                _ => {}
            }
            return Ok(CompiledExpr::Symbol(name.to_string()));
        }

        // Check environment
        if let Some(value) = self.env.get(name).cloned() {
            return self.compile(&value);
        }

        // Check registry (user-defined functions)
        if self.registry.contains_key(name) {
            return Ok(CompiledExpr::FunctionRef(name.to_string()));
        }

        // Allow interpreter builtin names as runtime-resolved symbols
        if is_builtin_name(name) {
            return Ok(CompiledExpr::Symbol(name.to_string()));
        }

        // Ellipsis marker for advanced indexing: (get tensor ... idx)
        if name == "..." {
            return Ok(CompiledExpr::Symbol("...".to_string()));
        }

        Err(SheafError::Compile {
            message: format!("Undefined symbol: {}", name),
            location: loc.clone(),
        })
    }

    // Special form compilation methods have been moved to src/forms/
    // See: binding.rs (defn, let, fn), control.rs (if, do), utils.rs (quote)

    /// Try to compile as a special form, return None if not a special form
    fn try_compile_special_form(
        &mut self,
        op: &str,
        args: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> Option<SheafResult<CompiledExpr>> {
        // Static dispatch to special forms
        use crate::forms::ml::{ValueAndGradForm, WithParamsForm};
        use crate::forms::*;

        let result = match op {
            "def" => DefForm.compile(self, args, loc),
            "defn" => DefnForm.compile(self, args, loc),
            "defmacro" => binding::DefmacroForm.compile(self, args, loc),
            "let" => LetForm.compile(self, args, loc),
            "fn" => FnForm.compile(self, args, loc),
            "if" => IfForm.compile(self, args, loc),
            "do" => DoForm.compile(self, args, loc),
            "quote" => QuoteForm.compile(self, args, loc),
            "case" => CaseForm.compile(self, args, loc),
            "while" => WhileForm.compile(self, args, loc),
            "repeat" => RepeatForm.compile(self, args, loc),
            "guard" => GuardForm.compile(self, args, loc),
            "->" => ThreadFirstForm.compile(self, args, loc),
            "as->" => ThreadAsForm.compile(self, args, loc),
            "get" => GetForm.compile(self, args, loc),
            "get-in" => GetInForm.compile(self, args, loc),
            "dict" => DictForm.compile(self, args, loc),
            "assoc" => AssocForm.compile(self, args, loc),
            "last" => LastForm.compile(self, args, loc),
            "use" => UseForm.compile(self, args, loc),
            "with-params" => WithParamsForm.compile(self, args, loc),
            "value-and-grad" => ValueAndGradForm.compile(self, args, loc),
            _ => return None, // Not a special form
        };

        Some(result)
    }

    /// Compile function call (op arg1 arg2 ...)
    fn compile_function_call(
        &mut self,
        elements: &[SheafValue],
        loc: &crate::core::error::SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        let op = &elements[0];
        let args = &elements[1..];

        // Get function name
        let func_name = op.as_symbol().ok_or_else(|| SheafError::Compile {
            message: format!("Function name must be a symbol, got: {}", op),
            location: loc.clone(),
        })?;

        // Compile arguments
        let compiled_args: SheafResult<Vec<CompiledExpr>> =
            args.iter().map(|arg| self.compile(arg)).collect();
        let compiled_args = compiled_args?;

        // If the symbol is a locally-bound lambda, emit a LambdaCall.
        // e.g. (let [f (fn [x] x)] (f 5))
        if self.local_vars.contains_key(func_name) {
            let callee = self.resolve_symbol(func_name, loc)?;
            if matches!(callee, CompiledExpr::Lambda { .. }) {
                return Ok(CompiledExpr::LambdaCall {
                    callee: Box::new(callee),
                    args: compiled_args,
                });
            }
        }

        Ok(CompiledExpr::FunctionCall {
            name: func_name.to_string(),
            args: compiled_args,
            loc: Some(loc.clone()),
        })
    }
}

/// Binding pattern for let bindings (simple or destructuring).
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BindingPattern {
    Simple(String),
    Destructure(Vec<BindingPattern>),
}

impl BindingPattern {
    pub fn as_simple(&self) -> Option<&str> {
        match self {
            BindingPattern::Simple(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl std::fmt::Display for BindingPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindingPattern::Simple(s) => write!(f, "{}", s),
            BindingPattern::Destructure(names) => {
                write!(f, "[")?;
                for (i, name) in names.iter().enumerate() {
                    if i > 0 { write!(f, " ")?; }
                    write!(f, "{}", name)?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Compiled expression - intermediate representation
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum CompiledExpr {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Nil,
    String(String),
    Keyword(String),
    Vector(Vec<CompiledExpr>),
    Tuple(Vec<CompiledExpr>),
    Dict(Vec<(CompiledExpr, CompiledExpr)>),
    Quoted(Box<SheafValue>),
    FunctionRef(String),
    FunctionCall {
        name: String,
        args: Vec<CompiledExpr>,
        loc: Option<crate::core::error::SourceLocation>,
    },
    Let {
        bindings: Vec<(BindingPattern, CompiledExpr)>,
        body: Box<CompiledExpr>,
    },
    If {
        condition: Box<CompiledExpr>,
        then_branch: Box<CompiledExpr>,
        else_branch: Option<Box<CompiledExpr>>,
    },
    Do(Vec<CompiledExpr>),
    Symbol(String),
    /// Tuple element access: get_tuple_element(param, [i, j, ...])
    /// Generated by with-params when resolving fields of a typed parameter.
    /// e.g. W resolved from (with-params [p :as Linear]) -> GetTupleElement { param: "p", indices: [0] }
    GetTupleElement {
        /// Name of the function parameter that holds the tuple
        param: String,
        /// Sequence of indices for nested tuple access
        indices: Vec<usize>,
    },
    /// Anonymous function (lambda). Pure -- captures no external state.
    /// Always inlined at call sites; never emitted as a separate MLIR function.
    ///
    /// Corresponds to (fn [params] body) in Sheaf.
    Lambda {
        params: Vec<String>,
        body: Box<CompiledExpr>,
    },
    /// Call of a lambda or a locally-bound function value.
    /// Generated when the callee is not a static function name but an expression.
    ///
    /// e.g. ((fn [x] (+ x 1)) 10)  or  (let [f (fn [x] x)] (f 5))
    LambdaCall {
        callee: Box<CompiledExpr>,
        args: Vec<CompiledExpr>,
    },
    /// Deferred value-and-grad computation.
    /// Records the intent to differentiate a function; codegen or interpreter handles execution.
    /// shape_config stores raw (param_name, dims) pairs -- no StableHLOType dependency.
    ValueAndGrad {
        fn_name: String,
        src_fn_name: String,
        wrt_params: Vec<String>,
        shape_config: Vec<(String, Vec<i64>)>,
    },
    /// Counted loop with accumulator: (repeat [i n] [acc init] body)
    /// Evaluates body `n` times, binding loop index to `i` and accumulator to `acc`.
    /// Returns final value of accumulator.
    Repeat {
        index_var: String,
        count: Box<CompiledExpr>,
        acc_var: String,
        acc_init: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },
    /// Conditional loop: (while cond [acc init] body)
    /// Evaluates body while cond is truthy, threading accumulator.
    /// The accumulator variable is visible in both cond and body.
    While {
        condition: Box<CompiledExpr>,
        acc_var: String,
        acc_init: Box<CompiledExpr>,
        body: Box<CompiledExpr>,
    },
    /// Runtime guard assertion: (guard :no-nan expr), (guard :range [lo hi] expr),
    /// (guard :shape [d1 d2] expr). Evaluates expr, checks the condition, returns
    /// the value transparently. Always active when written in source code.
    Guard {
        check: GuardCheck,
        expr: Box<CompiledExpr>,
    },
    /// Global constant binding: (def name value)
    /// Evaluates value once and binds the result to name in the global environment.
    Def {
        name: String,
        value: Box<CompiledExpr>,
    },
}

impl CompiledExpr {
    pub fn map_children(&self, mut f: impl FnMut(&CompiledExpr) -> CompiledExpr) -> CompiledExpr {
        match self {
            CompiledExpr::Integer(_)
            | CompiledExpr::Float(_)
            | CompiledExpr::Boolean(_)
            | CompiledExpr::Nil
            | CompiledExpr::String(_)
            | CompiledExpr::Keyword(_)
            | CompiledExpr::FunctionRef(_)
            | CompiledExpr::Symbol(_)
            | CompiledExpr::GetTupleElement { .. }
            | CompiledExpr::Quoted(_)
            | CompiledExpr::ValueAndGrad { .. } => self.clone(),
            CompiledExpr::Vector(elems) => CompiledExpr::Vector(elems.iter().map(&mut f).collect()),
            CompiledExpr::Tuple(elems) => CompiledExpr::Tuple(elems.iter().map(&mut f).collect()),
            CompiledExpr::Dict(pairs) => CompiledExpr::Dict(
                pairs.iter().map(|(k, v)| (f(k), f(v))).collect(),
            ),
            CompiledExpr::FunctionCall { name, args, .. } => CompiledExpr::FunctionCall {
                name: name.clone(),
                args: args.iter().map(&mut f).collect(),
                loc: None,
            },
            CompiledExpr::Let { bindings, body } => CompiledExpr::Let {
                bindings: bindings.iter().map(|(k, v)| (k.clone(), f(v))).collect(),
                body: Box::new(f(body)),
            },
            CompiledExpr::If { condition, then_branch, else_branch } => CompiledExpr::If {
                condition: Box::new(f(condition)),
                then_branch: Box::new(f(then_branch)),
                else_branch: else_branch.as_ref().map(|e| Box::new(f(e))),
            },
            CompiledExpr::Do(exprs) => CompiledExpr::Do(exprs.iter().map(&mut f).collect()),
            CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
                params: params.clone(),
                body: Box::new(f(body)),
            },
            CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
                callee: Box::new(f(callee)),
                args: args.iter().map(&mut f).collect(),
            },
            CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => CompiledExpr::Repeat {
                index_var: index_var.clone(),
                count: Box::new(f(count)),
                acc_var: acc_var.clone(),
                acc_init: Box::new(f(acc_init)),
                body: Box::new(f(body)),
            },
            CompiledExpr::While { condition, acc_var, acc_init, body } => CompiledExpr::While {
                condition: Box::new(f(condition)),
                acc_var: acc_var.clone(),
                acc_init: Box::new(f(acc_init)),
                body: Box::new(f(body)),
            },
            CompiledExpr::Guard { check, expr } => CompiledExpr::Guard {
                check: check.clone(),
                expr: Box::new(f(expr)),
            },
            CompiledExpr::Def { name, value } => CompiledExpr::Def {
                name: name.clone(),
                value: Box::new(f(value)),
            },
        }
    }
}

/// Manual Debug impl: omits `loc` from FunctionCall to keep cache keys stable.
impl std::fmt::Debug for CompiledExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Integer(n) => write!(f, "Integer({})", n),
            Self::Float(v) => write!(f, "Float({})", v),
            Self::Boolean(b) => write!(f, "Boolean({})", b),
            Self::Nil => write!(f, "Nil"),
            Self::String(s) => write!(f, "String({:?})", s),
            Self::Keyword(s) => write!(f, "Keyword({:?})", s),
            Self::Vector(v) => f.debug_tuple("Vector").field(v).finish(),
            Self::Tuple(v) => f.debug_tuple("Tuple").field(v).finish(),
            Self::Dict(v) => f.debug_tuple("Dict").field(v).finish(),
            Self::Quoted(v) => f.debug_tuple("Quoted").field(v).finish(),
            Self::FunctionRef(s) => write!(f, "FunctionRef({:?})", s),
            Self::FunctionCall { name, args, .. } => {
                f.debug_struct("FunctionCall")
                    .field("name", name)
                    .field("args", args)
                    .finish()
            }
            Self::Let { bindings, body } => {
                f.debug_struct("Let").field("bindings", bindings).field("body", body).finish()
            }
            Self::If { condition, then_branch, else_branch } => {
                f.debug_struct("If")
                    .field("condition", condition)
                    .field("then_branch", then_branch)
                    .field("else_branch", else_branch)
                    .finish()
            }
            Self::Do(v) => f.debug_tuple("Do").field(v).finish(),
            Self::Symbol(s) => write!(f, "Symbol({:?})", s),
            Self::GetTupleElement { param, indices } => {
                f.debug_struct("GetTupleElement").field("param", param).field("indices", indices).finish()
            }
            Self::Lambda { params, body } => {
                f.debug_struct("Lambda").field("params", params).field("body", body).finish()
            }
            Self::LambdaCall { callee, args } => {
                f.debug_struct("LambdaCall").field("callee", callee).field("args", args).finish()
            }
            Self::ValueAndGrad { fn_name, src_fn_name, wrt_params, shape_config } => {
                f.debug_struct("ValueAndGrad")
                    .field("fn_name", fn_name)
                    .field("src_fn_name", src_fn_name)
                    .field("wrt_params", wrt_params)
                    .field("shape_config", shape_config)
                    .finish()
            }
            Self::Repeat { index_var, count, acc_var, acc_init, body } => {
                f.debug_struct("Repeat")
                    .field("index_var", index_var)
                    .field("count", count)
                    .field("acc_var", acc_var)
                    .field("acc_init", acc_init)
                    .field("body", body)
                    .finish()
            }
            Self::While { condition, acc_var, acc_init, body } => {
                f.debug_struct("While")
                    .field("condition", condition)
                    .field("acc_var", acc_var)
                    .field("acc_init", acc_init)
                    .field("body", body)
                    .finish()
            }
            Self::Guard { check, expr } => {
                f.debug_struct("Guard").field("check", check).field("expr", expr).finish()
            }
            Self::Def { name, value } => {
                f.debug_struct("Def").field("name", name).field("value", value).finish()
            }
        }
    }
}

/// Guard check type for runtime assertions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GuardCheck {
    NoNan,
    Range { lo: f64, hi: f64 },
    Shape(Vec<i64>),
}

impl Default for CompilerContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SourceLocation;

    fn make_int(n: i64) -> SheafValue {
        SheafValue::Integer(n, SourceLocation::unknown())
    }

    fn make_symbol(s: &str) -> SheafValue {
        SheafValue::Symbol(s.to_string(), SourceLocation::unknown())
    }

    fn make_list(elems: Vec<SheafValue>) -> SheafValue {
        SheafValue::List(elems, SourceLocation::unknown())
    }

    #[test]
    fn canonical_ast_hash_ignores_source_locations() {
        let first = SheafValue::List(
            vec![SheafValue::String("x".into(), SourceLocation::new(1, 1, "a".into()))],
            SourceLocation::new(1, 1, "a".into()),
        );
        let second = SheafValue::List(
            vec![SheafValue::String("x".into(), SourceLocation::new(9, 4, "b".into()))],
            SourceLocation::new(9, 4, "b".into()),
        );

        assert_eq!(canonical_ast_hash(&first), canonical_ast_hash(&second));
    }

    #[test]
    fn canonical_ast_hash_distinguishes_equal_displayed_nans() {
        let first = SheafValue::Float(
            f64::from_bits(0x7ff8_0000_0000_0001),
            SourceLocation::unknown(),
        );
        let second = SheafValue::Float(
            f64::from_bits(0x7ff8_0000_0000_0002),
            SourceLocation::unknown(),
        );

        assert_eq!(first.to_string(), second.to_string());
        assert_ne!(canonical_ast_hash(&first), canonical_ast_hash(&second));
    }

    #[test]
    fn canonical_ast_hash_is_stable() {
        let body = make_list(vec![make_symbol("f"), make_int(1)]);
        assert_eq!(canonical_ast_hash(&body), canonical_ast_hash(&body));
    }

    #[test]
    fn test_compile_literal() {
        let mut ctx = CompilerContext::new();
        let expr = make_int(42);
        let result = ctx.compile(&expr).unwrap();
        assert!(matches!(result, CompiledExpr::Integer(42)));
    }

    #[test]
    fn test_compile_function_call() {
        let mut ctx = CompilerContext::new();
        // (+ 1 2)
        let expr = make_list(vec![make_symbol("+"), make_int(1), make_int(2)]);
        let result = ctx.compile(&expr).unwrap();

        match result {
            CompiledExpr::FunctionCall { name, args, .. } => {
                assert_eq!(name, "+");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_compile_let() {
        let mut ctx = CompilerContext::new();
        // (let [x 1 y 2] (+ x y))
        let expr = make_list(vec![
            make_symbol("let"),
            SheafValue::Vector(
                vec![make_symbol("x"), make_int(1), make_symbol("y"), make_int(2)],
                SourceLocation::unknown(),
            ),
            make_list(vec![make_symbol("+"), make_symbol("x"), make_symbol("y")]),
        ]);

        let result = ctx.compile(&expr).unwrap();

        match result {
            CompiledExpr::Let { bindings, body } => {
                assert_eq!(bindings.len(), 2);
                assert_eq!(bindings[0].0, BindingPattern::Simple("x".to_string()));
                assert_eq!(bindings[1].0, BindingPattern::Simple("y".to_string()));
                assert!(matches!(bindings[0].1, CompiledExpr::Integer(1)));
                assert!(matches!(bindings[1].1, CompiledExpr::Integer(2)));
                assert!(matches!(*body, CompiledExpr::FunctionCall { .. }));
            }
            _ => panic!("Expected let expression"),
        }
    }

    #[test]
    fn test_compile_defn() {
        let mut ctx = CompilerContext::new();
        // (defn add [x y] (+ x y))
        let expr = make_list(vec![
            make_symbol("defn"),
            make_symbol("add"),
            SheafValue::Vector(
                vec![make_symbol("x"), make_symbol("y")],
                SourceLocation::unknown(),
            ),
            make_list(vec![make_symbol("+"), make_symbol("x"), make_symbol("y")]),
        ]);

        let result = ctx.compile(&expr).unwrap();
        assert!(matches!(result, CompiledExpr::FunctionRef(_)));

        // Check function was registered
        assert!(ctx.registry.contains_key("add"));
        let func = &ctx.registry["add"];
        assert_eq!(func.params, vec!["x", "y"]);
    }

    #[test]
    fn test_compile_multi_function() {
        use crate::lowering::codegen::CodeGenerator;
        use crate::lowering::stablehlo::StableHLOEmitter;

        let mut ctx = CompilerContext::new();

        // (defn square [x] (* x x))
        let square_defn = make_list(vec![
            make_symbol("defn"),
            make_symbol("square"),
            SheafValue::Vector(vec![make_symbol("x")], SourceLocation::unknown()),
            make_list(vec![make_symbol("*"), make_symbol("x"), make_symbol("x")]),
        ]);
        ctx.compile(&square_defn).unwrap();

        // (defn add-squares [a b] (+ (square a) (square b)))
        let add_squares_defn = make_list(vec![
            make_symbol("defn"),
            make_symbol("add-squares"),
            SheafValue::Vector(
                vec![make_symbol("a"), make_symbol("b")],
                SourceLocation::unknown(),
            ),
            make_list(vec![
                make_symbol("+"),
                make_list(vec![make_symbol("square"), make_symbol("a")]),
                make_list(vec![make_symbol("square"), make_symbol("b")]),
            ]),
        ]);
        ctx.compile(&add_squares_defn).unwrap();

        // Now compile the main call: (add-squares 3.0 4.0)
        let main_expr = make_list(vec![
            make_symbol("add-squares"),
            SheafValue::Float(3.0, SourceLocation::unknown()),
            SheafValue::Float(4.0, SourceLocation::unknown()),
        ]);
        let main_compiled = ctx.compile(&main_expr).unwrap();

        // Generate code for all functions
        let mut func_declarations = Vec::new();

        // Generate square function
        let square_def = ctx.registry.get("square").unwrap();
        let square_body_compiled = square_def.body_compiled.clone().unwrap();
        let square_sig = square_def.signature.clone().unwrap();
        let square_params = square_def.params.clone();

        let codegen_square = CodeGenerator::with_function_params(
            &ctx.registry,
            &square_params,
            &square_sig.param_types,
        );

        let (square_decl, _) = codegen_square
            .emit_func_declaration(
                "square",
                &square_body_compiled,
                &square_sig.param_types,
                &square_sig.return_type,
            )
            .unwrap();
        func_declarations.push(square_decl);

        // Generate add-squares function
        let add_squares_def = ctx.registry.get("add-squares").unwrap();
        let add_squares_body_compiled = add_squares_def.body_compiled.clone().unwrap();
        let add_squares_sig = add_squares_def.signature.clone().unwrap();
        let add_squares_params = add_squares_def.params.clone();

        let codegen_add_squares = CodeGenerator::with_function_params(
            &ctx.registry,
            &add_squares_params,
            &add_squares_sig.param_types,
        );

        let (add_squares_decl, _) = codegen_add_squares
            .emit_func_declaration(
                "add-squares",
                &add_squares_body_compiled,
                &add_squares_sig.param_types,
                &add_squares_sig.return_type,
            )
            .unwrap();
        func_declarations.push(add_squares_decl);

        // Generate main function that calls add-squares
        let mut codegen_main = CodeGenerator::with_registry(&ctx.registry);
        let (_, result_ty) = codegen_main.generate(&main_compiled).unwrap();

        let (main_decl, _) = CodeGenerator::with_registry(&ctx.registry)
            .emit_func_declaration("main", &main_compiled, &[], &result_ty)
            .unwrap();
        func_declarations.push(main_decl);

        // Generate the complete module
        let module = StableHLOEmitter::emit_module(&func_declarations);

        // Verify the module contains all expected elements
        // Note: user-defined functions are inlined by the codegen,
        // so func.call is no longer emitted.
        assert!(module.contains("@main"));
        assert!(module.contains("stablehlo.multiply"));
        assert!(module.contains("stablehlo.add"));

        // Print the generated module for inspection
        println!("\nGenerated MLIR module:\n{}", module);
    }

    #[test]
    fn test_grad_form_linear() {
        // (grad (+ (@ x W) b) :wrt W)
        // Expected: symbolic gradient dL/dW = x^T @ 1  (simplified)
        let mut ctx = CompilerContext::new();

        // Introduce x, W, b as local symbols so they compile as Symbol nodes
        ctx.local_vars.insert(
            "x".to_string(),
            SheafValue::Symbol("x".to_string(), SourceLocation::unknown()),
        );
        ctx.local_vars.insert(
            "W".to_string(),
            SheafValue::Symbol("W".to_string(), SourceLocation::unknown()),
        );
        ctx.local_vars.insert(
            "b".to_string(),
            SheafValue::Symbol("b".to_string(), SourceLocation::unknown()),
        );

        // (grad (+ (@ x W) b) :wrt W)
        let expr = make_list(vec![
            make_symbol("grad"),
            make_list(vec![
                make_symbol("+"),
                make_list(vec![make_symbol("@"), make_symbol("x"), make_symbol("W")]),
                make_symbol("b"),
            ]),
            SheafValue::Keyword("wrt".to_string(), SourceLocation::unknown()),
            make_symbol("W"),
        ]);

        let result = ctx.compile(&expr).unwrap();
        println!("grad(linear, W) = {:?}", result);

        // Result should be a FunctionCall (matmul of transpose(x) @ upstream_grad)
        // After simplification:  (@ (transpose x) 1.0)  + 0.0  ->  (@ (transpose x) 1.0)
        assert!(
            matches!(result, CompiledExpr::FunctionCall { .. }),
            "Expected FunctionCall (matmul), got: {:?}",
            result
        );
    }

}
