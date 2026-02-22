// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf compiler - orchestrates compilation from AST to executable code
//!
//! Corresponds to Python sheaf/core/compiler.py

use crate::ast::SheafValue;
use crate::core::error::{SheafError, SheafResult};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// A leaf field in a parameter layout: a named tensor with its shape
#[derive(Debug, Clone)]
pub struct ParamField {
    /// Path of keys from the root: e.g. ["attn", "Wq"]
    pub path: Vec<String>,
    /// Shape of the tensor: e.g. [512, 512]
    pub shape: Vec<i64>,
    /// Tuple index at this level (set during layout resolution)
    pub tuple_index: Vec<usize>,
}

/// A resolved parameter layout for a `defparams` declaration.
/// The layout is a flat list of fields derived from the nested dict schema.
#[derive(Debug, Clone)]
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

    /// Parameter type registry (defparams declarations)
    pub param_types: HashMap<String, ParamLayout>,

    /// Param scope: maps symbol name -> (param_name, tuple_indices)
    /// Set by with-params to resolve symbols like W -> get_tuple_element(p, [0, 1])
    pub param_scope: HashMap<String, (String, Vec<usize>)>,

    /// Search paths for (use module): stdlib directories + current file's directory
    pub load_path: Vec<PathBuf>,

    /// Absolute paths of already-loaded modules (prevents duplicate loading)
    pub loaded_modules: HashSet<PathBuf>,

    /// Directory of the file currently being compiled (for relative use paths)
    pub current_dir: Option<PathBuf>,
}

/// Function definition
#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: SheafValue,
    pub body_compiled: Option<CompiledExpr>,
    pub signature: Option<crate::core::inference::FunctionSignature>,
}

fn is_builtin_name(name: &str) -> bool {
    matches!(name,
        "+" | "-" | "*" | "/" | "//" | "mod" | "%" | "**"
        | "abs" | "exp" | "log" | "sqrt" | "@"
        | "=" | "==" | "!=" | "<" | ">" | "<=" | ">="
        | "not" | "and" | "or"
        | "shape" | "ndim" | "len" | "count"
        | "int" | "float" | "str"
        | "relu" | "leaky-relu" | "sigmoid" | "tanh" | "gelu" | "selu" | "celu" | "silu"
        | "softmax" | "log-softmax"
        | "sum" | "mean" | "product" | "min" | "max" | "minimum" | "maximum"
        | "argmax" | "argmin"
        | "zeros" | "ones" | "arange" | "eye" | "one-hot" | "tril"
        | "reshape" | "transpose" | "concat" | "slice" | "where" | "roll" | "index-update"
        | "first" | "second" | "last" | "rest" | "nth" | "cons" | "append" | "empty?"
        | "get" | "get-in" | "assoc" | "dissoc" | "merge" | "keys" | "vals" | "dict"
        | "map" | "filter" | "reduce" | "apply" | "find"
        | "tensor" | "range" | "swapaxes" | "var" | "normalize" | "index-of"
        | "gensym" | "symbol?"
        | "einsum" | "append-and-roll"
        | "dynamic-slice" | "mse-loss" | "mae-loss" | "sparse-cross-entropy"
        | "tree-map" | "tree-map-zeros" | "tree-reduce" | "flatten"
    )
}

impl CompilerContext {
    pub fn new() -> Self {
        Self {
            env: Self::init_env(),
            registry: HashMap::new(),
            local_vars: HashMap::new(),
            param_types: HashMap::new(),
            param_scope: HashMap::new(),
            load_path: Self::default_load_path(),
            loaded_modules: HashSet::new(),
            current_dir: None,
        }
    }

    /// Build the default load path: stdlib dir (relative to binary) + cwd.
    fn default_load_path() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // 1. $SHEAF_LIB env var
        if let Ok(lib) = std::env::var("SHEAF_LIB") {
            paths.push(PathBuf::from(lib));
        }

        // 2. Stdlib relative to the running binary: <binary>/../lib/
        if let Ok(exe) = std::env::current_exe() {
            if let Some(bin_dir) = exe.parent() {
                let candidate = bin_dir.join("../lib");
                if candidate.exists() {
                    paths.push(candidate.canonicalize().unwrap_or(candidate));
                }
                // Also try <binary>/../../sheaf/lib/ for dev layout (rust/target/debug/sheaf)
                let dev_candidate = bin_dir.join("../../../sheaf/lib");
                if dev_candidate.exists() {
                    paths.push(dev_candidate.canonicalize().unwrap_or(dev_candidate));
                }
            }
        }

        // 3. Current working directory
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
        let env = HashMap::new();

        // Built-in constants
        // env.insert("true".to_string(), SheafValue::Boolean(true, SourceLocation::unknown()));
        // env.insert("false".to_string(), SheafValue::Boolean(false, SourceLocation::unknown()));
        // env.insert("nil".to_string(), SheafValue::Nil(SourceLocation::unknown()));

        // Built-in functions will be added as we port runtime ops

        env
    }

    /// Compile a Sheaf expression
    pub fn compile(&mut self, exp: &SheafValue) -> SheafResult<CompiledExpr> {
        match exp {
            // --- Literals ---
            SheafValue::Integer(n, _) => Ok(CompiledExpr::Integer(*n)),
            SheafValue::Float(x, _) => Ok(CompiledExpr::Float(*x)),
            SheafValue::Boolean(b, _) => Ok(CompiledExpr::Boolean(*b)),
            SheafValue::Nil(_) => Ok(CompiledExpr::Nil),
            SheafValue::String(s, _) => Ok(CompiledExpr::String(s.clone())),

            // --- Symbols ---
            SheafValue::Symbol(name, loc) => self.resolve_symbol(name, loc),

            // --- Keywords ---
            SheafValue::Keyword(k, _) => Ok(CompiledExpr::Keyword(k.clone())),

            // --- Lists (function calls, special forms) ---
            SheafValue::List(elements, loc) => {
                if elements.is_empty() {
                    return Err(SheafError::Compile {
                        message: "Cannot compile empty list".to_string(),
                        location: loc.clone(),
                    });
                }

                // Check for special forms
                if let Some(op) = elements[0].as_symbol() {
                    // Try to compile as special form, fall back to function call
                    match self.try_compile_special_form(op, &elements[1..], loc) {
                        Some(result) => result,
                        None => self.compile_function_call(elements, loc),
                    }
                } else {
                    // Head is not a symbol — compile it and treat as lambda call.
                    // e.g. ((fn [x] (+ x 1)) 10)
                    let callee = self.compile(&elements[0])?;
                    let args: SheafResult<Vec<CompiledExpr>> =
                        elements[1..].iter().map(|a| self.compile(a)).collect();
                    Ok(CompiledExpr::LambdaCall {
                        callee: Box::new(callee),
                        args: args?,
                    })
                }
            }

            // --- Vectors ---
            SheafValue::Vector(elements, _) => {
                // Compile each element
                let compiled: SheafResult<Vec<CompiledExpr>> =
                    elements.iter().map(|e| self.compile(e)).collect();
                Ok(CompiledExpr::Vector(compiled?))
            }

            // --- Dicts ---
            SheafValue::Dict(pairs, _) => {
                let compiled: SheafResult<Vec<(CompiledExpr, CompiledExpr)>> = pairs
                    .iter()
                    .map(|(k, v)| Ok((self.compile(k)?, self.compile(v)?)))
                    .collect();
                Ok(CompiledExpr::Dict(compiled?))
            }

            // --- Quotes ---
            SheafValue::Quote(inner, _) => {
                // Quote prevents evaluation
                Ok(CompiledExpr::Quoted(inner.clone()))
            }

            _ => Err(SheafError::Compile {
                message: format!("Unsupported expression type: {}", exp),
                location: exp.location().clone(),
            }),
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
            // If the value is a fn form, recompile it to get a Lambda.
            if let SheafValue::List(ref elems, _) = val {
                if elems.first().and_then(|e| e.as_symbol()) == Some("fn") {
                    return self.compile(&val);
                }
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
        use crate::forms::ml::{DefparamsForm, GradForm, ValueAndGradForm, WithParamsForm};
        use crate::forms::*;

        let result = match op {
            "defn" => DefnForm.compile(self, args, loc),
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
            "defparams" => DefparamsForm.compile(self, args, loc),
            "with-params" => WithParamsForm.compile(self, args, loc),
            "grad" => GradForm.compile(self, args, loc),
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
        })
    }
}

/// Compiled expression - intermediate representation
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Nil,
    String(String),
    Keyword(String),
    Vector(Vec<CompiledExpr>),
    Dict(Vec<(CompiledExpr, CompiledExpr)>),
    Quoted(Box<SheafValue>),
    FunctionRef(String),
    FunctionCall {
        name: String,
        args: Vec<CompiledExpr>,
    },
    Let {
        bindings: Vec<(String, CompiledExpr)>,
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
    /// Anonymous function (lambda). Pure — captures no external state.
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
    /// shape_config stores raw (param_name, dims) pairs — no StableHLOType dependency.
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
            CompiledExpr::FunctionCall { name, args } => {
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
                assert_eq!(bindings[0].0, "x");
                assert_eq!(bindings[1].0, "y");
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
        assert!(matches!(result, CompiledExpr::Nil));

        // Check function was registered
        assert!(ctx.registry.contains_key("add"));
        let func = &ctx.registry["add"];
        assert_eq!(func.params, vec!["x", "y"]);
    }

    #[test]
    fn test_compile_multi_function() {
        use crate::compiler::codegen::CodeGenerator;
        use crate::compiler::stablehlo::StableHLOEmitter;

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
            ctx.registry.clone(),
            &square_params,
            &square_sig.param_types,
        );

        let square_decl = codegen_square
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
            ctx.registry.clone(),
            &add_squares_params,
            &add_squares_sig.param_types,
        );

        let add_squares_decl = codegen_add_squares
            .emit_func_declaration(
                "add-squares",
                &add_squares_body_compiled,
                &add_squares_sig.param_types,
                &add_squares_sig.return_type,
            )
            .unwrap();
        func_declarations.push(add_squares_decl);

        // Generate main function that calls add-squares
        let mut codegen_main = CodeGenerator::with_registry(ctx.registry.clone());
        let (_, result_ty) = codegen_main.generate(&main_compiled).unwrap();

        let main_decl = CodeGenerator::with_registry(ctx.registry.clone())
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
        // After simplification:  (@ (transpose x) 1.0)  + 0.0  →  (@ (transpose x) 1.0)
        assert!(
            matches!(result, CompiledExpr::FunctionCall { .. }),
            "Expected FunctionCall (matmul), got: {:?}",
            result
        );
    }

    #[test]
    fn test_grad_form_wrt_b() {
        // (grad (+ (@ x W) b) :wrt b) = 1.0  (b enters via Add, so grad is 1)
        let mut ctx = CompilerContext::new();
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

        let expr = make_list(vec![
            make_symbol("grad"),
            make_list(vec![
                make_symbol("+"),
                make_list(vec![make_symbol("@"), make_symbol("x"), make_symbol("W")]),
                make_symbol("b"),
            ]),
            SheafValue::Keyword("wrt".to_string(), SourceLocation::unknown()),
            make_symbol("b"),
        ]);

        let result = ctx.compile(&expr).unwrap();
        println!("grad(linear, b) = {:?}", result);

        // After simplify: grad of (x @ W) wrt b = 0.0, grad of b wrt b = 1.0
        // Add(0.0, 1.0) → 1.0
        assert!(
            matches!(result, CompiledExpr::Float(f) if f == 1.0),
            "Expected Float(1.0), got: {:?}",
            result
        );
    }
}
