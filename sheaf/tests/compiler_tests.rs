// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Integration tests for the StableHLO compiler

use sheaf_compiler::{CodeGenerator, CompilerContext, StableHLOEmitter, parse};

#[test]
fn test_compile_add() {
    // (+ 1 2)
    let source = "(+ 1 2)";
    let exprs = parse(source, "<test>").unwrap();
    assert_eq!(exprs.len(), 1);

    let mut emitter = StableHLOEmitter::new();
    let mlir = emitter.emit_function("add", &exprs[0]);

    println!("{}", mlir);

    // Check structure
    assert!(mlir.contains("module {"));
    assert!(mlir.contains("func.func @add()"));
    assert!(mlir.contains("stablehlo.constant"));
    assert!(mlir.contains("dense<1.0>"));
    assert!(mlir.contains("dense<2.0>"));
    assert!(mlir.contains("stablehlo.add"));
    assert!(mlir.contains("return"));
    assert!(mlir.contains("tensor<f32>"));
}

#[test]
fn test_compile_nested() {
    // (* (+ 1 2) 4)
    let source = "(* (+ 1 2) 4)";
    let exprs = parse(source, "<test>").unwrap();
    assert_eq!(exprs.len(), 1);

    let mut emitter = StableHLOEmitter::new();
    let mlir = emitter.emit_function("add_then_mul", &exprs[0]);

    println!("{}", mlir);

    // Check operations appear in correct order
    assert!(mlir.contains("stablehlo.add"));
    assert!(mlir.contains("stablehlo.multiply"));

    // Check all constants (now with .0 for floats)
    assert!(mlir.contains("dense<1.0>"));
    assert!(mlir.contains("dense<2.0>"));
    assert!(mlir.contains("dense<4.0>"));
}

#[test]
fn test_compile_two_branches() {
    // (- (* 3 4) (+ 1 2))
    let source = "(- (* 3 4) (+ 1 2))";
    let exprs = parse(source, "<test>").unwrap();
    assert_eq!(exprs.len(), 1);

    let mut emitter = StableHLOEmitter::new();
    let mlir = emitter.emit_function("nested", &exprs[0]);

    println!("{}", mlir);

    // Check all operations
    assert!(mlir.contains("stablehlo.multiply"));
    assert!(mlir.contains("stablehlo.add"));
    assert!(mlir.contains("stablehlo.subtract"));

    // Check all constants
    assert!(mlir.contains("dense<3.0>"));
    assert!(mlir.contains("dense<4.0>"));
    assert!(mlir.contains("dense<1.0>"));
    assert!(mlir.contains("dense<2.0>"));
}

#[test]
fn test_compile_floats() {
    // (+ 1.5 2.5)
    let source = "(+ 1.5 2.5)";
    let exprs = parse(source, "<test>").unwrap();

    let mut emitter = StableHLOEmitter::new();
    let mlir = emitter.emit_function("add_floats", &exprs[0]);

    println!("{}", mlir);

    assert!(mlir.contains("dense<1.5>"));
    assert!(mlir.contains("dense<2.5>"));
}

#[test]
fn test_compile_division() {
    // (/ 10 2)
    let source = "(/ 10 2)";
    let exprs = parse(source, "<test>").unwrap();

    let mut emitter = StableHLOEmitter::new();
    let mlir = emitter.emit_function("divide", &exprs[0]);

    println!("{}", mlir);

    assert!(mlir.contains("stablehlo.divide"));
    assert!(mlir.contains("dense<10.0>"));
    assert!(mlir.contains("dense<2.0>"));
}

#[test]
fn test_compile_subtraction() {
    // (- 5 3)
    let source = "(- 5 3)";
    let exprs = parse(source, "<test>").unwrap();

    let mut emitter = StableHLOEmitter::new();
    let mlir = emitter.emit_function("subtract", &exprs[0]);

    println!("{}", mlir);

    assert!(mlir.contains("stablehlo.subtract"));
    assert!(mlir.contains("dense<5.0>"));
    assert!(mlir.contains("dense<3.0>"));
}

#[test]
fn test_write_mlir_to_file() {
    // Write the same examples as poc/emit.py
    use std::fs;
    use std::path::Path;

    let examples = vec![
        ("add", "(+ 1 2)"),
        ("add_then_mul", "(* (+ 1 2) 4)"),
        ("nested", "(- (* 3 4) (+ 1 2))"),
    ];

    let out_dir = Path::new("target/mlir");
    fs::create_dir_all(out_dir).unwrap();

    for (name, source) in examples {
        let exprs = parse(source, "<test>").unwrap();
        let mut emitter = StableHLOEmitter::new();
        let mlir = emitter.emit_function(name, &exprs[0]);

        let path = out_dir.join(format!("{}.mlir", name));
        fs::write(&path, mlir).unwrap();
        println!("[OK] {:?}", path);
    }
}

// Compile source, run codegen, return MLIR string.
fn compile_to_mlir(source: &str, fn_name: &str) -> String {
    let exprs = parse(source, "<test>").unwrap();
    let mut ctx = CompilerContext::new();
    for e in &exprs {
        ctx.compile(e).unwrap();
    }
    let func_def = ctx.registry.get(fn_name).unwrap().clone();
    let body = func_def.body_compiled.clone().unwrap();
    let sig = func_def.signature.clone().unwrap();
    let codegen = CodeGenerator::with_function_params(
        &ctx.registry,
        &func_def.params,
        &sig.param_types,
    );
    let (decl, _ty) = codegen
        .emit_func_declaration(fn_name, &body, &sig.param_types, &sig.return_type)
        .unwrap();
    StableHLOEmitter::emit_module(&[decl])
}

#[test]
fn test_fn_direct_call() {
    // ((fn [x] (+ x 1.0)) 10.0)
    let mlir = compile_to_mlir(
        "(defn test-fn-direct [a] ((fn [x] (+ x 1.0)) a))",
        "test-fn-direct",
    );
    println!("{}", mlir);
    assert!(mlir.contains("stablehlo.add"));
    assert!(mlir.contains("dense<1.0>"));
}

#[test]
fn test_fn_let_bound() {
    // (let [double (fn [n] (* n 2.0))] (double 21.0))
    let mlir = compile_to_mlir(
        "(defn test-fn-let [a] (let [double (fn [n] (* n 2.0))] (double a)))",
        "test-fn-let",
    );
    println!("{}", mlir);
    assert!(mlir.contains("stablehlo.multiply"));
    assert!(mlir.contains("dense<2.0>"));
}

#[test]
fn test_fn_higher_order() {
    // value-and-grad style: apply a loss fn to params
    // (defn apply-fn [f x] (f x))
    // (apply-fn (fn [x] (* x x)) 3.0)
    let mlir = compile_to_mlir("(defn test-fn-ho [a] ((fn [x] (* x x)) a))", "test-fn-ho");
    println!("{}", mlir);
    assert!(mlir.contains("stablehlo.multiply"));
}

// Destructuring let: JIT codegen coverage.

/// Helper for destructuring tests: creates a CompilerContext, compiles the source,
/// and returns the (parsed, registry) pair so callers can apply the destructuring
/// pipeline and codegen at will.
fn parse_to_registry(
    source: &str,
) -> (sheaf_compiler::CompilerContext, Vec<sheaf_compiler::SheafValue>) {
    let exprs = parse(source, "<test>").unwrap();
    let ctx = CompilerContext::new();
    (ctx, exprs)
}

/// Compile [1 2] classifier & destructuring pipeline + codegen, returns MLIR.
/// Mirrors what `compile_function` does in jit.rs for the destructuring part.
fn compile_with_destructure(source: &str, fn_name: &str) -> String {
    use std::collections::HashMap;
    let (mut ctx, exprs) = parse_to_registry(source);
    for e in &exprs {
        ctx.compile(e).unwrap();
    }
    let func_def = ctx.registry.get(fn_name).unwrap().clone();
    let body = func_def.body_compiled.as_deref().unwrap().clone();
    let sig = func_def.signature.clone().unwrap();
    let mut param_shapes: HashMap<String, Vec<i64>> = HashMap::new();
    for (p, ty) in func_def.params.iter().zip(sig.param_types.iter()) {
        let sh = ty.shape();
        if !sh.is_empty() {
            param_shapes.insert(p.clone(), sh.to_vec());
        }
    }
    let lowered = sheaf_compiler::lowering::transforms::lower_tuples_and_destructuring(
        body,
        &param_shapes,
    )
    .expect("destructuring let should lower");
    let codegen = CodeGenerator::with_function_params(
        &ctx.registry,
        &func_def.params,
        &sig.param_types,
    );
    let (decl, _ty) = codegen
        .emit_func_declaration(fn_name, &lowered, &sig.param_types, &sig.return_type)
        .expect("codegen");
    StableHLOEmitter::emit_module(&[decl])
}

#[test]
fn test_destructure_let_simple() {
    // Vector literal of scalars -> stays Vector/constant, destructuring emits
    // `slice` (shape propagation) instead of `get_tuple_element` for Vector 1-D.
    let mlir = compile_with_destructure(
        "(defn test-de-let [a] (let [[x y] [1 2]] (+ a (+ x y))))",
        "test-de-let",
    );
    println!("{}", mlir);
    // 1 2 gets constant-folded to dense<[1.0, 2.0]> and destructured via slice.
    assert!(mlir.contains("stablehlo.add"));
    assert!(mlir.contains("stablehlo.slice") || mlir.contains("stablehlo.get_tuple_element"),
        "destructuring a Vector literal must emit slice or get_tuple_element");
}

/// Like `compile_with_destructure`, but supplies the parameter shape explicitly.
/// The JIT resolves a parameter's shape from the caller's runtime argument (e.g.
/// passing `[1.0 2.0]` yields `tensor<2xf32>`); static signature inference cannot,
/// so we mirror the runtime-typed path here to exercise destructuring of a typed
/// Symbol source.
fn compile_typed_param_destructure(source: &str, fn_name: &str) -> String {
    use std::collections::HashMap;
    let exprs = parse(source, "<test>").unwrap();
    let mut ctx = CompilerContext::new();
    for e in &exprs {
        ctx.compile(e).unwrap();
    }
    let func_def = ctx.registry.get(fn_name).unwrap().clone();
    let body = func_def.body_compiled.as_deref().unwrap().clone();
    assert_eq!(
        func_def.params.len(),
        1,
        "this helper expects a single parameter"
    );
    let param_shapes: HashMap<String, Vec<i64>> =
        [(func_def.params[0].clone(), vec![2_i64])].into_iter().collect();
    let lowered = sheaf_compiler::lowering::transforms::lower_tuples_and_destructuring(
        body,
        &param_shapes,
    )
    .expect("typed-parameter destructuring should lower");
    let param_types = vec![sheaf_compiler::StableHLOType::f32_tensor(vec![2_i64])];
    let return_type = sheaf_compiler::StableHLOType::scalar_f32();
    let codegen = CodeGenerator::with_function_params(
        &ctx.registry,
        &func_def.params,
        &param_types,
    );
    let (decl, _ty) = codegen
        .emit_func_declaration(fn_name, &lowered, &param_types, &return_type)
        .expect("codegen");
    StableHLOEmitter::emit_module(&[decl])
}

#[test]
fn test_destructure_typed_param() {
    // Destructuring a typed parameter (v resolved as tensor<2xf32>, as a caller
    // passing [1.0 2.0] would) must lower and codegen to MLIR. Previously the
    // pass could not resolve a Symbol source's length, so the JIT skipped with
    // "tensor of dynamic length" and fell back to the interpreter.
    let mlir = compile_typed_param_destructure(
        "(defn test-de-param [v] (let [[a b] v] (+ a b)))",
        "test-de-param",
    );
    println!("{}", mlir);
    assert!(mlir.contains("stablehlo.add"));
    assert!(
        mlir.contains("stablehlo.slice") || mlir.contains("stablehlo.get_tuple_element"),
        "destructuring a typed tensor parameter must emit slice or get_tuple_element"
    );
}

#[test]
fn test_destructure_arity_mismatch_returns_compile_error() {
    // Arity mismatch must surface as a Result, never a panic.
    use sheaf_compiler::lowering::transforms as T;
    use std::collections::HashMap;
    // Build a minimal let-expression with an arity mismatch.
    use sheaf_compiler::CompiledExpr;
    let body = CompiledExpr::Let {
        bindings: vec![(
            sheaf_compiler::BindingPattern::Destructure(vec![
                sheaf_compiler::BindingPattern::Simple("a".into()),
                sheaf_compiler::BindingPattern::Simple("b".into()),
                sheaf_compiler::BindingPattern::Simple("c".into()),
            ]),
            CompiledExpr::Vector(vec![CompiledExpr::Integer(1), CompiledExpr::Integer(2)]),
        )],
        body: Box::new(CompiledExpr::Integer(0)),
    };
    let param_shapes: HashMap<String, Vec<i64>> = HashMap::new();
    let classified = T::classify_vectors(body, &param_shapes);
    let result = T::desugar_destructuring_lets(classified, &Default::default());
    assert!(result.is_err(), "arity mismatch must be an Err, not Ok");
    let err = result.err().unwrap();
    let msg = err.short_message();
    assert!(
        msg.contains("arity") || msg.contains("destructuring"),
        "expected arity error message, got: {}",
        msg
    );
}
