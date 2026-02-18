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
        ctx.registry.clone(),
        &func_def.params,
        &sig.param_types,
    );
    let decl = codegen
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
