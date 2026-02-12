// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Integration tests for the StableHLO compiler

use sheaf_compiler::{StableHLOEmitter, parse};

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
