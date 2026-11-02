// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use sheaf_frontend::{CompiledExpr, CompilerContext, parse};

const STDLIB: &[(&str, &str)] = &[
    ("macros.shf", include_str!("../lib/macros.shf")),
    ("misc.shf", include_str!("../lib/misc.shf")),
    ("nn.shf", include_str!("../lib/nn.shf")),
    ("optim.shf", include_str!("../lib/optim.shf")),
];

#[test]
fn compiles_the_complete_standard_library_without_runtime() {
    let mut context = CompilerContext::new_without_prelude();

    for (name, source) in STDLIB {
        context
            .compile_prelude_module(name, source)
            .unwrap_or_else(|error| panic!("failed to compile {name}: {error}"));
    }

    assert!(context.registry.contains_key("transform-layer"));
    assert!(context.registry.contains_key("layer-norm"));
    assert!(context.registry.contains_key("adamw-step"));
    assert!(context.macro_engine.macros.contains_key("defmodel"));
    assert!(context.macro_engine.macros.contains_key("defbatch"));
    assert_eq!(context.prelude_modules.len(), STDLIB.len());
}

#[test]
fn parses_special_forms_into_compiled_ir() {
    let source = "(defn add-one [x] (let [one 1] (+ x one)))";
    let expression = parse(source, "frontend-test.shf").unwrap().remove(0);
    let mut context = CompilerContext::new_without_prelude();

    let compiled = context.compile(&expression).unwrap();
    assert!(matches!(compiled, CompiledExpr::FunctionRef(ref name) if name == "add-one"));

    let function = &context.registry["add-one"];
    assert!(matches!(function.body_compiled.as_deref(), Some(CompiledExpr::Let { .. })));
}

#[test]
fn expands_prelude_macros_without_runtime() {
    let mut context = CompilerContext::new_without_prelude();
    context
        .compile_prelude_module("macros.shf", STDLIB[0].1)
        .unwrap();
    let expression = parse("(when true (+ 1 2))", "macro-test.shf")
        .unwrap()
        .remove(0);

    let compiled = context.compile(&expression).unwrap();
    assert!(matches!(compiled, CompiledExpr::If { .. }));
}
