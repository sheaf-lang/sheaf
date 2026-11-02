use super::*;

#[test]
fn test_static_tuple_get_preserves_nested_layout() {
    use crate::core::dtype::ElementType;
    use std::collections::{BTreeMap, HashMap};

    let registry = HashMap::new();
    let param_type = StableHLOType::Tuple(
        vec![StableHLOType::Tuple(
            vec![StableHLOType::Tuple(vec![StableHLOType::scalar_f32()], None)],
            None,
        )],
        None,
    );
    let body = CompiledExpr::Let {
        bindings: vec![
            (
                crate::core::expr::BindingPattern::Simple("blocks".to_string()),
                CompiledExpr::GetTupleElement {
                    param: "params".to_string(),
                    indices: vec![0],
                },
            ),
            (
                crate::core::expr::BindingPattern::Simple("block-p".to_string()),
                CompiledExpr::FunctionCall {
                    name: "get".to_string(),
                    args: vec![
                        CompiledExpr::Symbol("blocks".to_string()),
                        CompiledExpr::Integer(0),
                    ],
                    loc: None,
                },
            ),
        ],
        body: Box::new(CompiledExpr::FunctionCall {
            name: "get".to_string(),
            args: vec![
                CompiledExpr::Symbol("block-p".to_string()),
                CompiledExpr::Keyword("ln_1".to_string()),
            ],
            loc: None,
        }),
    };
    let mut codegen = CodeGenerator::with_function_params(
        &registry,
        &["params".to_string()],
        std::slice::from_ref(&param_type),
    );
    codegen.set_tuple_key_layouts(HashMap::from([
        ("params".to_string(), BTreeMap::from([("h".to_string(), 0)])),
        ("h".to_string(), BTreeMap::from([("0".to_string(), 0)])),
        ("0".to_string(), BTreeMap::from([("ln_1".to_string(), 0)])),
    ]));
    codegen.set_idx_to_key(HashMap::from([
        (("params".to_string(), 0), "h".to_string()),
        (("h".to_string(), 0), "0".to_string()),
    ]));
    let (_, result_type) = codegen
        .emit_func_declaration(
            "nested_get",
            &body,
            std::slice::from_ref(&param_type),
            &StableHLOType::ScalarF32,
        )
        .expect("static tuple get should preserve nested layout");
    assert_eq!(result_type.element_type(), Some(ElementType::F32));
}

#[test]
fn test_generate_constant() {
    let mut codegen = CodeGenerator::new();
    let expr = CompiledExpr::Integer(42);
    let result = codegen.generate(&expr);
    assert!(result.is_ok());
}

#[test]
fn string_dict_keys_are_preserved_and_sorted() {
    let mut codegen = CodeGenerator::new();
    let expr = CompiledExpr::Dict(vec![
        (
            CompiledExpr::String("p".to_string()),
            CompiledExpr::Float(1.0),
        ),
        (
            CompiledExpr::String("loss".to_string()),
            CompiledExpr::Float(2.0),
        ),
    ]);

    let (_, ty) = codegen.generate(&expr).expect("dictionary should compile");
    let StableHLOType::Tuple(_, Some(keys)) = ty else {
        panic!("dictionary should retain its keys");
    };
    assert_eq!(keys, vec!["loss".to_string(), "p".to_string()]);
}

#[test]
fn typed_dict_keys_take_precedence_over_external_layouts() {
    use std::collections::{BTreeMap, HashMap};

    let registry = HashMap::new();
    let ty = StableHLOType::Tuple(
        vec![StableHLOType::scalar_f32()],
        Some(vec!["w".to_string()]),
    );
    let mut codegen = CodeGenerator::with_function_params(
        &registry,
        &["block".to_string()],
        std::slice::from_ref(&ty),
    );
    codegen.set_tuple_key_layouts(HashMap::from([(
        "block".to_string(),
        BTreeMap::from([("other".to_string(), 0)]),
    )]));
    let missing = CompiledExpr::FunctionCall {
        name: "get".to_string(),
        args: vec![
            CompiledExpr::Symbol("block".to_string()),
            CompiledExpr::Keyword("other".to_string()),
        ],
        loc: None,
    };
    assert!(codegen.generate(&missing).is_err());

    let present = CompiledExpr::FunctionCall {
        name: "get".to_string(),
        args: vec![
            CompiledExpr::Symbol("block".to_string()),
            CompiledExpr::Keyword("w".to_string()),
        ],
        loc: None,
    };
    assert!(codegen.generate(&present).is_ok());
}

#[test]
fn assoc_preserves_dict_keys() {
    use std::collections::{BTreeMap, HashMap};

    let registry = HashMap::new();
    let state_type = StableHLOType::Tuple(
        vec![StableHLOType::scalar_f32(), StableHLOType::scalar_f32()],
        Some(vec!["loss".to_string(), "step".to_string()]),
    );
    let mut codegen = CodeGenerator::with_function_params(
        &registry,
        &["state".to_string()],
        std::slice::from_ref(&state_type),
    );
    codegen.set_tuple_key_layouts(HashMap::from([(
        "state".to_string(),
        BTreeMap::from([("loss".to_string(), 0), ("step".to_string(), 1)]),
    )]));
    let expr = CompiledExpr::FunctionCall {
        name: "assoc".to_string(),
        args: vec![
            CompiledExpr::Symbol("state".to_string()),
            CompiledExpr::Keyword("loss".to_string()),
            CompiledExpr::Float(2.0),
        ],
        loc: None,
    };

    let (_, ty) = codegen.generate(&expr).expect("assoc should compile");
    assert_eq!(ty, state_type);
}

#[test]
fn test_generate_binop() {
    let mut codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "+".to_string(),
        args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(2)],
        loc: None,
    };
    let result = codegen.generate(&expr);
    assert!(result.is_ok());
}

#[test]
fn test_emit_function() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "+".to_string(),
        args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(2)],
        loc: None,
    };
    let mlir = codegen.emit_function("test", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.add"));
    assert!(mlir_str.contains("@test"));
}

#[test]
fn test_generate_compare() {
    let mut codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: ">".to_string(),
        args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(2.0)],
        loc: None,
    };
    let result = codegen.generate(&expr);
    assert!(result.is_ok());
    let (_, ty) = result.unwrap();
    assert!(matches!(ty, StableHLOType::ScalarF32));
}

#[test]
fn test_emit_compare() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "=".to_string(),
        args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(1)],
        loc: None,
    };
    let mlir = codegen.emit_function("test_eq", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.sign"));
    assert!(mlir_str.contains("stablehlo.abs"));
    assert!(!mlir_str.contains("stablehlo.minimum"));
}

#[test]
fn test_emit_boolean_and() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "and".to_string(),
        args: vec![
            CompiledExpr::FunctionCall {
                name: ">".to_string(),
                args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(2.0)],
                loc: None,
            },
            CompiledExpr::FunctionCall {
                name: "<".to_string(),
                args: vec![CompiledExpr::Float(1.0), CompiledExpr::Float(3.0)],
                loc: None,
            },
        ],
        loc: None,
    };
    let mlir = codegen.emit_function("test_and", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.sign"));
    assert!(mlir_str.contains("stablehlo.multiply"));
}

#[test]
fn test_emit_boolean_not() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "not".to_string(),
        args: vec![CompiledExpr::FunctionCall {
            name: ">".to_string(),
            args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(10.0)],
            loc: None,
        }],
        loc: None,
    };
    let mlir = codegen.emit_function("test_not", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.sign"));
    assert!(mlir_str.contains("stablehlo.subtract"));
}

#[test]
fn test_reduction_preserves_dtype() {
    use crate::core::dtype::ElementType;

    let registry = HashMap::new();
    let param_type = StableHLOType::tensor(vec![2, 2], ElementType::F16);
    let result_type = StableHLOType::tensor(vec![2, 1], ElementType::F16);
    let codegen = CodeGenerator::with_function_params(
        &registry,
        &["x".to_string()],
        std::slice::from_ref(&param_type),
    );
    let body = CompiledExpr::FunctionCall {
        name: "mean".to_string(),
        args: vec![
            CompiledExpr::Symbol("x".to_string()),
            CompiledExpr::Keyword("axis".to_string()),
            CompiledExpr::Integer(-1),
            CompiledExpr::Keyword("keepdims".to_string()),
        ],
        loc: None,
    };
    let (mlir, actual_type) = codegen
        .emit_func_declaration(
            "mean",
            &body,
            std::slice::from_ref(&param_type),
            &result_type,
        )
        .unwrap();
    assert!(mlir.contains("stablehlo.reduce"));
    assert!(mlir.contains("stablehlo.constant dense<0.0> : tensor<f16>"));
    assert_eq!(actual_type, result_type);
}

#[test]
fn test_arithmetic_converts_weak_scalars() {
    use crate::core::dtype::ElementType;

    for (name, stablehlo_name) in [
        ("+", "add"),
        ("-", "subtract"),
        ("*", "multiply"),
        ("/", "divide"),
        ("**", "power"),
    ] {
        for dtype in [ElementType::F16, ElementType::BF16] {
            for shape in [vec![2], Vec::new()] {
                for scalar_first in [false, true] {
                    let registry = HashMap::new();
                    let param_type = StableHLOType::tensor(shape.clone(), dtype);
                    let codegen = CodeGenerator::with_function_params(
                        &registry,
                        &["x".to_string()],
                        std::slice::from_ref(&param_type),
                    );
                    let tensor = CompiledExpr::Symbol("x".to_string());
                    let scalar = CompiledExpr::Symbol("one".to_string());
                    let args = if scalar_first {
                        vec![scalar, tensor]
                    } else {
                        vec![tensor, scalar]
                    };
                    let body = CompiledExpr::Let {
                        bindings: vec![(
                            BindingPattern::Simple("one".to_string()),
                            CompiledExpr::Float(1.0),
                        )],
                        body: Box::new(CompiledExpr::FunctionCall {
                            name: name.to_string(),
                            args,
                            loc: None,
                        }),
                    };
                    let (mlir, result_type) = codegen
                        .emit_func_declaration(
                            "arithmetic",
                            &body,
                            std::slice::from_ref(&param_type),
                            &param_type,
                        )
                        .unwrap();
                    assert!(mlir.contains("stablehlo.convert"));
                    assert!(mlir.contains(&format!("-> tensor<{}>", dtype.to_mlir_str())));
                    assert!(mlir.contains(&format!("stablehlo.{}", stablehlo_name)));
                    assert_eq!(result_type, param_type);
                }
            }
        }
    }
}

fn call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
    CompiledExpr::FunctionCall {
        name: name.to_string(),
        args,
        loc: None,
    }
}

fn vag_call(loss: CompiledExpr, argument: &str) -> CompiledExpr {
    CompiledExpr::LambdaCall {
        callee: Box::new(call(
            "__value-and-grad-hof__",
            vec![CompiledExpr::lambda(vec!["p".to_string()], loss)],
        )),
        args: vec![CompiledExpr::Symbol(argument.to_string())],
    }
}

#[test]
fn test_cast_supports_f16() {
    use crate::core::dtype::ElementType;

    let registry = HashMap::new();
    let input_type = StableHLOType::f32_tensor(vec![2]);
    let codegen = CodeGenerator::with_function_params(
        &registry,
        &["x".to_string()],
        std::slice::from_ref(&input_type),
    );
    let body = call(
        "cast",
        vec![
            CompiledExpr::Symbol("x".to_string()),
            CompiledExpr::Keyword("f16".to_string()),
        ],
    );
    let result_type = StableHLOType::tensor(vec![2], ElementType::F16);
    let (mlir, actual_type) = codegen
        .emit_func_declaration(
            "cast_f16",
            &body,
            std::slice::from_ref(&input_type),
            &result_type,
        )
        .unwrap();

    assert!(mlir.contains("stablehlo.convert"));
    assert!(mlir.contains("-> tensor<2xf16>"));
    assert_eq!(actual_type, result_type);
}

#[test]
fn test_f16_value_and_grad_preserves_seed_dtype() {
    use crate::core::dtype::ElementType;

    let registry = HashMap::new();
    let param_type = StableHLOType::tensor(vec![4], ElementType::F16);
    let codegen = CodeGenerator::with_function_params(
        &registry,
        &["x".to_string()],
        std::slice::from_ref(&param_type),
    );
    let p = CompiledExpr::Symbol("p".to_string());
    let vag = vag_call(call("mean", vec![call("*", vec![p.clone(), p])]), "x");
    let result_type = StableHLOType::Tuple(
        vec![StableHLOType::f16_tensor(Vec::new()), param_type.clone()],
        None,
    );
    let (mlir, actual_type) = codegen
        .emit_func_declaration(
            "f16_vag",
            &vag,
            std::slice::from_ref(&param_type),
            &result_type,
        )
        .unwrap();

    assert!(mlir.contains("stablehlo.constant dense<1.0> : tensor<f16>"));
    assert!(mlir.contains("-> (tensor<f16>, tensor<4xf16>)"));
    assert_eq!(actual_type, result_type);
}

#[test]
fn test_f16_embedding_gradient_casts_one_hot() {
    use crate::core::dtype::ElementType;

    let registry = HashMap::new();
    let param_types = vec![
        StableHLOType::tensor(vec![4, 2], ElementType::F16),
        StableHLOType::f32_tensor(vec![1]),
    ];
    let codegen = CodeGenerator::with_function_params(
        &registry,
        &["table".to_string(), "index".to_string()],
        &param_types,
    );
    let lookup = call("get", vec![
        CompiledExpr::Symbol("p".to_string()),
        CompiledExpr::Symbol("index".to_string()),
    ]);
    let vag = vag_call(call("mean", vec![lookup]), "table");
    let result_type = StableHLOType::Tuple(
        vec![StableHLOType::f16_tensor(Vec::new()), param_types[0].clone()],
        None,
    );
    let (mlir, actual_type) = codegen
        .emit_func_declaration("f16_embedding_vag", &vag, &param_types, &result_type)
        .unwrap();

    assert!(mlir.contains("stablehlo.convert"));
    assert!(mlir.contains("tensor<4x2xf16>"));
    assert_eq!(actual_type, result_type);
}

#[test]
fn test_f16_comparison_and_slice_gradient_use_typed_constants() {
    use crate::core::dtype::ElementType;

    let registry = HashMap::new();
    let input_type = StableHLOType::tensor(vec![2, 2], ElementType::F16);
    let codegen = CodeGenerator::with_function_params(
        &registry,
        &["x".to_string()],
        std::slice::from_ref(&input_type),
    );
    let comparison = call("==", vec![
        CompiledExpr::Symbol("x".to_string()),
        CompiledExpr::Float(0.0),
    ]);
    let (mlir, actual_type) = codegen
        .emit_func_declaration(
            "f16_comparison",
            &comparison,
            std::slice::from_ref(&input_type),
            &input_type,
        )
        .unwrap();
    assert!(mlir.contains("stablehlo.constant dense<0.0> : tensor<f16>"));
    assert!(mlir.contains("stablehlo.constant dense<1.0> : tensor<f16>"));
    assert_eq!(actual_type, input_type);

    let adjoint_type = StableHLOType::tensor(vec![2, 2], ElementType::F16);
    let result_type = StableHLOType::tensor(vec![6, 2], ElementType::F16);
    let codegen = CodeGenerator::with_function_params(
        &registry,
        &["adjoint".to_string()],
        std::slice::from_ref(&adjoint_type),
    );
    let slice_grad = call("slice_grad", vec![
        CompiledExpr::Symbol("adjoint".to_string()),
        CompiledExpr::Vector(vec![
            CompiledExpr::Integer(6),
            CompiledExpr::Integer(2),
        ]),
        CompiledExpr::Integer(2),
    ]);
    let (mlir, actual_type) = codegen
        .emit_func_declaration(
            "f16_slice_grad",
            &slice_grad,
            std::slice::from_ref(&adjoint_type),
            &result_type,
        )
        .unwrap();
    assert!(mlir.contains("tensor<f16>) -> tensor<6x2xf16>"));
    assert_eq!(actual_type, result_type);
}

#[test]
fn test_concat_accepts_variadic_tensors_and_axis_keyword() {
    use crate::core::dtype::ElementType;

    let registry = HashMap::new();
    let param_types = vec![
        StableHLOType::tensor(vec![2, 1], ElementType::F32),
        StableHLOType::tensor(vec![2, 2], ElementType::F32),
    ];
    let codegen = CodeGenerator::with_function_params(
        &registry,
        &["x".to_string(), "y".to_string()],
        &param_types,
    );
    let body = CompiledExpr::FunctionCall {
        name: "concat".to_string(),
        args: vec![
            CompiledExpr::Symbol("x".to_string()),
            CompiledExpr::Symbol("y".to_string()),
            CompiledExpr::Keyword("axis".to_string()),
            CompiledExpr::Integer(1),
        ],
        loc: None,
    };
    let result_type = StableHLOType::tensor(vec![2, 3], ElementType::F32);
    let (mlir, actual_type) = codegen
        .emit_func_declaration("concatenate", &body, &param_types, &result_type)
        .unwrap();

    assert!(mlir.contains("stablehlo.concatenate"));
    assert!(mlir.contains("dim = 1"));
    assert_eq!(actual_type, result_type);
}

#[test]
fn test_dot_operations_preserve_dtypes() {
    use crate::core::dtype::ElementType;

    for name in ["@", "einsum"] {
        for dtype in [ElementType::F16, ElementType::BF16] {
            let registry = HashMap::new();
            let param_types = vec![
                StableHLOType::tensor(vec![1, 2], dtype),
                StableHLOType::tensor(vec![2, 1], dtype),
            ];
            let codegen = CodeGenerator::with_function_params(
                &registry,
                &["x".to_string(), "y".to_string()],
                &param_types,
            );
            let mut args = vec![
                CompiledExpr::Symbol("x".to_string()),
                CompiledExpr::Symbol("y".to_string()),
            ];
            if name == "einsum" {
                args.insert(0, CompiledExpr::String("ij,jk->ik".to_string()));
            }
            let body = CompiledExpr::FunctionCall {
                name: name.to_string(),
                args,
                loc: None,
            };
            let result_type = StableHLOType::tensor(vec![1, 1], dtype);
            let (mlir, actual_type) = codegen
                .emit_func_declaration("dot", &body, &param_types, &result_type)
                .unwrap();
            assert!(mlir.contains("stablehlo.dot_general"));
            assert!(mlir.contains(&format!("tensor<1x1x{}>", dtype.to_mlir_str())));
            assert_eq!(actual_type, result_type);
        }
    }
}

#[test]
fn test_arithmetic_rejects_strong_dtype_mismatches() {
    use crate::core::dtype::ElementType;

    let registry = HashMap::new();
    let param_types = vec![
        StableHLOType::tensor(vec![2], ElementType::F16),
        StableHLOType::tensor(vec![2], ElementType::BF16),
    ];
    for name in ["+", "-", "*", "/"] {
        let codegen = CodeGenerator::with_function_params(
            &registry,
            &["x".to_string(), "y".to_string()],
            &param_types,
        );
        let body = CompiledExpr::FunctionCall {
            name: name.to_string(),
            args: vec![
                CompiledExpr::Symbol("x".to_string()),
                CompiledExpr::Symbol("y".to_string()),
            ],
            loc: None,
        };
        let error = codegen
            .emit_func_declaration("arithmetic", &body, &param_types, &param_types[0])
            .unwrap_err();
        assert!(error.to_string().contains("dtype mismatch: f16 and bf16"));
    }
}
