// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use super::eval::Interpreter;
use super::value::Value;
use crate::runtime::jit::{JitCompileOutcome, JitCompiler};

fn numbers(value: &Value) -> Vec<f32> {
    match value.ensure_host().unwrap() {
        Value::Tensor { data, .. } => data.iter().copied().collect(),
        Value::List(values) | Value::Tuple(values) => values.iter().flat_map(numbers).collect(),
        value => vec![value.to_f64().expect("numeric scan result") as f32],
    }
}

fn assert_stacked(actual: &Value, outputs: &[Value]) {
    let Some(first) = outputs.first() else {
        assert!(matches!(actual, Value::Tuple(values) if values.is_empty()));
        return;
    };
    if let Value::Dict(fields) = first {
        let Value::Dict(actual) = actual else { panic!("expected stacked dict: {actual:?}") };
        assert_eq!(actual.len(), fields.len());
        for key in fields.keys() {
            let values: Vec<_> = outputs.iter().map(|output| {
                let Value::Dict(fields) = output else { panic!("expected dict output") };
                fields[key].clone()
            }).collect();
            assert_stacked(&actual[key], &values);
        }
        return;
    }
    let mut shape = vec![outputs.len()];
    if let Value::Tensor { data, .. } = first {
        shape.extend_from_slice(data.shape());
    }
    assert_eq!(actual.tensor_shape().unwrap(), shape.as_slice());
    assert_eq!(numbers(actual), outputs.iter().flat_map(numbers).collect::<Vec<_>>());
}

#[test]
fn vmfb_scan_matches_every_interpreted_output() {
    let device = std::env::var("SHEAF_SIGNATURE_DEVICE").unwrap_or_else(|_| "cpu".into());
    crate::core::config::init(0, Some(device), false);
    let mut interpreter = Interpreter::new();
    interpreter.env_mut().jit_compiler = None;
    interpreter.eval("(defn scan-pair [x xs]
        (scan (fn [c n] [(+ c n) (* 10.0 n)]) x xs))
        (defn range-scan [x]
          (scan (fn [c n] [(+ c n) (* 10.0 n)]) x (range 1 4)))
        (defn dict-scan [x xs]
          (scan (fn [c n] [(+ c n) {:value (* 10.0 n), :index n}]) x xs))
        (defn scan-outputs [x xs] (second (scan-pair x xs)))").unwrap();
    let session = crate::runtime::iree_session::shared_session().unwrap();
    let mut jit = JitCompiler::new();
    for (name, sources) in [
        ("scan-pair", vec!["[0.0 0.0]", "[[1.0 1.0] [2.0 2.0] [3.0 3.0]]"]),
        ("scan-pair", vec!["0.0", "[1.0 2.0 3.0]"]),
        ("scan-pair", vec!["[0.0 0.0]", "(range 1 4)"]),
        ("range-scan", vec!["[0.0 0.0]"]),
        ("dict-scan", vec!["[0.0 0.0]", "[1.0 2.0 3.0]"]),
        ("scan-pair", vec!["[0.0 0.0]", "[4.0]"]),
        ("scan-pair", vec!["[5.0 7.0]", "(zeros [0])"]),
        ("scan-pair", vec!["[5.0 7.0]", "'[]"]),
        ("scan-outputs", vec!["[0.0 0.0]", "[1.0 2.0 3.0]"]),
    ] {
        let call = format!("({name} {})", sources.join(" "));
        let expected = interpreter.eval(&call).unwrap();
        let args: Vec<_> = sources.iter().map(|source| interpreter.eval(source).unwrap()).collect();
        let function = interpreter.registry_get(name).unwrap().clone();
        let (prepared, args) = super::specialize::prepare_function(
            &function, &args, interpreter.env(),
        ).unwrap().expect_ready();
        let outcome = jit.try_jit_compile(&prepared, &args, &interpreter.env().registry, &session);
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{call}: {outcome:?}");
        let module = jit.module_for_args(&prepared, &args).unwrap();
        let actual = session.call_typed_device(
            &format!("{}.{}", module.module_name, name.replace('-', "_")),
            &args, &module.sig.return_type,
        ).unwrap().ensure_host().unwrap();
        if name == "scan-outputs" {
            let Value::List(outputs) = expected else { panic!("expected output list") };
            assert_stacked(&actual, &outputs);
            continue;
        }
        let Value::Tuple(expected) = expected else { panic!("expected interpreted scan pair") };
        let Value::Tuple(actual) = actual else { panic!("expected VMFB scan pair: {actual:?}") };
        assert_eq!(actual.len(), 2);
        assert_eq!(numbers(&actual[0]), numbers(&expected[0]), "{call}");
        let Value::List(outputs) = &expected[1] else { panic!("expected output list") };
        assert_stacked(&actual[1], outputs);
    }
}
