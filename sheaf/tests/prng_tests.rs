use sheaf_compiler::core::{config, prng};
use sheaf_compiler::core::inference::{FunctionSignature, reconstruct_jit_result};
use sheaf_compiler::interpreter::eval::Interpreter;
use sheaf_compiler::interpreter::value::Value;
use sheaf_compiler::runtime::iree_session::{IreeSession, shared_session};
use sheaf_compiler::runtime::jit::{
    JitCompileOutcome, JitCompiler, cache_key_for_function, module_name_for,
};
use std::sync::Arc;

struct Vmfb {
    session: Arc<IreeSession>,
    name: String,
    signature: FunctionSignature,
}

impl Vmfb {
    fn compile(interpreter: &Interpreter, name: &str, key: &Value) -> Self {
        let function = interpreter.registry_get(name).unwrap();
        let registry = &interpreter.env().registry;
        let session = shared_session().unwrap();
        let args = std::slice::from_ref(key);
        let signature = match JitCompiler::new().try_jit_compile(function, args, registry, &session) {
            JitCompileOutcome::Compiled(signature) => signature,
            outcome => panic!("{name}: expected compilation, got {outcome:?}"),
        };
        let cache_key = cache_key_for_function(function, args, registry).unwrap();
        Self { session, name: format!("{}.{name}", module_name_for(name, &cache_key)), signature }
    }

    fn call(&self, key: &Value) -> Value {
        // A direct VMFB call cannot fall back to the interpreter.
        let sig = &self.signature;
        let mut result = self.session.call_typed_device(&self.name, std::slice::from_ref(key), &sig.return_type).unwrap();
        if !sig.arg_type_layouts.is_empty() {
            result = reconstruct_jit_result(result, &sig.return_type, &sig.arg_type_layouts);
        }
        if let Some(layout) = &sig.return_layout { result = layout.reconstruct(result); }
        result
    }
}

fn compare(actual: &Value, expected: &Value, tolerance: f32) {
    let actual = actual.ensure_host().unwrap();
    let expected = expected.ensure_host().unwrap();
    match (&actual, &expected) {
        (Value::Tensor { data: a, .. }, Value::Tensor { data: e, .. }) => {
            assert_eq!(a.shape(), e.shape());
            for (&a, &e) in a.iter().zip(e.iter()) {
                assert!(a.is_finite() && e.is_finite(), "nonfinite sample: {a}, {e}");
                if tolerance == 0.0 { assert_eq!(a.to_bits(), e.to_bits()); }
                else { assert!((a - e).abs() <= tolerance * (1.0 + e.abs()), "{a} != {e}"); }
            }
        }
        (Value::List(a) | Value::Tuple(a), Value::List(e) | Value::Tuple(e)) => {
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e.iter()) { compare(a, e, tolerance); }
        }
        (_, Value::Int(n)) if actual.to_f32().is_some() => {
            assert_eq!(actual.to_f32().unwrap(), *n as f32);
        }
        _ => panic!("unexpected results: {actual:?}, {expected:?}"),
    }
}

#[test]
fn interpreter_vmfb_prng_parity() {
    let device = std::env::var("SHEAF_PRNG_DEVICE").unwrap_or_else(|_| "cpu".into());
    config::init(0, Some(device), false);
    let mut interpreter = Interpreter::new();
    interpreter.env_mut().jit_compiler = None;
    interpreter.eval("(def input-key (random-key 42)) (def child-key (random-key 0))").unwrap();
    let initial = interpreter.eval("input-key").unwrap();
    let mut cases = vec![
        ("literal_key".to_string(), "(random-key -1)".to_string(), 0.0),
        ("float_seed".to_string(), "(random-key -1.0)".to_string(), 0.0),
        ("float_bounds".to_string(), "(random-randint key '[7] -3.0 7.0)".to_string(), 0.0),
        ("float_split_count".to_string(), "(random-split key 3.0)".to_string(), 0.0),
        ("split_keys".to_string(), "(random-split key 5)".to_string(), 0.0),
        ("split_empty".to_string(), "(len (random-split key 0))".to_string(), 0.0),
        ("integer_seed".to_string(), "(random-uniform 16777217 '[7])".to_string(), 0.0),
        ("nested_split".to_string(), "(random-split (get (random-split key 3) 2) 3)".to_string(), 0.0),
        ("split_samples".to_string(), "(random-uniform (get (random-split key) 1) '[7])".to_string(), 0.0),
        ("select_key".to_string(), "(random-uniform (if (> (get key 0) 0.0) key (get (random-split key) 0)) '[7])".to_string(), 0.0),
        ("scan_keys".to_string(), "(first (scan (fn [k i] (let [next (get (random-split k) 1)] [next next])) key (arange 3)))".to_string(), 0.0),
        ("choice_sample".to_string(), "(choice key 4 :p [0.0 0.25 0.25 0.5])".to_string(), 0.0),
    ];
    for (i, shape) in ["'[]", "'[1]", "'[7]", "'[2 3]", "'[257]"].iter().enumerate() {
        for (operation, suffix, tolerance) in [
            ("random-uniform", "", 0.0), ("random-normal", "", 2e-5),
            ("random-randint", " -3 7", 0.0),
        ] {
            cases.push((format!("{}_{}", operation.replace('-', "_"), i),
                format!("({operation} key {shape}{suffix})"), tolerance));
        }
    }
    let cases: Vec<_> = cases.into_iter().map(|(name, body, tolerance)| {
        interpreter.eval(&format!("(defn {name} [key] {body})")).unwrap();
        let vmfb = Vmfb::compile(&interpreter, &name, &initial);
        (name, vmfb, tolerance)
    }).collect();
    for seed in [0, 42, 16777217, 4294967295, -1, i64::MIN, i64::MAX] {
        let key = interpreter.eval(&format!("(random-key {seed})")).unwrap();
        interpreter.env_mut().set_global("input-key", key.clone());
        for (name, vmfb, tolerance) in &cases {
            let expected = interpreter.eval(&format!("({name} input-key)")).unwrap();
            compare(&vmfb.call(&key), &expected, *tolerance);
        }
    }

    // CPU -> device -> CPU, then device -> another VMFB, with identical child keys.
    let key = interpreter.eval("(random-key -1)").unwrap();
    interpreter.env_mut().set_global("input-key", key.clone());
    let split = &cases.iter().find(|(name, _, _)| name == "split_keys").unwrap().1;
    let children = split.call(&key);
    let children = match children {
        Value::Tuple(children) | Value::List(children) => children,
        other => panic!("expected split keys, got {other:?}"),
    };
    for child in children {
        let host = child.ensure_host().unwrap();
        interpreter.env_mut().set("child-key", host.clone());
        let expected = interpreter.eval("(random-uniform child-key '[257])").unwrap();
        let uniform = &cases.iter().find(|(name, _, _)| name == "random_uniform_4").unwrap().1;
        compare(&uniform.call(&child), &expected, 0.0);
        compare(&uniform.call(&host), &expected, 0.0);
        let expected = interpreter.eval("(random-split child-key 5)").unwrap();
        compare(&split.call(&child), &expected, 0.0);
    }

    // Changing the shape must not change the prefix of a flattened stream.
    for operation in ["random-uniform", "random-normal"] {
        let values = |interpreter: &mut Interpreter, n| {
            let value = interpreter.eval(&format!("({operation} input-key '[{n}])")).unwrap();
            let Value::Tensor { data, .. } = value else { unreachable!() };
            data.iter().copied().collect::<Vec<_>>()
        };
        assert_eq!(values(&mut interpreter, 7), values(&mut interpreter, 257)[..7]);
    }
    let Value::Tensor { data, .. } = interpreter.eval("(random-uniform input-key '[65536])").unwrap()
        else { unreachable!() };
    assert!(data.iter().all(|&x| (0.0..1.0).contains(&x)));
    let Value::Tensor { data, .. } = interpreter.eval("(random-normal input-key '[65536])").unwrap()
        else { unreachable!() };
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / data.len() as f64;
    assert!(mean.abs() < 0.025 && (0.96..1.04).contains(&variance));
    assert!(interpreter.eval("(random-split input-key 4294967297)").is_err());
    assert!(interpreter.eval("(random-normal (tensor [0 0]) '[1])").is_err());
    assert!(interpreter.eval("(random-normal (tensor [65536 0 0 0]) '[1])").is_err());
    assert!(interpreter.eval("(random-key 1.5)").is_err());
    assert_eq!(prng::KEY_SIZE, 4);
}

#[test]
fn prng_argument_validation() {
    let mut interpreter = Interpreter::new();
    interpreter.env_mut().jit_compiler = None;
    for (name, valid_args) in [
        ("random-key", vec!["42"]),
        ("random-normal", vec!["(random-key 42)", "'[2]"]),
        ("random-uniform", vec!["(random-key 42)", "'[2]"]),
        ("random-randint", vec!["(random-key 42)", "'[2]", "0", "10"]),
    ] {
        for args in [valid_args[..valid_args.len() - 1].join(" "), format!("{} 0", valid_args.join(" "))] {
            for source in [format!("({name} {args})"), format!("(apply {name} [{args}])")] {
                let error = interpreter.eval(&source).expect_err(&source);
                assert!(error.to_string().contains(name), "{source}: {error}");
            }
        }
    }
    for source in ["(random-split)", "(random-split (random-key 42) 2 0)", "(choice)", "(choice (random-key 42))"] {
        assert!(interpreter.eval(source).is_err(), "{source}");
    }
    let integer_count = interpreter.eval("(random-split (random-key 42) 3)").unwrap();
    let float_count = interpreter.eval("(random-split (random-key 42) 3.0)").unwrap();
    compare(&integer_count, &float_count, 0.0);
    compare(&interpreter.eval("(random-key -1.0)").unwrap(), &interpreter.eval("(random-key -1)").unwrap(), 0.0);
}
