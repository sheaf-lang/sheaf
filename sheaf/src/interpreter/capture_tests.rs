// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use super::*;

fn run(source: &str) -> Result<Value, SheafError> {
    let mut interpreter = crate::interpreter::eval::Interpreter::new();
    #[cfg(iree_runtime)]
    { interpreter.env_mut().jit_compiler = None; }
    interpreter.eval(source)
}

#[cfg(iree_runtime)]
#[test]
fn prepared_closures_refresh_float_captures_without_recompiling() {
    use crate::runtime::jit::{JitCompileOutcome, JitCompiler};
    let device = std::env::var("SHEAF_SIGNATURE_DEVICE").unwrap_or_else(|_| "cpu".into());
    crate::core::config::init(0, Some(device), false);
    let mut interpreter = crate::interpreter::eval::Interpreter::new();
    interpreter.env_mut().jit_compiler = None;
    interpreter.eval("(defn make_shift [s] (fn [y] (+ s y)))
        (defn call_shift [f x] (f x))").unwrap();
    let function = interpreter.registry_get("call_shift").unwrap().clone();
    let input = interpreter.eval("[1.0 2.0]").unwrap();
    let session = crate::runtime::iree_session::shared_session().unwrap();
    let mut jit = JitCompiler::new();
    let mut modules = Vec::new();
    for (capture, expected) in [(2.0, [3.0, 4.0]), (5.0, [6.0, 7.0])] {
        let closure = interpreter.eval(&format!("(make_shift {capture:.1})")).unwrap();
        let (prepared, args) = super::specialize::prepare_function(
            &function, &[closure, input.clone()], interpreter.env(),
        ).unwrap().expect_ready();
        let outcome = jit.try_jit_compile(&prepared, &args, &interpreter.env().registry, &session);
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{outcome:?}");
        let module = jit.module_for_args(&prepared, &args).unwrap();
        let result = session.call_typed_device(
            &format!("{}.call_shift", module.module_name), &args, &module.sig.return_type,
        ).unwrap().ensure_host().unwrap();
        let Value::Tensor { data, .. } = result else { panic!() };
        assert_eq!(data.as_slice().unwrap(), &expected);
        modules.push(module.module_name);
    }
    assert_eq!(modules[0], modules[1]);
}

#[cfg(iree_runtime)]
#[test]
fn global_capture_updates_and_code_replacement_invalidate_only_needed_variants() {
    use crate::runtime::jit::{JitCompileOutcome, JitCompiler};
    let device = std::env::var("SHEAF_SIGNATURE_DEVICE").unwrap_or_else(|_| "cpu".into());
    crate::core::config::init(0, Some(device), false);
    let mut interpreter = crate::interpreter::eval::Interpreter::new();
    interpreter.env_mut().jit_compiler = None;
    interpreter.eval("(defn make_shift [s] (fn [y] (+ s y)))
        (defn make_subtract [s] (fn [y] (- y s)))
        (def shift (make_shift 2.0)) (defn run_global [x] (shift x))").unwrap();
    let function = interpreter.registry_get("run_global").unwrap().clone();
    let input = interpreter.eval("[1.0 2.0]").unwrap();
    let session = crate::runtime::iree_session::shared_session().unwrap();
    let mut jit = JitCompiler::new();
    let mut modules = Vec::new();
    for (source, expected) in [
        ("(make_shift 2.0)", [3.0, 4.0]),
        ("(make_shift 5.0)", [6.0, 7.0]),
        ("(make_shift [10.0 20.0])", [11.0, 22.0]),
        ("(make_shift [30.0 40.0])", [31.0, 42.0]),
        ("(make_shift 2.0)", [3.0, 4.0]),
        ("(make_subtract 2.0)", [-1.0, 0.0]),
    ] {
        let closure = interpreter.eval(source).unwrap();
        interpreter.env_mut().set_global("shift", closure);
        let (prepared, args) = super::specialize::prepare_function(
            &function, std::slice::from_ref(&input), interpreter.env(),
        ).unwrap().expect_ready();
        let outcome = jit.try_jit_compile(&prepared, &args, &interpreter.env().registry, &session);
        assert!(matches!(outcome, JitCompileOutcome::Compiled(_)), "{source}: {outcome:?}");
        let module = jit.module_for_args(&prepared, &args).unwrap();
        let result = session.call_typed_device(
            &format!("{}.run_global", module.module_name), &args, &module.sig.return_type,
        ).unwrap().ensure_host().unwrap();
        let Value::Tensor { data, .. } = result else { panic!() };
        assert_eq!(data.as_slice().unwrap(), &expected, "{source}");
        modules.push(module.module_name);
    }
    assert_eq!(modules[0], modules[1]);
    assert_ne!(modules[1], modules[2]);
    assert_eq!(modules[2], modules[3]);
    assert_eq!(modules[0], modules[4]);
    assert_ne!(modules[4], modules[5]);
}

#[cfg(iree_runtime)]
#[test]
fn preparation_cache_is_bounded_and_does_not_retain_captured_tensor_data() {
    let mut interpreter = crate::interpreter::eval::Interpreter::new();
    interpreter.env_mut().jit_compiler = None;
    interpreter.eval("(defn make_shift [s] (fn [y] (+ s y)))
        (defn call_shift [f x] (f x))").unwrap();
    let function = interpreter.registry_get("call_shift").unwrap().clone();
    let input = interpreter.eval("[1.0 2.0]").unwrap();
    let closure = interpreter.eval("(make_shift [10.0 20.0])").unwrap();
    let Value::Function { closure: captures, .. } = &closure else { panic!() };
    let Value::Tensor { data, .. } = &captures[0].1 else { panic!() };
    let data = std::sync::Arc::downgrade(data);
    let args = [closure, input.clone()];
    let prepared = super::specialize::prepare_function(&function, &args, interpreter.env()).unwrap().expect_ready();
    drop((prepared, args));
    assert!(data.upgrade().is_none(), "preparation must not retain tensor payloads");
    for n in 0..40 {
        let closure = interpreter.eval(&format!("(make_shift {n})")).unwrap();
        super::specialize::prepare_function(&function, &[closure, input.clone()], interpreter.env())
            .unwrap().expect_ready();
    }
    assert_eq!(interpreter.env().preparation_cache.borrow().len(), 32);
}

#[cfg(iree_runtime)]
#[test]
fn generated_closures_fall_back_without_changing_results() {
    let device = std::env::var("SHEAF_SIGNATURE_DEVICE").unwrap_or_else(|_| "cpu".into());
    crate::core::config::init(0, Some(device), false);
    let cases = [
        ("global VAG", "(def generated (value-and-grad (fn [p] (sum (* p p)))))
            (defn call_generated [x] (generated x))
            (call_generated [1.0 2.0])", true),
        ("argument VAG", "(defn call_generated [f x] (f x))
            (call_generated (value-and-grad (fn [p] (sum (* p p)))) [1.0 2.0])", true),
        ("global vmap", "(def generated (vmap (fn [x] (* x x))))
            (defn call_generated [x] (generated x))
            (call_generated [1.0 2.0])", false),
        ("argument vmap", "(defn call_generated [f x] (f x))
            (call_generated (vmap (fn [x] (* x x))) [1.0 2.0])", false),
        ("vmap captured by loss", "(let [generated (vmap (fn [x] (* x x)))]
            ((value-and-grad (fn [p] (+ (sum (* p p)) (sum (generated [3.0 4.0])))))
             [1.0 2.0]))", true),
    ];
    let mut errors = Vec::new();
    for (name, source, is_vag) in cases {
        for jit_enabled in [false, true] {
            let mut interpreter = crate::interpreter::eval::Interpreter::new();
            if !jit_enabled {
                interpreter.env_mut().jit_compiler = None;
            }
            let result = match interpreter.eval(source) {
                Ok(result) => result,
                Err(error) => {
                    errors.push(format!("{name}, JIT={jit_enabled}: {error}"));
                    continue;
                }
            };
            let tensor = if is_vag {
                let values = match result {
                    Value::Tuple(values) | Value::List(values) => values,
                    _ => panic!("{name}: {result:?}"),
                };
                let loss = if name == "vmap captured by loss" { 30.0 } else { 5.0 };
                assert_eq!(
                    values[0].to_f64().unwrap(), loss, "{name}, JIT={jit_enabled}",
                );
                values[1].ensure_host().unwrap()
            } else {
                result.ensure_host().unwrap()
            };
            let Value::Tensor { data, .. } = tensor else {
                panic!("{name}: {tensor:?}");
            };
            let expected = if is_vag { [2.0, 4.0] } else { [1.0, 4.0] };
            assert_eq!(data.shape(), &[2], "{name}, JIT={jit_enabled}");
            assert_eq!(data.as_slice().unwrap(), &expected, "{name}, JIT={jit_enabled}");
        }
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
fn traced_input_is_not_confused_with_an_equal_scalar_capture() {
    let result = run("(let [s 2.0 generated (vmap (fn [x] (* x x)))]
        ((value-and-grad (fn [p] (+ (* s (* p p)) (sum (generated [3.0 4.0]))))) 2.0))")
        .unwrap();
    let Value::List(values) = result else { panic!("{result:?}") };
    assert_eq!(values[0].to_f64().unwrap(), 33.0);
    assert_eq!(values[1].to_f64().unwrap(), 8.0);
}

#[test]
fn returned_function_retains_its_lexical_value() {
    let value = run("(defn make_shift [x] (fn [y] (+ x y)))
        (let [f (make_shift 10)] (f 1))").unwrap();
    assert!(matches!(value, Value::Int(11)), "{value:?}");
}

#[test]
fn captures_do_not_follow_later_bindings_or_caller_locals() {
    let value = run("(let [x 2 f (fn [y] (+ x y)) x 100] (f 1))").unwrap();
    assert!(matches!(value, Value::Int(3)), "{value:?}");
    let value = run("(def scale 2.0) (defn global_scale [x] (* scale x))
        (defn caller [scale] (global_scale 3.0)) (caller 100.0)").unwrap();
    assert!(matches!(value, Value::Float(6.0)), "{value:?}");
}

#[test]
fn interpreted_gradients_keep_nested_capture_values() {
    for source in [
        "(defn make_loss [s] (fn [p] (sum (* s (* p p)))))
         ((value-and-grad (make_loss 2.0)) [1.0 2.0])",
        "(defn make_scale [s] (fn [p] (* s p)))
         (defn make_loss [f] (fn [p] (sum (* p (f p)))))
         ((value-and-grad (make_loss (make_scale 2.0))) [1.0 2.0])",
    ] {
        let result = run(source).unwrap();
        let values = match result {
            Value::Tuple(values) | Value::List(values) => values,
            _ => panic!("{result:?}"),
        };
        assert_eq!(values[0].to_f64().unwrap(), 10.0);
        let value = values[1].ensure_host().unwrap();
        let Value::Tensor { data, .. } = value else { panic!("{value:?}") };
        assert_eq!(data.as_slice().unwrap(), &[4.0, 8.0], "{source}");
    }
}

#[test]
fn user_capture_names_do_not_select_internal_function_dispatch() {
    for name in ["__vag_fn__", "__vmap_fn__"] {
        let value = run(&format!("(let [{name} 2] ((fn [x] (+ {name} x)) 1))")).unwrap();
        assert!(matches!(value, Value::Int(3)), "{name}: {value:?}");
    }
}

#[test]
fn imported_definitions_keep_their_lexical_initialization_scope() {
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("sheaf-capture-module-{nonce}.shf"));
    std::fs::write(&path, "(let [scale 2.0] (defn imported_scale [x] (* scale x)))").unwrap();
    let result = run(&format!("(use {:?}) (imported_scale 3.0)", path.to_string_lossy()));
    std::fs::remove_file(path).unwrap();
    let value = result.unwrap();
    assert!(matches!(value, Value::Float(6.0)), "{value:?}");
}

#[test]
fn capture_layout_contains_only_referenced_lexical_values() {
    let value = run("(def factor 3.0)
        (let [x 2.0 unused [1.0 2.0]] (fn [y] (+ (* x y) factor)))").unwrap();
    let Value::Function { closure, .. } = value else { panic!() };
    assert_eq!(closure.len(), 1);
    assert_eq!(closure[0].0, "x");
    assert!(matches!(closure[0].1, Value::Float(2.0)));
}
