// Copyright (c) 2025-2026 Damien Boureille
// Licensed under the MIT License.

//! Interpreted value-and-grad using the shared reverse-mode engine.

use crate::autodiff::reverse::{GradientOutput, reverse_grad};
use crate::autodiff::trace::{LeafMap, numeric_shape, trace_function};
use crate::core::error::SheafError;
use crate::core::expr::CompiledExpr;
use crate::interpreter::env::{Env, runtime_error};
use crate::interpreter::value::Value;
use std::collections::{BTreeMap, HashMap};

pub(super) fn eval_value_and_grad_hof(args: &[Value], _env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 1 {
        return Err(runtime_error("value-and-grad: expected exactly 1 argument (the function)"));
    }
    Ok(Value::Function {
        kind: super::value::FunctionKind::ValueAndGrad,
        name: None,
        params: vec!["__vag_params__".to_string()],
        body: CompiledExpr::Symbol("__vag_params__".to_string()).into(),
        closure: vec![("__vag_fn__".to_string(), args[0].clone())],
    })
}

pub(super) fn eval_value_and_grad_call(
    func: &Value,
    params: &Value,
    env: &mut Env,
) -> Result<Value, SheafError> {
    #[cfg(iree_runtime)]
    if env.tracer.is_none() && env.jit_compiler.is_some() {
        use crate::runtime::jit::JitVagOutcome;
        match super::iree_dispatch::try_jit_vag(func, params, env) {
            JitVagOutcome::Success(result) => return result,
            JitVagOutcome::NotAttempted | JitVagOutcome::NotCompiled(
                crate::core::jit_diagnostic::JitDiagnostic::Interpreted(_)
                | crate::core::jit_diagnostic::JitDiagnostic::Unsupported(_, _)) => {}
            JitVagOutcome::NotCompiled(crate::core::jit_diagnostic::JitDiagnostic::UserError(error)) => return Err(error),
            JitVagOutcome::NotCompiled(crate::core::jit_diagnostic::JitDiagnostic::Environment { message, hint }) =>
                return Err(runtime_error(format!("{message}\n  = hint: {hint}"))),
            JitVagOutcome::NotCompiled(crate::core::jit_diagnostic::JitDiagnostic::Internal(reason)) => return Err(runtime_error(format!(
                "value-and-grad: JIT compilation failed: {reason} \
                 (run with -vv for details). This is a bug in Sheaf.",
            ))),
        }
    }
    let Value::Function { params: fn_params, .. } = func else {
        return Err(runtime_error("value-and-grad: expected a function"));
    };
    if fn_params.len() != 1 {
        return Err(runtime_error("value-and-grad: function must take one parameter"));
    }

    env.push_scope();
    let result = (|| {
        let mut leaves = LeafMap::new(env);
        let tree = ParameterTree::bind(params, &mut leaves, env);
        let wrt = leaves.leaves.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
        let traced = trace_function(func, tree.expression(), env, &mut leaves)?;
        let loss = scalar_from_value(&traced.value)?;
        let reverse = reverse_grad(&traced.bindings, &traced.output, &wrt, &traced.shapes)?;
        for (name, expr) in reverse.backward_bindings {
            let value = super::eval(&expr, env)?;
            env.set(&name, value);
        }
        let gradient = tree.gradient(&reverse.gradients, env)?;
        Ok(Value::List(vec![Value::Float(loss), gradient]))
    })();
    env.pop_scope();
    result
}

fn scalar_from_value(value: &Value) -> Result<f32, SheafError> {
    match &*value.ensure_host_cow()? {
        Value::Float(value) => Ok(*value),
        Value::Int(value) => Ok(*value as f32),
        Value::Tensor { data, .. } if data.ndim() == 0 => Ok(data[ndarray::IxDyn(&[])]),
        _ => Err(runtime_error("value-and-grad: loss must return a scalar")),
    }
}

enum ParameterTree {
    Leaf(String, Value),
    Dict(Vec<(String, ParameterTree)>),
    List(String, Vec<ParameterTree>),
    Tuple(Vec<ParameterTree>),
    Constant(String),
}

impl ParameterTree {
    fn bind(value: &Value, leaves: &mut LeafMap, env: &mut Env) -> Self {
        match value {
            Value::Tensor { .. } | Value::DeviceBuffer(_) | Value::Float(_) | Value::Int(_) => {
                let name = leaves.register(value.clone(), env);
                Self::Leaf(name, value.clone())
            }
            Value::Dict(values) => Self::Dict(values.iter().map(|(key, value)| {
                (key.clone(), Self::bind(value, leaves, env))
            }).collect()),
            Value::List(values) => {
                let items = values.iter().map(|value| Self::bind(value, leaves, env)).collect::<Vec<_>>();
                let expressions = items.iter().map(Self::expression).collect();
                let name = leaves.store_list(value.clone(), expressions, env);
                Self::List(name, items)
            }
            Value::Tuple(values) => Self::Tuple(values.iter().map(|value| {
                Self::bind(value, leaves, env)
            }).collect()),
            value => Self::Constant(leaves.store(value.clone(), env)),
        }
    }

    fn expression(&self) -> CompiledExpr {
        match self {
            Self::Leaf(name, _) | Self::Constant(name) => CompiledExpr::Symbol(name.clone()),
            Self::Dict(values) => CompiledExpr::Dict(values.iter().map(|(key, tree)| {
                (CompiledExpr::Keyword(key.clone()), tree.expression())
            }).collect()),
            Self::List(name, _) => CompiledExpr::Symbol(name.clone()),
            Self::Tuple(values) => CompiledExpr::Tuple(values.iter().map(Self::expression).collect()),
        }
    }

    fn gradient(
        &self,
        outputs: &HashMap<String, GradientOutput>,
        env: &mut Env,
    ) -> Result<Value, SheafError> {
        match self {
            Self::Leaf(name, value) => {
                let gradient = match crate::autodiff::reverse::gradient_output(outputs, name)? {
                    GradientOutput::Computed(name) => env.get(name)?,
                    GradientOutput::ProvenZero => crate::interpreter::builtins::tree_zeros(value),
                };
                let gradient_shape = numeric_shape(&gradient);
                if gradient_shape != numeric_shape(value) {
                    return Err(runtime_error(format!(
                        "value-and-grad: gradient shape {gradient_shape:?} does not match parameter {:?}",
                        numeric_shape(value),
                    )));
                }
                if matches!(value, Value::Float(_) | Value::Int(_)) {
                    return gradient.to_f32().map(Value::Float)
                        .ok_or_else(|| runtime_error("value-and-grad: expected a scalar gradient"));
                }
                crate::interpreter::builtins::cast_like(&gradient, value)
            }
            Self::Dict(values) => values.iter().map(|(key, tree)| {
                Ok((key.clone(), tree.gradient(outputs, env)?))
            }).collect::<Result<BTreeMap<_, _>, SheafError>>().map(Value::Dict),
            Self::List(_, values) => values.iter().map(|tree| {
                tree.gradient(outputs, env)
            }).collect::<Result<Vec<_>, _>>().map(Value::List),
            Self::Tuple(values) => values.iter().map(|tree| {
                tree.gradient(outputs, env)
            }).collect::<Result<Vec<_>, _>>().map(Value::Tuple),
            Self::Constant(_) => Ok(Value::Nil),
        }
    }
}
