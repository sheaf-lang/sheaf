// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Exact integer and boolean inputs shared by specialization and cache lookup.

use crate::core::bindings::{Replacements, replace_bindings};
use crate::core::expr::CompiledExpr;
use crate::interpreter::value::Value;
use crate::core::trace::value_to_param_layout;
use crate::lowering::config::{layout_to_index_map, lower_get_calls};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum StaticScalar {
    Integer(i64),
    Boolean(bool),
}

impl StaticScalar {
    pub(crate) fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Int(value) => Some(Self::Integer(*value)),
            Value::Bool(value) => Some(Self::Boolean(*value)),
            _ => None,
        }
    }

    pub(crate) fn expression(&self) -> CompiledExpr {
        match self {
            Self::Integer(value) => CompiledExpr::Integer(*value),
            Self::Boolean(value) => CompiledExpr::Boolean(*value),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct StaticInput {
    pub argument: usize,
    pub indices: Vec<usize>,
    pub value: StaticScalar,
}

pub(crate) fn collect(args: &[Value]) -> Vec<StaticInput> {
    fn visit(value: &Value, argument: usize, indices: &mut Vec<usize>, output: &mut Vec<StaticInput>) {
        if let Some(value) = StaticScalar::from_value(value) {
            output.push(StaticInput { argument, indices: indices.clone(), value });
            return;
        }
        match value {
            Value::Dict(values) => {
                for (index, value) in values.values().enumerate() {
                    indices.push(index);
                    visit(value, argument, indices, output);
                    indices.pop();
                }
            }
            // Numeric lists use the tensor ABI, not scalar option slots.
            Value::List(_) if value.numeric_list_elements().is_some() => {}
            Value::List(values) | Value::Tuple(values) => {
                for (index, value) in values.iter().enumerate() {
                    indices.push(index);
                    visit(value, argument, indices, output);
                    indices.pop();
                }
            }
            _ => {}
        }
    }
    let mut output = Vec::new();
    for (argument, value) in args.iter().enumerate() {
        visit(value, argument, &mut Vec::new(), &mut output);
    }
    output
}

pub(crate) fn specialize(body: &CompiledExpr, params: &[String], args: &[Value]) -> CompiledExpr {
    let inputs = collect(args);
    let mut body = body.clone();
    for (argument, param) in params.iter().enumerate() {
        if !inputs.iter().any(|input| input.argument == argument) {
            continue;
        }
        let Some(value) = args.get(argument) else { continue };
        if let Some(layout) = value_to_param_layout(param, value) {
            body = lower_get_calls(&body, param, &layout_to_index_map(&layout));
        }
    }
    apply(&body, params, &inputs)
}

pub(crate) fn apply(body: &CompiledExpr, params: &[String], inputs: &[StaticInput]) -> CompiledExpr {
    let mut replacements = Replacements::new();
    for input in inputs {
        let Some(param) = params.get(input.argument) else { continue };
        replacements.entry(param.clone()).or_default().insert(input.indices.clone(), input.value.expression());
    }
    replace_bindings(body, &replacements)
}
