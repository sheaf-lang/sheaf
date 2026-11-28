// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use crate::core::bindings::{FreshBindings, pattern_names};
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::inference::infer_type_with_context;
use crate::lowering::stablehlo::StableHLOType;
use std::collections::{HashMap, HashSet};

/// Expand maps with a statically known collection length, evaluating the source once.
pub fn unroll_maps(expr: &CompiledExpr, types: &[(String, StableHLOType)]) -> CompiledExpr {
    let mut fresh = FreshBindings::default();
    fresh.reserve(expr);
    for (name, _) in types { fresh.reserve_name(name); }
    unroll(expr, &types.iter().cloned().collect(), &mut fresh)
}

fn unroll(
    expr: &CompiledExpr,
    types: &HashMap<String, StableHLOType>,
    fresh: &mut FreshBindings,
) -> CompiledExpr {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            let mut scope = types.clone();
            let bindings = bindings.iter().map(|(pattern, value)| {
                let value = unroll(value, &scope, fresh);
                let ty = super::tuples::random_split_length(&value).map(|n| {
                    StableHLOType::Tuple(vec![StableHLOType::f32_tensor(
                        vec![crate::core::prng::KEY_SIZE as i64]); n], None)
                }).or_else(|| infer_type_with_context(&value, &scope).ok());
                let mut names = HashSet::new();
                pattern_names(pattern, &mut names);
                for name in names { scope.remove(&name); }
                if let (BindingPattern::Simple(name), Some(ty)) = (pattern, ty) {
                    scope.insert(name.clone(), ty);
                }
                (pattern.clone(), value)
            }).collect();
            CompiledExpr::Let { bindings, body: Box::new(unroll(body, &scope, fresh)) }
        }
        CompiledExpr::Lambda { params, body, .. } => {
            let mut scope = types.clone();
            for param in params { scope.remove(param); }
            CompiledExpr::lambda(params.clone(), unroll(body, &scope, fresh))
        }
        CompiledExpr::FunctionCall { name, args, .. } if name == "map" && args.len() == 2 => {
            let function = unroll(&args[0], types, fresh);
            let source = unroll(&args[1], types, fresh);
            let layout = super::tuples::random_split_length(&source).map(|n| (n, true))
                .or_else(|| match &source {
                    CompiledExpr::Tuple(elements) => Some((elements.len(), true)),
                    CompiledExpr::Vector(elements) => Some((elements.len(), false)),
                    _ => infer_type_with_context(&source, types).ok().and_then(|ty| match ty {
                        StableHLOType::Tuple(elements, _) => Some((elements.len(), true)),
                        _ => None,
                    }),
                });
            let Some((length, tuple)) = layout else { return expr.clone() };
            let collection = fresh.fresh_name("map_source");
            let mut bindings = vec![(BindingPattern::Simple(collection.clone()), source)];
            let mut results = Vec::with_capacity(length);
            for index in 0..length {
                let element = if tuple {
                    CompiledExpr::GetTupleElement { param: collection.clone(), indices: vec![index] }
                } else {
                    CompiledExpr::FunctionCall {
                        name: "get".into(), args: vec![CompiledExpr::Symbol(collection.clone()),
                            CompiledExpr::Integer(index as i64)], loc: None,
                    }
                };
                let result = fresh.fresh_name("map_result");
                bindings.push((BindingPattern::Simple(result.clone()), CompiledExpr::LambdaCall {
                    callee: Box::new(function.clone()), args: vec![element],
                }));
                results.push(CompiledExpr::Symbol(result));
            }
            CompiledExpr::Let { bindings, body: Box::new(CompiledExpr::Tuple(results)) }
        }
        _ => expr.map_children(|child| unroll(child, types, fresh)),
    }
}
