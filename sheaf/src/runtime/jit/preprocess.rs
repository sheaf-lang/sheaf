// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! JIT preprocessing for nested value-and-grad lambdas.

use super::support::inject_tuple_shapes;
use super::*;

pub(super) struct VagPreprocessContext<'a> {
    pub(super) registry: &'a HashMap<String, FunctionDef>,
    pub(super) constants: &'a HashMap<(String, Vec<usize>), i64>,
    pub(super) param_shapes: &'a HashMap<String, Vec<i64>>,
    pub(super) param_index_maps: &'a [ParamIndexMap],
    pub(super) known_types: &'a [(String, StableHLOType)],
    pub(super) arity_err: &'a std::cell::Cell<Option<SheafError>>,
}

impl JitCompiler {
    /// Preprocess VAG lambda bodies found in a function body.
    ///
    /// When the body contains `((value-and-grad loss-fn) params)`, the Lambda
    /// inside `__value-and-grad-hof__` hasn't been preprocessed by the standard
    /// pipeline (inline_function_calls doesn't recurse into Lambdas). This method
    /// finds the Lambda, applies the same preprocessing (inline, lower_gets,
    /// resolve_static_constants), and puts it back.
    pub(super) fn preprocess_vag_lambda(
        &self,
        body: &CompiledExpr,
        context: &VagPreprocessContext<'_>,
    ) -> CompiledExpr {
        preprocess_vag_lambda_rec(body, context)
    }
}

fn preprocess_vag_lambda_rec(
    expr: &CompiledExpr,
    context: &VagPreprocessContext<'_>,
) -> CompiledExpr {
    let recurse = |e: &CompiledExpr| preprocess_vag_lambda_rec(e, context);

    match expr {
        CompiledExpr::LambdaCall { callee, args } => {
            let new_callee = recurse(callee);
            let new_args: Vec<CompiledExpr> = args.iter().map(&recurse).collect();

            if let CompiledExpr::FunctionCall {
                name,
                args: inner_args,
                ..
            } = &new_callee
                && name == "__value-and-grad-hof__" && inner_args.len() == 1 {
                    let preprocessed_lambda =
                        preprocess_one_vag_lambda(&inner_args[0], context);
                    return CompiledExpr::LambdaCall {
                        callee: Box::new(CompiledExpr::FunctionCall {
                            name: "__value-and-grad-hof__".to_string(),
                            args: vec![preprocessed_lambda],
                            loc: None,
                        }),
                        args: new_args,
                    };
                }
            CompiledExpr::LambdaCall {
                callee: Box::new(new_callee),
                args: new_args,
            }
        }
        _ => expr.map_children(recurse),
    }
}

fn preprocess_one_vag_lambda(
    lambda_expr: &CompiledExpr,
    context: &VagPreprocessContext<'_>,
) -> CompiledExpr {
    let registry = context.registry;
    let constants = context.constants;
    let param_shapes = context.param_shapes;
    let param_index_maps = context.param_index_maps;
    let known_types = context.known_types;
    let arity_err = context.arity_err;
    let (params, body) = match lambda_expr {
        CompiledExpr::Lambda { params, body, .. } => (params.clone(), body.as_ref().clone()),
        _ => return lambda_expr.clone(),
    };

    let mut lambda_param_shapes: HashMap<String, Vec<i64>> = param_shapes.clone();
    for (name, ty) in known_types {
        if params.contains(name) {
            inject_tuple_shapes(name, ty, &[], &mut lambda_param_shapes);
        }
    }

    // Propagate destructuring errors through the recursive traversal.
    let body = match preprocess_body(
        &body,
        registry,
        param_index_maps,
        constants,
        &lambda_param_shapes,
        false,
    ) {
        Ok(b) => b,
        Err(err) => {
            // Preserve the original lambda while the caller reports the error.
            arity_err.set(Some(err));
            return lambda_expr.clone();
        }
    };

    CompiledExpr::Lambda {
        params,
        body: Box::new(body),
    }
}

fn preprocess_body(
    body: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
    param_index_maps: &[ParamIndexMap],
    constants: &HashMap<(String, Vec<usize>), i64>,
    param_shapes: &HashMap<String, Vec<i64>>,
    skip_lambda: bool,
) -> Result<CompiledExpr, SheafError> {
    let mut body = crate::autodiff::inline_function_calls(body, registry);

    for (param_name, index_map) in param_index_maps {
        body = lower_get_calls(&body, param_name, index_map);
    }
    body = lower_inlined_gets(&body, param_index_maps);

    body = resolve_static_constants(&body, constants, param_shapes, skip_lambda);

    // Arity mismatch and other destructuring errors bubble up as SheafError::Compile.
    crate::lowering::transforms::lower_tuples_and_destructuring(body, param_shapes)
}
