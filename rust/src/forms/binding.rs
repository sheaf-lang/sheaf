// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Binding special forms: defn, let, fn

use crate::ast::SheafValue;
use crate::core::compiler::{CompiledExpr, CompilerContext, FunctionDef};
use crate::core::error::{SheafError, SheafResult, SourceLocation};
use crate::forms::base::{SpecialForm, check_min_arity, expect_symbol, expect_vector};

/// defn - Function definition: (defn name [params] body)
pub struct DefnForm;

impl SpecialForm for DefnForm {
    fn name(&self) -> &'static str {
        "defn"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // (defn name [params] body)
        check_min_arity("defn", args, 3, loc)?;

        let name = expect_symbol(&args[0], "defn name", loc)?;
        let params_vec = expect_vector(&args[1], "defn parameters", loc)?;

        // Extract parameter names, handling typed params:
        //   - symbol: simple param  e.g. `x`
        //   - (p :as TypeName): defparams-typed param
        //   - (x [4 2]): shape-annotated param → tensor<4x2xf32>
        let mut params: Vec<String> = Vec::new();
        let mut type_annotations: Vec<(String, String)> = Vec::new(); // (param, type_name)
        let mut shape_annotations: Vec<(String, Vec<i64>)> = Vec::new(); // (param, shape)

        for p in params_vec {
            match p {
                // Simple symbol param: x
                SheafValue::Symbol(s, _) => {
                    params.push(s.clone());
                }
                // Typed or shape-annotated param: a list
                SheafValue::List(elems, inner_loc) => {
                    if elems.len() == 3 {
                        // (p :as TypeName)
                        let pname = expect_symbol(&elems[0], "typed param name", inner_loc)?;
                        match &elems[1] {
                            SheafValue::Keyword(k, _) if k == "as" => {}
                            other => {
                                return Err(SheafError::Compile {
                                    message: format!(
                                        "defn typed param: expected :as keyword, got {}",
                                        other
                                    ),
                                    location: inner_loc.clone(),
                                });
                            }
                        };
                        let type_name = expect_symbol(&elems[2], "param type name", inner_loc)?;
                        params.push(pname.to_string());
                        type_annotations.push((pname.to_string(), type_name.to_string()));
                    } else if elems.len() == 2 {
                        // (x [4 2]) — shape annotation
                        let pname = expect_symbol(&elems[0], "shape-annotated param name", inner_loc)?;
                        let shape = match &elems[1] {
                            SheafValue::Vector(dims, dim_loc) => {
                                dims.iter()
                                    .map(|d| match d {
                                        SheafValue::Float(n, _) => Ok(*n as i64),
                                        SheafValue::Integer(n, _) => Ok(*n),
                                        other => Err(SheafError::Compile {
                                            message: format!(
                                                "defn shape annotation: expected integer dimension, got {}",
                                                other
                                            ),
                                            location: dim_loc.clone(),
                                        }),
                                    })
                                    .collect::<SheafResult<Vec<i64>>>()?
                            }
                            other => {
                                return Err(SheafError::Compile {
                                    message: format!(
                                        "defn param: expected shape vector [dim ...], got {}",
                                        other
                                    ),
                                    location: inner_loc.clone(),
                                });
                            }
                        };
                        params.push(pname.to_string());
                        shape_annotations.push((pname.to_string(), shape));
                    } else {
                        return Err(SheafError::Compile {
                            message: format!(
                                "defn param: expected (name :as Type) or (name [shape]), got list with {} elements",
                                elems.len()
                            ),
                            location: inner_loc.clone(),
                        });
                    }
                }
                other => {
                    return Err(SheafError::Compile {
                        message: format!(
                            "defn param: expected symbol, (name :as Type), or (name [shape]), got {}",
                            other
                        ),
                        location: loc.clone(),
                    });
                }
            }
        }

        // Body is the third argument - compile it
        let body_ast = args[2].clone();

        // Add parameters to local scope temporarily
        let saved_locals = compiler.local_vars.clone();
        for param in &params {
            compiler.local_vars.insert(param.clone(), body_ast.clone()); // Placeholder
        }

        // Register type annotations as __type__<param> in local_vars
        // so with-params can look them up
        for (param, type_name) in &type_annotations {
            let key = format!("__type__{}", param);
            compiler
                .local_vars
                .insert(key, SheafValue::Symbol(type_name.clone(), loc.clone()));
        }

        // Compile body
        let body_compiled = compiler.compile(&body_ast)?;

        // Restore local scope
        compiler.local_vars = saved_locals;

        // Build known param types from defparams and shape annotations
        let mut known_param_types: Vec<(String, crate::compiler::stablehlo::StableHLOType)> =
            Vec::new();
        for (param, type_name) in &type_annotations {
            if let Some(layout) = compiler.param_types.get(type_name) {
                let tuple_ty = crate::forms::ml::param_layout_to_stablehlo_type(layout);
                known_param_types.push((param.clone(), tuple_ty));
            }
        }
        for (param, shape) in &shape_annotations {
            known_param_types.push((
                param.clone(),
                crate::compiler::stablehlo::StableHLOType::f32_tensor(shape.clone()),
            ));
        }

        // Infer function signature with known types for typed params
        let mut signature = crate::core::inference::infer_function_signature_with_known(
            compiler,
            &params,
            &body_compiled,
            &known_param_types,
        )?;

        // Override param types from known_param_types (infer may have defaulted them)
        for (param, tuple_ty) in &known_param_types {
            if let Some(idx) = params.iter().position(|p| p == param) {
                signature.param_types[idx] = tuple_ty.clone();
            }
        }

        // If body returns a Dict, store sorted keys for IREE result reconstruction
        if let CompiledExpr::Dict(pairs) = &body_compiled {
            let mut keys: Vec<String> = pairs
                .iter()
                .filter_map(|(k, _)| {
                    if let CompiledExpr::Keyword(s) = k {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect();
            keys.sort();
            signature.return_dict_keys = Some(keys);
        }

        // Register the function in the compiler with compiled body and signature
        compiler.registry.insert(
            name.to_string(),
            FunctionDef {
                name: name.to_string(),
                params,
                body: body_ast,
                body_compiled: Some(body_compiled),
                signature: Some(signature),
                vmfb_session_idx: None,
            },
        );

        // defn returns nil (side effect: registers function)
        Ok(CompiledExpr::Nil)
    }
}

/// let - Local bindings: (let [var1 val1 var2 val2] body)
pub struct LetForm;

impl SpecialForm for LetForm {
    fn name(&self) -> &'static str {
        "let"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // (let [bindings] body)
        check_min_arity("let", args, 2, loc)?;

        let bindings_vec = expect_vector(&args[0], "let bindings", loc)?;

        if bindings_vec.len() % 2 != 0 {
            return Err(crate::core::error::SheafError::Compile {
                message: "let bindings must have even number of elements (name value pairs)"
                    .to_string(),
                location: loc.clone(),
            });
        }

        // Save current local_vars state
        let saved_locals = compiler.local_vars.clone();

        // Process bindings in pairs
        let mut compiled_bindings = Vec::new();
        for i in (0..bindings_vec.len()).step_by(2) {
            let pattern = &bindings_vec[i];
            let value = &bindings_vec[i + 1];
            let compiled_value = compiler.compile(value)?;

            match pattern {
                // Simple symbol binding: [x expr]
                SheafValue::Symbol(name, _) => {
                    compiler.local_vars.insert(name.clone(), value.clone());
                    compiled_bindings.push((name.clone(), compiled_value));
                }
                // Vector destructuring: [[a b] expr]
                SheafValue::Vector(names, inner_loc) => {
                    let sym_names: Vec<String> = names
                        .iter()
                        .map(|n| {
                            expect_symbol(n, "let destructuring name", inner_loc)
                                .map(|s| s.to_string())
                        })
                        .collect::<SheafResult<_>>()?;
                    // Register each name in local scope as a symbol
                    for n in &sym_names {
                        compiler.local_vars.insert(
                            n.clone(),
                            SheafValue::Symbol(n.clone(), inner_loc.clone()),
                        );
                    }
                    // Encode the pattern as "[a b c]" — interpreter decodes it
                    let pattern_key = format!("[{}]", sym_names.join(" "));
                    compiled_bindings.push((pattern_key, compiled_value));
                }
                other => {
                    return Err(SheafError::Compile {
                        message: format!(
                            "let binding name must be a symbol or destructuring vector, got {}",
                            other
                        ),
                        location: loc.clone(),
                    });
                }
            }
        }

        // Compile body with bindings in scope
        let body = &args[1];
        let compiled_body = compiler.compile(body)?;

        // Restore local_vars (let is scoped)
        compiler.local_vars = saved_locals;

        Ok(CompiledExpr::Let {
            bindings: compiled_bindings,
            body: Box::new(compiled_body),
        })
    }
}

/// fn - Anonymous function: (fn [params] body)
pub struct FnForm;

impl SpecialForm for FnForm {
    fn name(&self) -> &'static str {
        "fn"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // (fn [params] body)
        check_min_arity("fn", args, 2, loc)?;

        let params_vec = expect_vector(&args[0], "fn parameters", loc)?;
        let param_names: Vec<String> = params_vec
            .iter()
            .map(|p| {
                p.as_symbol()
                    .map(|s| s.to_string())
                    .ok_or_else(|| SheafError::Compile {
                        message: format!("fn: parameter must be a symbol, got {}", p),
                        location: loc.clone(),
                    })
            })
            .collect::<SheafResult<_>>()?;

        // Compile the body with params registered as local variables so they
        // resolve as Symbol nodes (not FunctionRef).
        let saved_locals = compiler.local_vars.clone();
        for p in &param_names {
            compiler.local_vars.insert(
                p.clone(),
                crate::ast::SheafValue::Symbol(p.clone(), loc.clone()),
            );
        }

        let body = if args.len() == 2 {
            compiler.compile(&args[1])?
        } else {
            // Multiple body expressions → implicit do
            let exprs: SheafResult<Vec<CompiledExpr>> =
                args[1..].iter().map(|e| compiler.compile(e)).collect();
            CompiledExpr::Do(exprs?)
        };

        compiler.local_vars = saved_locals;

        Ok(CompiledExpr::Lambda {
            params: param_names,
            body: Box::new(body),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SourceLocation;

    #[test]
    fn test_defn_form_name() {
        let form = DefnForm;
        assert_eq!(form.name(), "defn");
    }

    #[test]
    fn test_let_form_name() {
        let form = LetForm;
        assert_eq!(form.name(), "let");
    }
}
