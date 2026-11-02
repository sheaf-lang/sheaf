// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Binding special forms: defn, let, fn

use crate::core::ast::SheafValue;
use crate::core::expr::{BindingPattern, CompiledExpr, CompilerContext, FunctionDef};
use crate::core::error::{SheafError, SheafResult, SourceLocation};
use crate::core::macro_engine::MacroDef;
use crate::forms::base::{SpecialForm, check_min_arity, expect_symbol, expect_vector};

// AST-only helper errors must not hide invalid forms later in the body.
fn validate_function_body(expr: &SheafValue) -> SheafResult<()> {
    match expr {
        SheafValue::List(values, loc) => {
            if let Some(SheafValue::Symbol(name, _)) = values.first() {
                let message = match name.as_str() {
                    "quote" | "quasiquote" => return Ok(()),
                    "def" => Some("def is not allowed in a function body; define globals at module level"),
                    "defn" => Some("defn is not allowed in a function body; define functions at module level"),
                    "fn" if matches!(values.get(1), Some(SheafValue::Symbol(_, _))) => {
                        Some("named fn is not supported, use defn at top level for recursion")
                    }
                    _ => None,
                };
                if let Some(message) = message {
                    return Err(SheafError::InvalidSourceForm {
                        message: message.to_string(), location: loc.clone(),
                    });
                }
            }
            values.iter().try_for_each(validate_function_body)
        }
        SheafValue::Vector(values, _) => values.iter().try_for_each(validate_function_body),
        SheafValue::Dict(values, _) => values.iter().try_for_each(|(key, value)| {
            validate_function_body(key)?;
            validate_function_body(value)
        }),
        SheafValue::Unquote(value, _) | SheafValue::UnquoteSplicing(value, _) => {
            validate_function_body(value)
        }
        _ => Ok(()),
    }
}

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
        if compiler.function_depth > 0 {
            return Err(SheafError::InvalidSourceForm {
                message: "defn is not allowed in a function body; define functions at module level".to_string(),
                location: loc.clone(),
            });
        }
        // (defn name [params] body)
        check_min_arity("defn", args, 3, loc)?;

        let name = expect_symbol(&args[0], "defn name", loc)?;

        // Prevent redefinition of user-defined functions
        if compiler.registry.contains_key(name) {
            return Err(SheafError::Compile {
                message: format!(
                    "Function '{}' is already defined. Redefinition is not allowed to prevent shadowing bugs.",
                    name
                ),
                location: loc.clone(),
            });
        }

        let params_vec = expect_vector(&args[1], "defn parameters", loc)?;

        // Extract parameter names, handling typed params:
        //   - symbol: simple param  e.g. `x`
        //   - (p :as TypeName): typed param
        //   - (x [4 2]): shape-annotated param -> tensor<4x2xf32>
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
                        // (x [4 2]) -- shape annotation
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

        // Body is everything after [params], with optional :jit skip.
        // Multiple body forms get an implicit (do ...) wrapper.
        let body_start = if let SheafValue::Keyword(k, _) = &args[2] {
            if k == "jit" && args.len() > 3 {
                eprintln!("warning: :jit annotation is not required and deprecated in Sheaf v2");
                3
            } else {
                2
            }
        } else {
            2
        };

        let body_ast = if args.len() - body_start == 1 {
            args[body_start].clone()
        } else {
            // Multiple body forms -> implicit do
            let mut do_forms = vec![SheafValue::Symbol("do".to_string(), loc.clone())];
            do_forms.extend(args[body_start..].iter().cloned());
            SheafValue::List(do_forms, loc.clone())
        };

        // Add parameters to local scope temporarily as Symbol placeholders so
        // they resolve to CompiledExpr::Symbol nodes (not the body itself,
        // which would cause infinite recompilation). Mirrors FnForm.
        let saved_locals = compiler.local_vars.clone();
        let was_global = compiler.global_names.contains(name);
        compiler.global_names.insert(name.to_string());
        compiler.local_vars.remove(name);
        for param in &params {
            compiler.local_vars.insert(
                param.clone(),
                SheafValue::Symbol(param.clone(), loc.clone()),
            );
        }

        // Register type annotations as __type__<param> in local_vars
        // so with-params can look them up
        for (param, type_name) in &type_annotations {
            let key = format!("__type__{}", param);
            compiler
                .local_vars
                .insert(key, SheafValue::Symbol(type_name.clone(), loc.clone()));
        }

        // Compile body. May fail for macro helper functions that reference
        // symbols only available at macro expansion time (e.g., transform-layer
        // uses bare `p` as AST data). In that case, register with body_compiled: None;
        // the mini-eval can still call the function via its AST body.
        compiler.function_depth += 1;
        let compile_result = validate_function_body(&body_ast)
            .and_then(|()| compiler.compile(&body_ast));
        compiler.function_depth -= 1;

        // Restore local scope (must happen even if compile failed)
        compiler.local_vars = saved_locals;

        let (body_compiled_opt, signature_opt, known_param_types, compile_err) = match compile_result {
            Ok(body_compiled) => {
                // Build known param types from type and shape annotations
                let mut known_param_types: Vec<(
                    String,
                    crate::lowering::stablehlo::StableHLOType,
                )> = Vec::new();
                for (param, type_name) in &type_annotations {
                    if let Some(layout) = compiler.param_types.get(type_name) {
                        let tuple_ty = crate::forms::ml::param_layout_to_stablehlo_type(layout);
                        known_param_types.push((param.clone(), tuple_ty));
                    }
                }
                for (param, shape) in &shape_annotations {
                    known_param_types.push((
                        param.clone(),
                        crate::lowering::stablehlo::StableHLOType::f32_tensor(shape.clone()),
                    ));
                }

                // Infer function signature with known types for typed params
                let mut signature =
                    crate::core::inference::infer_function_signature_with_known(
                        &params,
                        &body_compiled,
                        &known_param_types,
                    )
                    .ok();

                if let Some(ref mut sig) = signature {
                    // Override param types from known_param_types
                    for (param, tuple_ty) in &known_param_types {
                        if let Some(idx) = params.iter().position(|p| p == param) {
                            sig.param_types[idx] = tuple_ty.clone();
                        }
                    }

                    // If the body returns a dict, store its sorted keys for IREE result reconstruction.
                    if let Some(keys) =
                        crate::core::inference::find_return_dict_keys(&body_compiled)
                    {
                        sig.return_dict_keys = Some(keys);
                    }
                }

                (Some(body_compiled), signature, known_param_types, None)
            }
            Err(error @ SheafError::InvalidSourceForm { .. }) => {
                if !was_global {
                    compiler.global_names.remove(name);
                }
                return Err(error);
            }
            Err(_) => {
                // Body compilation failed -- register as AST-only function.
                // The mini-eval can still call it via its AST body (macro helpers).
                (None, None, vec![], None)
            }
        };

        let body_compiled = body_compiled_opt.map(std::rc::Rc::new);
        let definition = body_compiled.as_ref().map(|body| {
            CompiledExpr::lambda_with_shared_body(params.clone(), body.clone())
        });
        let initialization = definition.filter(|value| {
            matches!(value, CompiledExpr::Lambda { captures, .. } if !captures.is_empty())
        }).map(|value| CompiledExpr::Def {
            name: name.to_string(),
            value: Box::new(value),
        });

        // Register the function in the compiler with compiled body and signature
        compiler.registry.insert(
            name.to_string(),
            FunctionDef {
                name: name.to_string(),
                params,
                body: body_ast,
                body_compiled,
                requires_initialization: initialization.is_some(),
                signature: signature_opt.filter(|_| initialization.is_none()),
                vmfb_module_name: None,
                known_param_types,
                compile_error: compile_err,
            },
        );

        Ok(initialization.unwrap_or_else(|| CompiledExpr::FunctionRef(name.to_string())))
    }
}

/// def - Global constant: (def name value)
pub struct DefForm;

impl SpecialForm for DefForm {
    fn name(&self) -> &'static str {
        "def"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        if compiler.function_depth > 0 {
            return Err(SheafError::InvalidSourceForm {
                message: "def is not allowed in a function body; define globals at module level".to_string(),
                location: loc.clone(),
            });
        }
        check_min_arity("def", args, 2, loc)?;

        let name = expect_symbol(&args[0], "def name", loc)?;

        if compiler.registry.contains_key(name) || compiler.global_names.contains(name)
            || compiler.local_vars.contains_key(name) {
            return Err(SheafError::Compile {
                message: format!(
                    "'{}' is already defined. Redefinition is not allowed.",
                    name
                ),
                location: loc.clone(),
            });
        }

        let value = compiler.compile(&args[1])?;

        compiler.global_names.insert(name.to_string());
        compiler.env.insert(name.to_string(), args[1].clone());
        if let CompiledExpr::Lambda { params, captures, body } = &value
            && captures.is_empty() {
            compiler.registry.insert(name.to_string(), FunctionDef {
                name: name.to_string(),
                params: params.clone(),
                body: args[1].clone(),
                body_compiled: Some(body.clone()),
                requires_initialization: true,
                signature: None,
                vmfb_module_name: None,
                known_param_types: Vec::new(),
                compile_error: None,
            });
        }

        Ok(CompiledExpr::Def {
            name: name.to_string(),
            value: Box::new(value),
        })
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
                    compiled_bindings.push((BindingPattern::Simple(name.clone()), compiled_value));
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
                    let pattern = BindingPattern::Destructure(
                        sym_names.iter().map(|n| BindingPattern::Simple(n.clone())).collect()
                    );
                    compiled_bindings.push((pattern, compiled_value));
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

        // Compile body with bindings in scope (implicit do for multiple body exprs)
        let compiled_body = if args.len() == 2 {
            compiler.compile(&args[1])?
        } else {
            let exprs: SheafResult<Vec<CompiledExpr>> =
                args[1..].iter().map(|e| compiler.compile(e)).collect();
            CompiledExpr::Do(exprs?)
        };

        // Restore local_vars (let is scoped)
        compiler.local_vars = saved_locals;

        Ok(CompiledExpr::Let {
            bindings: compiled_bindings,
            body: Box::new(compiled_body),
        })
    }
}

/// defmacro - Macro definition: (defmacro name [params] body-template)
pub struct DefmacroForm;

impl SpecialForm for DefmacroForm {
    fn name(&self) -> &'static str {
        "defmacro"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        check_min_arity("defmacro", args, 3, loc)?;

        let name = expect_symbol(&args[0], "defmacro name", loc)?;
        let params_vec = expect_vector(&args[1], "defmacro parameters", loc)?;

        // Parse parameters, separating positional from rest (&)
        let mut positional_params = Vec::new();
        let mut rest_param = None;
        let mut saw_ampersand = false;

        for p in params_vec {
            let sym = expect_symbol(p, "defmacro parameter", loc)?;
            if sym == "&" {
                saw_ampersand = true;
            } else if saw_ampersand {
                rest_param = Some(sym.to_string());
                break;
            } else {
                positional_params.push(sym.to_string());
            }
        }

        let body_template = args[2].clone();

        compiler.macro_engine.macros.insert(
            name.to_string(),
            MacroDef {
                name: name.to_string(),
                positional_params,
                rest_param,
                body_template,
            },
        );

        Ok(CompiledExpr::Nil)
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
        if matches!(args.first(), Some(SheafValue::Symbol(_, _))) {
            return Err(SheafError::InvalidSourceForm {
                message: "named fn is not supported, use defn at top level for recursion".to_string(),
                location: loc.clone(),
            });
        }
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
                crate::core::ast::SheafValue::Symbol(p.clone(), loc.clone()),
            );
        }

        compiler.function_depth += 1;
        let result = args[1..].iter().try_for_each(validate_function_body).and_then(|()| {
            if args.len() == 2 {
                compiler.compile(&args[1])
            } else {
                args[1..].iter().map(|expr| compiler.compile(expr))
                    .collect::<SheafResult<Vec<_>>>().map(CompiledExpr::Do)
            }
        });
        compiler.function_depth -= 1;
        compiler.local_vars = saved_locals;
        result.map(|body| CompiledExpr::lambda(param_names, body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_function_forms_are_not_retained_as_ast_helpers() {
        for (source, expected) in [
            ("(fn self [n] n)", "named fn is not supported"),
            ("(defn outer [] (fn self [n] n))", "named fn is not supported"),
            ("(fn [x] (def hidden x))", "def is not allowed in a function body"),
            ("(defn outer [x] (def hidden x))", "def is not allowed in a function body"),
            ("(defn outer [] (get) (def hidden 1))", "def is not allowed in a function body"),
            ("(defn outer [] (get) (defn inner [] 1))", "defn is not allowed"),
            ("(defn outer [] (get) (fn self [n] n))", "named fn is not supported"),
            ("(fn [x] (defn inner [] x))", "defn is not allowed in a function body"),
            ("(defn outer [] (if false (defn inner [] 1) 2))", "defn is not allowed"),
            ("(defmacro define-x [] '(def hidden 1)) (fn [x] (define-x))", "def is not allowed"),
        ] {
            let mut compiler = CompilerContext::new();
            let parsed = crate::core::parse(source, "<source-forms>").unwrap();
            let error = parsed.iter().try_for_each(|expr| compiler.compile(expr).map(|_| ())).unwrap_err();
            assert!(matches!(error, SheafError::InvalidSourceForm { .. }), "{error}");
            assert!(error.short_message().contains(expected), "{error}");
            assert!(crate::core::error_format::format_error(&error).contains(expected));
            assert_eq!(compiler.function_depth, 0);
            assert!(compiler.local_vars.is_empty());
            for name in ["outer", "inner", "hidden"] {
                assert!(!compiler.registry.contains_key(name));
                assert!(!compiler.env.contains_key(name));
            }
            let valid = crate::core::parse("(defn valid [x] x)", "<source-forms>").unwrap();
            compiler.compile(&valid[0]).unwrap();
        }
    }

    #[test]
    fn quoted_definitions_and_macro_helpers_remain_valid() {
        let mut compiler = CompilerContext::new();
        for source in [
            "(fn [] '(def x 1))", "(fn [] (quote (defn x [] 1)))",
            "(defn helper [x] macro-data)",
            "(defmacro identity-macro [x] x)", "(fn [x] (identity-macro x))",
        ] {
            let parsed = crate::core::parse(source, "<source-forms>").unwrap();
            compiler.compile(&parsed[0]).unwrap();
            assert_eq!(compiler.function_depth, 0);
            assert!(compiler.local_vars.is_empty());
        }
        assert_eq!(compiler.registry["helper"].body.as_symbol(), Some("macro-data"));
    }

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
