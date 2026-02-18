// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! ML-specific special forms: defparams, with-params

use crate::ast::SheafValue;
use crate::compiler::stablehlo::StableHLOType;
use crate::core::compiler::{CompiledExpr, CompilerContext, ParamField, ParamLayout};
use crate::core::error::{SheafError, SheafResult, SourceLocation};
use crate::forms::base::{SpecialForm, check_min_arity};

// ---------------------------------------------------------------------------
// defparams
// ---------------------------------------------------------------------------

/// defparams - Declare a named parameter layout schema.
///
/// Syntax:
///   (defparams Name {:key [shape] ...})
///   (defparams Name {:group {:key [shape] ...} ...})
///
/// Example:
///   (defparams Linear {:W [4 8] :b [8]})
///   (defparams GPT-Layer {:attn {:Wq [512 512] :Wk [512 512]}
///                          :mlp  {:W1 [512 2048] :W2 [2048 512]}})
pub struct DefparamsForm;

impl SpecialForm for DefparamsForm {
    fn name(&self) -> &'static str {
        "defparams"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        check_min_arity("defparams", args, 2, loc)?;

        // First arg: name (symbol)
        let name = args[0].as_symbol().ok_or_else(|| SheafError::Compile {
            message: "defparams: first argument must be a symbol name".to_string(),
            location: loc.clone(),
        })?;

        // Second arg: schema dict
        let schema = &args[1];

        // Parse the schema into a flat list of ParamFields
        let mut fields: Vec<ParamField> = Vec::new();
        parse_schema(schema, &[], &mut fields, loc)?;

        // Assign flat tuple indices (for the outer level of nesting)
        // We use a flat index into the top-level tuple
        assign_tuple_indices(&mut fields);

        let layout = ParamLayout {
            name: name.to_string(),
            fields,
        };

        compiler.param_types.insert(name.to_string(), layout);

        // defparams is a compile-time declaration, returns Nil at runtime
        Ok(CompiledExpr::Nil)
    }
}

/// Recursively parse a schema dict into flat ParamFields.
/// path: current key path (e.g. ["attn"] while parsing :Wq)
fn parse_schema(
    value: &SheafValue,
    path: &[String],
    fields: &mut Vec<ParamField>,
    loc: &SourceLocation,
) -> SheafResult<()> {
    match value {
        // Nested dict: recurse into each key
        SheafValue::Dict(pairs, _) => {
            for (key_val, sub_val) in pairs {
                let key = key_as_string(key_val, loc)?;
                let mut sub_path = path.to_vec();
                sub_path.push(key);
                parse_schema(sub_val, &sub_path, fields, loc)?;
            }
        }

        // Shape vector [dim1 dim2 ...]: this is a leaf field
        SheafValue::Vector(elems, _) => {
            let shape: SheafResult<Vec<i64>> = elems
                .iter()
                .map(|e| match e {
                    SheafValue::Integer(n, _) => Ok(*n),
                    _ => Err(SheafError::Compile {
                        message: format!(
                            "defparams: shape dimensions must be integers, got: {}",
                            e
                        ),
                        location: loc.clone(),
                    }),
                })
                .collect();

            fields.push(ParamField {
                path: path.to_vec(),
                shape: shape?,
                tuple_index: vec![], // filled by assign_tuple_indices
            });
        }

        other => {
            return Err(SheafError::Compile {
                message: format!("defparams: expected dict or shape vector, got: {}", other),
                location: loc.clone(),
            });
        }
    }
    Ok(())
}

/// Extract a string key from a keyword or symbol value
fn key_as_string(val: &SheafValue, loc: &SourceLocation) -> SheafResult<String> {
    match val {
        SheafValue::Keyword(k, _) => Ok(k.clone()),
        SheafValue::Symbol(s, _) => Ok(s.clone()),
        other => Err(SheafError::Compile {
            message: format!("defparams: dict key must be a keyword, got: {}", other),
            location: loc.clone(),
        }),
    }
}

/// Assign flat tuple indices to fields based on their declaration order.
/// For a 2-level nested dict, produces hierarchical indices like [0, 1], [0, 2], [1, 0], etc.
/// For now we use a simple flat index.
fn assign_tuple_indices(fields: &mut Vec<ParamField>) {
    // Group by top-level key to assign hierarchical indices
    // E.g. {:attn {:Wq :Wk} :mlp {:W1}} -> [0,0], [0,1], [1,0]
    let mut top_level_seen: Vec<String> = Vec::new();

    for field in fields.iter_mut() {
        if field.path.is_empty() {
            continue;
        }

        let top_key = &field.path[0];
        let top_idx = match top_level_seen.iter().position(|k| k == top_key) {
            Some(i) => i,
            None => {
                top_level_seen.push(top_key.clone());
                top_level_seen.len() - 1
            }
        };

        if field.path.len() == 1 {
            // Flat field: index is just [top_idx]
            field.tuple_index = vec![top_idx];
        } else {
            // Nested field: need sub-index within the group
            // Count how many fields with the same top_key came before this one
            // This is computed after the full iteration, so we store the top_idx for now
            field.tuple_index = vec![top_idx]; // refined below
        }
    }

    // Refine sub-indices for nested fields
    // For each top-level group, assign sequential indices to children
    let groups: Vec<String> = {
        let mut seen = Vec::new();
        for f in fields.iter() {
            if let Some(k) = f.path.first() {
                if !seen.contains(k) {
                    seen.push(k.clone());
                }
            }
        }
        seen
    };

    for (top_idx, top_key) in groups.iter().enumerate() {
        let mut sub_idx = 0usize;
        for field in fields.iter_mut() {
            if field.path.first().map(|k| k == top_key).unwrap_or(false) {
                if field.path.len() == 1 {
                    field.tuple_index = vec![top_idx];
                } else {
                    field.tuple_index = vec![top_idx, sub_idx];
                    sub_idx += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// with-params
// ---------------------------------------------------------------------------

/// with-params - Destructure a typed parameter tuple into local scope.
///
/// Syntax:
///   (with-params [param-name] body...)
///   (with-params [param-name :key] body...)   ; access sub-dict
///
/// The parameter must be typed with :as in the enclosing defn signature,
/// or the type must be inferable from context.
///
/// Example:
///   (defparams Linear {:W [4 8] :b [8]})
///
///   (defn linear [x (p :as Linear)]
///     (with-params [p]
///       (+ (@ x W) b)))   ; W and b resolved from p via tuple indices
pub struct WithParamsForm;

impl SpecialForm for WithParamsForm {
    fn name(&self) -> &'static str {
        "with-params"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        check_min_arity("with-params", args, 2, loc)?;

        // First arg: binding vector [param-name] or [param-name :key]
        let binding = &args[0];
        let body = &args[1..];

        let (param_name, opt_key) = parse_with_params_binding(binding, loc)?;

        // Find the layout for this param
        // Look up the param's type annotation from local_vars metadata
        // For now: look up by convention (the param must be annotated with :as SomeType)
        let layout_name = compiler
            .local_vars
            .get(&format!("__type__{}", param_name))
            .and_then(|v| v.as_symbol())
            .map(|s| s.to_string())
            .ok_or_else(|| SheafError::Compile {
                message: format!(
                    "with-params: parameter '{}' has no type annotation. \
                     Use (defn f [(p :as MyLayout)] ...) to declare its type.",
                    param_name
                ),
                location: loc.clone(),
            })?;

        let layout = compiler
            .param_types
            .get(&layout_name)
            .cloned()
            .ok_or_else(|| SheafError::Compile {
                message: format!(
                    "with-params: unknown param type '{}'. \
                     Did you forget (defparams {})?",
                    layout_name, layout_name
                ),
                location: loc.clone(),
            })?;

        // Determine which fields to expose based on optional key filter
        let fields_to_expose: Vec<&ParamField> = if let Some(ref key) = opt_key {
            layout.fields_under(key)
        } else {
            layout.fields.iter().collect()
        };

        // Save current param_scope to restore after body
        let saved_scope = compiler.param_scope.clone();

        // Populate param_scope: last path component -> (param_name, tuple_indices)
        for field in &fields_to_expose {
            let local_name = field.path.last().unwrap().clone();
            compiler
                .param_scope
                .insert(local_name, (param_name.clone(), field.tuple_index.clone()));
        }

        // Compile body with extended scope
        let compiled_body: SheafResult<Vec<CompiledExpr>> =
            body.iter().map(|e| compiler.compile(e)).collect();
        let compiled_body = compiled_body?;

        // Restore scope
        compiler.param_scope = saved_scope;

        // Return last expression (like do)
        if compiled_body.len() == 1 {
            Ok(compiled_body.into_iter().next().unwrap())
        } else {
            Ok(CompiledExpr::Do(compiled_body))
        }
    }
}

/// Parse the binding vector of with-params.
/// Returns (param_name, optional_sub_key)
fn parse_with_params_binding(
    binding: &SheafValue,
    loc: &SourceLocation,
) -> SheafResult<(String, Option<String>)> {
    match binding {
        SheafValue::Vector(elems, _) => {
            if elems.is_empty() {
                return Err(SheafError::Compile {
                    message: "with-params: binding vector must not be empty".to_string(),
                    location: loc.clone(),
                });
            }

            let param_name = elems[0].as_symbol().ok_or_else(|| SheafError::Compile {
                message: "with-params: first element of binding must be a symbol".to_string(),
                location: loc.clone(),
            })?;

            let opt_key = if elems.len() >= 2 {
                match &elems[1] {
                    SheafValue::Keyword(k, _) => Some(k.clone()),
                    other => {
                        return Err(SheafError::Compile {
                            message: format!(
                                "with-params: second element of binding must be a keyword, got: {}",
                                other
                            ),
                            location: loc.clone(),
                        });
                    }
                }
            } else {
                None
            };

            Ok((param_name.to_string(), opt_key))
        }
        other => Err(SheafError::Compile {
            message: format!(
                "with-params: expected binding vector [param] or [param :key], got: {}",
                other
            ),
            location: loc.clone(),
        }),
    }
}

// ---------------------------------------------------------------------------
// grad
// ---------------------------------------------------------------------------

/// grad - Symbolic differentiation of a compiled expression.
///
/// Syntax:
///   (grad body :wrt param)
///
/// Returns a `CompiledExpr` representing d(body)/d(param), simplified.
///
/// Example:
///   (defn linear-grad [x W b]
///     (grad (+ (@ x W) b) :wrt W))
///
/// This expands to the symbolic gradient of `(+ (@ x W) b)` with respect to `W`.
pub struct GradForm;

impl SpecialForm for GradForm {
    fn name(&self) -> &'static str {
        "grad"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // Syntax: (grad body :wrt param)
        // args = [body, :wrt, param]
        if args.len() < 3 {
            return Err(SheafError::Compile {
                message: "grad: expected (grad body :wrt param)".to_string(),
                location: loc.clone(),
            });
        }

        // Find :wrt keyword and the following param name
        let mut body_exprs: Vec<&SheafValue> = Vec::new();
        let mut wrt_param: Option<String> = None;
        let mut i = 0;
        while i < args.len() {
            match &args[i] {
                SheafValue::Keyword(k, _) if k == "wrt" => {
                    i += 1;
                    if i >= args.len() {
                        return Err(SheafError::Compile {
                            message: "grad: :wrt must be followed by a symbol".to_string(),
                            location: loc.clone(),
                        });
                    }
                    let param = args[i].as_symbol().ok_or_else(|| SheafError::Compile {
                        message: "grad: :wrt value must be a symbol".to_string(),
                        location: loc.clone(),
                    })?;
                    wrt_param = Some(param.to_string());
                }
                other => body_exprs.push(other),
            }
            i += 1;
        }

        let wrt = wrt_param.ok_or_else(|| SheafError::Compile {
            message: "grad: missing :wrt param".to_string(),
            location: loc.clone(),
        })?;

        if body_exprs.is_empty() {
            return Err(SheafError::Compile {
                message: "grad: missing body expression".to_string(),
                location: loc.clone(),
            });
        }

        // Compile body expression(s) — use last one as the differentiable expr
        let compiled_bodies: SheafResult<Vec<CompiledExpr>> =
            body_exprs.iter().map(|e| compiler.compile(e)).collect();
        let compiled_bodies = compiled_bodies?;
        let body_expr = compiled_bodies.into_iter().last().unwrap();

        // Compute symbolic gradient and simplify
        let gradient = crate::autodiff::grad_simplified(&body_expr, &wrt);

        Ok(gradient)
    }
}

// ---------------------------------------------------------------------------
// ParamLayout → StableHLOType conversion
// ---------------------------------------------------------------------------

/// Convert a ParamLayout into a StableHLO tuple type.
///
/// Flat layout:  {:W [4 8] :b [8]}
///   → tuple<tensor<4x8xf32>, tensor<8xf32>>
///
/// Nested layout: {:attn {:Wq [512 512] :Wk [512 512]} :mlp {:W1 [512 2048]}}
///   → tuple<tuple<tensor<512x512xf32>, tensor<512x512xf32>>, tuple<tensor<512x2048xf32>>>
pub fn param_layout_to_stablehlo_type(layout: &ParamLayout) -> StableHLOType {
    // Group fields by their first path segment
    let mut top_keys: Vec<String> = Vec::new();
    for field in &layout.fields {
        if let Some(k) = field.path.first() {
            if !top_keys.contains(k) {
                top_keys.push(k.clone());
            }
        }
    }

    // Build element types for each top-level group
    let elements: Vec<StableHLOType> = top_keys
        .iter()
        .map(|key| {
            let children: Vec<&ParamField> = layout
                .fields
                .iter()
                .filter(|f| f.path.first().map(|k| k == key).unwrap_or(false))
                .collect();

            if children.len() == 1 && children[0].path.len() == 1 {
                // Leaf: simple tensor
                field_to_tensor_type(children[0])
            } else {
                // Group: sub-tuple
                let sub_elements: Vec<StableHLOType> =
                    children.iter().map(|f| field_to_tensor_type(f)).collect();
                StableHLOType::Tuple(sub_elements)
            }
        })
        .collect();

    StableHLOType::Tuple(elements)
}

fn field_to_tensor_type(field: &ParamField) -> StableHLOType {
    if field.shape.is_empty() {
        StableHLOType::scalar_f32()
    } else {
        StableHLOType::f32_tensor(field.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SourceLocation;

    fn loc() -> SourceLocation {
        SourceLocation::unknown()
    }

    fn make_keyword(k: &str) -> SheafValue {
        SheafValue::Keyword(k.to_string(), loc())
    }

    fn make_int(n: i64) -> SheafValue {
        SheafValue::Integer(n, loc())
    }

    fn make_vec(elems: Vec<SheafValue>) -> SheafValue {
        SheafValue::Vector(elems, loc())
    }

    fn make_dict(pairs: Vec<(SheafValue, SheafValue)>) -> SheafValue {
        SheafValue::Dict(pairs, loc())
    }

    #[test]
    fn test_defparams_flat() {
        let mut ctx = CompilerContext::new();
        let args = vec![
            SheafValue::Symbol("Linear".to_string(), loc()),
            make_dict(vec![
                (make_keyword("W"), make_vec(vec![make_int(4), make_int(8)])),
                (make_keyword("b"), make_vec(vec![make_int(8)])),
            ]),
        ];
        let result = DefparamsForm.compile(&mut ctx, &args, &loc());
        assert!(result.is_ok(), "defparams should compile: {:?}", result);

        let layout = ctx.param_types.get("Linear").unwrap();
        assert_eq!(layout.fields.len(), 2);
        assert_eq!(layout.fields[0].path, vec!["W"]);
        assert_eq!(layout.fields[0].shape, vec![4, 8]);
        assert_eq!(layout.fields[0].tuple_index, vec![0]);
        assert_eq!(layout.fields[1].path, vec!["b"]);
        assert_eq!(layout.fields[1].shape, vec![8]);
        assert_eq!(layout.fields[1].tuple_index, vec![1]);
    }

    #[test]
    fn test_defparams_nested() {
        let mut ctx = CompilerContext::new();
        let args = vec![
            SheafValue::Symbol("Layer".to_string(), loc()),
            make_dict(vec![
                (
                    make_keyword("attn"),
                    make_dict(vec![
                        (
                            make_keyword("Wq"),
                            make_vec(vec![make_int(512), make_int(512)]),
                        ),
                        (
                            make_keyword("Wk"),
                            make_vec(vec![make_int(512), make_int(512)]),
                        ),
                    ]),
                ),
                (
                    make_keyword("mlp"),
                    make_dict(vec![(
                        make_keyword("W1"),
                        make_vec(vec![make_int(512), make_int(2048)]),
                    )]),
                ),
            ]),
        ];
        let result = DefparamsForm.compile(&mut ctx, &args, &loc());
        assert!(
            result.is_ok(),
            "defparams nested should compile: {:?}",
            result
        );

        let layout = ctx.param_types.get("Layer").unwrap();
        assert_eq!(layout.fields.len(), 3);

        // attn/Wq -> [0, 0], attn/Wk -> [0, 1], mlp/W1 -> [1, 0]
        assert_eq!(layout.fields[0].tuple_index, vec![0, 0]);
        assert_eq!(layout.fields[1].tuple_index, vec![0, 1]);
        assert_eq!(layout.fields[2].tuple_index, vec![1, 0]);
    }
}
