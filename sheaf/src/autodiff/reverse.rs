// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Reverse-mode autodiff on Administrative Normal Form.
//!
//! The forward pass gives every intermediate a name. The backward pass then
//! walks those intermediates in reverse order to build their gradients.

use crate::autodiff::replace_symbol;
use crate::core::error::{SheafError, SheafResult};
use crate::core::expr::{BindingPattern, CompiledExpr};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
struct AnfNameGenerator {
    next: usize,
}

impl AnfNameGenerator {
    fn fresh(&mut self) -> String {
        let name = format!("__anf_{}", self.next);
        self.next += 1;
        name
    }
}

fn is_trivial(expr: &CompiledExpr) -> bool {
    matches!(
        expr,
        CompiledExpr::Symbol(_)
            | CompiledExpr::Float(_)
            | CompiledExpr::Integer(_)
            | CompiledExpr::Boolean(_)
            | CompiledExpr::Nil
            | CompiledExpr::String(_)
            | CompiledExpr::Keyword(_)
            | CompiledExpr::Quoted(_)
            | CompiledExpr::GetTupleElement { .. }
    )
}

/// Convert an expression to ANF.
///
/// Vectors stay inline because codegen reads them directly as shapes.
pub fn to_anf(expr: &CompiledExpr) -> CompiledExpr {
    let mut bindings = Vec::new();
    let mut names = AnfNameGenerator::default();
    let body = anf_rec(expr, &mut bindings, &mut names);
    if bindings.is_empty() {
        body
    } else {
        CompiledExpr::Let {
            bindings,
            body: Box::new(body),
        }
    }
}

fn anf_rec(
    expr: &CompiledExpr,
    out: &mut Vec<(BindingPattern, CompiledExpr)>,
    names: &mut AnfNameGenerator,
) -> CompiledExpr {
    match expr {
        _ if is_trivial(expr) => expr.clone(),

        CompiledExpr::FunctionCall { name, args, loc } => {
            let anf_args: Vec<CompiledExpr> =
                args.iter().map(|arg| anf_rec(arg, out, names)).collect();
            let result = CompiledExpr::FunctionCall {
                name: name.clone(),
                args: anf_args,
                loc: loc.clone(),
            };
            let sym = names.fresh();
            out.push((BindingPattern::Simple(sym.clone()), result));
            CompiledExpr::Symbol(sym)
        }

        // Give each let binding a unique name so shadowed variables keep the right value.
        CompiledExpr::Let { bindings, body } => {
            let mut rename_map: HashMap<String, String> = HashMap::new();
            for (name, value) in bindings {
                let mut val = value.clone();
                for (old, new_name) in &rename_map {
                    val = replace_symbol(&val, old, &CompiledExpr::Symbol(new_name.clone()));
                }

                let anf_val = anf_rec(&val, out, names);
                let fresh = names.fresh();
                out.push((BindingPattern::Simple(fresh.clone()), anf_val));
                // Destructured bindings should already have been replaced with simple names.
                debug_assert!(
                    matches!(name, BindingPattern::Simple(_)),
                    "ANF: expected Simple binding pattern (destructure should be desugared), got {:?}",
                    name
                );
                let simple_name = match name {
                    BindingPattern::Simple(s) => s.clone(),
                    BindingPattern::Destructure(_) => String::new(),
                };
                rename_map.insert(simple_name, fresh);
            }
            let mut renamed_body = body.as_ref().clone();
            for (old, new_name) in &rename_map {
                renamed_body = replace_symbol(&renamed_body, old, &CompiledExpr::Symbol(new_name.clone()));
            }
            anf_rec(&renamed_body, out, names)
        }

        CompiledExpr::Do(exprs) => {
            let mut last = CompiledExpr::Nil;
            for e in exprs {
                last = anf_rec(e, out, names);
            }
            last
        }

        CompiledExpr::Vector(elems) => {
            let anf_elems: Vec<CompiledExpr> =
                elems.iter().map(|elem| anf_rec(elem, out, names)).collect();
            CompiledExpr::Vector(anf_elems)
        }

        other => {
            let sym = names.fresh();
            out.push((BindingPattern::Simple(sym.clone()), other.clone()));
            CompiledExpr::Symbol(sym)
        }
    }
}

fn fresh_grad_name(bindings: &[(String, CompiledExpr)]) -> String {
    format!("__grad_{}", bindings.len())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientOutput {
    /// Binding that contains the computed gradient.
    Computed(String),
    /// No gradient reaches this parameter.
    ProvenZero,
}

#[derive(Debug)]
pub struct ReverseGradResult {
    /// Bindings that compute the gradients.
    pub backward_bindings: Vec<(String, CompiledExpr)>,
    /// Result for each requested parameter.
    pub gradients: HashMap<String, GradientOutput>,
}

impl ReverseGradResult {
    pub fn gradient_output(&self, symbol: &str) -> SheafResult<&GradientOutput> {
        gradient_output(&self.gradients, symbol)
    }
}

pub fn gradient_output<'a>(
    gradients: &'a HashMap<String, GradientOutput>,
    symbol: &str,
) -> SheafResult<&'a GradientOutput> {
    gradients.get(symbol).ok_or_else(|| SheafError::AutodiffMissingGradientOutput {
        symbol: symbol.to_string(),
    })
}

struct AdjointState<'a> {
    dependencies: &'a HashMap<String, bool>,
    names: HashMap<String, String>,
    reached_by_vjp: HashSet<String>,
    zero_contributions: HashSet<String>,
}

/// Differentiate an ANF expression with respect to `wrt`.
///
/// The caller must emit the forward and backward bindings in the same scope.
pub fn reverse_grad(
    anf_bindings: &[(String, CompiledExpr)],
    anf_body: &CompiledExpr,
    wrt: &[String],
    shapes: &HashMap<String, Vec<i64>>,
) -> SheafResult<ReverseGradResult> {
    let requested: HashSet<String> = wrt.iter().cloned().collect();
    let dependencies = analyze_anf_dependencies(anf_bindings, &requested);
    let mut adj_names = AdjointState {
        dependencies: &dependencies,
        names: HashMap::new(),
        reached_by_vjp: HashSet::new(),
        zero_contributions: HashSet::new(),
    };
    let mut backward_bindings: Vec<(String, CompiledExpr)> = Vec::new();

    if let CompiledExpr::Symbol(s) = anf_body
        && dependencies.get(s).copied().unwrap_or(false) {
        let seed_name = fresh_grad_name(&backward_bindings);
        backward_bindings.push((
            seed_name.clone(),
            call("__ones-like", vec![anf_body.clone()]),
        ));
        adj_names.names.insert(s.clone(), seed_name);
    }

    for (name, value) in anf_bindings.iter().rev() {
        // A requested binding is a differentiation input, even when defined by an alias.
        if requested.contains(name) {
            continue;
        }
        let adj_sym = match adj_names.names.get(name) {
            Some(s) => CompiledExpr::Symbol(s.clone()),
            None => continue,
        };

        let context = ReverseContext {
            shapes,
            fwd_name: name,
            dependencies: &dependencies,
        };
        distribute_adjoint_named(
            value,
            &adj_sym,
            &mut adj_names,
            &mut backward_bindings,
            &context,
        )?;
    }

    // A gradient path still exists when its contribution simplifies to zero.
    let grad_map: HashMap<String, GradientOutput> = wrt
        .iter()
        .map(|param| {
            let output = match adj_names.names.get(param) {
                Some(adj_name) => GradientOutput::Computed(adj_name.clone()),
                None if !adj_names.reached_by_vjp.contains(param) => GradientOutput::ProvenZero,
                None if adj_names.zero_contributions.contains(param) => GradientOutput::ProvenZero,
                None => return Err(SheafError::AutodiffMissingGradientOutput {
                    symbol: param.clone(),
                }),
            };
            Ok((param.clone(), output))
        })
        .collect::<SheafResult<_>>()?;

    retain_gradient_bindings(&mut backward_bindings, &grad_map);
    Ok(ReverseGradResult {
        backward_bindings,
        gradients: grad_map,
    })
}

/// Keep only the pure backward operations used by a requested gradient.
fn retain_gradient_bindings(
    bindings: &mut Vec<(String, CompiledExpr)>,
    gradients: &HashMap<String, GradientOutput>,
) {
    let mut live: HashSet<String> = gradients.values().filter_map(|gradient| match gradient {
        GradientOutput::Computed(name) => Some(name.clone()),
        GradientOutput::ProvenZero => None,
    }).collect();
    for (name, expr) in bindings.iter().rev() {
        if live.contains(name) {
            crate::autodiff::collect_free_vars(expr, &mut live);
        }
    }
    bindings.retain(|(name, _)| live.contains(name));
}

/// Mark every binding that depends on a parameter being differentiated.
///
/// Propagate dependencies through each binding using the shared scope analysis.
fn analyze_anf_dependencies(
    anf_bindings: &[(String, CompiledExpr)],
    wrt: &HashSet<String>,
) -> HashMap<String, bool> {
    let mut dependencies: HashMap<String, bool> = wrt
        .iter()
        .map(|name| (name.clone(), true))
        .collect();

    for (name, value) in anf_bindings {
        let depends = wrt.contains(name)
            || crate::autodiff::analysis::depends_on_in_env(value, &dependencies);
        dependencies.insert(name.clone(), depends);
    }

    dependencies
}

fn accumulate_named(
    var_name: &str,
    contribution: CompiledExpr,
    adj_names: &mut AdjointState,
    bindings: &mut Vec<(String, CompiledExpr)>,
) {
    if !adj_names.dependencies.get(var_name).copied().unwrap_or(false) {
        return;
    }
    adj_names.reached_by_vjp.insert(var_name.to_string());
    if is_zero(&contribution) {
        adj_names.zero_contributions.insert(var_name.to_string());
        return;
    }

    match adj_names.names.get(var_name) {
        None => {
            let grad_name = fresh_grad_name(bindings);
            bindings.push((grad_name.clone(), contribution));
            adj_names.names.insert(var_name.to_string(), grad_name);
        }
        Some(existing) => {
            let contrib_name = fresh_grad_name(bindings);
            bindings.push((contrib_name.clone(), contribution));
            let sum_name = fresh_grad_name(bindings);
            bindings.push((
                sum_name.clone(),
                CompiledExpr::FunctionCall {
                    name: "+".to_string(),
                    args: vec![
                        CompiledExpr::Symbol(existing.clone()),
                        CompiledExpr::Symbol(contrib_name),
                    ],
                    loc: None,
                },
            ));
            adj_names.names.insert(var_name.to_string(), sum_name);
        }
    }
}

fn is_zero(expr: &CompiledExpr) -> bool {
    matches!(expr, CompiledExpr::Float(v) if *v == 0.0)
        || matches!(expr, CompiledExpr::Integer(0))
}

fn sym(name: &str) -> CompiledExpr {
    CompiledExpr::Symbol(name.to_string())
}

fn float(v: f64) -> CompiledExpr {
    CompiledExpr::Float(v)
}

fn call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
    CompiledExpr::FunctionCall {
        name: name.to_string(),
        args,
        loc: None,
    }
}

fn shape_vec(shape: &[i64]) -> CompiledExpr {
    CompiledExpr::Vector(
        shape
            .iter()
            .map(|&d| CompiledExpr::Integer(d))
            .collect(),
    )
}

fn maybe_unbroadcast(
    adj: CompiledExpr,
    target_arg: &CompiledExpr,
    shapes: &HashMap<String, Vec<i64>>,
    bindings: &mut Vec<(String, CompiledExpr)>,
) -> CompiledExpr {
    if let Some(target_shape) = arg_shape(target_arg, shapes) {
        // Codegen drops this reduction when the shapes already match.
        let reduced = emit_binding(
            bindings,
            call("sum_to_shape", vec![adj, shape_vec(&target_shape)]),
        );
        sym(&reduced)
    } else {
        adj
    }
}

fn reject_missing_rule_if_dependent(
    operation: &str,
    fwd_name: &str,
    dependencies: &HashMap<String, bool>,
    location: Option<crate::core::error::SourceLocation>,
) -> SheafResult<()> {
    if dependencies.get(fwd_name).copied().unwrap_or(true) {
        Err(SheafError::AutodiffMissingRule {
            operation: operation.to_string(),
            location,
        })
    } else {
        Ok(())
    }
}

fn anf_form_name(expr: &CompiledExpr) -> &'static str {
    match expr {
        CompiledExpr::Integer(_) => "integer",
        CompiledExpr::Float(_) => "float",
        CompiledExpr::Boolean(_) => "boolean",
        CompiledExpr::Nil => "nil",
        CompiledExpr::String(_) => "string",
        CompiledExpr::Keyword(_) => "keyword",
        CompiledExpr::Vector(_) => "vector",
        CompiledExpr::Tuple(_) => "tuple",
        CompiledExpr::Dict(_) => "dict",
        CompiledExpr::Quoted(_) => "quoted",
        CompiledExpr::FunctionRef(_) => "function reference",
        CompiledExpr::GlobalRef(_) => "module reference",
        CompiledExpr::Let { .. } => "let",
        CompiledExpr::If { .. } => "if",
        CompiledExpr::Do(_) => "do",
        CompiledExpr::Lambda { .. } => "lambda",
        CompiledExpr::LambdaCall { .. } => "lambda call",
        CompiledExpr::ValueAndGrad { .. } => "value-and-grad",
        CompiledExpr::Repeat { .. } => "repeat",
        CompiledExpr::While { .. } => "while",
        CompiledExpr::Guard { .. } => "guard",
        CompiledExpr::Def { .. } => "def",
        CompiledExpr::Symbol(_) => "symbol",
        CompiledExpr::GetTupleElement { .. } => "tuple element",
        CompiledExpr::FunctionCall { .. } => "function call",
    }
}

struct ReverseContext<'a> {
    shapes: &'a HashMap<String, Vec<i64>>,
    fwd_name: &'a str,
    dependencies: &'a HashMap<String, bool>,
}

fn distribute_adjoint_named(
    expr: &CompiledExpr,
    adj_sym: &CompiledExpr,
    adj_names: &mut AdjointState,
    bindings: &mut Vec<(String, CompiledExpr)>,
    context: &ReverseContext<'_>,
) -> SheafResult<()> {
    let fwd_name = context.fwd_name;
    let dependencies = context.dependencies;
    if !dependencies.get(fwd_name).copied().unwrap_or(true) {
        return Ok(());
    }
    match expr {
        CompiledExpr::Symbol(s) => {
            accumulate_named(s, adj_sym.clone(), adj_names, bindings);
            Ok(())
        }
        CompiledExpr::GetTupleElement { param, .. } => {
            accumulate_named(param, adj_sym.clone(), adj_names, bindings);
            Ok(())
        }
        CompiledExpr::FunctionCall { name, args, loc } => distribute_fn_adjoint_named(
            name,
            args,
            adj_sym,
            adj_names,
            bindings,
            context,
            loc.clone(),
        ),
        _ => reject_missing_rule_if_dependent(
            anf_form_name(expr),
            fwd_name,
            dependencies,
            None,
        ),
    }
}

fn distribute_fn_adjoint_named(
    name: &str,
    args: &[CompiledExpr],
    adj: &CompiledExpr,
    adj_names: &mut AdjointState,
    bindings: &mut Vec<(String, CompiledExpr)>,
    context: &ReverseContext<'_>,
    location: Option<crate::core::error::SourceLocation>,
) -> SheafResult<()> {
    let shapes = context.shapes;
    let fwd_name = context.fwd_name;
    let dependencies = context.dependencies;
    match name {
        "stop-gradient" => {}

        "+" if !args.is_empty() => {
            for arg in args {
                let contribution = maybe_unbroadcast(adj.clone(), arg, shapes, bindings);
                acc_arg(arg, contribution, adj_names, bindings);
            }
        }

        "-" if args.len() == 1 => {
            let neg = emit_binding(bindings, call("*", vec![float(-1.0), adj.clone()]));
            acc_arg(&args[0], sym(&neg), adj_names, bindings);
        }

        "-" if args.len() >= 2 => {
            let first = maybe_unbroadcast(adj.clone(), &args[0], shapes, bindings);
            acc_arg(&args[0], first, adj_names, bindings);
            let neg = emit_binding(bindings, call("*", vec![float(-1.0), adj.clone()]));
            for arg in &args[1..] {
                let contribution = maybe_unbroadcast(sym(&neg), arg, shapes, bindings);
                acc_arg(arg, contribution, adj_names, bindings);
            }
        }

        "*" if args.len() >= 2 => {
            let mut prefixes = vec![args[0].clone()];
            for arg in &args[1..args.len() - 1] {
                let product = emit_binding(
                    bindings,
                    call("*", vec![prefixes.last().unwrap().clone(), arg.clone()]),
                );
                prefixes.push(sym(&product));
            }

            let mut suffixes = vec![None; args.len()];
            suffixes[args.len() - 1] = Some(args[args.len() - 1].clone());
            for i in (1..args.len() - 1).rev() {
                let product = emit_binding(
                    bindings,
                    call("*", vec![args[i].clone(), suffixes[i + 1].clone().unwrap()]),
                );
                suffixes[i] = Some(sym(&product));
            }

            for (i, arg) in args.iter().enumerate() {
                let other_factors = if i == 0 {
                    suffixes[1].clone().unwrap()
                } else if i + 1 == args.len() {
                    prefixes[i - 1].clone()
                } else {
                    let product = emit_binding(
                        bindings,
                        call("*", vec![
                            prefixes[i - 1].clone(),
                            suffixes[i + 1].clone().unwrap(),
                        ]),
                    );
                    sym(&product)
                };
                let contribution = emit_binding(
                    bindings,
                    call("*", vec![adj.clone(), other_factors]),
                );
                let contribution =
                    maybe_unbroadcast(sym(&contribution), arg, shapes, bindings);
                acc_arg(arg, contribution, adj_names, bindings);
            }
        }

        "/" if args.len() >= 2 => {
            let denominator = if args.len() == 2 {
                args[1].clone()
            } else {
                let product = emit_binding(bindings, call("*", args[1..].to_vec()));
                sym(&product)
            };
            let numerator_grad = emit_binding(
                bindings,
                call("/", vec![adj.clone(), denominator]),
            );
            let numerator_grad =
                maybe_unbroadcast(sym(&numerator_grad), &args[0], shapes, bindings);
            acc_arg(&args[0], numerator_grad, adj_names, bindings);

            for arg in &args[1..] {
                let scaled = emit_binding(
                    bindings,
                    call("*", vec![adj.clone(), sym(fwd_name)]),
                );
                let divided = emit_binding(
                    bindings,
                    call("/", vec![sym(&scaled), arg.clone()]),
                );
                let negated = emit_binding(
                    bindings,
                    call("*", vec![float(-1.0), sym(&divided)]),
                );
                let contribution =
                    maybe_unbroadcast(sym(&negated), arg, shapes, bindings);
                acc_arg(arg, contribution, adj_names, bindings);
            }
        }

        "@" => {
            let a_ndim = arg_shape(&args[0], shapes).map(|s| s.len()).unwrap_or(2);
            let b_ndim = arg_shape(&args[1], shapes).map(|s| s.len()).unwrap_or(2);

            if b_ndim == 1 {
                if a_ndim == 1 {
                    let da = emit_binding(
                        bindings,
                        call("*", vec![adj.clone(), args[1].clone()]),
                    );
                    acc_arg(&args[0], sym(&da), adj_names, bindings);
                } else if let Some(a_shape) = arg_shape(&args[0], shapes) {
                    let mut expanded_shape = a_shape[..a_shape.len() - 1].to_vec();
                    expanded_shape.push(1);
                    let expanded_adj = emit_binding(
                        bindings,
                        call("reshape", vec![adj.clone(), shape_vec(&expanded_shape)]),
                    );
                    let da = emit_binding(
                        bindings,
                        call("*", vec![sym(&expanded_adj), args[1].clone()]),
                    );
                    acc_arg(&args[0], sym(&da), adj_names, bindings);
                } else {
                    let da = emit_binding(
                        bindings,
                        call("*", vec![adj.clone(), args[1].clone()]),
                    );
                    let da = maybe_unbroadcast(sym(&da), &args[0], shapes, bindings);
                    acc_arg(&args[0], da, adj_names, bindings);
                }
            } else if a_ndim == 1 {
                // Reshape the incoming gradient to [1, n] before the matrix multiplication.
                if let Some(b_shape) = arg_shape(&args[1], shapes) {
                    let n = b_shape[b_shape.len() - 1];
                    let adj_row = emit_binding(bindings, call("reshape", vec![
                        adj.clone(),
                        CompiledExpr::Vector(vec![CompiledExpr::Integer(1), CompiledExpr::Integer(n)]),
                    ]));
                    let bt = emit_binding(bindings, call("swapaxes", vec![
                        args[1].clone(), CompiledExpr::Integer(-1), CompiledExpr::Integer(-2),
                    ]));
                    let da_2d = emit_binding(bindings, call("@", vec![sym(&adj_row), sym(&bt)]));
                    let m = b_shape[0];
                    let da = emit_binding(bindings, call("reshape", vec![
                        sym(&da_2d),
                        CompiledExpr::Vector(vec![CompiledExpr::Integer(m)]),
                    ]));
                    acc_arg(&args[0], sym(&da), adj_names, bindings);
                } else {
                    let bt = emit_binding(bindings, call("swapaxes", vec![
                        args[1].clone(), CompiledExpr::Integer(-1), CompiledExpr::Integer(-2),
                    ]));
                    let da = emit_binding(bindings, call("@", vec![adj.clone(), sym(&bt)]));
                    let da_ub = maybe_unbroadcast(sym(&da), &args[0], shapes, bindings);
                    acc_arg(&args[0], da_ub, adj_names, bindings);
                }
            } else {
                let bt = emit_binding(bindings, call("swapaxes", vec![
                    args[1].clone(), CompiledExpr::Integer(-1), CompiledExpr::Integer(-2),
                ]));
                let da = emit_binding(bindings, call("@", vec![adj.clone(), sym(&bt)]));
                let da_ub = maybe_unbroadcast(sym(&da), &args[0], shapes, bindings);
                acc_arg(&args[0], da_ub, adj_names, bindings);
            }

            if a_ndim == 1 && b_ndim == 1 {
                let db = emit_binding(
                    bindings,
                    call("*", vec![adj.clone(), args[0].clone()]),
                );
                acc_arg(&args[1], sym(&db), adj_names, bindings);
            } else if a_ndim == 1 {
                // Reshape both vectors so matmul computes their outer product.
                if let (Some(a_shape), Some(b_shape)) = (arg_shape(&args[0], shapes), arg_shape(&args[1], shapes)) {
                    let m = a_shape[0];
                    let n = b_shape[b_shape.len() - 1];
                    let a_col = emit_binding(bindings, call("reshape", vec![
                        args[0].clone(),
                        CompiledExpr::Vector(vec![CompiledExpr::Integer(m), CompiledExpr::Integer(1)]),
                    ]));
                    let g_row = emit_binding(bindings, call("reshape", vec![
                        adj.clone(),
                        CompiledExpr::Vector(vec![CompiledExpr::Integer(1), CompiledExpr::Integer(n)]),
                    ]));
                    let db = emit_binding(bindings, call("@", vec![sym(&a_col), sym(&g_row)]));
                    let db_ub = maybe_unbroadcast(sym(&db), &args[1], shapes, bindings);
                    acc_arg(&args[1], db_ub, adj_names, bindings);
                } else {
                    let at = emit_binding(bindings, call("swapaxes", vec![
                        args[0].clone(), CompiledExpr::Integer(-1), CompiledExpr::Integer(-2),
                    ]));
                    let db = emit_binding(bindings, call("@", vec![sym(&at), adj.clone()]));
                    let db_ub = maybe_unbroadcast(sym(&db), &args[1], shapes, bindings);
                    acc_arg(&args[1], db_ub, adj_names, bindings);
                }
            } else {
                let at = emit_binding(bindings, call("swapaxes", vec![
                    args[0].clone(), CompiledExpr::Integer(-1), CompiledExpr::Integer(-2),
                ]));
                let db = emit_binding(bindings, call("@", vec![sym(&at), adj.clone()]));
                let db_ub = maybe_unbroadcast(sym(&db), &args[1], shapes, bindings);
                acc_arg(&args[1], db_ub, adj_names, bindings);
            }
        }

        "einsum" => {
            // Build each operand's gradient from the output gradient and the other operand.
            let Some(CompiledExpr::String(sub)) = args.first() else {
                return reject_missing_rule_if_dependent("einsum", fwd_name, dependencies, location);
            };
            if args.len() != 3 {
                return reject_missing_rule_if_dependent("einsum", fwd_name, dependencies, location);
            }
            let Some((inputs, output)) = sub.split_once("->") else {
                return reject_missing_rule_if_dependent("einsum", fwd_name, dependencies, location);
            };
            let parts: Vec<&str> = inputs.split(',').collect();
            if parts.len() != 2 {
                return reject_missing_rule_if_dependent("einsum", fwd_name, dependencies, location);
            }
            let lhs_sub = parts[0].trim();
            let rhs_sub = parts[1].trim();
            let out_sub = output.trim();

            let grad_lhs_sub = format!("{},{}->{}", out_sub, rhs_sub, lhs_sub);
            let da = emit_binding(bindings, call("einsum", vec![
                CompiledExpr::String(grad_lhs_sub),
                adj.clone(),
                args[2].clone(),
            ]));
            acc_arg(&args[1], sym(&da), adj_names, bindings);

            let grad_rhs_sub = format!("{},{}->{}", lhs_sub, out_sub, rhs_sub);
            let db = emit_binding(bindings, call("einsum", vec![
                CompiledExpr::String(grad_rhs_sub),
                args[1].clone(),
                adj.clone(),
            ]));
            acc_arg(&args[2], sym(&db), adj_names, bindings);
        }

        "transpose" | "tr" => {
            let dt = emit_binding(bindings, call("transpose", vec![adj.clone()]));
            acc_arg(&args[0], sym(&dt), adj_names, bindings);
        }

        "__cast-like" if args.len() == 2 => {
            let cast = emit_binding(
                bindings,
                call("__cast-like", vec![adj.clone(), args[0].clone()]),
            );
            acc_arg(&args[0], sym(&cast), adj_names, bindings);
        }

        "concat" if !args.is_empty() => {
            let Some(shape) = arg_shape(&args[0], shapes) else {
                return reject_missing_rule_if_dependent(name, fwd_name, dependencies, location);
            };
            let axis = parse_keyword_int(args, "axis").unwrap_or(0);
            let axis = if axis < 0 { shape.len() as i64 + axis } else { axis };
            let Some(axis) = usize::try_from(axis).ok().filter(|&axis| axis < shape.len()) else {
                return reject_missing_rule_if_dependent(name, fwd_name, dependencies, location);
            };
            let mut start = 0;
            for arg in args.iter().take_while(|arg| !matches!(arg, CompiledExpr::Keyword(_))) {
                let Some(shape) = arg_shape(arg, shapes) else {
                    return reject_missing_rule_if_dependent(name, fwd_name, dependencies, location);
                };
                let Some(&width) = shape.get(axis) else {
                    return reject_missing_rule_if_dependent(name, fwd_name, dependencies, location);
                };
                let end = start + width;
                let gradient = emit_binding(bindings, call("slice", vec![
                    adj.clone(), CompiledExpr::Integer(start), CompiledExpr::Integer(end),
                    CompiledExpr::Keyword("axis".into()), CompiledExpr::Integer(axis as i64),
                ]));
                acc_arg(arg, sym(&gradient), adj_names, bindings);
                start = end;
            }
        }

        "reshape" => {
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let shape_vec = CompiledExpr::Vector(
                    input_shape.iter().map(|&d| CompiledExpr::Integer(d)).collect()
                );
                let dr = emit_binding(bindings, call("reshape", vec![adj.clone(), shape_vec]));
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            } else {
                acc_arg(&args[0], adj.clone(), adj_names, bindings);
            }
        }

        "swapaxes" if args.len() == 3 => {
            let dr = emit_binding(bindings, call("swapaxes", vec![adj.clone(), args[1].clone(), args[2].clone()]));
            acc_arg(&args[0], sym(&dr), adj_names, bindings);
        }

        "relu" => {
            let cond = emit_binding(bindings, call(">", vec![args[0].clone(), float(0.0)]));
            let dr = emit_binding(bindings, call("where", vec![sym(&cond), adj.clone(), float(0.0)]));
            acc_arg(&args[0], sym(&dr), adj_names, bindings);
        }

        "gelu" => {
            // This must match the tanh approximation used by the forward GELU.
            let x = &args[0];
            let x2 = emit_binding(bindings, call("*", vec![x.clone(), x.clone()]));
            let x3 = emit_binding(bindings, call("*", vec![sym(&x2), x.clone()]));
            let coeff_x3 = emit_binding(bindings, call("*", vec![float(0.044715), sym(&x3)]));
            let inner = emit_binding(bindings, call("+", vec![x.clone(), sym(&coeff_x3)]));
            let k = emit_binding(bindings, call("*", vec![float(0.7978845608), sym(&inner)]));
            let tanh_k = emit_binding(bindings, call("tanh", vec![sym(&k)]));
            let one_plus_tanh = emit_binding(bindings, call("+", vec![float(1.0), sym(&tanh_k)]));
            let half_term1 = emit_binding(bindings, call("*", vec![float(0.5), sym(&one_plus_tanh)]));
            let tanh_sq = emit_binding(bindings, call("*", vec![sym(&tanh_k), sym(&tanh_k)]));
            let sech2 = emit_binding(bindings, call("-", vec![float(1.0), sym(&tanh_sq)]));
            let coeff_x2 = emit_binding(bindings, call("*", vec![float(0.134145), sym(&x2)]));
            let kp_inner = emit_binding(bindings, call("+", vec![float(1.0), sym(&coeff_x2)]));
            let kp = emit_binding(bindings, call("*", vec![float(0.7978845608), sym(&kp_inner)]));
            let x_sech2 = emit_binding(bindings, call("*", vec![x.clone(), sym(&sech2)]));
            let x_sech2_kp = emit_binding(bindings, call("*", vec![sym(&x_sech2), sym(&kp)]));
            let half_term2 = emit_binding(bindings, call("*", vec![float(0.5), sym(&x_sech2_kp)]));
            let gelu_grad = emit_binding(bindings, call("+", vec![sym(&half_term1), sym(&half_term2)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&gelu_grad)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "softmax" => {
            let axis = parse_keyword_int(args, "axis").unwrap_or(-1);
            let sm = fwd_name.to_string();
            let prod = emit_binding(bindings, call("*", vec![adj.clone(), sym(&sm)]));
            let s = emit_binding(bindings, call("sum", vec![
                sym(&prod),
                CompiledExpr::Keyword("axis".to_string()),
                CompiledExpr::Integer(axis),
                CompiledExpr::Keyword("keepdims".to_string()),
                CompiledExpr::Boolean(true),
            ]));
            let diff = emit_binding(bindings, call("-", vec![adj.clone(), sym(&s)]));
            let dr = emit_binding(bindings, call("*", vec![sym(&sm), sym(&diff)]));
            acc_arg(&args[0], sym(&dr), adj_names, bindings);
        }

        "log-softmax" => {
            let axis = parse_keyword_int(args, "axis").unwrap_or(-1);
            let fwd_args: Vec<CompiledExpr> = args.to_vec();
            let lsm = emit_binding(bindings, call("log-softmax", fwd_args));
            let sm = emit_binding(bindings, call("exp", vec![sym(&lsm)]));
            let s = emit_binding(bindings, call("sum", vec![
                adj.clone(),
                CompiledExpr::Keyword("axis".to_string()),
                CompiledExpr::Integer(axis),
                CompiledExpr::Keyword("keepdims".to_string()),
                CompiledExpr::Boolean(true),
            ]));
            let prod = emit_binding(bindings, call("*", vec![sym(&sm), sym(&s)]));
            let dr = emit_binding(bindings, call("-", vec![adj.clone(), sym(&prod)]));
            acc_arg(&args[0], sym(&dr), adj_names, bindings);
        }

        "sum" | "mean" if !args.is_empty() => {
            let Some(input_shape) = arg_shape(&args[0], shapes) else {
                if parse_keyword_int(args, "axis").is_some() {
                    return reject_missing_rule_if_dependent(name, fwd_name, dependencies, location);
                }
                let ones = emit_binding(bindings, call("__ones-like", vec![args[0].clone()]));
                let contribution = if name == "mean" {
                    let count = emit_binding(bindings, call("sum", vec![sym(&ones)]));
                    let scaled = emit_binding(bindings, call("/", vec![adj.clone(), sym(&count)]));
                    sym(&scaled)
                } else { adj.clone() };
                let gradient = emit_binding(bindings, call("*", vec![contribution, sym(&ones)]));
                acc_arg(&args[0], sym(&gradient), adj_names, bindings);
                return Ok(());
            };
            let axis = parse_keyword_int(args, "axis").map(|axis| {
                let axis = if axis < 0 { axis + input_shape.len() as i64 } else { axis };
                usize::try_from(axis).ok().filter(|&axis| axis < input_shape.len())
                    .ok_or_else(|| SheafError::Compile {
                        message: format!("{name}: reduction axis out of range"),
                        location: location.clone().unwrap_or_else(crate::core::error::SourceLocation::unknown),
                    })
            }).transpose()?;
            let mut contribution = adj.clone();
            if let Some(axis) = axis
                && shapes.get(fwd_name).is_some_and(|shape| shape.len() < input_shape.len()) {
                let mut expanded = input_shape.clone();
                expanded[axis] = 1;
                let reshaped = emit_binding(bindings, call("reshape", vec![contribution, shape_vec(&expanded)]));
                contribution = sym(&reshaped);
            }
            if name == "mean" {
                let count = axis.map(|axis| input_shape[axis]).unwrap_or_else(|| input_shape.iter().product());
                let scaled = emit_binding(bindings, call("/", vec![contribution, float(count as f64)]));
                contribution = sym(&scaled);
            }
            let broadcast = emit_binding(bindings, call("broadcast", vec![contribution, shape_vec(&input_shape)]));
            let cast = emit_binding(bindings, call("__cast-like", vec![sym(&broadcast), args[0].clone()]));
            acc_arg(&args[0], sym(&cast), adj_names, bindings);
        }

        "var" => {
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let axis = parse_keyword_int(args, "axis");
                let mut mean_args = vec![args[0].clone()];
                if let Some(ax) = axis {
                    mean_args.push(CompiledExpr::Keyword("axis".to_string()));
                    mean_args.push(CompiledExpr::Integer(ax));
                }
                mean_args.push(CompiledExpr::Keyword("keepdims".to_string()));
                mean_args.push(CompiledExpr::Boolean(true));
                let m = emit_binding(bindings, call("mean", mean_args));
                let diff = emit_binding(bindings, call("-", vec![args[0].clone(), sym(&m)]));
                let ndim = input_shape.len();
                let n: i64 = if let Some(ax) = axis {
                    let ax_usize = if ax < 0 { (ndim as i64 + ax) as usize } else { ax as usize };
                    input_shape[ax_usize]
                } else {
                    input_shape.iter().product()
                };
                let scaled = emit_binding(bindings, call("*", vec![float(2.0 / n as f64), sym(&diff)]));
                let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&scaled)]));
                let ub = maybe_unbroadcast(sym(&contrib), &args[0], shapes, bindings);
                acc_arg(&args[0], ub, adj_names, bindings);
            } else {
                acc_arg(&args[0], adj.clone(), adj_names, bindings);
            }
        }

        "max" | "min" => {
            // Every matching maximum or minimum receives the full gradient.
            let axis = parse_keyword_int(args, "axis");
            let mut fwd_args = vec![args[0].clone()];
            if let Some(ax) = axis {
                fwd_args.push(CompiledExpr::Keyword("axis".to_string()));
                fwd_args.push(CompiledExpr::Integer(ax));
                fwd_args.push(CompiledExpr::Keyword("keepdims".to_string()));
                fwd_args.push(CompiledExpr::Boolean(true));
            }
            let fwd_val = emit_binding(bindings, call(name, fwd_args));
            let mask = emit_binding(bindings, call("==", vec![args[0].clone(), sym(&fwd_val)]));
            let mask = emit_binding(bindings, call("__cast-like", vec![sym(&mask), args[0].clone()]));
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let sv = shape_vec(&input_shape);
                let adj_bc = emit_binding(bindings, call("broadcast", vec![adj.clone(), sv]));
                let adj_bc = emit_binding(bindings, call("__cast-like", vec![sym(&adj_bc), args[0].clone()]));
                let dr = emit_binding(bindings, call("*", vec![sym(&adj_bc), sym(&mask)]));
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            } else {
                let dr = emit_binding(bindings, call("*", vec![adj.clone(), sym(&mask)]));
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            }
        }

        "sigmoid" => {
            let sig = fwd_name.to_string();
            let one_minus_sig = emit_binding(bindings, call("-", vec![float(1.0), sym(&sig)]));
            let local = emit_binding(bindings, call("*", vec![sym(&sig), sym(&one_minus_sig)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&local)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "exp" => {
            let ex = fwd_name.to_string();
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&ex)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "log" => {
            let inv = emit_binding(bindings, call("/", vec![float(1.0), args[0].clone()]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&inv)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "sqrt" => {
            let sq = emit_binding(bindings, call("sqrt", vec![args[0].clone()]));
            let two_sq = emit_binding(bindings, call("*", vec![float(2.0), sym(&sq)]));
            let inv = emit_binding(bindings, call("/", vec![float(1.0), sym(&two_sq)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&inv)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "tanh" => {
            let t = fwd_name.to_string();
            let t2 = emit_binding(bindings, call("*", vec![sym(&t), sym(&t)]));
            let local = emit_binding(bindings, call("-", vec![float(1.0), sym(&t2)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&local)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "sin" => {
            let c = emit_binding(bindings, call("cos", vec![args[0].clone()]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&c)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "cos" => {
            let s = emit_binding(bindings, call("sin", vec![args[0].clone()]));
            let neg = emit_binding(bindings, call("*", vec![float(-1.0), sym(&s)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&neg)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "tan" => {
            let t = fwd_name.to_string();
            let t2 = emit_binding(bindings, call("*", vec![sym(&t), sym(&t)]));
            let local = emit_binding(bindings, call("+", vec![float(1.0), sym(&t2)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&local)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "maximum" if args.len() == 2 => {
            let cond = emit_binding(bindings, call(">=", vec![args[0].clone(), args[1].clone()]));
            let da = emit_binding(bindings, call("where", vec![sym(&cond), adj.clone(), float(0.0)]));
            let db = emit_binding(bindings, call("where", vec![sym(&cond), float(0.0), adj.clone()]));
            acc_arg(&args[0], sym(&da), adj_names, bindings);
            acc_arg(&args[1], sym(&db), adj_names, bindings);
        }

        "minimum" if args.len() == 2 => {
            let cond = emit_binding(bindings, call("<=", vec![args[0].clone(), args[1].clone()]));
            let da = emit_binding(bindings, call("where", vec![sym(&cond), adj.clone(), float(0.0)]));
            let db = emit_binding(bindings, call("where", vec![sym(&cond), float(0.0), adj.clone()]));
            acc_arg(&args[0], sym(&da), adj_names, bindings);
            acc_arg(&args[1], sym(&db), adj_names, bindings);
        }

        "where" if args.len() == 3 => {
            let da = emit_binding(bindings, call("where", vec![args[0].clone(), adj.clone(), float(0.0)]));
            let db = emit_binding(bindings, call("where", vec![args[0].clone(), float(0.0), adj.clone()]));
            acc_arg(&args[1], sym(&da), adj_names, bindings);
            acc_arg(&args[2], sym(&db), adj_names, bindings);
        }

        "slice" if args.len() >= 3 => {
            if let Some(input_shape) = arg_shape(&args[0], shapes) {
                let target_vec = shape_vec(&input_shape);
                let dr = emit_binding(
                    bindings,
                    call("slice_grad", vec![adj.clone(), target_vec, args[1].clone()]),
                );
                acc_arg(&args[0], sym(&dr), adj_names, bindings);
            } else {
                acc_arg(&args[0], adj.clone(), adj_names, bindings);
            }
        }

        "neg" => {
            let neg = emit_binding(bindings, call("*", vec![float(-1.0), adj.clone()]));
            acc_arg(&args[0], sym(&neg), adj_names, bindings);
        }

        "abs" => {
            let cond = emit_binding(bindings, call(">=", vec![args[0].clone(), float(0.0)]));
            let sign_f32 = emit_binding(
                bindings,
                call("where", vec![sym(&cond), float(1.0), float(-1.0)]),
            );
            let sign = emit_binding(
                bindings,
                call("__cast-like", vec![sym(&sign_f32), adj.clone()]),
            );
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&sign)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "**" if args.len() == 2 => {
            let n = match &args[1] {
                CompiledExpr::Float(n) => *n,
                CompiledExpr::Integer(n) => *n as f64,
                _ => return reject_missing_rule_if_dependent("**", fwd_name, dependencies, location),
            };
            let pow = emit_binding(bindings, call("**", vec![args[0].clone(), float(n - 1.0)]));
            let local = emit_binding(bindings, call("*", vec![float(n), sym(&pow)]));
            let contrib = emit_binding(bindings, call("*", vec![adj.clone(), sym(&local)]));
            acc_arg(&args[0], sym(&contrib), adj_names, bindings);
        }

        "first" if args.len() == 1 => {
            acc_arg(&args[0], adj.clone(), adj_names, bindings);
        }

        "second" if args.len() == 1 => {
            acc_arg(&args[0], adj.clone(), adj_names, bindings);
        }

        // A field lookup only selects a value, so its gradient passes through unchanged.
        "get" if args.len() == 2 && matches!(&args[1], CompiledExpr::Keyword(_)) => {
            acc_arg(&args[0], adj.clone(), adj_names, bindings);
        }
        "get" if args.len() == 2 => {
            let is_scalar_index = matches!(&args[1],
                CompiledExpr::Integer(_) | CompiledExpr::Float(_)
            ) || matches!(&args[1], CompiledExpr::Symbol(s) if {
                shapes.get(s.as_str()).is_some_and(|sh| sh.is_empty() || sh == &[1])
            });

            let v = emit_binding(bindings, call("shape", vec![args[0].clone(), CompiledExpr::Integer(0)]));
            let oh_f32 = emit_binding(bindings, call("one-hot", vec![args[1].clone(), sym(&v)]));
            let oh = emit_binding(
                bindings,
                call("__cast-like", vec![sym(&oh_f32), adj.clone()]),
            );

            if is_scalar_index && arg_shape(&args[0], shapes).is_some_and(|shape| shape.len() == 1) {
                let input_shape = arg_shape(&args[0], shapes).unwrap();
                let mask = emit_binding(bindings, call("reshape", vec![sym(&oh), shape_vec(&input_shape)]));
                let gradient = emit_binding(bindings, call("*", vec![sym(&mask), adj.clone()]));
                acc_arg(&args[0], sym(&gradient), adj_names, bindings);
            } else if is_scalar_index {
                let oh_col = emit_binding(bindings, call("reshape", vec![
                    sym(&oh),
                    CompiledExpr::Vector(vec![sym(&v), CompiledExpr::Integer(1)]),
                ]));
                let adj_row = emit_binding(bindings, call("reshape", vec![
                    adj.clone(),
                    CompiledExpr::Vector(vec![CompiledExpr::Integer(1), CompiledExpr::Integer(-1)]),
                ]));
                let grad = emit_binding(bindings, call("@", vec![sym(&oh_col), sym(&adj_row)]));
                let grad = maybe_unbroadcast(sym(&grad), &args[0], shapes, bindings);
                acc_arg(&args[0], grad, adj_names, bindings);
            } else {
                let Some(input_shape) = arg_shape(&args[0], shapes) else {
                    return reject_missing_rule_if_dependent(name, fwd_name, dependencies, location);
                };
                let width = input_shape[1..].iter().product();
                let oh_flat = emit_binding(bindings, call("reshape", vec![sym(&oh),
                    CompiledExpr::Vector(vec![CompiledExpr::Integer(-1), sym(&v)])]));
                let oh_t = emit_binding(bindings, call("tr", vec![sym(&oh_flat)]));
                let adj_flat = emit_binding(bindings, call("reshape", vec![adj.clone(),
                    CompiledExpr::Vector(vec![CompiledExpr::Integer(-1), CompiledExpr::Integer(width)])]));
                let grad = emit_binding(bindings, call("@", vec![sym(&oh_t), sym(&adj_flat)]));
                let grad = emit_binding(bindings, call("reshape", vec![sym(&grad), shape_vec(&input_shape)]));
                acc_arg(&args[0], sym(&grad), adj_names, bindings);
            }
        }

        "scan" if args.len() == 3 => {
            let vjp_result = emit_binding(bindings, call("__scan_vjp__", vec![
                args[0].clone(),
                args[1].clone(),
                args[2].clone(),
                adj.clone(),
            ]));
            let adj_init = emit_binding(bindings, call("first", vec![sym(&vjp_result)]));
            let adj_coll = emit_binding(bindings, call("second", vec![sym(&vjp_result)]));
            acc_arg(&args[1], sym(&adj_init), adj_names, bindings);
            acc_arg(&args[2], sym(&adj_coll), adj_names, bindings);
        }

        "reduce" if args.len() == 3 => {
            let vjp_result = emit_binding(bindings, call("__scan_vjp__", vec![
                args[0].clone(),
                args[1].clone(),
                args[2].clone(),
                adj.clone(),
            ]));
            let adj_init = emit_binding(bindings, call("first", vec![sym(&vjp_result)]));
            let adj_coll = emit_binding(bindings, call("second", vec![sym(&vjp_result)]));
            acc_arg(&args[1], sym(&adj_init), adj_names, bindings);
            acc_arg(&args[2], sym(&adj_coll), adj_names, bindings);
        }

        _ => return reject_missing_rule_if_dependent(name, fwd_name, dependencies, location),
    }
    Ok(())
}

fn parse_keyword_int(args: &[CompiledExpr], key: &str) -> Option<i64> {
    for (i, arg) in args.iter().enumerate() {
        if let CompiledExpr::Keyword(k) = arg
            && k == key
            && let Some(CompiledExpr::Integer(n)) = args.get(i + 1) {
                    return Some(*n);
                }
    }
    None
}

fn arg_shape(arg: &CompiledExpr, shapes: &HashMap<String, Vec<i64>>) -> Option<Vec<i64>> {
    match arg {
        CompiledExpr::Symbol(s) => shapes.get(s).cloned(),
        CompiledExpr::Integer(_) | CompiledExpr::Float(_) | CompiledExpr::Boolean(_) => Some(Vec::new()),
        CompiledExpr::Vector(values) => {
            let element_shape = match values.first() {
                Some(value) => arg_shape(value, shapes)?,
                None => Vec::new(),
            };
            if values.iter().any(|value| arg_shape(value, shapes).as_ref() != Some(&element_shape)) { return None; }
            let mut shape = vec![values.len() as i64];
            shape.extend(element_shape);
            Some(shape)
        }
        _ => None,
    }
}

fn emit_binding(
    bindings: &mut Vec<(String, CompiledExpr)>,
    value: CompiledExpr,
) -> String {
    let name = fresh_grad_name(bindings);
    bindings.push((name.clone(), value));
    name
}

fn acc_arg(
    arg: &CompiledExpr,
    contribution: CompiledExpr,
    adj_names: &mut AdjointState,
    bindings: &mut Vec<(String, CompiledExpr)>,
) {
    match arg {
        CompiledExpr::Symbol(s) => {
            accumulate_named(s, contribution, adj_names, bindings);
        }
        CompiledExpr::GetTupleElement { param, .. } => {
            accumulate_named(param, contribution, adj_names, bindings);
        }
        _ => {} // Literal: no gradient
    }
}

#[cfg(test)]
mod tests {
    use super::{GradientOutput, ReverseGradResult, reverse_grad, to_anf};
    use crate::core::error::{SheafError, SourceLocation};
    use crate::core::expr::{BindingPattern, CompiledExpr};
    use std::collections::HashMap;
    use std::rc::Rc;

    fn symbol(name: &str) -> CompiledExpr {
        CompiledExpr::Symbol(name.to_string())
    }

    fn call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: name.to_string(),
            args,
            loc: None,
        }
    }

    fn missing_rule(error: SheafError, operation: &str) {
        assert!(matches!(
            error,
            SheafError::AutodiffMissingRule {
                operation: actual, ..
            } if actual == operation
        ));
    }

    #[test]
    fn only_requested_operands_receive_adjoints() {
        let bindings = vec![("result".to_string(), call("*", vec![symbol("x"), symbol("weight")]))];
        let shapes = HashMap::from([
            ("x".to_string(), Vec::new()),
            ("weight".to_string(), Vec::new()),
            ("result".to_string(), Vec::new()),
        ]);
        let requested = reverse_grad(&bindings, &symbol("result"), &["x".to_string()], &shapes).unwrap();
        assert!(matches!(requested.gradients["x"], GradientOutput::Computed(_)));
        assert!(!requested.backward_bindings.iter().any(|(_, expr)| {
            matches!(expr, CompiledExpr::FunctionCall { name, args, .. }
                if name == "*" && args.iter().any(|arg| matches!(arg, CompiledExpr::Symbol(name) if name == "x")))
        }));
        // The constant weight is still a primal operand of the requested gradient.
        assert!(requested.backward_bindings.iter().any(|(_, expr)| {
            matches!(expr, CompiledExpr::FunctionCall { name, args, .. }
                if name == "*" && args.iter().any(|arg| matches!(arg, CompiledExpr::Symbol(name) if name == "weight")))
        }));
        let both = reverse_grad(
            &bindings, &symbol("result"), &["x".to_string(), "weight".to_string()], &shapes,
        ).unwrap();
        assert!(matches!(both.gradients["weight"], GradientOutput::Computed(_)));
    }

    #[test]
    fn a_parameter_independent_output_needs_no_backward_bindings() {
        let bindings = vec![("result".to_string(), call("unknown", vec![symbol("constant")]))];
        let result = reverse_grad(
            &bindings, &symbol("result"), &["x".to_string()], &HashMap::new(),
        ).unwrap();
        assert!(result.backward_bindings.is_empty());
        assert_eq!(result.gradients["x"], GradientOutput::ProvenZero);
    }

    #[test]
    fn rejects_unknown_operation_depending_on_parameter() {
        let bindings = vec![("result".to_string(), call("unknown", vec![symbol("x")]))];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "unknown");
    }

    #[test]
    fn tracks_dependency_through_multiple_anf_bindings() {
        let bindings = vec![
            ("a".to_string(), call("unknown", vec![symbol("x")])),
            ("result".to_string(), call("+", vec![symbol("a"), CompiledExpr::Float(1.0)])),
        ];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "unknown");
    }

    #[test]
    fn rejects_nonliteral_power_exponent() {
        let bindings = vec![(
            "result".to_string(),
            call("**", vec![symbol("x"), symbol("exponent")]),
        )];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "**");
    }

    #[test]
    fn rejects_unsupported_einsum_form() {
        let bindings = vec![(
            "result".to_string(),
            call("einsum", vec![CompiledExpr::String("ij".to_string()), symbol("x"), symbol("y")]),
        )];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "einsum");
    }

    #[test]
    fn rejects_active_non_call_anf_form() {
        let bindings = vec![(
            "result".to_string(),
            CompiledExpr::If {
                condition: Box::new(CompiledExpr::Boolean(true)),
                then_branch: Box::new(symbol("x")),
                else_branch: Some(Box::new(CompiledExpr::Float(0.0))),
            },
        )];
        let error = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        missing_rule(error, "if");
    }

    #[test]
    fn permits_unknown_operation_independent_of_parameters() {
        let bindings = vec![("result".to_string(), call("unknown", vec![CompiledExpr::Float(1.0)]))];
        let ReverseGradResult {
            backward_bindings,
            gradients,
        } = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert!(backward_bindings.is_empty());
        assert!(matches!(
            gradients.get("x"),
            Some(GradientOutput::ProvenZero)
        ));
    }

    #[test]
    fn permits_unknown_operation_with_a_static_shape_argument() {
        let bindings = vec![
            ("shape".to_string(), call("shape", vec![symbol("x")])),
            ("classes".to_string(), call("last", vec![symbol("shape")])),
            (
                "result".to_string(),
                call("one-hot", vec![CompiledExpr::Integer(0), symbol("classes")]),
            ),
        ];
        let ReverseGradResult { gradients, .. } = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert!(matches!(
            gradients.get("x"),
            Some(GradientOutput::ProvenZero)
        ));
    }

    #[test]
    fn ignores_unknown_operation_in_dead_binding() {
        let bindings = vec![
            ("dead".to_string(), call("unknown", vec![symbol("x")])),
            ("result".to_string(), call("+", vec![symbol("x"), CompiledExpr::Float(1.0)])),
        ];
        let ReverseGradResult { gradients, .. } = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert!(matches!(
            gradients.get("x"),
            Some(GradientOutput::Computed(_))
        ));
    }

    #[test]
    fn stop_gradient_is_an_explicit_zero_gradient() {
        let bindings = vec![("result".to_string(), call("stop-gradient", vec![symbol("x")]))];
        let ReverseGradResult { gradients, .. } = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert!(matches!(
            gradients.get("x"),
            Some(GradientOutput::ProvenZero)
        ));
    }

    #[test]
    fn reports_computed_and_proven_zero_outputs_for_multiple_parameters() {
        let bindings = vec![(
            "result".to_string(),
            call("*", vec![symbol("active"), CompiledExpr::Float(2.0)]),
        )];
        let ReverseGradResult { gradients, .. } = reverse_grad(
            &bindings,
            &symbol("result"),
            &["active".to_string(), "unused".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert!(matches!(
            gradients.get("active"),
            Some(GradientOutput::Computed(_))
        ));
        assert!(matches!(
            gradients.get("unused"),
            Some(GradientOutput::ProvenZero)
        ));
    }

    #[test]
    fn rejects_a_missing_gradient_output() {
        let result = ReverseGradResult {
            backward_bindings: vec![],
            gradients: HashMap::new(),
        };
        let error = result.gradient_output("x").unwrap_err();
        assert!(matches!(
            error,
            SheafError::AutodiffMissingGradientOutput { symbol } if symbol == "x"
        ));
    }

    #[test]
    fn generates_deterministic_anf_names() {
        let expr = call("exp", vec![symbol("x")]);

        for anf in [to_anf(&expr), to_anf(&expr)] {
            let CompiledExpr::Let { bindings, body } = anf else {
                panic!("expected ANF binding");
            };
            assert_eq!(bindings.len(), 1);
            assert_eq!(bindings[0].0.as_simple(), Some("__anf_0"));
            assert!(matches!(*body, CompiledExpr::Symbol(ref name) if name == "__anf_0"));
        }
    }

    #[test]
    fn generates_deterministic_gradient_names() {
        let expr = call("exp", vec![symbol("x")]);
        let bindings = vec![("result".to_string(), expr)];
        let first = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();
        let second = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert_eq!(first.backward_bindings.len(), second.backward_bindings.len());
        for ((first_name, _), (second_name, _)) in first
            .backward_bindings
            .iter()
            .zip(&second.backward_bindings)
        {
            assert_eq!(first_name, second_name);
        }
        assert_eq!(first.gradients, second.gradients);
    }

    #[test]
    fn reuses_the_current_forward_result() {
        let bindings = vec![("result".to_string(), call("exp", vec![symbol("x")]))];
        let result = reverse_grad(
            &bindings,
            &symbol("result"),
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap();

        assert!(result.backward_bindings.iter().any(|(_, expr)| {
            let CompiledExpr::FunctionCall { name, args, .. } = expr else {
                return false;
            };
            name == "*"
                && args.iter().any(|arg| {
                    matches!(arg, CompiledExpr::Symbol(symbol) if symbol == "result")
                })
        }));
    }

    #[test]
    fn preserves_unknown_operation_name_and_location() {
        let location = SourceLocation::new(7, 3, Rc::from("loss.shf"));
        let expr = CompiledExpr::Let {
            bindings: vec![(BindingPattern::Simple("input".to_string()), symbol("x"))],
            body: Box::new(CompiledExpr::FunctionCall {
                name: "unknown".to_string(),
                args: vec![symbol("input")],
                loc: Some(location.clone()),
            }),
        };
        let anf = to_anf(&expr);
        let CompiledExpr::Let { bindings, body } = anf else {
            panic!("expected ANF binding");
        };
        let bindings: Vec<(String, CompiledExpr)> = bindings.into_iter().map(|(name, value)| {
            (name.as_simple().unwrap().to_string(), value)
        }).collect();
        let error = reverse_grad(
            &bindings,
            &body,
            &["x".to_string()],
            &HashMap::new(),
        ).unwrap_err();

        assert!(matches!(
            error,
            SheafError::AutodiffMissingRule {
                operation,
                location: Some(actual),
            } if operation == "unknown" && actual == location
        ));
    }
}
