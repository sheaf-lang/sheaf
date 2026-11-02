// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Function inlining and dictionary access folding.

use crate::core::bindings::{FreshBindings, pattern_names};
use crate::core::expr::{BindingPattern, CompiledExpr, FunctionDef};
use std::collections::{HashMap, HashSet};

pub fn inline_function_calls(
    expr: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
) -> CompiledExpr {
    let mut fresh = FreshBindings::default();
    let mut pending = vec![expr];
    let mut seen = HashSet::new();
    while let Some(body) = pending.pop() {
        fresh.reserve(body);
        crate::lowering::walk::walk_expr(body, &mut |node| {
            let name = match node {
                CompiledExpr::FunctionCall { name, .. } | CompiledExpr::FunctionRef(name) => name,
                _ => return,
            };
            if seen.insert(name.clone()) && let Some(function) = registry.get(name) {
                for param in &function.params { fresh.reserve_name(param); }
                if let Some(body) = &function.body_compiled { pending.push(body); }
            }
        });
    }
    let expr = fresh.rename(expr);
    Inliner { registry, fresh }.inline(&expr, &HashMap::new(), 0)
}

const MAX_INLINE_DEPTH: usize = 16;
type LocalFunctions = HashMap<String, CompiledExpr>;

struct Inliner<'a> {
    registry: &'a HashMap<String, FunctionDef>,
    fresh: FreshBindings,
}

fn shadow_pattern(pattern: &BindingPattern, locals: &mut LocalFunctions) {
    let mut names = HashSet::new();
    pattern_names(pattern, &mut names);
    for name in names { locals.remove(&name); }
}

impl Inliner<'_> {
    fn inline(&mut self, expr: &CompiledExpr, locals: &LocalFunctions, depth: usize) -> CompiledExpr {
        if depth > MAX_INLINE_DEPTH { return expr.clone(); }
        match expr {
            CompiledExpr::Symbol(name) => match locals.get(name) {
                Some(function @ CompiledExpr::Lambda { .. }) => function.clone(),
                _ => expr.clone(),
            },
            CompiledExpr::FunctionCall { name, args, loc } => {
                let args: Vec<_> = args.iter().map(|arg| self.inline(arg, locals, depth)).collect();
                if let Some(function) = self.registry.get(name)
                    && let Some(body) = &function.body_compiled {
                    let lambda = CompiledExpr::lambda_with_shared_body(function.params.clone(), body.clone());
                    let CompiledExpr::Lambda { params, body, .. } = self.fresh.rename(&lambda) else { unreachable!() };
                    let bindings = params.into_iter().zip(args)
                        .map(|(param, arg)| (BindingPattern::Simple(param), arg)).collect();
                    return self.inline(&CompiledExpr::Let {
                        bindings, body: Box::new(body.as_ref().clone()),
                    }, locals, depth + 1);
                }
                CompiledExpr::FunctionCall { name: name.clone(), args, loc: loc.clone() }
            }
            CompiledExpr::Let { bindings, body } => {
                let mut scoped = locals.clone();
                let mut new_bindings = Vec::with_capacity(bindings.len());
                for (pattern, value) in bindings {
                    let mut value = self.inline(value, &scoped, depth);
                    if let CompiledExpr::Symbol(name) = &value
                        && let Some(function) = scoped.get(name) {
                        value = function.clone();
                    }
                    shadow_pattern(pattern, &mut scoped);
                    if let BindingPattern::Simple(name) = pattern
                        && matches!(value, CompiledExpr::Lambda { .. }
                            | CompiledExpr::GlobalRef(_) | CompiledExpr::FunctionRef(_)) {
                        scoped.insert(name.clone(), value.clone());
                    }
                    new_bindings.push((pattern.clone(), value));
                }
                CompiledExpr::Let {
                    bindings: new_bindings,
                    body: Box::new(self.inline(body, &scoped, depth)),
                }
            }
            CompiledExpr::Lambda { params, body, .. } => {
                let mut scoped = locals.clone();
                for param in params { scoped.remove(param); }
                CompiledExpr::lambda(params.clone(), self.inline(body, &scoped, depth))
            }
            CompiledExpr::LambdaCall { callee, args } => {
                let args: Vec<_> = args.iter().map(|arg| self.inline(arg, locals, depth)).collect();
                let mut callee = self.inline(callee, locals, depth);
                if let CompiledExpr::Symbol(name) = &callee
                    && let Some(function) = locals.get(name) {
                    callee = function.clone();
                }
                match &callee {
                    CompiledExpr::GlobalRef(name) | CompiledExpr::FunctionRef(name) => {
                        self.inline(&CompiledExpr::FunctionCall {
                            name: name.clone(), args, loc: None,
                        }, locals, depth + 1)
                    }
                    CompiledExpr::Lambda { params, body, .. } if params.len() == args.len() => {
                        let bindings = params.iter().zip(args)
                            .map(|(param, arg)| (BindingPattern::Simple(param.clone()), arg)).collect();
                        self.inline(&CompiledExpr::Let {
                            bindings, body: Box::new(body.as_ref().clone()),
                        }, locals, depth + 1)
                    }
                    CompiledExpr::Let { bindings, body } => self.inline(&CompiledExpr::Let {
                        bindings: bindings.clone(),
                        body: Box::new(CompiledExpr::LambdaCall { callee: body.clone(), args }),
                    }, locals, depth + 1),
                    _ => CompiledExpr::LambdaCall { callee: Box::new(callee), args },
                }
            }
            CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => {
                let mut scoped = locals.clone();
                scoped.remove(index_var);
                scoped.remove(acc_var);
                CompiledExpr::Repeat {
                    index_var: index_var.clone(), count: Box::new(self.inline(count, locals, depth)),
                    acc_var: acc_var.clone(), acc_init: Box::new(self.inline(acc_init, locals, depth)),
                    body: Box::new(self.inline(body, &scoped, depth)),
                }
            }
            CompiledExpr::While { condition, acc_var, acc_init, body } => {
                let mut scoped = locals.clone();
                scoped.remove(acc_var);
                CompiledExpr::While {
                    condition: Box::new(self.inline(condition, &scoped, depth)),
                    acc_var: acc_var.clone(), acc_init: Box::new(self.inline(acc_init, locals, depth)),
                    body: Box::new(self.inline(body, &scoped, depth)),
                }
            }
            _ => expr.map_children(|child| self.inline(child, locals, depth)),
        }
    }
}

/// Fold `(get <dict_literal> :key)` into the dict value directly.
pub fn fold_dict_gets(expr: &CompiledExpr) -> CompiledExpr {
    fold_dict_gets_rec(expr, &HashMap::new())
}

fn fold_dict_gets_rec(
    expr: &CompiledExpr,
    dict_bindings: &HashMap<String, Vec<(CompiledExpr, CompiledExpr)>>,
) -> CompiledExpr {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            let mut new_dicts = dict_bindings.clone();
            let new_bindings: Vec<(BindingPattern, CompiledExpr)> = bindings
                .iter()
                .map(|(k, v)| {
                    let folded = fold_dict_gets_rec(v, &new_dicts);
                    if let (BindingPattern::Simple(name), CompiledExpr::Dict(pairs)) = (k, &folded) {
                        new_dicts.insert(name.clone(), pairs.clone());
                    }
                    (k.clone(), folded)
                })
                .collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(fold_dict_gets_rec(body, &new_dicts)),
            }
        }

        CompiledExpr::FunctionCall { name, args, loc } if name == "get" && args.len() == 2 => {
            // (get sym :key) where sym is bound to a dict literal
            if let (CompiledExpr::Symbol(dict_name), CompiledExpr::Keyword(key)) =
                (&args[0], &args[1])
                && let Some(pairs) = dict_bindings.get(dict_name.as_str())
            {
                for (k, v) in pairs
                {
                    if let CompiledExpr::Keyword(kw) = k
                        && kw == key
                    {
                        return fold_dict_gets_rec(v, dict_bindings);
                    }
                }
            }
            // Not foldable, recurse into args
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|a| fold_dict_gets_rec(a, dict_bindings)).collect(),
                loc: loc.clone(),
            }
        }

        CompiledExpr::FunctionCall { name, args, loc } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(|a| fold_dict_gets_rec(a, dict_bindings)).collect(),
            loc: loc.clone(),
        },

        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs.iter().map(|e| fold_dict_gets_rec(e, dict_bindings)).collect(),
        ),

        CompiledExpr::If { condition, then_branch, else_branch } => CompiledExpr::If {
            condition: Box::new(fold_dict_gets_rec(condition, dict_bindings)),
            then_branch: Box::new(fold_dict_gets_rec(then_branch, dict_bindings)),
            else_branch: else_branch.as_ref().map(|e| Box::new(fold_dict_gets_rec(e, dict_bindings))),
        },

        CompiledExpr::Lambda { params, body, .. } => CompiledExpr::lambda(
            params.clone(), fold_dict_gets_rec(body, dict_bindings),
        ),

        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems.iter().map(|e| fold_dict_gets_rec(e, dict_bindings)).collect(),
        ),

        _ => expr.clone(),
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    #[test]
    fn unrelated_definitions_do_not_change_local_names() {
        let expr = CompiledExpr::Let {
            bindings: vec![(BindingPattern::Simple("x".into()), CompiledExpr::Integer(1))],
            body: Box::new(CompiledExpr::Symbol("x".into())),
        };
        let mut compiler = crate::core::expr::CompilerContext::new();
        let source = crate::core::parse("(defn unused [__lexical_0_x] __lexical_0_x)", "<test>").unwrap();
        compiler.compile(&source[0]).unwrap();
        let without = inline_function_calls(&expr, &HashMap::new());
        let with = inline_function_calls(&expr, &compiler.registry);
        let (CompiledExpr::Let { bindings: a, .. }, CompiledExpr::Let { bindings: b, .. }) = (without, with)
            else { panic!() };
        assert_eq!(a[0].0, b[0].0);
    }
}
