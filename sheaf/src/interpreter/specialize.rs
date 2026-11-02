// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Read module dependencies for a compilation request, without changing runtime closures.

use super::{Env, Value};
use crate::core::error::SheafError;
use super::env::runtime_error;
use crate::core::expr::{BindingPattern, CompiledExpr};
#[cfg(iree_runtime)]
use crate::core::expr::FunctionDef;
use crate::core::signature::OpId;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

type Locals = HashMap<String, CompiledExpr>;

#[derive(Clone)]
struct Code(Rc<CompiledExpr>);

impl PartialEq for Code {
    fn eq(&self, other: &Self) -> bool { Rc::ptr_eq(&self.0, &other.0) }
}

#[derive(PartialEq)]
enum ValueKey {
    Integer(i64),
    Boolean(bool),
    Function(super::value::FunctionKind, Code, Vec<String>, Vec<(String, ValueKey)>),
    Builtin(String, usize),
    Keyword(String),
    String(String),
    Nil,
    Dict(Vec<(String, ValueKey)>),
    List(Vec<ValueKey>),
    Tuple(Vec<ValueKey>),
    Data(Option<crate::lowering::stablehlo::StableHLOType>),
}

impl ValueKey {
    fn of(value: &Value) -> Self {
        match value {
            Value::Int(value) => Self::Integer(*value),
            Value::Bool(value) => Self::Boolean(*value),
            Value::Function { kind, params, body, closure, .. } => Self::Function(
                *kind, Code(body.clone()), params.clone(),
                closure.iter().map(|(name, value)| (name.clone(), Self::of(value))).collect(),
            ),
            Value::BuiltinFn { name, func } => Self::Builtin(name.clone(), *func as usize),
            Value::Keyword(value) => Self::Keyword(value.clone()),
            Value::String(value) => Self::String(value.clone()),
            Value::Nil => Self::Nil,
            Value::Dict(values) => Self::Dict(values.iter().map(|(name, value)| (name.clone(), Self::of(value))).collect()),
            Value::List(_) if value.numeric_list_elements().is_some() => {
                Self::Data(crate::core::trace::value_to_stablehlo_type(value).ok())
            }
            Value::List(values) => Self::List(values.iter().map(Self::of).collect()),
            Value::Tuple(values) => Self::Tuple(values.iter().map(Self::of).collect()),
            _ => Self::Data(crate::core::trace::value_to_stablehlo_type(value).ok()),
        }
    }
}

#[derive(Clone)]
enum InputSource {
    Argument(usize),
    Global(String),
    Capture(Box<InputSource>, usize),
}

impl InputSource {
    fn read(&self, args: &[Value], env: &Env) -> Result<Value, SheafError> {
        match self {
            Self::Argument(index) => args.get(*index).cloned().ok_or_else(|| runtime_error("invalid prepared argument")),
            Self::Global(name) => env.module_value(name),
            Self::Capture(parent, index) => {
                let Value::Function { closure, .. } = parent.read(args, env)? else {
                    return Err(runtime_error("invalid prepared capture"));
                };
                closure.get(*index).map(|(_, value)| value.clone())
                    .ok_or_else(|| runtime_error("invalid prepared capture index"))
            }
        }
    }
}

#[cfg(iree_runtime)]
pub(super) struct CachedPreparation {
    source: Code,
    parameters: Vec<String>,
    arguments: Vec<ValueKey>,
    globals: HashMap<String, Option<ValueKey>>,
    inputs: Vec<InputSource>,
    definition: FunctionDef,
    shape_positions: Vec<(String, Vec<usize>)>,
}

#[cfg(iree_runtime)]
impl CachedPreparation {
    fn values(&self, args: &[Value], env: &Env) -> Result<Vec<Value>, SheafError> {
        self.inputs.iter().map(|source| source.read(args, env)).collect()
    }
}

pub(super) enum Preparation<T> {
    Ready(T),
    UnboundGlobal(String),
    RecursiveGlobal(String),
    GeneratedHigherOrderClosure,
}

impl<T> Preparation<T> {
    pub(super) fn ready(self) -> Option<T> {
        match self {
            Self::Ready(value) => Some(value),
            Self::UnboundGlobal(name) => {
                if crate::core::config::verbosity() > 0 {
                    crate::sheaf_msg!("jit: skipped (unbound module symbol: {})", name);
                }
                None
            }
            Self::RecursiveGlobal(name) => {
                if crate::core::config::verbosity() > 0 {
                    crate::sheaf_msg!("jit: skipped (recursive call through '{}')", name);
                }
                None
            }
            Self::GeneratedHigherOrderClosure => {
                if crate::core::config::verbosity() > 0 {
                    crate::sheaf_msg!("jit: skipped (generated higher-order closure)");
                }
                None
            }
        }
    }

    #[cfg(test)]
    pub(super) fn expect_ready(self) -> T {
        match self {
            Self::Ready(value) => value,
            Self::UnboundGlobal(name) => panic!("unbound module symbol: {}", name),
            Self::RecursiveGlobal(name) => panic!("recursive module function: {}", name),
            Self::GeneratedHigherOrderClosure => panic!("generated higher-order closure"),
        }
    }
}

struct Snapshot<'a> {
    env: &'a Env,
    inputs: Vec<(String, Value, InputSource)>,
    globals: HashMap<String, CompiledExpr>,
    dependencies: HashMap<String, Option<ValueKey>>,
    changed: bool,
    active_globals: HashSet<String>,
    unbound_global: Option<String>,
    recursive_global: Option<String>,
    generated_higher_order_closure: bool,
    next_input: usize,
}

impl Snapshot<'_> {
    fn input(&mut self, value: &Value, source: InputSource) -> CompiledExpr {
        // The tokenizer cannot produce a symbol containing a parenthesis.
        let name = format!("(module_input_{})", self.next_input);
        self.next_input += 1;
        self.inputs.push((name.clone(), value.clone(), source));
        self.changed = true;
        CompiledExpr::Symbol(name)
    }

    fn value(&mut self, value: &Value, source: InputSource) -> Result<CompiledExpr, SheafError> {
        match value {
            Value::Int(integer) => {
                self.input(value, source);
                Ok(CompiledExpr::Integer(*integer))
            }
            Value::Bool(boolean) => {
                self.input(value, source);
                Ok(CompiledExpr::Boolean(*boolean))
            }
            Value::Function { kind, params, body, closure, .. } => {
                if *kind != super::value::FunctionKind::User {
                    self.generated_higher_order_closure = true;
                    return Ok(CompiledExpr::Nil);
                }
                let mut locals = Locals::new();
                for (index, (name, value)) in closure.iter().enumerate() {
                    locals.insert(name.clone(), self.value(value, InputSource::Capture(Box::new(source.clone()), index))?);
                }
                for param in params {
                    locals.remove(param);
                }
                self.rewrite(body, &locals).map(|body| CompiledExpr::lambda(params.clone(), body))
            }
            Value::BuiltinFn { name, .. } => Ok(CompiledExpr::FunctionRef(name.clone())),
            Value::Keyword(value) => Ok(CompiledExpr::Keyword(value.clone())),
            Value::String(value) => Ok(CompiledExpr::String(value.clone())),
            Value::Nil => Ok(CompiledExpr::Nil),
            _ => Ok(self.input(value, source)),
        }
    }

    fn global(&mut self, name: &str) -> Result<CompiledExpr, SheafError> {
        if let Some(expr) = self.globals.get(name) {
            return Ok(expr.clone());
        }
        let value = self.env.module_value(name);
        self.dependencies.insert(name.to_string(), value.as_ref().ok().map(ValueKey::of));
        let value = match value {
            Ok(value) => value,
            Err(_) if OpId::resolve(name).is_some() => {
                return Ok(CompiledExpr::FunctionRef(name.to_string()));
            }
            Err(_) => {
                self.unbound_global.get_or_insert_with(|| name.to_string());
                return Ok(CompiledExpr::GlobalRef(name.to_string()));
            }
        };
        if !self.active_globals.insert(name.to_string()) {
            self.recursive_global.get_or_insert_with(|| name.to_string());
            return Ok(CompiledExpr::FunctionRef(name.to_string()));
        }
        let expr = self.value(&value, InputSource::Global(name.to_string()))?;
        self.active_globals.remove(name);
        self.globals.insert(name.to_string(), expr.clone());
        Ok(expr)
    }

    fn shadow(pattern: &BindingPattern, locals: &mut Locals) {
        let mut names = HashSet::new();
        crate::core::bindings::pattern_names(pattern, &mut names);
        for name in names {
            locals.remove(&name);
        }
    }

    fn rewrite(&mut self, expr: &CompiledExpr, locals: &Locals) -> Result<CompiledExpr, SheafError> {
        match expr {
            CompiledExpr::Symbol(name) => {
                if let Some(value) = locals.get(name) {
                    self.changed = true;
                    Ok(value.clone())
                } else {
                    Ok(expr.clone())
                }
            }
            CompiledExpr::GetTupleElement { param, indices } => {
                if let Some(CompiledExpr::Symbol(name)) = locals.get(param) {
                    self.changed = true;
                    Ok(CompiledExpr::GetTupleElement { param: name.clone(), indices: indices.clone() })
                } else {
                    Ok(expr.clone())
                }
            }
            CompiledExpr::GlobalRef(name) | CompiledExpr::FunctionRef(name) => {
                self.changed = true;
                self.global(name)
            }
            CompiledExpr::FunctionCall { name, args, loc } => {
                self.dependencies.entry(name.clone()).or_insert_with(|| self.env.module_value(name).ok().as_ref().map(ValueKey::of));
                if let Some(Value::BuiltinFn { name: builtin, .. }) = self.env.get_global(name) {
                    self.changed |= builtin != *name;
                    let args = args.iter().map(|arg| self.rewrite(arg, locals)).collect::<Result<Vec<_>, _>>()?;
                    return Ok(CompiledExpr::FunctionCall { name: builtin, args, loc: loc.clone() });
                }
                if self.env.get_global(name).is_none() && OpId::resolve(name).is_some() {
                    let args = args.iter().map(|arg| self.rewrite(arg, locals)).collect::<Result<Vec<_>, _>>()?;
                    return Ok(CompiledExpr::FunctionCall { name: name.clone(), args, loc: loc.clone() });
                }
                let callee = self.global(name)?;
                if matches!(callee, CompiledExpr::GlobalRef(_)) {
                    let args = args.iter().map(|arg| self.rewrite(arg, locals)).collect::<Result<Vec<_>, _>>()?;
                    return Ok(CompiledExpr::FunctionCall { name: name.clone(), args, loc: loc.clone() });
                }
                self.changed = true;
                let mut positional = Vec::new();
                let mut evaluated_options = Vec::new();
                let mut index = 0;
                while index < args.len() {
                    if super::uses_kwargs(name) && matches!(&args[index], CompiledExpr::Keyword(_)) {
                        index += 1;
                        if index < args.len() && !matches!(&args[index], CompiledExpr::Keyword(_)) {
                            evaluated_options.push(self.rewrite(&args[index], locals)?);
                            index += 1;
                        }
                    } else {
                        positional.push(self.rewrite(&args[index], locals)?);
                        index += 1;
                    }
                }
                let call = CompiledExpr::LambdaCall { callee: Box::new(callee), args: positional };
                if evaluated_options.is_empty() {
                    Ok(call)
                } else {
                    evaluated_options.push(call);
                    Ok(CompiledExpr::Do(evaluated_options))
                }
            }
            CompiledExpr::Let { bindings, body } => {
                let mut scoped = locals.clone();
                let mut rewritten = Vec::with_capacity(bindings.len());
                for (pattern, value) in bindings {
                    rewritten.push((pattern.clone(), self.rewrite(value, &scoped)?));
                    Self::shadow(pattern, &mut scoped);
                }
                Ok(CompiledExpr::Let { bindings: rewritten, body: Box::new(self.rewrite(body, &scoped)?) })
            }
            CompiledExpr::Lambda { params, body, .. } => {
                let mut scoped = locals.clone();
                for param in params {
                    scoped.remove(param);
                }
                Ok(CompiledExpr::lambda(params.clone(), self.rewrite(body, &scoped)?))
            }
            CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => {
                let mut scoped = locals.clone();
                scoped.remove(index_var);
                scoped.remove(acc_var);
                Ok(CompiledExpr::Repeat {
                    index_var: index_var.clone(), count: Box::new(self.rewrite(count, locals)?),
                    acc_var: acc_var.clone(), acc_init: Box::new(self.rewrite(acc_init, locals)?),
                    body: Box::new(self.rewrite(body, &scoped)?),
                })
            }
            CompiledExpr::While { condition, acc_var, acc_init, body } => {
                let mut scoped = locals.clone();
                scoped.remove(acc_var);
                Ok(CompiledExpr::While {
                    condition: Box::new(self.rewrite(condition, &scoped)?),
                    acc_var: acc_var.clone(), acc_init: Box::new(self.rewrite(acc_init, locals)?),
                    body: Box::new(self.rewrite(body, &scoped)?),
                })
            }
            CompiledExpr::If { condition, then_branch, else_branch } => {
                let condition = self.rewrite(condition, locals)?;
                if let Some(branch) = crate::core::bindings::constant_branch(
                    &condition, then_branch, else_branch.as_deref(),
                ) {
                    self.changed = true;
                    self.rewrite(&branch, locals)
                } else {
                    Ok(CompiledExpr::If {
                        condition: Box::new(condition),
                        then_branch: Box::new(self.rewrite(then_branch, locals)?),
                        else_branch: else_branch.as_ref()
                            .map(|branch| self.rewrite(branch, locals).map(Box::new)).transpose()?,
                    })
                }
            }
            _ => {
                let mut error = None;
                let rewritten = expr.map_children(|child| match self.rewrite(child, locals) {
                    Ok(child) => child,
                    Err(err) => {
                        error = Some(err);
                        child.clone()
                    }
                });
                match error {
                    Some(error) => Err(error),
                    None => Ok(rewritten),
                }
            }
        }
    }
}

fn snapshot(env: &Env) -> Snapshot<'_> {
    Snapshot {
        env,
        inputs: Vec::new(),
        globals: HashMap::new(),
        dependencies: HashMap::new(),
        changed: false,
        active_globals: HashSet::new(),
        unbound_global: None,
        recursive_global: None,
        generated_higher_order_closure: false,
        next_input: 0,
    }
}

pub(super) fn prepare_vag(function: &Value, env: &Env) -> Result<Preparation<Value>, SheafError> {
    let mut snapshot = snapshot(env);
    let lambda = snapshot.value(function, InputSource::Argument(0))?;
    if snapshot.generated_higher_order_closure {
        return Ok(Preparation::GeneratedHigherOrderClosure);
    }
    let CompiledExpr::Lambda { params, body, .. } = lambda else {
        return Err(runtime_error("value-and-grad: expected a function"));
    };
    if let Some(name) = snapshot.unbound_global {
        return Ok(Preparation::UnboundGlobal(name));
    }
    if let Some(name) = snapshot.recursive_global {
        return Ok(Preparation::RecursiveGlobal(name));
    }
    let closure = snapshot.inputs.into_iter().map(|(name, _, source)| {
        source.read(std::slice::from_ref(function), env).map(|value| (name, value))
    }).collect::<Result<Vec<_>, _>>()?;
    Ok(Preparation::Ready(Value::Function { kind: super::value::FunctionKind::User, name: None, params, body, closure }))
}

#[cfg(iree_runtime)]
pub(super) fn prepare_function(
    function: &FunctionDef,
    args: &[Value],
    env: &Env,
) -> Result<Preparation<(FunctionDef, Vec<Value>)>, SheafError> {
    let keys: Vec<_> = args.iter().map(ValueKey::of).collect();
    let mut cache = env.preparation_cache.borrow_mut();
    let hit = cache.iter().position(|entry| {
        entry.definition.name == function.name && entry.parameters == function.params
            && function.body_compiled.as_ref().is_some_and(|body| Rc::ptr_eq(body, &entry.source.0))
            && entry.arguments == keys
            && entry.globals.iter().all(|(name, key)| &env.module_value(name).ok().as_ref().map(ValueKey::of) == key)
    });
    if let Some(index) = hit {
        let entry = cache.remove(index).expect("preparation cache index");
        let values = entry.values(args, env)?;
        crate::runtime::jit::validate_shape_positions(&entry.definition.params, &values, entry.shape_positions.clone())?;
        let prepared = entry.definition.clone();
        cache.push_front(entry);
        return Ok(Preparation::Ready((prepared, values)));
    }
    drop(cache);
    let mut snapshot = snapshot(env);
    let mut locals = Locals::new();
    let mut params = Vec::new();
    let mut values = Vec::new();
    let mut inputs = Vec::new();
    for (index, (param, value)) in function.params.iter().zip(args).enumerate() {
        if matches!(value, Value::Function { .. } | Value::BuiltinFn { .. }) {
            locals.insert(param.clone(), snapshot.value(value, InputSource::Argument(index))?);
        } else {
            params.push(param.clone());
            values.push(value.clone());
            inputs.push(InputSource::Argument(index));
        }
    }
    let mut prepared = function.clone();
    if let Some(body) = &function.body_compiled {
        let specialized = crate::runtime::jit::static_inputs::specialize(body, &function.params, args);
        prepared.body_compiled = Some(Rc::new(snapshot.rewrite(&specialized, &locals)?));
    }
    if snapshot.generated_higher_order_closure {
        return Ok(Preparation::GeneratedHigherOrderClosure);
    }
    if let Some(name) = snapshot.unbound_global {
        return Ok(Preparation::UnboundGlobal(name));
    }
    if let Some(name) = snapshot.recursive_global {
        return Ok(Preparation::RecursiveGlobal(name));
    }
    let mut structured_params = Vec::new();
    let mut structured_values = Vec::new();
    for (name, value, source) in snapshot.inputs {
        if matches!(value, Value::Dict(_) | Value::Tuple(_) | Value::List(_)) {
            structured_params.push(name.clone());
            structured_values.push(value.clone());
        }
        params.push(name);
        values.push(value);
        inputs.push(source);
    }
    let has_static_inputs = !crate::runtime::jit::static_inputs::collect(&values).is_empty();
    if !structured_params.is_empty() && let Some(body) = &prepared.body_compiled {
        prepared.body_compiled = Some(Rc::new(
            crate::runtime::jit::static_inputs::specialize(body, &structured_params, &structured_values),
        ));
    }
    let changed = snapshot.changed || has_static_inputs || params != function.params;
    prepared.params = params;
    if changed {
        prepared.signature = None;
        prepared.known_param_types.clear();
    } else {
        prepared.body_compiled = function.body_compiled.clone();
    }
    let shape_positions = crate::runtime::jit::shape_argument_positions(&prepared, &values, &env.registry)
        .unwrap_or_default();
    crate::runtime::jit::validate_shape_positions(&prepared.params, &values, shape_positions.clone())?;
    if let Some(body) = &function.body_compiled {
        let mut cache = env.preparation_cache.borrow_mut();
        cache.push_front(CachedPreparation { source: Code(body.clone()), parameters: function.params.clone(),
            arguments: keys, globals: snapshot.dependencies, inputs, definition: prepared.clone(), shape_positions });
        cache.truncate(32);
    }
    Ok(Preparation::Ready((prepared, values)))
}
