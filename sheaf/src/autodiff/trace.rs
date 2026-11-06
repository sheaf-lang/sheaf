// Copyright (c) 2025-2026 Damien Boureille
// Licensed under the MIT License.

//! Concrete execution and ANF recording for interpreted reverse-mode autodiff.

use crate::core::error::{SheafError, SourceLocation};
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::signature::OpId;
use crate::interpreter::env::{Env, runtime_error};
use crate::interpreter::eval;
use crate::interpreter::hof::dict_scan_length;
use crate::interpreter::value::Value;
use std::collections::{HashMap, HashSet};

type Scope = HashMap<String, CompiledExpr>;

/// Numeric parameter leaves and a shared allocator for trace-local names.
pub(crate) struct LeafMap {
    pub leaves: Vec<(String, Value)>,
    reserved: HashSet<String>,
    lists: HashMap<String, Vec<CompiledExpr>>,
    dependencies: HashMap<String, bool>,
    next: usize,
}

impl LeafMap {
    pub fn new(env: &Env) -> Self {
        Self {
            leaves: Vec::new(),
            reserved: env.all_names().into_iter().collect(),
            lists: HashMap::new(),
            dependencies: HashMap::new(),
            next: 0,
        }
    }

    fn fresh(&mut self) -> String {
        loop {
            let name = format!("__ad_trace_{}__", self.next);
            self.next += 1;
            if self.reserved.insert(name.clone()) {
                return name;
            }
        }
    }

    pub fn register(&mut self, value: Value, env: &mut Env) -> String {
        let name = self.store(value.clone(), env);
        self.leaves.push((name.clone(), value));
        self.dependencies.insert(name.clone(), true);
        name
    }

    pub fn store_list(&mut self, value: Value, items: Vec<CompiledExpr>, env: &mut Env) -> String {
        let dependent = items.iter().any(|item| {
            crate::autodiff::analysis::depends_on_in_env(item, &self.dependencies)
        });
        let name = self.store(value, env);
        self.dependencies.insert(name.clone(), dependent);
        self.lists.insert(name.clone(), items);
        name
    }

    pub fn store(&mut self, value: Value, env: &mut Env) -> String {
        let name = self.fresh();
        env.set(&name, value);
        name
    }
}

/// Each binding has already been evaluated once, in recording order.
pub(crate) struct TracedProgram {
    pub bindings: Vec<(String, CompiledExpr)>,
    pub output: CompiledExpr,
    pub value: Value,
    pub shapes: HashMap<String, Vec<i64>>,
}

pub(crate) fn numeric_shape(value: &Value) -> Option<Vec<i64>> {
    match value {
        Value::Tensor { data, .. } => Some(data.shape().iter().map(|&size| size as i64).collect()),
        Value::DeviceBuffer(buffer) => Some(buffer.shape.iter().map(|&size| size as i64).collect()),
        Value::Float(_) | Value::Int(_) | Value::Bool(_) => Some(Vec::new()),
        _ => None,
    }
}

/// Trace a function value with its lexical closure, without constructing an expression tree.
pub(crate) fn trace_function(
    function: &Value,
    argument: CompiledExpr,
    env: &mut Env,
    leaves: &mut LeafMap,
) -> Result<TracedProgram, SheafError> {
    let mut shapes = HashMap::new();
    let dependencies = leaves.dependencies.clone();
    for (name, value) in &leaves.leaves {
        if let Some(shape) = numeric_shape(value) {
            shapes.insert(name.clone(), shape);
        }
    }
    let mut recorder = Recorder {
        env,
        leaves,
        bindings: Vec::new(),
        shapes,
        dependencies,
        closures: HashMap::new(),
        depth: 0,
    };
    let function = recorder.import(function)?;
    let output = recorder.invoke(&function, vec![argument])?;
    let value = recorder.value(&output)?;
    Ok(TracedProgram {
        bindings: recorder.bindings,
        output,
        value,
        shapes: recorder.shapes,
    })
}

struct Recorder<'a> {
    env: &'a mut Env,
    leaves: &'a mut LeafMap,
    bindings: Vec<(String, CompiledExpr)>,
    shapes: HashMap<String, Vec<i64>>,
    dependencies: HashMap<String, bool>,
    closures: HashMap<String, Scope>,
    depth: usize,
}

impl Recorder<'_> {
    fn value(&mut self, expr: &CompiledExpr) -> Result<Value, SheafError> {
        // Operands are literals, aggregate references or names of already executed operations.
        eval(expr, self.env)
    }

    fn dependent(&self, expr: &CompiledExpr) -> bool {
        crate::autodiff::analysis::depends_on_in_env(expr, &self.dependencies)
    }

    fn active_list_argument(&self, expr: &CompiledExpr) -> bool {
        if let CompiledExpr::Symbol(name) = expr {
            return self.leaves.lists.contains_key(name)
                && self.dependencies.get(name).copied().unwrap_or(false);
        }
        let mut found = false;
        expr.for_each_child(|child| found |= self.active_list_argument(child));
        found
    }

    fn store(&mut self, value: Value) -> CompiledExpr {
        let shape = numeric_shape(&value);
        let name = self.leaves.store(value, self.env);
        if let Some(shape) = shape {
            self.shapes.insert(name.clone(), shape);
        }
        CompiledExpr::Symbol(name)
    }

    fn import(&mut self, value: &Value) -> Result<CompiledExpr, SheafError> {
        match value {
            Value::Dict(values) => {
                let mut pairs = Vec::with_capacity(values.len());
                for (key, value) in values {
                    pairs.push((CompiledExpr::Keyword(key.clone()), self.import(value)?));
                }
                Ok(CompiledExpr::Dict(pairs))
            }
            Value::List(values) => {
                let items = values.iter().map(|value| self.import(value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.list(items)
            }
            Value::Tuple(values) => {
                let values = values.iter().map(|value| self.import(value))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(CompiledExpr::Tuple(values))
            }
            _ => Ok(self.store(value.clone())),
        }
    }

    fn list(&mut self, items: Vec<CompiledExpr>) -> Result<CompiledExpr, SheafError> {
        let dependent = items.iter().any(|item| self.dependent(item));
        let values = items.iter().map(|item| self.value(item)).collect::<Result<Vec<_>, _>>()?;
        let name = self.leaves.store_list(Value::List(values), items, self.env);
        self.dependencies.insert(name.clone(), dependent);
        Ok(CompiledExpr::Symbol(name))
    }

    fn emit(&mut self, expr: CompiledExpr) -> Result<CompiledExpr, SheafError> {
        let dependent = self.dependent(&expr);
        let value = self.value(&expr)?;
        if matches!(value, Value::Function { .. }) {
            return Err(runtime_error("autodiff: an opaque operation returned a function"));
        }
        let CompiledExpr::Symbol(name) = self.store(value) else { unreachable!() };
        self.dependencies.insert(name.clone(), dependent);
        self.bindings.push((name.clone(), expr));
        Ok(CompiledExpr::Symbol(name))
    }

    fn atomic_call(
        &mut self,
        name: &str,
        args: Vec<CompiledExpr>,
        loc: Option<SourceLocation>,
    ) -> Result<CompiledExpr, SheafError> {
        let mut normalized = Vec::with_capacity(args.len());
        let mut option = None;
        for (index, arg) in args.into_iter().enumerate() {
            if let CompiledExpr::Keyword(keyword) = &arg {
                option = Some(keyword.clone());
                normalized.push(arg);
                continue;
            }
            if matches!(option.as_deref(), Some("axis" | "keepdims")) {
                let value = self.value(&arg)?;
                normalized.push(static_expr(&value)?);
            } else if matches!(&arg, CompiledExpr::Vector(_)) && self.dependent(&arg)
                && !crate::core::signature::OpId::resolve(name)
                    .is_some_and(|op| op.is_shape_vector_argument(index))
            {
                normalized.push(self.tensor_vector(&arg)?);
            } else {
                normalized.push(arg);
            }
            option = None;
        }
        let expr = CompiledExpr::FunctionCall { name: name.to_string(), args: normalized, loc };
        if matches!(name, "shape" | "ndim" | "len" | "count") {
            return static_expr(&self.value(&expr)?);
        }
        if self.active_list_argument(&expr) {
            let location = match &expr {
                CompiledExpr::FunctionCall { loc, .. } => loc.clone(),
                _ => None,
            };
            return Err(SheafError::AutodiffMissingRule { operation: name.to_string(), location });
        }
        self.emit(expr)
    }

    fn tensor_vector(&mut self, expr: &CompiledExpr) -> Result<CompiledExpr, SheafError> {
        let CompiledExpr::Vector(values) = expr else { return Ok(expr.clone()) };
        if values.is_empty() {
            return Ok(expr.clone());
        }
        let mut rows = Vec::with_capacity(values.len());
        for value in values {
            let value = if matches!(value, CompiledExpr::Vector(_)) {
                self.tensor_vector(value)?
            } else {
                value.clone()
            };
            let concrete = self.value(&value)?;
            let mut shape = vec![1];
            shape.extend(numeric_shape(&concrete)
                .ok_or_else(|| runtime_error("autodiff: tensor vector contains a nonnumeric value"))?);
            let row = self.atomic_call("reshape", vec![value, shape_expr(&shape)], None)?;
            rows.push(row);
        }
        self.atomic_call("concat", rows, None)
    }

    fn lookup(&mut self, name: &str, scope: &Scope) -> Result<CompiledExpr, SheafError> {
        if let Some(value) = scope.get(name) {
            return Ok(value.clone());
        }
        Err(runtime_error(format!("Undefined symbol: {name}")))
    }

    fn global(&mut self, name: &str) -> Result<CompiledExpr, SheafError> {
        let value = self.env.module_value(name)?;
        self.import(&value)
    }

    fn invoke(
        &mut self,
        callee: &CompiledExpr,
        args: Vec<CompiledExpr>,
    ) -> Result<CompiledExpr, SheafError> {
        let function = self.value(callee)?;
        match function {
            Value::BuiltinFn { name, .. } => self.builtin(&name, args, None),
            Value::Function { kind, params, body, closure, .. } => {
                if kind == crate::interpreter::value::FunctionKind::Vmap {
                    let function = closure.iter().find(|(name, _)| name == "__vmap_fn__")
                        .ok_or_else(|| runtime_error("autodiff: vmap function is missing"))?;
                    let function = self.import(&function.1)?;
                    let axes = closure.iter().find(|(name, _)| name == "__vmap_axes__")
                        .map(|(_, value)| value);
                    return self.vmap(&function, axes, args);
                }
                if kind != crate::interpreter::value::FunctionKind::User {
                    return Err(runtime_error(
                        "autodiff: tracing a generated higher-order closure is not supported",
                    ));
                }
                if params.len() != args.len() {
                    return Err(runtime_error(format!(
                        "autodiff: function expected {} arguments, got {}", params.len(), args.len(),
                    )));
                }
                if self.depth >= 1024 {
                    return Err(runtime_error("autodiff: function trace recursion limit exceeded"));
                }
                let mut scope = match callee {
                    CompiledExpr::Symbol(name) => self.closures.get(name).cloned().unwrap_or_default(),
                    _ => Scope::new(),
                };
                for (name, value) in closure {
                    if let std::collections::hash_map::Entry::Vacant(entry) = scope.entry(name) {
                        entry.insert(self.import(&value)?);
                    }
                }
                scope.extend(params.into_iter().zip(args));
                self.depth += 1;
                let result = self.trace(&body, &scope);
                self.depth -= 1;
                result
            }
            value => Err(runtime_error(format!(
                "autodiff: expected a function, got {}", value.type_name(),
            ))),
        }
    }

    fn vmap(
        &mut self,
        function: &CompiledExpr,
        axes: Option<&Value>,
        args: Vec<CompiledExpr>,
    ) -> Result<CompiledExpr, SheafError> {
        let mapped = match axes {
            None | Some(Value::Int(0)) => vec![true; args.len()],
            Some(Value::List(values)) => values.iter().map(|axis| match axis {
                Value::Nil => Ok(false),
                Value::Int(0) => Ok(true),
                _ => Err(runtime_error("autodiff: traced vmap currently requires axis 0 or nil")),
            }).collect::<Result<Vec<_>, _>>()?,
            _ => return Err(runtime_error("autodiff: traced vmap currently requires axis 0")),
        };
        if mapped.len() != args.len() {
            return Err(runtime_error("vmap: in-axes count does not match argument count"));
        }
        let mut batch = None;
        let mut arguments = Vec::with_capacity(args.len());
        for (arg, mapped) in args.iter().zip(mapped) {
            let value = self.value(arg)?;
            if mapped && let Some(shape) = numeric_shape(&value).filter(|shape| !shape.is_empty()) {
                let len = shape[0] as usize;
                if batch.is_some_and(|batch| batch != len) {
                    return Err(runtime_error("vmap: mapped arguments have different batch sizes"));
                }
                batch = Some(len);
                arguments.push(Some(self.items(arg)?));
            } else {
                arguments.push(None);
            }
        }
        let batch = batch.ok_or_else(|| runtime_error("vmap: expected a mapped tensor"))?;
        let mut outputs = Vec::with_capacity(batch);
        for index in 0..batch {
            let args = args.iter().zip(&arguments).map(|(arg, items)| {
                items.as_ref().map(|items| items[index].clone()).unwrap_or_else(|| arg.clone())
            }).collect();
            outputs.push(self.invoke(function, args)?);
        }
        self.tensor_vector(&CompiledExpr::Vector(outputs))
    }

    fn lambda(
        &mut self,
        params: &[String],
        captures: &[String],
        body: &std::rc::Rc<CompiledExpr>,
        scope: &Scope,
    ) -> Result<CompiledExpr, SheafError> {
        let mut closure = Scope::new();
        for name in captures {
            closure.insert(name.clone(), self.lookup(name, scope)?);
        }
        let expr = self.store(Value::Function {
            kind: crate::interpreter::value::FunctionKind::User,
            name: None, params: params.to_vec(), body: body.clone(), closure: Vec::new(),
        });
        if let CompiledExpr::Symbol(name) = &expr {
            self.closures.insert(name.clone(), closure);
        }
        Ok(expr)
    }

    fn trace(&mut self, expr: &CompiledExpr, scope: &Scope) -> Result<CompiledExpr, SheafError> {
        match expr {
            CompiledExpr::Symbol(name) => self.lookup(name, scope),
            CompiledExpr::GlobalRef(name) | CompiledExpr::FunctionRef(name) => self.global(name),
            CompiledExpr::Integer(_) | CompiledExpr::Float(_) | CompiledExpr::Boolean(_)
            | CompiledExpr::Nil | CompiledExpr::String(_) | CompiledExpr::Keyword(_) => Ok(expr.clone()),
            CompiledExpr::Quoted(_) => {
                let value = self.value(expr)?;
                self.import(&value)
            }
            CompiledExpr::FunctionCall { name, args, loc } => {
                if matches!(name.as_str(), "and" | "or") {
                    let mut last = CompiledExpr::Boolean(name == "and");
                    for arg in args {
                        last = self.trace(arg, scope)?;
                        if self.value(&last)?.is_truthy() == (name == "or") {
                            break;
                        }
                    }
                    return Ok(last);
                }
                let args = args.iter().map(|arg| self.trace(arg, scope))
                    .collect::<Result<Vec<_>, _>>()?;
                if matches!(OpId::resolve(name), Some(
                    OpId::Map | OpId::Reduce | OpId::Scan | OpId::Apply | OpId::Filter | OpId::Find
                    | OpId::TreeMap | OpId::TreeReduce | OpId::Vmap | OpId::ValueAndGrad
                )) {
                    return self.builtin(name, args, loc.clone());
                }
                let callee = self.global(name)?;
                if let Value::BuiltinFn { name, .. } = self.value(&callee)? {
                    self.builtin(&name, args, loc.clone())
                } else {
                    let args = if crate::interpreter::uses_kwargs(name) {
                        positional_args(args)
                    } else {
                        args
                    };
                    self.invoke(&callee, args)
                }
            }
            CompiledExpr::LambdaCall { callee, args } => {
                let callee = self.trace(callee, scope)?;
                let args = args.iter().map(|arg| self.trace(arg, scope))
                    .collect::<Result<Vec<_>, _>>()?;
                self.invoke(&callee, args)
            }
            CompiledExpr::Lambda { params, captures, body } => self.lambda(params, captures, body, scope),
            CompiledExpr::Let { bindings, body } => {
                let mut scope = scope.clone();
                for (pattern, value) in bindings {
                    let value = self.trace(value, &scope)?;
                    self.bind(pattern, &value, &mut scope)?;
                }
                self.trace(body, &scope)
            }
            CompiledExpr::If { condition, then_branch, else_branch } => {
                let condition = self.trace(condition, scope)?;
                if self.value(&condition)?.is_truthy() {
                    self.trace(then_branch, scope)
                } else if let Some(branch) = else_branch {
                    self.trace(branch, scope)
                } else {
                    Ok(CompiledExpr::Nil)
                }
            }
            CompiledExpr::Do(values) => {
                let mut output = CompiledExpr::Nil;
                for value in values {
                    output = self.trace(value, scope)?;
                }
                Ok(output)
            }
            CompiledExpr::Vector(values) | CompiledExpr::Tuple(values) => {
                let traced = values.iter().map(|value| self.trace(value, scope))
                    .collect::<Result<Vec<_>, _>>()?;
                if matches!(expr, CompiledExpr::Vector(_)) {
                    Ok(CompiledExpr::Vector(traced))
                } else {
                    Ok(CompiledExpr::Tuple(traced))
                }
            }
            CompiledExpr::Dict(values) => {
                let values = values.iter().map(|(key, value)| {
                    Ok((self.trace(key, scope)?, self.trace(value, scope)?))
                }).collect::<Result<Vec<_>, SheafError>>()?;
                Ok(CompiledExpr::Dict(values))
            }
            CompiledExpr::GetTupleElement { param, indices } => {
                let mut value = self.lookup(param, scope)?;
                for &index in indices {
                    value = self.select(&value, Value::Int(index as i64))?;
                }
                Ok(value)
            }
            _ => {
                let form = match expr {
                    CompiledExpr::While { .. } => "while",
                    CompiledExpr::Repeat { .. } => "repeat",
                    CompiledExpr::Guard { .. } => "guard",
                    CompiledExpr::Def { .. } => "def",
                    CompiledExpr::ValueAndGrad { .. } => "value-and-grad",
                    _ => "expression",
                };
                Err(runtime_error(format!("autodiff: tracing {form} is not supported")))
            }
        }
    }

    fn bind(
        &mut self,
        pattern: &BindingPattern,
        expr: &CompiledExpr,
        scope: &mut Scope,
    ) -> Result<(), SheafError> {
        match pattern {
            BindingPattern::Simple(name) => {
                scope.insert(name.clone(), expr.clone());
            }
            BindingPattern::Destructure(patterns) => {
                let value = self.value(expr)?;
                let len = match value {
                    Value::List(values) | Value::Tuple(values) => values.len(),
                    Value::Tensor { data, .. } if data.ndim() > 0 => data.shape()[0],
                    _ => return Err(runtime_error("autodiff: cannot destructure this value")),
                };
                if len != patterns.len() {
                    return Err(runtime_error("autodiff: destructuring arity mismatch"));
                }
                for (index, pattern) in patterns.iter().enumerate() {
                    let selected = self.select(expr, Value::Int(index as i64))?;
                    self.bind(pattern, &selected, scope)?;
                }
            }
        }
        Ok(())
    }

    fn select(&mut self, container: &CompiledExpr, key: Value) -> Result<CompiledExpr, SheafError> {
        if let CompiledExpr::Symbol(name) = container
            && let Some(items) = self.leaves.lists.get(name)
            && let Value::Int(index) = &key
        {
            let index = if *index < 0 { items.len() as i64 + index } else { *index };
            return usize::try_from(index).ok().and_then(|index| items.get(index).cloned())
                .ok_or_else(|| runtime_error("autodiff: collection index out of range"));
        }
        if let CompiledExpr::Dict(values) = container
            && let Value::String(key) | Value::Keyword(key) = &key {
            return values.iter().find_map(|(name, value)| match name {
                CompiledExpr::String(name) | CompiledExpr::Keyword(name) if name == key => Some(value.clone()),
                _ => None,
            }).ok_or_else(|| runtime_error(format!("autodiff: missing dictionary key: {key}")));
        }
        if let CompiledExpr::Vector(values) | CompiledExpr::Tuple(values) = container
            && let Value::Int(index) = key {
            let index = if index < 0 { values.len() as i64 + index } else { index };
            return usize::try_from(index).ok().and_then(|index| values.get(index).cloned())
                .ok_or_else(|| runtime_error("autodiff: collection index out of range"));
        }
        let key = self.import(&key)?;
        self.atomic_call("get", vec![container.clone(), key], None)
    }

    fn builtin(
        &mut self,
        name: &str,
        args: Vec<CompiledExpr>,
        loc: Option<SourceLocation>,
    ) -> Result<CompiledExpr, SheafError> {
        match (name, args.as_slice()) {
            ("tensor", [input]) => {
                let value = self.value(input)?;
                if matches!(value, Value::Tensor { .. } | Value::DeviceBuffer(_)) {
                    return Ok(input.clone());
                }
                let items = match input {
                    CompiledExpr::Symbol(name) => self.leaves.lists.get(name).cloned(),
                    _ => None,
                };
                if let (Value::List(values), Some(items)) = (&value, items)
                    && !values.is_empty()
                    && values.iter().all(|value| matches!(value, Value::Float(_) | Value::Int(_)))
                {
                    let tensor = self.tensor_vector(&CompiledExpr::Vector(items))?;
                    if matches!(self.value(&tensor)?, Value::Tensor {
                        dtype: crate::core::dtype::ElementType::F32, ..
                    }) {
                        return Ok(tensor);
                    }
                    return self.atomic_call("cast", vec![tensor, CompiledExpr::Keyword("f32".into())], loc);
                }
                self.atomic_call(name, args, loc)
            }
            ("get", [container, key]) => {
                let key = self.value(key)?;
                self.select(container, key)
            }
            ("first" | "second" | "last", [container]) => {
                let index = match name { "first" => 0, "second" => 1, _ => -1 };
                self.select(container, Value::Int(index))
            }
            ("reduce" | "scan", [function, init, collection]) => {
                let items = self.items(collection)?;
                let mut carry = init.clone();
                let mut outputs = Vec::with_capacity(items.len());
                for item in items {
                    let step = self.invoke(function, vec![carry, item])?;
                    if name == "scan" {
                        carry = self.select(&step, Value::Int(0))?;
                        outputs.push(self.select(&step, Value::Int(1))?);
                    } else {
                        carry = step;
                    }
                }
                if name == "scan" {
                    let outputs = self.list(outputs)?;
                    Ok(CompiledExpr::Tuple(vec![carry, outputs]))
                } else {
                    Ok(carry)
                }
            }
            ("map", [function, collection]) => {
                let mut outputs = Vec::new();
                for item in self.items(collection)? {
                    outputs.push(self.invoke(function, vec![item])?);
                }
                self.list(outputs)
            }
            ("apply", [function, collection]) => {
                let items = self.items(collection)?;
                self.invoke(function, items)
            }
            _ if matches!(OpId::resolve(name), Some(
                OpId::Map | OpId::Reduce | OpId::Scan | OpId::Apply | OpId::Filter | OpId::Find
                | OpId::TreeMap | OpId::TreeReduce | OpId::Vmap | OpId::ValueAndGrad
            )) => Err(runtime_error(format!("autodiff: unsupported traced higher-order operation: {name}"))),
            _ => self.atomic_call(name, args, loc),
        }
    }

    fn items(&mut self, collection: &CompiledExpr) -> Result<Vec<CompiledExpr>, SheafError> {
        let value = self.value(collection)?;
        let len = match &value {
            Value::List(values) | Value::Tuple(values) => values.len(),
            Value::Tensor { data, .. } if data.ndim() > 0 => data.shape()[0],
            Value::DeviceBuffer(buffer) if !buffer.shape.is_empty() => buffer.shape[0],
            Value::Dict(values) => dict_scan_length(values)?,
            _ => return Err(runtime_error("autodiff: expected an iterable collection")),
        };
        let mut items = Vec::with_capacity(len);
        for index in 0..len {
            let key = Value::Int(index as i64);
            if let CompiledExpr::Dict(values) = collection {
                let mut row = Vec::with_capacity(values.len());
                for (name, value) in values {
                    row.push((name.clone(), self.select(value, key.clone())?));
                }
                items.push(CompiledExpr::Dict(row));
            } else {
                items.push(self.select(collection, key)?);
            }
        }
        Ok(items)
    }
}

fn positional_args(args: Vec<CompiledExpr>) -> Vec<CompiledExpr> {
    let mut args = args.into_iter().peekable();
    let mut positional = Vec::new();
    while let Some(arg) = args.next() {
        if matches!(arg, CompiledExpr::Keyword(_)) {
            if args.peek().is_some_and(|next| !matches!(next, CompiledExpr::Keyword(_))) {
                args.next();
            }
        } else {
            positional.push(arg);
        }
    }
    positional
}

fn shape_expr(shape: &[i64]) -> CompiledExpr {
    CompiledExpr::Vector(shape.iter().copied().map(CompiledExpr::Integer).collect())
}

fn static_expr(value: &Value) -> Result<CompiledExpr, SheafError> {
    match value {
        Value::Int(value) => Ok(CompiledExpr::Integer(*value)),
        Value::Float(value) => Ok(CompiledExpr::Float(*value as f64)),
        Value::Bool(value) => Ok(CompiledExpr::Boolean(*value)),
        Value::Tensor { data, .. } if data.ndim() == 1 => Ok(shape_expr(
            &data.iter().map(|value| *value as i64).collect::<Vec<_>>(),
        )),
        _ => Err(runtime_error("autodiff: expected a static numeric value")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::eval::Interpreter;

    #[test]
    fn forward_operations_execute_once_and_are_not_replayed_by_backward() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        fn record_once(_: &[Value], _: &std::collections::BTreeMap<String, Value>)
            -> Result<Value, SheafError>
        {
            CALLS.fetch_add(1, Ordering::SeqCst);
            Ok(Value::Float(3.0))
        }
        let mut interpreter = Interpreter::new();
        interpreter.env_mut().set_builtin("record-once", record_once);
        let function = interpreter.eval("(fn [p] (let [unused (record-once)] (sum (* p p))))").unwrap();
        let env = interpreter.env_mut();
        env.push_scope();
        let mut leaves = LeafMap::new(env);
        let input = Value::tensor_f32(ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2]), vec![1.0, 2.0],
        ).unwrap());
        let parameter = leaves.register(input, env);
        let program = trace_function(&function, CompiledExpr::Symbol(parameter.clone()), env, &mut leaves)
            .unwrap();
        assert_eq!(program.value.to_f64().unwrap(), 5.0);
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        let reverse = crate::autodiff::reverse::reverse_grad(
            &program.bindings, &program.output, &[parameter], &program.shapes,
        ).unwrap();
        for (name, expr) in reverse.backward_bindings {
            let value = eval(&expr, env).unwrap();
            env.set(&name, value);
        }
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        env.pop_scope();
    }

    #[test]
    fn constant_dataset_does_not_receive_dense_adjoints() {
        for steps in [16, 64, 128, 1024] {
            let mut interpreter = Interpreter::new();
            let function = interpreter.eval(&format!(
                "(fn [p] (sum (reduce (fn [c row] (+ (* c 0.5) (* (* c 0.5) (get row :x)))) \
                 p {{:x (ones [{steps} 2])}})))",
            )).unwrap();
            let env = interpreter.env_mut();
            env.push_scope();
            let mut leaves = LeafMap::new(env);
            let value = Value::tensor_f32(ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2]), vec![1.0, 2.0],
            ).unwrap());
            let parameter = leaves.register(value, env);
            let program = trace_function(&function, CompiledExpr::Symbol(parameter.clone()), env, &mut leaves)
                .unwrap();
            assert!(program.bindings.len() <= 5 * steps + 2);
            let reverse = crate::autodiff::reverse::reverse_grad(
                &program.bindings, &program.output, std::slice::from_ref(&parameter), &program.shapes,
            ).unwrap();
            assert!(reverse.backward_bindings.len() <= 20 * steps + 8,
                "{} backward bindings", reverse.backward_bindings.len());
            assert!(!reverse.backward_bindings.iter().any(|(_, expr)| {
                matches!(expr, CompiledExpr::FunctionCall { name, .. } if name == "one-hot")
            }), "a constant dataset received gather adjoints");
            for (name, expr) in reverse.backward_bindings {
                let value = eval(&expr, env).unwrap();
                env.set(&name, value);
            }
            let crate::autodiff::reverse::GradientOutput::Computed(gradient) = &reverse.gradients[&parameter]
                else { panic!("expected computed gradient") };
            let Value::Tensor { data, .. } = env.get(gradient).unwrap()
                else { panic!("expected a tensor gradient") };
            assert_eq!(data.shape(), &[2]);
            assert!(data.iter().all(|value| *value == 1.0));
            env.pop_scope();
        }
    }

    #[test]
    fn repeated_accumulator_records_a_linear_graph() {
        for steps in [16, 64, 128] {
            let mut interpreter = Interpreter::new();
            let function = interpreter.eval(&format!(
                "(fn [p] (reduce (fn [c x] (+ (* c 0.5) (* c 0.5))) p {{:x (ones [{steps} 1])}}))",
            )).unwrap();
            let env = interpreter.env_mut();
            env.push_scope();
            let mut leaves = LeafMap::new(env);
            let parameter = leaves.register(Value::Float(3.0), env);
            let program = trace_function(&function, CompiledExpr::Symbol(parameter.clone()), env, &mut leaves)
                .unwrap();
            assert!(program.bindings.len() <= 4 * steps + 2, "{} bindings", program.bindings.len());
            assert!(matches!(program.value, Value::Float(value) if value == 3.0));
            let reverse = crate::autodiff::reverse::reverse_grad(
                &program.bindings, &program.output, std::slice::from_ref(&parameter), &program.shapes,
            ).unwrap();
            assert!(reverse.backward_bindings.len() <= 12 * steps + 4,
                "{} backward bindings", reverse.backward_bindings.len());
            for (name, expr) in reverse.backward_bindings {
                let value = eval(&expr, env).unwrap();
                env.set(&name, value);
            }
            let crate::autodiff::reverse::GradientOutput::Computed(gradient) = &reverse.gradients[&parameter]
                else { panic!("expected computed gradient") };
            assert!(matches!(env.get(gradient).unwrap(), Value::Float(value) if value == 1.0));
            env.pop_scope();
        }
    }
}
