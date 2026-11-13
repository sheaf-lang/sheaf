// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! JIT call graph.

use crate::core::expr::{CompiledExpr, FunctionDef};
use crate::core::jit_diagnostic::{Construct, InterpretedReason, JitDiagnostic};
use crate::lowering::effects::{collect_effects, collect_hof_calls};
use std::collections::{HashMap, HashSet};

pub type NodeId = usize;

pub fn direct_callees(body: &CompiledExpr) -> HashSet<String> {
    let mut callees = HashSet::new();
    crate::core::bindings::visit_module_references(body, &mut |name, kind| {
        if kind == crate::core::bindings::ModuleReference::Call { callees.insert(name.to_string()); }
    });
    callees
}

pub struct CallGraph {
    name_to_id: HashMap<String, NodeId>,
    id_to_name: Vec<String>,
    edges: Vec<Vec<NodeId>>,
}

impl CallGraph {
    pub fn from_registry(registry: &HashMap<String, FunctionDef>) -> Self {
        let id_to_name: Vec<String> = registry.keys().cloned().collect();
        let name_to_id: HashMap<String, NodeId> = id_to_name
            .iter()
            .enumerate()
            .map(|(id, name)| (name.clone(), id))
            .collect();

        let n = id_to_name.len();
        let mut edges: Vec<Vec<NodeId>> = vec![Vec::new(); n];

        for (id, name) in id_to_name.iter().enumerate() {
            let Some(fd) = registry.get(name) else { continue };
            let Some(body) = &fd.body_compiled else { continue };

            let mut succs: Vec<NodeId> = direct_callees(body)
                .into_iter()
                .filter_map(|callee| name_to_id.get(&callee).copied())
                .collect();
            succs.sort_unstable();
            succs.dedup();
            edges[id] = succs;
        }

        Self { name_to_id, id_to_name, edges }
    }

    pub fn node_count(&self) -> usize {
        self.id_to_name.len()
    }

    pub fn name_of(&self, id: NodeId) -> &str {
        &self.id_to_name[id]
    }

    pub fn reachable(&self, root: &str) -> Vec<NodeId> {
        let Some(&start) = self.name_to_id.get(root) else {
            return Vec::new();
        };
        let n = self.id_to_name.len();

        let mut visited = vec![false; n];
        visited[start] = true;
        let mut count = 1;

        let mut stack: Vec<NodeId> = Vec::with_capacity(n);
        stack.push(start);
        while let Some(v) = stack.pop() {
            for &w in self.edges[v].iter() {
                if !visited[w] {
                    visited[w] = true;
                    count += 1;
                    stack.push(w);
                }
            }
        }

        let mut out: Vec<NodeId> = Vec::with_capacity(count);
        for (v, &is_visited) in visited.iter().enumerate() {
            if is_visited {
                out.push(v);
            }
        }
        out
    }

    pub fn nodes_in_cycles(&self) -> Vec<NodeId> {
        let n = self.id_to_name.len();
        let mut in_cycle: Vec<bool> = vec![false; n];

        let mut disc: Vec<usize> = vec![0; n];
        let mut discovered: Vec<bool> = vec![false; n];
        let mut low: Vec<usize> = vec![0; n];
        let mut on_stack: Vec<bool> = vec![false; n];
        let mut scc_stack: Vec<NodeId> = Vec::with_capacity(n);
        let mut next_disc: usize = 1;

        struct Frame {
            v: NodeId,
            iter_pos: usize,
        }
        let mut work: Vec<Frame> = Vec::with_capacity(n);

        for start in 0..n {
            if discovered[start] {
                continue;
            }
            discovered[start] = true;
            disc[start] = next_disc;
            low[start] = next_disc;
            next_disc += 1;
            scc_stack.push(start);
            on_stack[start] = true;
            work.push(Frame { v: start, iter_pos: 0 });

            while let Some(frame) = work.last_mut() {
                let v = frame.v;
                if frame.iter_pos < self.edges[v].len() {
                    let w = self.edges[v][frame.iter_pos];
                    frame.iter_pos += 1;
                    if !discovered[w] {
                        discovered[w] = true;
                        disc[w] = next_disc;
                        low[w] = next_disc;
                        next_disc += 1;
                        scc_stack.push(w);
                        on_stack[w] = true;
                        work.push(Frame { v: w, iter_pos: 0 });
                    } else if on_stack[w] {
                        let dw = disc[w];
                        if dw < low[v] {
                            low[v] = dw;
                        }
                    }
                } else {
                    let popped_v = v;
                    let popped_low = low[v];
                    let is_scc_root = low[popped_v] == disc[popped_v];
                    work.pop();

                    if let Some(parent) = work.last()
                        && popped_low < low[parent.v] {
                            low[parent.v] = popped_low;
                        }

                    if is_scc_root {
                        let mut size: usize = 0;
                        loop {
                            let m = scc_stack.pop().expect("SCC root on stack");
                            on_stack[m] = false;
                            in_cycle[m] = true;
                            size += 1;
                            if m == popped_v {
                                break;
                            }
                        }
                        // A single-node component needs a self-loop.
                        if size == 1 {
                            in_cycle[popped_v] =
                                self.edges[popped_v].contains(&popped_v);
                        }
                    }
                }
            }
        }

        let mut out: Vec<NodeId> = Vec::new();
        for (v, &is_in_cycle) in in_cycle.iter().enumerate() {
            if is_in_cycle {
                out.push(v);
            }
        }
        out
    }
}

pub fn jit_eligibility(
    root: &str,
    registry: &HashMap<String, FunctionDef>,
) -> Result<(), JitDiagnostic> {
    let graph = CallGraph::from_registry(registry);
    let reach = graph.reachable(root);

    if reach.is_empty() {
        return Err(JitDiagnostic::Internal(format!("Unknown function '{root}' in the compilation registry")));
    }

    let cyclic = graph.nodes_in_cycles();
    let n = graph.node_count();
    let mut is_cyclic = vec![false; n];
    for &c in &cyclic {
        is_cyclic[c] = true;
    }
    for &r in &reach {
        if is_cyclic[r] {
            return Err(JitDiagnostic::Interpreted(InterpretedReason::RecursiveCall(graph.name_of(r).into())));
        }
    }

    let mut effects: Vec<String> = Vec::new();
    let mut hofs: Vec<String> = Vec::new();
    for &id in &reach {
        let name = graph.name_of(id);
        let Some(fd) = registry.get(name) else { continue };
        let Some(body) = &fd.body_compiled else { continue };
        for s in collect_effects(body) {
            effects.push(s.name);
        }
        for h in collect_hof_calls(body) {
            hofs.push(h);
        }
    }

    if !effects.is_empty() {
        effects.sort();
        effects.dedup();
        let effects = effects.iter().map(|name| Construct::named(name)
            .ok_or_else(|| JitDiagnostic::Internal(format!("Unknown effect '{name}'"))))
            .collect::<Result<Vec<_>, _>>()?;
        return Err(JitDiagnostic::Interpreted(InterpretedReason::Effects(effects)));
    }
    if !hofs.is_empty() {
        hofs.sort();
        hofs.dedup();
        let construct = Construct::named(&hofs[0])
            .ok_or_else(|| JitDiagnostic::Internal(format!("Unknown higher-order operation '{}'", hofs[0])))?;
        return Err(JitDiagnostic::from_hof(construct));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::jit_diagnostic::UnsupportedReason;
    use crate::core::expr::FunctionDef;
    use crate::core::ast::SheafValue;
    use crate::core::error::SourceLocation;

    fn loc() -> SourceLocation {
        SourceLocation::new(0, 0, "".into())
    }

    fn nil_body() -> SheafValue {
        SheafValue::Nil(loc())
    }

    fn node_names<'a>(graph: &'a CallGraph, ids: &[NodeId]) -> HashSet<&'a str> {
        ids.iter().map(|&id| graph.name_of(id)).collect()
    }

    fn fn_def(name: &str, body: CompiledExpr) -> FunctionDef {
        FunctionDef {
            name: name.to_string(),
            params: vec!["x".to_string()],
            body: nil_body(),
            body_compiled: Some(body.into()),
            requires_initialization: false,
            signature: None,
            known_param_types: Vec::new(),
            compile_error: None,
        }
    }

    fn call(target: &str, arg: CompiledExpr) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: target.to_string(),
            args: vec![arg],
            loc: None,
        }
    }

    fn sym(name: &str) -> CompiledExpr {
        CompiledExpr::Symbol(name.to_string())
    }

    #[test]
    fn self_recursion_is_cyclic() {
        let body = call("f", sym("x"));
        let mut registry = HashMap::new();
        registry.insert("f".to_string(), fn_def("f", body));
        let graph = CallGraph::from_registry(&registry);
        let cycles = graph.nodes_in_cycles();
        let cycle_names = node_names(&graph, &cycles);
        assert!(cycle_names.contains("f"), "self-recursion must be detected as a cycle");
        assert!(jit_eligibility("f", &registry).is_err());
    }

    #[test]
    fn mutual_recursion_is_cyclic() {
        let f_body = call("g", sym("x"));
        let g_body = call("f", sym("x"));
        let mut registry = HashMap::new();
        registry.insert("f".to_string(), fn_def("f", f_body));
        registry.insert("g".to_string(), fn_def("g", g_body));
        let graph = CallGraph::from_registry(&registry);
        let cycles = graph.nodes_in_cycles();
        let cycle_names = node_names(&graph, &cycles);
        assert!(cycle_names.contains("f") && cycle_names.contains("g"),
            "mutual recursion: both nodes must appear in cycles");
        assert_eq!(cycles.len(), 2);

        let err = jit_eligibility("f", &registry).unwrap_err();
        assert!(matches!(err, JitDiagnostic::Interpreted(InterpretedReason::RecursiveCall(_))));
    }

    #[test]
    fn transitive_hof_is_rejected() {
        let f_body = call("g", sym("x"));
        let g_body = CompiledExpr::FunctionCall {
            name: "map".to_string(),
            args: vec![sym("x"), sym("x")],
            loc: None,
        };
        let mut registry = HashMap::new();
        registry.insert("f".to_string(), fn_def("f", f_body));
        registry.insert("g".to_string(), fn_def("g", g_body));

        let err = jit_eligibility("f", &registry).unwrap_err();
        assert!(matches!(err, JitDiagnostic::Unsupported(
            UnsupportedReason::Construct(Construct::Operation(crate::core::signature::OpId::Map)), None)));
    }

    #[test]
    fn pure_chain_is_eligible() {
        let mut registry = HashMap::new();
        registry.insert("h".to_string(), fn_def("h", sym("x")));
        registry.insert("g".to_string(), fn_def("g", call("h", sym("x"))));
        registry.insert("f".to_string(), fn_def("f", call("g", sym("x"))));

        assert!(jit_eligibility("f", &registry).is_ok());

        let reach = CallGraph::from_registry(&registry).reachable("f");
        assert_eq!(reach.len(), 3, "f, g, h reachable");
    }

    #[test]
    fn tuple_callee_is_checked() {
        let mut registry = HashMap::new();
        registry.insert(
            "g".to_string(),
            fn_def("g", call("print", sym("x"))),
        );
        registry.insert(
            "f".to_string(),
            fn_def("f", CompiledExpr::Tuple(vec![call("g", sym("x"))])),
        );

        let err = jit_eligibility("f", &registry).unwrap_err();
        assert!(matches!(err, JitDiagnostic::Interpreted(InterpretedReason::Effects(effects))
            if effects == vec![Construct::Operation(crate::core::signature::OpId::Print)]));
    }

    #[test]
    fn diamond_does_not_create_false_cycle() {
        let mut registry = HashMap::new();
        registry.insert("h".to_string(), fn_def("h", sym("x")));
        registry.insert(
            "g".to_string(),
            fn_def("g", call("h", sym("x"))),
        );
        let f_body = CompiledExpr::Do(vec![
            call("g", sym("x")),
            call("h", sym("x")),
        ]);
        registry.insert("f".to_string(), fn_def("f", f_body));

        let graph = CallGraph::from_registry(&registry);
        assert!(graph.nodes_in_cycles().is_empty(),
            "diamond must not be flagged as cyclic");

        let reach = graph.reachable("f");
        assert_eq!(reach.len(), 3, "f, g, h reachable");

        assert!(jit_eligibility("f", &registry).is_ok());
    }

    #[test]
    fn lambda_within_hof_is_detected_via_root_body() {
        let g_body = sym("x");
        let lambda = CompiledExpr::lambda(vec!["y".to_string()], call("g", sym("y")));
        let f_body = CompiledExpr::FunctionCall {
            name: "map".to_string(),
            args: vec![sym("x"), lambda],
            loc: None,
        };
        let mut registry = HashMap::new();
        registry.insert("f".to_string(), fn_def("f", f_body));
        registry.insert("g".to_string(), fn_def("g", g_body));

        let direct = direct_callees(registry["f"].body_compiled.as_ref().unwrap());
        assert!(direct.contains("map"));
        assert!(direct.contains("g"));

        let err = jit_eligibility("f", &registry).unwrap_err();
        assert!(matches!(err, JitDiagnostic::Unsupported(
            UnsupportedReason::Construct(Construct::Operation(crate::core::signature::OpId::Map)), None)));
    }

    #[test]
    fn unknown_root_returns_error() {
        let registry = HashMap::new();
        assert!(matches!(jit_eligibility("nope", &registry), Err(JitDiagnostic::Internal(_))));
    }
}
