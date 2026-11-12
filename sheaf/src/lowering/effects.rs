// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! JIT effect and higher-order call analysis.

use crate::core::expr::CompiledExpr;
use crate::lowering::walk::walk_expr;

const EFFECTFUL_BUILTINS: &[&str] = &["print", "io"];

pub const HOF_BUILTINS: &[&str] = &[
    "map", "filter", "sort",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectSite {
    pub name: String,
}

impl EffectSite {
    fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }
}

pub fn has_side_effects(expr: &CompiledExpr) -> bool {
    !collect_effects(expr).is_empty()
}

pub fn collect_hof_calls(expr: &CompiledExpr) -> Vec<String> {
    let mut found = Vec::new();
    walk_expr(expr, &mut |expr| match expr {
        CompiledExpr::FunctionCall { name, .. }
            if HOF_BUILTINS.contains(&name.as_str()) =>
        {
            found.push(name.clone());
        }
        CompiledExpr::ValueAndGrad { .. } => found.push("value-and-grad".to_string()),
        _ => {}
    });
    found.sort();
    found.dedup();
    found
}

pub fn collect_effects(expr: &CompiledExpr) -> Vec<EffectSite> {
    let mut sites = Vec::new();
    walk_expr(expr, &mut |expr| match expr {
        CompiledExpr::FunctionCall { name, .. }
            if EFFECTFUL_BUILTINS.contains(&name.as_str()) =>
        {
            sites.push(EffectSite::new(name));
        }
        CompiledExpr::Guard { .. } => sites.push(EffectSite::new("guard")),
        CompiledExpr::Def { .. } => sites.push(EffectSite::new("def")),
        _ => {}
    });
    sites
}

pub fn format_effects(sites: &[EffectSite]) -> String {
    let mut seen = std::collections::BTreeSet::new();
    for site in sites {
        seen.insert(site.name.clone());
    }
    seen.into_iter().collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: name.to_string(),
            args: vec![CompiledExpr::Nil],
            loc: None,
        }
    }

    #[test]
    fn finds_effect_in_tuple() {
        let expr = CompiledExpr::Tuple(vec![call("print")]);
        assert_eq!(collect_effects(&expr), vec![EffectSite::new("print")]);
    }

    #[test]
    fn finds_hof_in_nested_collections_and_loops() {
        let expr = CompiledExpr::Dict(vec![(
            CompiledExpr::Nil,
            CompiledExpr::While {
                condition: Box::new(CompiledExpr::Boolean(true)),
                acc_var: "acc".to_string(),
                acc_init: Box::new(CompiledExpr::Tuple(vec![CompiledExpr::Nil])),
                body: Box::new(call("map")),
            },
        )]);
        assert_eq!(collect_hof_calls(&expr), vec!["map"]);
    }
}
