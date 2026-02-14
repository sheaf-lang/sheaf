// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Special forms - Sheaf language constructs
//!
//! ## Module Organization
//!
//! - `base`: Trait SpecialForm and utilities
//! - `binding`: defn, let, fn (function definition and binding)
//! - `control`: if, do, case, while, repeat, guard (control flow)
//! - `flow`: ->, as-> (threading macros)
//! - `utils`: get, dict, last, use, quote (utilities)
//! - `ml`: vmap, scan, with-params, static (ML-specific, future)

pub mod base;
pub mod binding;
pub mod control;
pub mod flow;
pub mod utils;
// pub mod ml;  // Phase 3 - ML specific forms

use std::collections::HashMap;

pub use base::SpecialForm;
pub use binding::{DefnForm, FnForm, LetForm};
pub use control::{CaseForm, DoForm, GuardForm, IfForm, RepeatForm, WhileForm};
pub use flow::{ThreadAsForm, ThreadFirstForm};
pub use utils::{AssocForm, DictForm, GetForm, GetInForm, LastForm, QuoteForm, UseForm};

/// Create the registry of all special forms
///
/// This function creates a hashmap of all available special forms,
/// mapping their names to boxed trait objects.
///
/// # Example
/// ```
/// use sheaf_compiler::forms::special_forms_registry;
/// let registry = special_forms_registry();
/// assert!(registry.contains_key("defn"));
/// assert!(registry.contains_key("let"));
/// ```
pub fn special_forms_registry() -> HashMap<&'static str, Box<dyn SpecialForm>> {
    let mut forms: HashMap<&'static str, Box<dyn SpecialForm>> = HashMap::new();

    // Binding forms
    forms.insert("defn", Box::new(DefnForm));
    forms.insert("let", Box::new(LetForm));
    forms.insert("fn", Box::new(FnForm));

    // Control flow forms
    forms.insert("if", Box::new(IfForm));
    forms.insert("do", Box::new(DoForm));
    forms.insert("case", Box::new(CaseForm));
    forms.insert("while", Box::new(WhileForm));
    forms.insert("repeat", Box::new(RepeatForm));
    forms.insert("guard", Box::new(GuardForm));

    // Flow forms
    forms.insert("->", Box::new(ThreadFirstForm));
    forms.insert("as->", Box::new(ThreadAsForm));

    // Utility forms
    forms.insert("quote", Box::new(QuoteForm));
    forms.insert("get", Box::new(GetForm));
    forms.insert("get-in", Box::new(GetInForm));
    forms.insert("dict", Box::new(DictForm));
    forms.insert("assoc", Box::new(AssocForm));
    forms.insert("last", Box::new(LastForm));
    forms.insert("use", Box::new(UseForm));

    forms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_contains_core_forms() {
        let registry = special_forms_registry();

        // Check binding forms
        assert!(registry.contains_key("defn"));
        assert!(registry.contains_key("let"));
        assert!(registry.contains_key("fn"));

        // Check control forms
        assert!(registry.contains_key("if"));
        assert!(registry.contains_key("do"));

        // Check utils
        assert!(registry.contains_key("quote"));
    }

    #[test]
    fn test_registry_form_names() {
        let registry = special_forms_registry();

        assert_eq!(registry.get("defn").unwrap().name(), "defn");
        assert_eq!(registry.get("let").unwrap().name(), "let");
        assert_eq!(registry.get("if").unwrap().name(), "if");
    }
}
