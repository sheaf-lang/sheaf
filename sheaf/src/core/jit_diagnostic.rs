// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use super::error::SheafError;
use super::signature::OpId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LanguageForm {
    Repeat, While, Destructuring, Guard, Def,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Construct {
    Operation(OpId),
    Form(LanguageForm),
    // Only source constructs are serialized; Other is produced at the JIT boundary.
    #[serde(skip)]
    Other(#[serde(skip)] &'static str),
}

impl Construct {
    pub fn named(name: &str) -> Option<Self> {
        match name {
            "guard" => Some(Self::Form(LanguageForm::Guard)),
            "def" => Some(Self::Form(LanguageForm::Def)),
            "value-and-grad" => Some(Self::Operation(OpId::ValueAndGrad)),
            _ => OpId::resolve(name).map(Self::Operation),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InterpretedReason {
    ScalarOnly,
    Effects(Vec<Construct>),
    RecursiveCall(String),
    UninitializedGlobal(String),
    GeneratedClosure,
    NoCompiledBody,
    NonTensorArgument(String),
    DataDependent(Construct),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnsupportedReason {
    Construct(Construct),
    GraphLimit { phase: GraphPhase, nodes: usize, limit: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphPhase { Inlining, Unrolling }

#[derive(Debug, Clone)]
pub enum JitDiagnostic {
    Interpreted(InterpretedReason),
    Unsupported(UnsupportedReason, Option<Box<SheafError>>),
    UserError(SheafError),
    Environment { message: String, hint: &'static str },
    Internal(String),
}

impl JitDiagnostic {
    pub fn from_hof(construct: Construct) -> Self {
        match construct {
            Construct::Operation(OpId::Filter) => Self::Interpreted(InterpretedReason::DataDependent(construct)),
            _ => Self::Unsupported(UnsupportedReason::Construct(construct), None),
        }
    }

    pub fn codegen_panic(payload: Box<dyn std::any::Any + Send>) -> Self {
        let detail = payload.downcast_ref::<String>().map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied()).unwrap_or("unknown panic");
        Self::Internal(format!("Codegen panicked: {detail}"))
    }

    pub fn from_codegen(error: SheafError) -> Self {
        match &error {
            SheafError::CompilerUnsupported { construct, .. } =>
                Self::Unsupported(UnsupportedReason::Construct(construct.clone()), Some(Box::new(error))),
            SheafError::CompilerInvariant { .. } | SheafError::AutodiffMissingGradientOutput { .. } =>
                Self::Internal(error.short_message().into_owned()),
            SheafError::InvalidShapeDimension { .. } | SheafError::InvalidSourceForm { .. }
                | SheafError::AutodiffMissingRule { .. } => Self::UserError(error),
            _ => Self::Unsupported(UnsupportedReason::Construct(Construct::Other("compiled code")), Some(Box::new(error))),
        }
    }

    pub fn io(action: &str, error: std::io::Error) -> Self {
        Self::Environment {
            message: format!("{action}: {error}"),
            hint: "Check file permissions and available disk space.",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SourceLocation;

    fn location() -> SourceLocation { SourceLocation::new(7, 3, "model.shf".into()) }

    #[test]
    fn generic_codegen_errors_are_not_user_errors_or_internal_by_message() {
        let errors = [
            SheafError::Compile { message: "iree-compile failed".into(), location: location() },
            SheafError::Runtime { message: "internal panic".into(), location: Some(location()) },
            SheafError::Parse { message: "invalid syntax".into(), location: location() },
            SheafError::Io("permission denied".into()),
        ];
        for error in errors {
            let text = error.to_string();
            assert!(matches!(JitDiagnostic::from_codegen(error), JitDiagnostic::Unsupported(
                UnsupportedReason::Construct(Construct::Other("compiled code")), Some(original)
            ) if original.to_string() == text));
        }
    }

    #[test]
    fn only_explicit_user_error_variants_cross_the_boundary_as_user_errors() {
        let errors = [
            SheafError::InvalidShapeDimension { message: "float dimension".into(), location: location() },
            SheafError::InvalidSourceForm { message: "invalid arity".into(), location: location() },
            SheafError::AutodiffMissingRule { operation: "custom".into(), location: Some(location()) },
        ];
        for error in errors {
            let text = error.to_string();
            assert!(matches!(JitDiagnostic::from_codegen(error), JitDiagnostic::UserError(original)
                if original.to_string() == text));
        }
    }

    #[test]
    fn explicit_unsupported_constructs_keep_their_identity_and_source() {
        for construct in [Construct::Operation(OpId::Flip), Construct::Form(LanguageForm::Repeat),
            Construct::Form(LanguageForm::While), Construct::Form(LanguageForm::Destructuring)] {
            let error = SheafError::CompilerUnsupported {
                construct: construct.clone(), message: "unsupported construct".into(), location: location(),
            };
            let serialized = serde_json::to_string(&error).unwrap();
            let restored: SheafError = serde_json::from_str(&serialized).unwrap();
            assert!(matches!(JitDiagnostic::from_codegen(restored), JitDiagnostic::Unsupported(
                UnsupportedReason::Construct(actual), Some(original)
            ) if actual == construct && original.to_string() == error.to_string()));
        }
        assert!(serde_json::to_string(&Construct::Other("compiled code")).is_err());
        assert!(serde_json::from_str::<Construct>(r#"{"Other":"compiled code"}"#).is_err());
    }

    #[test]
    fn hof_categories_are_typed_and_dead_names_are_not_constructs() {
        assert!(matches!(JitDiagnostic::from_hof(Construct::Operation(OpId::Filter)),
            JitDiagnostic::Interpreted(InterpretedReason::DataDependent(Construct::Operation(OpId::Filter)))));
        for op in [OpId::Map, OpId::Sort] {
            assert!(matches!(JitDiagnostic::from_hof(Construct::Operation(op)),
                JitDiagnostic::Unsupported(UnsupportedReason::Construct(Construct::Operation(actual)), None) if actual == op));
        }
        assert_eq!(Construct::named("grad"), None);
        assert_eq!(Construct::named("jit"), None);
    }

    #[test]
    fn codegen_panic_preserves_the_payload_without_changing_its_category() {
        let detail = "index out of bounds: the len is 1 but the index is 1";
        for payload in [Box::new(detail) as Box<dyn std::any::Any + Send>, Box::new(detail.to_string())] {
            assert!(matches!(JitDiagnostic::codegen_panic(payload), JitDiagnostic::Internal(message)
                if message == format!("Codegen panicked: {detail}")));
        }
    }

    #[test]
    fn invariants_and_missing_gradient_outputs_are_internal() {
        for error in [
            SheafError::CompilerInvariant { message: "invalid projection".into(), location: location() },
            SheafError::AutodiffMissingGradientOutput { symbol: "p".into() },
        ] {
            assert!(matches!(JitDiagnostic::from_codegen(error), JitDiagnostic::Internal(_)));
        }
    }
}
