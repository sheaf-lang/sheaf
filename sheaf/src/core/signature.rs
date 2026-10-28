// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Native operation names and aliases shared by the compiler and interpreter.

macro_rules! operations {
    ($($public:literal { $($id:ident => ($name:literal, [$($alias:literal),*])),* $(,)? })*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum OpId { $($($id,)*)* }

        impl OpId {
            pub const ALL: &'static [Self] = &[$($(Self::$id,)*)*];

            pub const fn name(self) -> &'static str {
                match self { $($(Self::$id => $name,)*)* }
            }

            pub const fn aliases(self) -> &'static [&'static str] {
                match self { $($(Self::$id => &[$($alias),*],)*)* }
            }

            pub const fn is_public(self) -> bool {
                match self { $($(Self::$id => $public,)*)* }
            }

            pub fn resolve(name: &str) -> Option<Self> {
                Self::ALL.iter().copied().find(|op| {
                    op.name() == name || op.aliases().contains(&name)
                })
            }
        }
    };
}

operations! {
    true {
        Add => ("+", []),
        Subtract => ("-", []),
        Multiply => ("*", []),
        Divide => ("/", []),
        FloorDivide => ("//", []),
        Modulo => ("mod", ["%"]),
        Power => ("**", []),
        Abs => ("abs", []),
        ArithmeticShift => ("ash", []),
        Exp => ("exp", []),
        Log => ("log", []),
        Sqrt => ("sqrt", []),
        Round => ("round", []),
        Ceil => ("ceil", []),
        Floor => ("floor", []),
        Sin => ("sin", []),
        Cos => ("cos", []),
        Tan => ("tan", []),
        Matmul => ("@", []),
        Einsum => ("einsum", []),
        AppendAndRoll => ("append-and-roll", []),
        Tanh => ("tanh", []),
        Equal => ("=", []),
        ElementEqual => ("==", []),
        NotEqual => ("!=", []),
        Less => ("<", []),
        Greater => (">", []),
        LessEqual => ("<=", []),
        GreaterEqual => (">=", []),
        Not => ("not", []),
        Shape => ("shape", []),
        Ndim => ("ndim", []),
        Len => ("len", ["count"]),
        Int => ("int", []),
        Float => ("float", []),
        Reshape => ("reshape", []),
        Transpose => ("transpose", ["tr"]),
        Concat => ("concat", []),
        Slice => ("slice", []),
        Get => ("get", []),
        Where => ("where", []),
        Roll => ("roll", []),
        IndexUpdate => ("index-update", []),
        Swapaxes => ("swapaxes", []),
        DynamicSlice => ("dynamic-slice", []),
        DynamicUpdateSlice => ("dynamic-update-slice", []),
        TensorSplit => ("tensor-split", []),
        Flip => ("flip", []),
        First => ("first", []),
        Second => ("second", []),
        Last => ("last", []),
        Rest => ("rest", []),
        Nth => ("nth", []),
        Cons => ("cons", []),
        Append => ("append", []),
        Empty => ("empty?", []),
        GetIn => ("get-in", []),
        Assoc => ("assoc", []),
        Dissoc => ("dissoc", []),
        Merge => ("merge", []),
        Keys => ("keys", []),
        Vals => ("vals", []),
        Dict => ("dict", []),
        Sort => ("sort", []),
        Chars => ("chars", []),
        IndexOf => ("index-of", []),
        Sum => ("sum", []),
        Mean => ("mean", []),
        Product => ("product", []),
        Min => ("min", []),
        Max => ("max", []),
        Minimum => ("minimum", []),
        Maximum => ("maximum", []),
        Argmax => ("argmax", []),
        Argmin => ("argmin", []),
        Var => ("var", []),
        Normalize => ("normalize", []),
        Print => ("print", ["println"]),
        Str => ("str", []),
        StrCall => ("str-call", []),
        Io => ("io", []),
        Gensym => ("gensym", []),
        SymbolPredicate => ("symbol?", []),
        Time => ("time", []),
        TreeMapZeros => ("tree-map-zeros", []),
        Zeros => ("zeros", []),
        Ones => ("ones", []),
        Arange => ("arange", []),
        Eye => ("eye", []),
        OneHot => ("one-hot", []),
        Tril => ("tril", []),
        Tensor => ("tensor", []),
        Range => ("range", []),
        Cast => ("cast", []),
        RandomKey => ("random-key", []),
        RandomSplit => ("random-split", []),
        RandomNormal => ("random-normal", []),
        RandomUniform => ("random-uniform", []),
        RandomRandint => ("random-randint", []),
        Choice => ("choice", []),
        TopK => ("top_k", []),
        StopGradient => ("stop-gradient", []),
        Map => ("map", []),
        Filter => ("filter", []),
        Reduce => ("reduce", []),
        Scan => ("scan", []),
        Apply => ("apply", []),
        Find => ("find", []),
        TreeMap => ("tree-map", []),
        TreeReduce => ("tree-reduce", []),
        Flatten => ("flatten", []),
        Vmap => ("vmap", []),
        And => ("and", []),
        Or => ("or", []),
    }
    false {
        MatmulGradLhs => ("@-grad-lhs", []),
        MatmulGradRhs => ("@-grad-rhs", []),
        CastLike => ("__cast-like", []),
        ValueAndGrad => ("__value-and-grad-hof__", []),
        Negate => ("neg", []),
        Broadcast => ("broadcast", []),
        OnesLike => ("__ones-like", []),
        SumToShape => ("sum_to_shape", []),
        SliceGrad => ("slice_grad", []),
        ScanVjp => ("__scan_vjp__", []),
    }
}

impl OpId {
    pub const fn is_shape_vector_argument(self, index: usize) -> bool {
        match self {
            Self::Zeros | Self::Ones => index == 0,
            Self::Reshape | Self::RandomNormal | Self::RandomUniform | Self::RandomRandint
            | Self::Broadcast | Self::SumToShape | Self::SliceGrad => index == 1,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::OpId;
    use std::collections::HashSet;

    #[test]
    fn operation_names_and_aliases_are_unique() {
        let mut seen = HashSet::new();
        for &op in OpId::ALL {
            for name in std::iter::once(op.name()).chain(op.aliases().iter().copied()) {
                assert!(seen.insert(name), "duplicate operation name: {name}");
                assert_eq!(OpId::resolve(name), Some(op));
            }
        }
        assert_eq!(OpId::resolve("not-a-native-operation"), None);
    }

    #[test]
    fn internal_operations_are_not_public() {
        for op in [OpId::MatmulGradLhs, OpId::CastLike, OpId::SumToShape, OpId::ScanVjp] {
            assert!(!op.is_public());
        }
        assert!(OpId::Sum.is_public());
        assert!(OpId::Mean.is_public());
    }
}
