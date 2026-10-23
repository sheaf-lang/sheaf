// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! IR transforms used by AOT and JIT compilation.

mod constants;
mod dict_layout;
mod reduce;
mod tuples;

pub(crate) use constants::{collect_shape_gtes, filter_constants_for_shape_positions};
pub use constants::{
    extract_scalar_constants, resolve_static_constants, try_infer_shape,
};
pub use dict_layout::{lower_inlined_gets, propagate_let_layouts};
pub use reduce::unroll_reduces;
pub use tuples::{classify_vectors, desugar_destructuring_lets, lower_tuples_and_destructuring};
