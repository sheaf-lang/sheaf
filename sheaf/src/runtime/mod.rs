// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Sheaf runtime: IREE session management, JIT compilation, buffer handling.

#[cfg(iree_runtime)]
pub mod iree_ffi;
#[cfg(iree_runtime)]
pub mod device_buffer;
#[cfg(iree_runtime)]
pub mod buffer_cache;
#[cfg(iree_runtime)]
pub mod buffer_convert;
#[cfg(iree_runtime)]
pub mod signature;
#[cfg(iree_runtime)]
pub mod iree_session;
#[cfg(iree_runtime)]
pub mod jit;
#[cfg(iree_runtime)]
pub mod toolchain;

pub fn report_jit_profile() {
    #[cfg(iree_runtime)]
    iree_session::report_shared_session_profile();
}
