// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Global runtime configuration, set once from CLI flags.

use std::sync::OnceLock;

static CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

pub struct RuntimeConfig {
    pub verbosity: u8,
    pub device: Option<String>,
    pub jit_profile: bool,
    pub jit_strict: bool,
}

pub fn init(verbosity: u8, device: Option<String>, jit_profile: bool) {
    init_with_jit_strict(verbosity, device, jit_profile, false);
}

pub fn init_with_jit_strict(verbosity: u8, device: Option<String>, jit_profile: bool, jit_strict: bool) {
    let _ = CONFIG.set(RuntimeConfig { verbosity, device, jit_profile, jit_strict });
}

pub fn verbosity() -> u8 {
    CONFIG.get().map(|c| c.verbosity).unwrap_or(0)
}

pub fn device_override() -> Option<&'static str> {
    CONFIG.get().and_then(|c| c.device.as_deref())
}

pub fn jit_profile() -> bool {
    CONFIG.get().map(|c| c.jit_profile).unwrap_or(false)
}

pub fn jit_strict() -> bool {
    CONFIG.get().map(|c| c.jit_strict).unwrap_or(false)
}

pub fn jit_module_warning_threshold() -> usize {
    64
}
