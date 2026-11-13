#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static CACHED_BACKEND: OnceLock<String> = OnceLock::new();

static SESSION_CREATION_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

static LIVE_SESSION_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn session_creation_attempt_count() -> usize {
    SESSION_CREATION_ATTEMPTS.load(Ordering::Relaxed)
}

pub fn live_session_count() -> usize {
    LIVE_SESSION_COUNT.load(Ordering::Relaxed)
}

fn record_session_creation_attempt() {
    SESSION_CREATION_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
}

fn record_live_session() {
    LIVE_SESSION_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn record_session_drop() {
    let _ = LIVE_SESSION_COUNT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
        count.checked_sub(1)
    });
}

static SHARED_SESSION: OnceLock<Arc<IreeSession>> = OnceLock::new();
static SHARED_SESSION_INIT: Mutex<()> = Mutex::new(());

pub fn shared_session() -> Result<Arc<IreeSession>, SheafError> {
    if let Some(session) = SHARED_SESSION.get() {
        return Ok(Arc::clone(session));
    }
    let _init = SHARED_SESSION_INIT.lock().map_err(|_| SheafError::Runtime {
        message: "IREE shared session initialization lock is poisoned".to_string(),
        location: None,
    })?;
    if let Some(session) = SHARED_SESSION.get() {
        return Ok(Arc::clone(session));
    }
    let session = Arc::new(IreeSession::new()?);
    SHARED_SESSION.set(Arc::clone(&session)).map_err(|_| SheafError::Runtime {
        message: "failed to publish IREE shared session".to_string(),
        location: None,
    })?;
    Ok(session)
}

pub fn initialized_shared_session() -> Option<Arc<IreeSession>> {
    SHARED_SESSION.get().map(Arc::clone)
}

pub fn report_shared_session_profile() {
    if let Some(session) = SHARED_SESSION.get() {
        session.report_profile();
    }
}

use crate::core::error::SheafError;
use crate::sheaf_msg;
use crate::interpreter::value::{Dtype, Value};
use crate::runtime::iree_ffi::*;

use super::buffer_cache::{BufferViewCache, TensorFingerprint};
use super::buffer_convert::{
    buffer_view_to_value, flatten_values, iree_err, unflatten_value, value_to_buffer_view,
};
use super::device_buffer::{libc_stderr, suppress_stderr, restore_stderr};

pub use super::device_buffer::{DeviceBufferInner, IreeDeviceHandle};
pub use super::signature::{
    args_match_signature, check_shapes_match, count_arg_tensors, count_signature_tensors,
};

const INITIAL_BUFFER_VIEW_CACHE_CAPACITY: usize = 512;

fn decode_scalar_value(dtype: Dtype, raw: [u8; 4]) -> Value {
    match dtype {
        Dtype::F16 => {
            let bits = u16::from_le_bytes([raw[0], raw[1]]);
            Value::Float(crate::core::dtype::f16_bits_to_f32(bits))
        }
        Dtype::BF16 => {
            let bits = u16::from_le_bytes([raw[0], raw[1]]);
            Value::Float(crate::core::dtype::bf16_bits_to_f32(bits))
        }
        Dtype::I32 => Value::Int(i32::from_le_bytes(raw) as i64),
        _ => Value::Float(f32::from_le_bytes(raw)),
    }
}

pub struct IreeSession {
    instance: *mut iree_runtime_instance_t,
    device_handle: Arc<IreeDeviceHandle>,
    session: *mut iree_runtime_session_t,
    // Retains source spans referenced by IREE.
    _vmfb_data: Mutex<Vec<Vec<u8>>>,
    driver_name: String,
    buffer_cache: Mutex<BufferViewCache>,
    profile: bool,
    profile_reported: AtomicBool,
    t_flatten_ns: AtomicU64,
    t_buffers_ns: AtomicU64,
    t_call_ns: AtomicU64,
    t_output_ns: AtomicU64,
    n_calls: AtomicU64,
    n_cache_hits: AtomicU64,
    n_cache_misses: AtomicU64,
}

unsafe impl Send for IreeSession {}
unsafe impl Sync for IreeSession {}

impl IreeSession {
    pub fn new() -> Result<Self, SheafError> {
        record_session_creation_attempt();
        unsafe {
            let alloc = system_allocator();

            let mut opts: iree_runtime_instance_options_t = std::mem::zeroed();
            iree_runtime_instance_options_initialize(&mut opts);
            iree_runtime_instance_options_use_all_available_drivers(&mut opts);

            let mut instance: *mut iree_runtime_instance_t = std::ptr::null_mut();
            let status = iree_runtime_instance_create(&opts, alloc, &mut instance);
            if !iree_status_is_ok(status) {
                return Err(iree_err("failed to create IREE instance"));
            }

            let device_override = crate::core::config::device_override();
            let driver_names: Vec<&str> = match device_override {
                Some("cpu") => vec!["local-task"],
                Some(d) => vec![d, "local-task"],
                None => vec!["cuda", "metal", "vulkan", "local-task"],
            };
            let mut device: *mut iree_hal_device_t = std::ptr::null_mut();
            let mut chosen_driver = "";
            for name in &driver_names {
                let driver = iree_string_view_t::from_utf8(name);
                let status =
                    iree_runtime_instance_try_create_default_device(instance, driver, &mut device);
                if iree_status_is_ok(status) {
                    chosen_driver = name;
                    break;
                } else if crate::core::config::verbosity() >= 2 {
                    sheaf_msg!("jit: driver '{}' not available", name);
                }
            }
            if device.is_null() {
                iree_runtime_instance_release(instance);
                let tried = driver_names.join(", ");
                return Err(iree_err(&format!(
                    "failed to create IREE device (tried: {})", tried
                )));
            }
            let backend = match chosen_driver {
                "metal" => "metal-spirv",
                "vulkan" => "vulkan-spirv",
                "cuda" => "cuda",
                _ => "llvm-cpu",
            };
            let _ = CACHED_BACKEND.set(backend.to_string());

            {
                use std::sync::atomic::AtomicBool;
                static PRINTED: AtomicBool = AtomicBool::new(false);
                if !PRINTED.swap(true, Ordering::Relaxed) {
                    let display = match chosen_driver {
                        "local-task" => "CPU".to_string(),
                        "metal" => "Metal GPU".to_string(),
                        "cuda" => match super::jit::detect_cuda_target() {
                            Some(target) => format!("CUDA GPU ({})", target),
                            None => "CUDA GPU".to_string(),
                        },
                        "vulkan" => "Vulkan GPU".to_string(),
                        other => other.to_string(),
                    };
                    sheaf_msg!("sheaf: running on {}", display);
                }
            }

            // IreeDeviceHandle owns a retained device reference.
            iree_hal_device_retain(device);
            let device_handle = Arc::new(IreeDeviceHandle { device });

            let mut session_opts: iree_runtime_session_options_t = std::mem::zeroed();
            iree_runtime_session_options_initialize(&mut session_opts);

            let mut session: *mut iree_runtime_session_t = std::ptr::null_mut();
            let status = iree_runtime_session_create_with_device(
                instance,
                &session_opts,
                device,
                alloc,
                &mut session,
            );
            if !iree_status_is_ok(status) {
                drop(device_handle);
                iree_hal_device_release(device);
                iree_runtime_instance_release(instance);
                return Err(iree_err("failed to create IREE session"));
            }

            let result = IreeSession {
                instance,
                device_handle,
                session,
                _vmfb_data: Mutex::new(Vec::new()),
                driver_name: chosen_driver.to_string(),
                buffer_cache: Mutex::new(BufferViewCache::new(INITIAL_BUFFER_VIEW_CACHE_CAPACITY)),
                profile: crate::core::config::jit_profile(),
                profile_reported: AtomicBool::new(false),
                t_flatten_ns: AtomicU64::new(0),
                t_buffers_ns: AtomicU64::new(0),
                t_call_ns: AtomicU64::new(0),
                t_output_ns: AtomicU64::new(0),
                n_calls: AtomicU64::new(0),
                n_cache_hits: AtomicU64::new(0),
                n_cache_misses: AtomicU64::new(0),
            };
            record_live_session();
            Ok(result)
        }
    }

    pub fn target_backend(&self) -> &str {
        match self.driver_name.as_str() {
            "metal" => "metal-spirv",
            "vulkan" => "vulkan-spirv",
            "cuda" => "cuda",
            _ => "llvm-cpu",
        }
    }

    pub fn cached_target_backend() -> Option<&'static str> {
        CACHED_BACKEND.get().map(|s| s.as_str())
    }

    pub fn driver_name(&self) -> &str {
        &self.driver_name
    }

    fn report_profile(&self) {
        if !self.profile {
            return;
        }
        let n = self.n_calls.load(Ordering::Relaxed);
        if n == 0 || self.profile_reported.swap(true, Ordering::Relaxed) {
            return;
        }

        let flatten = self.t_flatten_ns.load(Ordering::Relaxed) as f64 / 1e6;
        let buffers = self.t_buffers_ns.load(Ordering::Relaxed) as f64 / 1e6;
        let call = self.t_call_ns.load(Ordering::Relaxed) as f64 / 1e6;
        let output = self.t_output_ns.load(Ordering::Relaxed) as f64 / 1e6;
        let total = flatten + buffers + call + output;
        let hits = self.n_cache_hits.load(Ordering::Relaxed);
        let misses = self.n_cache_misses.load(Ordering::Relaxed);
        let cache_size = self
            .buffer_cache
            .lock()
            .map(|cache| (cache.len(), cache.capacity(), cache.evictions()))
            .unwrap_or((0, 0, 0));
        sheaf_msg!(
            "\njit: dispatch profile ({} calls, {:.1}ms total):",
            n,
            total
        );
        sheaf_msg!(
            "  flatten:  {:7.1}ms ({:4.1}%)",
            flatten,
            flatten / total * 100.0
        );
        sheaf_msg!(
            "  buffers:  {:7.1}ms ({:4.1}%)  [hits: {}, misses: {}, cache: {}/{}, evictions: {}]",
            buffers,
            buffers / total * 100.0,
            hits,
            misses,
            cache_size.0,
            cache_size.1,
            cache_size.2
        );
        sheaf_msg!("  call:     {:7.1}ms ({:4.1}%)", call, call / total * 100.0);
        sheaf_msg!(
            "  output:   {:7.1}ms ({:4.1}%)",
            output,
            output / total * 100.0
        );
    }

    pub fn device_handle(&self) -> &Arc<IreeDeviceHandle> {
        &self.device_handle
    }

    pub fn device_allocator_ptr(&self) -> *mut crate::runtime::iree_ffi::iree_hal_allocator_t {
        unsafe { iree_runtime_session_device_allocator(self.session) }
    }

    pub fn load_vmfb(&self, data: Vec<u8>) -> Result<(), SheafError> {
        let mut vmfb_data = self._vmfb_data.lock().map_err(|_| SheafError::Runtime {
            message: "IREE VMFB retention lock is poisoned".to_string(),
            location: None,
        })?;
        vmfb_data.push(data);
        unsafe {
            let bytes = vmfb_data
                .last()
                .ok_or_else(|| iree_err("failed to retain VMFB source data"))?;
            let span = iree_const_byte_span_t::from_slice(bytes);
            let null_alloc = iree_allocator_t {
                self_: std::ptr::null_mut(),
                ctl: None,
            };
            let suppress = crate::core::config::verbosity() < 2;
            let saved_stderr = if suppress { suppress_stderr() } else { None };
            let status = iree_runtime_session_append_bytecode_module_from_memory(
                self.session,
                span,
                null_alloc,
            );
            if let Some(fd) = saved_stderr { restore_stderr(fd); }
            if !iree_status_is_ok(status) {
                if crate::core::config::verbosity() >= 2 {
                    iree_status_fprint(libc_stderr(), status);
                }
                vmfb_data.pop();
                return Err(iree_err("failed to load compiled module"));
            }

            Ok(())
        }
    }

    pub fn call(&self, fn_name: &str, inputs: &[Value]) -> Result<Value, SheafError> {
        unsafe {
            let alloc = system_allocator();
            let device = iree_runtime_session_device(self.session);
            let device_alloc = iree_runtime_session_device_allocator(self.session);

            let t0 = if self.profile { Some(std::time::Instant::now()) } else { None };

            let flat_inputs = flatten_values(inputs)?;

            let t1 = t0.map(|_| std::time::Instant::now());

            let mut input_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let variant_type = iree_vm_type_def_t { value: 0 };
            let status =
                iree_vm_list_create(variant_type, flat_inputs.len(), alloc, &mut input_list);
            if !iree_status_is_ok(status) {
                return Err(iree_err("failed to create input list"));
            }

            let mut cache = self.buffer_cache.lock().unwrap();
            cache.ensure_capacity(flat_inputs.len());

            for val in &flat_inputs {
                let bv = if let Some(buffer_view) = cache.get(val) {
                    if self.profile { self.n_cache_hits.fetch_add(1, Ordering::Relaxed); }
                    buffer_view
                } else {
                    if self.profile { self.n_cache_misses.fetch_add(1, Ordering::Relaxed); }
                    let new_bv = match value_to_buffer_view(device, device_alloc, val) {
                        Ok(view) => view,
                        Err(error) => {
                            // The list releases retained arguments on failure.
                            iree_vm_list_release(input_list);
                            return Err(error);
                        }
                    };
                    if let Some(fingerprint) = TensorFingerprint::from_value(val)
                        && let Some(evicted) = cache.insert(fingerprint, new_bv)
                    {
                        iree_hal_buffer_view_release(evicted);
                    }
                    new_bv
                };

                let mut ref_ = iree_hal_buffer_view_retain_ref(bv);
                let status = iree_vm_list_push_ref_retain(input_list, &ref_);
                iree_vm_ref_release(&mut ref_);
                if !iree_status_is_ok(status) {
                    iree_vm_list_release(input_list);
                    return Err(iree_err("failed to push input to list"));
                }
            }

            // Do not hold the cache lock during dispatch.
            drop(cache);

            let t2 = t0.map(|_| std::time::Instant::now());

            let mut output_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let status =
                iree_vm_list_create(variant_type, 16, alloc, &mut output_list);
            if !iree_status_is_ok(status) {
                iree_vm_list_release(input_list);
                return Err(iree_err("failed to create output list"));
            }

            let name = iree_string_view_t::from_utf8(fn_name);
            let status = iree_runtime_session_call_by_name(
                self.session,
                name,
                input_list,
                output_list,
            );
            iree_vm_list_release(input_list);
            if !iree_status_is_ok(status) {
                iree_vm_list_release(output_list);
                let clean_name = fn_name.strip_prefix("module.").unwrap_or(fn_name).replace('_', "-");
                let got = inputs.iter().map(|v| v.short_desc()).collect::<Vec<_>>().join(", ");
                return Err(iree_err(&format!(
                    "{}: shape mismatch (compiled signature does not match arguments).\n  Called with: ({})",
                    clean_name, got
                )));
            }

            let t3 = t0.map(|_| std::time::Instant::now());

            let n_outputs = iree_vm_list_size(output_list);
            let mut results = Vec::with_capacity(n_outputs);
            for i in 0..n_outputs {
                let mut ref_: iree_vm_ref_t = std::mem::zeroed();
                let status = iree_vm_list_get_ref_retain(output_list, i, &mut ref_);
                if !iree_status_is_ok(status) {
                    iree_vm_list_release(output_list);
                    return Err(iree_err("failed to get output from list"));
                }
                let bv = ref_.ptr as *mut iree_hal_buffer_view_t;
                let val = match buffer_view_to_value(device, bv) {
                    Ok(value) => value,
                    Err(error) => {
                        iree_vm_ref_release(&mut ref_);
                        iree_vm_list_release(output_list);
                        return Err(error);
                    }
                };
                iree_vm_ref_release(&mut ref_);
                results.push(val);
            }
            iree_vm_list_release(output_list);

            if let (Some(t0), Some(t1), Some(t2), Some(t3)) = (t0, t1, t2, t3) {
                let t4 = std::time::Instant::now();
                self.t_flatten_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                self.t_buffers_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
                self.t_call_ns.fetch_add((t3 - t2).as_nanos() as u64, Ordering::Relaxed);
                self.t_output_ns.fetch_add((t4 - t3).as_nanos() as u64, Ordering::Relaxed);
                self.n_calls.fetch_add(1, Ordering::Relaxed);
            }

            match results.len() {
                0 => Ok(Value::Nil),
                1 => Ok(results.into_iter().next().unwrap()),
                _ => Ok(Value::Tuple(results)),
            }
        }
    }

    pub fn call_typed(
        &self,
        fn_name: &str,
        inputs: &[Value],
        return_type: &crate::lowering::stablehlo::StableHLOType,
    ) -> Result<Value, SheafError> {
        let flat_result = self.call(fn_name, inputs)?;
        let flat_values = match flat_result {
            Value::Tuple(vals) => vals,
            other => vec![other],
        };
        let mut cursor = 0;
        let structured = unflatten_value(return_type, &flat_values, &mut cursor)?;
        Ok(structured)
    }

    pub fn call_device(&self, fn_name: &str, inputs: &[Value]) -> Result<Value, SheafError> {
        unsafe {
            let alloc = system_allocator();
            let device = iree_runtime_session_device(self.session);
            let device_alloc = iree_runtime_session_device_allocator(self.session);

            let t0 = if self.profile { Some(std::time::Instant::now()) } else { None };

            let flat_inputs = flatten_values(inputs)?;

            let t1 = t0.map(|_| std::time::Instant::now());

            let mut input_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let variant_type = iree_vm_type_def_t { value: 0 };
            let status =
                iree_vm_list_create(variant_type, flat_inputs.len(), alloc, &mut input_list);
            if !iree_status_is_ok(status) {
                return Err(iree_err("failed to create input list"));
            }

            let all_device = flat_inputs.iter().all(|v| matches!(v, Value::DeviceBuffer(_)));

            if all_device {
                for val in flat_inputs.iter() {
                    if let Value::DeviceBuffer(db) = val {
                        if self.profile { self.n_cache_hits.fetch_add(1, Ordering::Relaxed); }
                        let mut ref_ = iree_hal_buffer_view_retain_ref(db.buffer_view());
                        let status = iree_vm_list_push_ref_retain(input_list, &ref_);
                        iree_vm_ref_release(&mut ref_);
                        if !iree_status_is_ok(status) {
                            iree_vm_list_release(input_list);
                            return Err(iree_err("failed to push input to list"));
                        }
                    }
                }
            } else {
                let mut cache = self.buffer_cache.lock().unwrap();
                cache.ensure_capacity(flat_inputs.len());

                for val in &flat_inputs {
                    let bv = match val {
                        Value::DeviceBuffer(db) => {
                            if self.profile { self.n_cache_hits.fetch_add(1, Ordering::Relaxed); }
                            db.buffer_view()
                        }
                        _ => {
                            if let Some(buffer_view) = cache.get(val) {
                                if self.profile { self.n_cache_hits.fetch_add(1, Ordering::Relaxed); }
                                buffer_view
                            } else {
                                if self.profile { self.n_cache_misses.fetch_add(1, Ordering::Relaxed); }
                                let new_bv = match value_to_buffer_view(device, device_alloc, val) {
                                    Ok(view) => view,
                                    Err(error) => {
                                        // The list releases retained arguments on failure.
                                        iree_vm_list_release(input_list);
                                        return Err(error);
                                    }
                                };
                                if let Some(fingerprint) = TensorFingerprint::from_value(val)
                                    && let Some(evicted) = cache.insert(fingerprint, new_bv)
                                {
                                    iree_hal_buffer_view_release(evicted);
                                }
                                new_bv
                            }
                        }
                    };

                    let mut ref_ = iree_hal_buffer_view_retain_ref(bv);
                    let status = iree_vm_list_push_ref_retain(input_list, &ref_);
                    iree_vm_ref_release(&mut ref_);
                    if !iree_status_is_ok(status) {
                        iree_vm_list_release(input_list);
                        return Err(iree_err("failed to push input to list"));
                    }
                }

                drop(cache);
            }

            let t2 = t0.map(|_| std::time::Instant::now());

            let mut output_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let status =
                iree_vm_list_create(variant_type, 16, alloc, &mut output_list);
            if !iree_status_is_ok(status) {
                iree_vm_list_release(input_list);
                return Err(iree_err("failed to create output list"));
            }

            let name = iree_string_view_t::from_utf8(fn_name);
            let status = iree_runtime_session_call_by_name(
                self.session,
                name,
                input_list,
                output_list,
            );
            iree_vm_list_release(input_list);
            if !iree_status_is_ok(status) {
                iree_vm_list_release(output_list);
                let clean_name = fn_name.strip_prefix("module.").unwrap_or(fn_name).replace('_', "-");
                let got = inputs.iter().map(|v| v.short_desc()).collect::<Vec<_>>().join(", ");
                return Err(iree_err(&format!(
                    "{}: shape mismatch (compiled signature does not match arguments).\n  Called with: ({})",
                    clean_name, got
                )));
            }

            let t3 = t0.map(|_| std::time::Instant::now());

            let n_outputs = iree_vm_list_size(output_list);
            let mut results = Vec::with_capacity(n_outputs);
            for i in 0..n_outputs {
                let mut ref_: iree_vm_ref_t = std::mem::zeroed();
                let status = iree_vm_list_get_ref_retain(output_list, i, &mut ref_);
                if !iree_status_is_ok(status) {
                    iree_vm_list_release(output_list);
                    return Err(iree_err("failed to get output from list"));
                }
                let bv = ref_.ptr as *mut iree_hal_buffer_view_t;

                let rank = iree_hal_buffer_view_shape_rank(bv);
                let shape: Vec<usize> = (0..rank)
                    .map(|j| iree_hal_buffer_view_shape_dim(bv, j) as usize)
                    .collect();
                let elem_type = iree_hal_buffer_view_element_type(bv);
                let dtype = if elem_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 {
                    Dtype::F16
                } else if elem_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 {
                    Dtype::BF16
                } else if elem_type == IREE_HAL_ELEMENT_TYPE_INT_32 {
                    Dtype::I32
                } else {
                    Dtype::F32
                };

                // Keep scalar control flow on the host.
                if shape.is_empty() {
                    let buf = iree_hal_buffer_view_buffer(bv);
                    let byte_len = if matches!(dtype, Dtype::F16 | Dtype::BF16) { 2u64 } else { 4u64 };
                    let mut raw = [0u8; 4];
                    let status = iree_hal_device_transfer_d2h(
                        self.device_handle.device,
                        buf,
                        0,
                        raw.as_mut_ptr() as *mut std::ffi::c_void,
                        byte_len,
                        0,
                        iree_timeout_t::infinite(),
                    );
                    iree_hal_buffer_view_release(bv);
                    ref_.ptr = std::ptr::null_mut();
                    iree_vm_ref_release(&mut ref_);
                    if !iree_status_is_ok(status) {
                        let message = status_message(status);
                        iree_status_free(status);
                        iree_vm_list_release(output_list);
                        return Err(iree_err(&format!("failed to read a scalar result: {message}")));
                    }
                    results.push(decode_scalar_value(dtype, raw));
                } else {
                    let db = Arc::new(DeviceBufferInner::new(
                        bv,
                        Arc::clone(&self.device_handle),
                        shape,
                        dtype,
                    ));
                    // DeviceBuffer owns the retained buffer view.
                    ref_.ptr = std::ptr::null_mut();
                    iree_vm_ref_release(&mut ref_);

                    results.push(Value::DeviceBuffer(db));
                }
            }
            iree_vm_list_release(output_list);

            if let (Some(t0), Some(t1), Some(t2), Some(t3)) = (t0, t1, t2, t3) {
                let t4 = std::time::Instant::now();
                self.t_flatten_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                self.t_buffers_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
                self.t_call_ns.fetch_add((t3 - t2).as_nanos() as u64, Ordering::Relaxed);
                self.t_output_ns.fetch_add((t4 - t3).as_nanos() as u64, Ordering::Relaxed);
                self.n_calls.fetch_add(1, Ordering::Relaxed);
            }

            match results.len() {
                0 => Ok(Value::Nil),
                1 => Ok(results.into_iter().next().unwrap()),
                _ => Ok(Value::Tuple(results)),
            }
        }
    }

    pub fn call_typed_device(
        &self,
        fn_name: &str,
        inputs: &[Value],
        return_type: &crate::lowering::stablehlo::StableHLOType,
    ) -> Result<Value, SheafError> {
        let flat_result = self.call_device(fn_name, inputs)?;
        let flat_values = match flat_result {
            Value::Tuple(vals) => vals,
            other => vec![other],
        };
        let mut cursor = 0;
        let structured = unflatten_value(return_type, &flat_values, &mut cursor)?;
        Ok(structured)
    }
}


impl Drop for IreeSession {
    fn drop(&mut self) {
        record_session_drop();
        self.report_profile();
        unsafe {
            // Cached buffer views must not outlive the session.
            if let Ok(cache) = self.buffer_cache.lock() {
                for buffer_view in cache.buffer_views() {
                    iree_hal_buffer_view_release(buffer_view);
                }
            }
            if !self.session.is_null() {
                iree_runtime_session_release(self.session);
            }
            // device_handle keeps the device alive for outstanding buffers.
            if !self.instance.is_null() {
                iree_runtime_instance_release(self.instance);
            }
        }
    }
}

/// Format an IREE status for an error message, without consuming it.
unsafe fn status_message(status: iree_status_t) -> String {
    let mut buffer = [0u8; 1024];
    let mut length = 0usize;
    unsafe {
        iree_status_format(status, buffer.len(), buffer.as_mut_ptr().cast(), &mut length);
    }
    // A truncated message keeps its first characters before the null terminator.
    let length = length.min(buffer.len() - 1);
    if length == 0 {
        return "unknown IREE error".to_string();
    }
    String::from_utf8_lossy(&buffer[..length]).into_owned()
}

#[cfg(test)]
mod lifetime_counter_tests {
    use super::{
        LIVE_SESSION_COUNT, SESSION_CREATION_ATTEMPTS, decode_scalar_value, live_session_count,
        record_live_session, record_session_creation_attempt, record_session_drop,
        session_creation_attempt_count,
    };
    use crate::interpreter::value::{Dtype, Value};
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, OnceLock};

    fn counter_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct CounterRestore {
        attempts: usize,
        live: usize,
    }

    impl Drop for CounterRestore {
        fn drop(&mut self) {
            SESSION_CREATION_ATTEMPTS.store(self.attempts, Ordering::Relaxed);
            LIVE_SESSION_COUNT.store(self.live, Ordering::Relaxed);
        }
    }

    #[test]
    fn scalar_i32_results_preserve_their_bits() {
        assert!(matches!(
            decode_scalar_value(Dtype::I32, 262i32.to_le_bytes()),
            Value::Int(262)
        ));
        assert!(matches!(
            decode_scalar_value(Dtype::I32, (-7i32).to_le_bytes()),
            Value::Int(-7)
        ));
    }

    #[test]
    fn lifetime_counters_use_deltas_and_live_count_does_not_underflow() {
        let _lock = counter_test_lock().lock().unwrap();
        let restore = CounterRestore {
            attempts: session_creation_attempt_count(),
            live: live_session_count(),
        };
        record_session_creation_attempt();
        assert_eq!(session_creation_attempt_count(), restore.attempts + 1);
        assert_eq!(live_session_count(), restore.live);
        record_live_session();
        assert_eq!(live_session_count(), restore.live + 1);
        record_session_drop();
        assert_eq!(live_session_count(), restore.live);
        LIVE_SESSION_COUNT.store(0, Ordering::Relaxed);
        record_session_drop();
        assert_eq!(live_session_count(), 0);
    }
}
