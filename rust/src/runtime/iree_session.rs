#![allow(dead_code)]

use std::ffi::c_void;

use crate::core::error::SheafError;
use crate::interpreter::value::{Dtype, Value};
use crate::runtime::iree_ffi::*;
use ndarray::ArrayD;

unsafe fn libc_stderr() -> *mut c_void {
    unsafe {
        unsafe extern "C" {
            static __stderrp: *mut c_void;
        }
        __stderrp
    }
}

pub struct IreeSession {
    instance: *mut iree_runtime_instance_t,
    device: *mut iree_hal_device_t,
    session: *mut iree_runtime_session_t,
    _vmfb_data: Option<Vec<u8>>,
}

unsafe impl Send for IreeSession {}
unsafe impl Sync for IreeSession {}

impl IreeSession {
    pub fn new() -> Result<Self, SheafError> {
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

            let driver = iree_string_view_t::from_str("local-task");
            let mut device: *mut iree_hal_device_t = std::ptr::null_mut();
            let status =
                iree_runtime_instance_try_create_default_device(instance, driver, &mut device);
            if !iree_status_is_ok(status) {
                iree_runtime_instance_release(instance);
                return Err(iree_err("failed to create local-task device"));
            }

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
                iree_hal_device_release(device);
                iree_runtime_instance_release(instance);
                return Err(iree_err("failed to create IREE session"));
            }

            Ok(IreeSession {
                instance,
                device,
                session,
                _vmfb_data: None,
            })
        }
    }

    pub fn load_vmfb(&mut self, data: Vec<u8>) -> Result<(), SheafError> {
        unsafe {
            self._vmfb_data = Some(data);
            let bytes = self._vmfb_data.as_ref().unwrap();
            let span = iree_const_byte_span_t::from_slice(bytes);
            let null_alloc = iree_allocator_t {
                self_: std::ptr::null_mut(),
                ctl: None,
            };
            let status = iree_runtime_session_append_bytecode_module_from_memory(
                self.session,
                span,
                null_alloc,
            );
            if !iree_status_is_ok(status) {
                return Err(iree_err("failed to load VMFB module"));
            }
            Ok(())
        }
    }

    pub fn call(&self, fn_name: &str, inputs: &[Value]) -> Result<Value, SheafError> {
        unsafe {
            let alloc = system_allocator();
            let device = iree_runtime_session_device(self.session);
            let device_alloc = iree_runtime_session_device_allocator(self.session);

            // Flatten tuples/dicts into individual tensor leaves for IREE
            let flat_inputs = flatten_values(inputs)?;

            let mut input_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let variant_type = iree_vm_type_def_t { value: 0 };
            let status =
                iree_vm_list_create(variant_type, flat_inputs.len(), alloc, &mut input_list);
            if !iree_status_is_ok(status) {
                return Err(iree_err("failed to create input list"));
            }

            for val in &flat_inputs {
                let bv = value_to_buffer_view(device, device_alloc, val)?;
                let ref_ = iree_hal_buffer_view_retain_ref(bv);
                let status = iree_vm_list_push_ref_retain(input_list, &ref_);
                iree_hal_buffer_view_release(bv);
                if !iree_status_is_ok(status) {
                    iree_vm_list_release(input_list);
                    return Err(iree_err("failed to push input to list"));
                }
            }

            let mut output_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let status =
                iree_vm_list_create(variant_type, 16, alloc, &mut output_list);
            if !iree_status_is_ok(status) {
                iree_vm_list_release(input_list);
                return Err(iree_err("failed to create output list"));
            }

            let name = iree_string_view_t::from_str(fn_name);
            let status = iree_runtime_session_call_by_name(
                self.session,
                name,
                input_list,
                output_list,
            );
            iree_vm_list_release(input_list);
            if !iree_status_is_ok(status) {
                iree_status_fprint(libc_stderr(), status);
                iree_vm_list_release(output_list);
                return Err(iree_err(&format!("IREE call '{}' failed", fn_name)));
            }

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
                let val = buffer_view_to_value(bv)?;
                iree_hal_buffer_view_release(bv);
                results.push(val);
            }
            iree_vm_list_release(output_list);

            match results.len() {
                0 => Ok(Value::Nil),
                1 => Ok(results.into_iter().next().unwrap()),
                _ => Ok(Value::Tuple(results)),
            }
        }
    }

    /// Call with a known return type to reconstruct nested tuple/dict structure
    /// from IREE's flattened output buffers.
    pub fn call_typed(
        &self,
        fn_name: &str,
        inputs: &[Value],
        return_type: &crate::compiler::stablehlo::StableHLOType,
    ) -> Result<Value, SheafError> {
        let flat_result = self.call(fn_name, inputs)?;
        // Unpack the flat result into the expected structure
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
        unsafe {
            if !self.session.is_null() {
                iree_runtime_session_release(self.session);
            }
            if !self.device.is_null() {
                iree_hal_device_release(self.device);
            }
            if !self.instance.is_null() {
                iree_runtime_instance_release(self.instance);
            }
        }
    }
}

unsafe fn value_to_buffer_view(
    device: *mut iree_hal_device_t,
    allocator: *mut iree_hal_allocator_t,
    val: &Value,
) -> Result<*mut iree_hal_buffer_view_t, SheafError> {
    unsafe {
        match val {
            Value::Tensor { data, dtype } => {
                let shape: Vec<iree_hal_dim_t> =
                    data.shape().iter().map(|&d| d as iree_hal_dim_t).collect();

                let (element_type, byte_data) = match dtype {
                    Dtype::F32 => {
                        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                        let bytes: Vec<u8> = f32_data
                            .iter()
                            .flat_map(|f| f.to_ne_bytes())
                            .collect();
                        (IREE_HAL_ELEMENT_TYPE_FLOAT_32, bytes)
                    }
                    _ => {
                        return Err(iree_err(&format!(
                            "unsupported dtype {:?} for IREE buffer",
                            dtype
                        )));
                    }
                };

                let params = iree_hal_buffer_params_t {
                    usage: 3 | 3072,   // TRANSFER | DISPATCH_STORAGE
                    access: 7,         // ALL (read|write|discard)
                    type_: 50,         // DEVICE_LOCAL | HOST_VISIBLE
                    queue_affinity: IREE_HAL_QUEUE_AFFINITY_ANY,
                    min_alignment: 0,
                };

                let span = iree_const_byte_span_t {
                    data: byte_data.as_ptr(),
                    data_length: byte_data.len(),
                };

                let mut bv: *mut iree_hal_buffer_view_t = std::ptr::null_mut();
                let status = iree_hal_buffer_view_allocate_buffer_copy(
                    device,
                    allocator,
                    shape.len(),
                    shape.as_ptr(),
                    element_type,
                    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                    params,
                    span,
                    &mut bv,
                );
                if !iree_status_is_ok(status) {
                    return Err(iree_err("failed to allocate IREE buffer view"));
                }
                Ok(bv)
            }
            Value::Float(f) => {
                let tensor = Value::Tensor {
                    data: ArrayD::from_elem(vec![], *f),
                    dtype: Dtype::F32,
                };
                value_to_buffer_view(device, allocator, &tensor)
            }
            Value::Int(n) => {
                let tensor = Value::Tensor {
                    data: ArrayD::from_elem(vec![], *n as f64),
                    dtype: Dtype::F32,
                };
                value_to_buffer_view(device, allocator, &tensor)
            }
            _ => Err(iree_err(&format!(
                "cannot convert {} to IREE buffer",
                val.type_name()
            ))),
        }
    }
}

unsafe fn buffer_view_to_value(
    bv: *mut iree_hal_buffer_view_t,
) -> Result<Value, SheafError> {
    unsafe {
        let rank = iree_hal_buffer_view_shape_rank(bv);
        let shape: Vec<usize> = (0..rank)
            .map(|i| iree_hal_buffer_view_shape_dim(bv, i) as usize)
            .collect();
        let elem_type = iree_hal_buffer_view_element_type(bv);

        if elem_type != IREE_HAL_ELEMENT_TYPE_FLOAT_32 {
            return Err(iree_err(&format!(
                "unsupported IREE element type: 0x{:08x}",
                elem_type
            )));
        }

        let n_elems: usize = shape.iter().product::<usize>().max(1);
        let byte_len = n_elems * 4;
        let mut f32_buf: Vec<f32> = vec![0.0; n_elems];

        let buf = iree_hal_buffer_view_buffer(bv);
        let status = iree_hal_buffer_map_read(
            buf,
            0,
            f32_buf.as_mut_ptr() as *mut c_void,
            byte_len as u64,
        );
        if !iree_status_is_ok(status) {
            return Err(iree_err("failed to read IREE buffer data"));
        }

        let f64_data: Vec<f64> = f32_buf.iter().map(|&x| x as f64).collect();
        let data = ArrayD::from_shape_vec(shape, f64_data)
            .map_err(|e| iree_err(&format!("shape mismatch: {}", e)))?;

        Ok(Value::Tensor {
            data,
            dtype: Dtype::F32,
        })
    }
}

/// Flatten a list of values into individual tensor leaves.
/// Dicts are sorted by key (matching codegen convention), then recursed.
/// Tuples are recursed. Scalars/tensors pass through.
fn flatten_values(inputs: &[Value]) -> Result<Vec<Value>, SheafError> {
    let mut flat = Vec::new();
    for val in inputs {
        flatten_value(val, &mut flat)?;
    }
    Ok(flat)
}

fn flatten_value(val: &Value, out: &mut Vec<Value>) -> Result<(), SheafError> {
    match val {
        Value::Dict(map) => {
            // Keys are already sorted (BTreeMap)
            for v in map.values() {
                flatten_value(v, out)?;
            }
            Ok(())
        }
        Value::Tuple(elems) => {
            for v in elems {
                flatten_value(v, out)?;
            }
            Ok(())
        }
        Value::Tensor { .. } | Value::Float(_) | Value::Int(_) => {
            out.push(val.clone());
            Ok(())
        }
        _ => Err(iree_err(&format!(
            "cannot flatten {} for IREE call",
            val.type_name()
        ))),
    }
}

/// Reconstruct a nested Value from a flat list of tensor Values,
/// guided by a StableHLOType structure.
fn unflatten_value(
    ty: &crate::compiler::stablehlo::StableHLOType,
    flat: &[Value],
    cursor: &mut usize,
) -> Result<Value, SheafError> {
    use crate::compiler::stablehlo::StableHLOType;
    match ty {
        StableHLOType::Tuple(elem_tys) => {
            let mut elems = Vec::new();
            for elem_ty in elem_tys {
                elems.push(unflatten_value(elem_ty, flat, cursor)?);
            }
            Ok(Value::Tuple(elems))
        }
        _ => {
            if *cursor < flat.len() {
                let val = flat[*cursor].clone();
                *cursor += 1;
                Ok(val)
            } else {
                Err(iree_err("not enough IREE outputs to reconstruct tuple structure"))
            }
        }
    }
}

fn iree_err(msg: &str) -> SheafError {
    SheafError::Runtime {
        message: msg.to_string(),
        location: None,
    }
}
