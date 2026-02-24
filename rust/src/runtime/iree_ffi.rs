#![allow(non_camel_case_types, non_upper_case_globals, dead_code)]

use std::ffi::c_void;
use std::os::raw::c_char;

pub type iree_host_size_t = usize;
pub type iree_device_size_t = u64;
pub type iree_hal_dim_t = u64;
pub type iree_hal_element_type_t = u32;
pub type iree_hal_encoding_type_t = u32;
pub type iree_hal_buffer_usage_t = u32;
pub type iree_hal_memory_access_t = u16;
pub type iree_hal_memory_type_t = u32;
pub type iree_hal_queue_affinity_t = u64;
pub type iree_vm_ref_type_t = usize;

pub type iree_status_t = *mut c_void;

pub const IREE_HAL_ELEMENT_TYPE_FLOAT_32: u32 = (0x21 << 24) | 32;
pub const IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR: u32 = 1;
pub const IREE_HAL_QUEUE_AFFINITY_ANY: u64 = u64::MAX;

pub enum iree_runtime_instance_t {}
pub enum iree_runtime_session_t {}
pub enum iree_hal_device_t {}
pub enum iree_hal_allocator_t {}
pub enum iree_hal_buffer_t {}
pub enum iree_hal_buffer_view_t {}
pub enum iree_hal_driver_registry_t {}
pub enum iree_vm_list_t {}
pub enum iree_vm_module_t {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct iree_allocator_t {
    pub self_: *mut c_void,
    pub ctl: Option<unsafe extern "C" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> iree_status_t>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct iree_string_view_t {
    pub data: *const c_char,
    pub size: iree_host_size_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct iree_const_byte_span_t {
    pub data: *const u8,
    pub data_length: iree_host_size_t,
}

#[repr(C)]
pub struct iree_runtime_instance_options_t {
    pub driver_registry: *mut iree_hal_driver_registry_t,
}

#[repr(C)]
pub struct iree_runtime_session_options_t {
    pub context_flags: u32,
    pub builtin_modules: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct iree_vm_function_t {
    pub module: *mut iree_vm_module_t,
    pub linkage: u16,
    pub ordinal: u16,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct iree_vm_ref_t {
    pub ptr: *mut c_void,
    pub type_: iree_vm_ref_type_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct iree_vm_type_def_t {
    pub value: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct iree_hal_buffer_params_t {
    pub usage: iree_hal_buffer_usage_t,
    pub access: iree_hal_memory_access_t,
    pub type_: iree_hal_memory_type_t,
    pub queue_affinity: iree_hal_queue_affinity_t,
    pub min_alignment: iree_device_size_t,
}

impl iree_string_view_t {
    pub fn from_str(s: &str) -> Self {
        Self {
            data: s.as_ptr() as *const c_char,
            size: s.len(),
        }
    }
}

impl iree_const_byte_span_t {
    pub fn from_slice(s: &[u8]) -> Self {
        Self {
            data: s.as_ptr(),
            data_length: s.len(),
        }
    }
}

pub fn iree_status_is_ok(status: iree_status_t) -> bool {
    status.is_null()
}

unsafe extern "C" {
    pub fn iree_allocator_libc_ctl(
        self_: *mut c_void,
        command: u32,
        params: *const c_void,
        inout_ptr: *mut *mut c_void,
    ) -> iree_status_t;

    // Instance
    pub fn iree_runtime_instance_options_initialize(
        out: *mut iree_runtime_instance_options_t,
    );
    pub fn iree_runtime_instance_options_use_all_available_drivers(
        opts: *mut iree_runtime_instance_options_t,
    );
    pub fn iree_runtime_instance_create(
        opts: *const iree_runtime_instance_options_t,
        alloc: iree_allocator_t,
        out: *mut *mut iree_runtime_instance_t,
    ) -> iree_status_t;
    pub fn iree_runtime_instance_release(instance: *mut iree_runtime_instance_t);
    pub fn iree_runtime_instance_try_create_default_device(
        instance: *mut iree_runtime_instance_t,
        driver: iree_string_view_t,
        out: *mut *mut iree_hal_device_t,
    ) -> iree_status_t;

    // Device
    pub fn iree_hal_device_release(device: *mut iree_hal_device_t);

    // Session
    pub fn iree_runtime_session_options_initialize(
        out: *mut iree_runtime_session_options_t,
    );
    pub fn iree_runtime_session_create_with_device(
        instance: *mut iree_runtime_instance_t,
        opts: *const iree_runtime_session_options_t,
        device: *mut iree_hal_device_t,
        alloc: iree_allocator_t,
        out: *mut *mut iree_runtime_session_t,
    ) -> iree_status_t;
    pub fn iree_runtime_session_release(session: *mut iree_runtime_session_t);
    pub fn iree_runtime_session_device(
        session: *const iree_runtime_session_t,
    ) -> *mut iree_hal_device_t;
    pub fn iree_runtime_session_device_allocator(
        session: *const iree_runtime_session_t,
    ) -> *mut iree_hal_allocator_t;

    // Module loading
    pub fn iree_runtime_session_append_bytecode_module_from_memory(
        session: *mut iree_runtime_session_t,
        data: iree_const_byte_span_t,
        alloc: iree_allocator_t,
    ) -> iree_status_t;

    // Function call
    pub fn iree_runtime_session_call_by_name(
        session: *mut iree_runtime_session_t,
        name: iree_string_view_t,
        inputs: *mut iree_vm_list_t,
        outputs: *mut iree_vm_list_t,
    ) -> iree_status_t;

    // Buffer views
    pub fn iree_hal_buffer_view_allocate_buffer_copy(
        device: *mut iree_hal_device_t,
        queue_affinity: iree_hal_queue_affinity_t,
        params: iree_hal_buffer_params_t,
        shape_rank: iree_host_size_t,
        shape: *const iree_hal_dim_t,
        element_type: iree_hal_element_type_t,
        encoding_type: iree_hal_encoding_type_t,
        data: iree_const_byte_span_t,
        out: *mut *mut iree_hal_buffer_view_t,
    ) -> iree_status_t;
    pub fn iree_hal_buffer_view_release(view: *mut iree_hal_buffer_view_t);
    pub fn iree_hal_buffer_view_buffer(
        view: *const iree_hal_buffer_view_t,
    ) -> *mut iree_hal_buffer_t;
    pub fn iree_hal_buffer_view_shape_rank(
        view: *const iree_hal_buffer_view_t,
    ) -> iree_host_size_t;
    pub fn iree_hal_buffer_view_shape_dim(
        view: *const iree_hal_buffer_view_t,
        idx: iree_host_size_t,
    ) -> iree_hal_dim_t;
    pub fn iree_hal_buffer_view_element_type(
        view: *const iree_hal_buffer_view_t,
    ) -> iree_hal_element_type_t;

    // Buffer read
    pub fn iree_hal_buffer_map_read(
        buf: *mut iree_hal_buffer_t,
        offset: iree_device_size_t,
        target: *mut c_void,
        len: iree_device_size_t,
    ) -> iree_status_t;

    // VM list
    pub fn iree_vm_list_create(
        element_type: iree_vm_type_def_t,
        capacity: iree_host_size_t,
        alloc: iree_allocator_t,
        out: *mut *mut iree_vm_list_t,
    ) -> iree_status_t;
    pub fn iree_vm_list_release(list: *mut iree_vm_list_t);
    pub fn iree_vm_list_size(list: *const iree_vm_list_t) -> iree_host_size_t;
    pub fn iree_vm_list_push_ref_retain(
        list: *mut iree_vm_list_t,
        value: *const iree_vm_ref_t,
    ) -> iree_status_t;
    pub fn iree_vm_list_get_ref_retain(
        list: *const iree_vm_list_t,
        i: iree_host_size_t,
        out: *mut iree_vm_ref_t,
    ) -> iree_status_t;

    // HAL buffer_view VM ref helpers (generated by IREE_VM_DEFINE_TYPE_ADAPTERS)
    pub fn iree_hal_buffer_view_retain_ref(
        value: *mut iree_hal_buffer_view_t,
    ) -> iree_vm_ref_t;
}

pub fn system_allocator() -> iree_allocator_t {
    iree_allocator_t {
        self_: std::ptr::null_mut(),
        ctl: Some(iree_allocator_libc_ctl),
    }
}
