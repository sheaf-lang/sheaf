#![allow(dead_code)]

use std::ffi::c_void;
use std::sync::Arc;

use crate::core::error::SheafError;
use crate::interpreter::value::Dtype;
use crate::runtime::iree_ffi::*;
use ndarray::ArrayD;

/// Shared handle to the IREE device, allowing DeviceBuffers to outlive
/// the IreeSession safely. Released when the last reference drops.
pub struct IreeDeviceHandle {
    pub(crate) device: *mut iree_hal_device_t,
}

unsafe impl Send for IreeDeviceHandle {}
unsafe impl Sync for IreeDeviceHandle {}

impl Drop for IreeDeviceHandle {
    fn drop(&mut self) {
        unsafe { iree_hal_device_release(self.device); }
    }
}

/// An IREE buffer view living on device. Wraps raw FFI pointers behind Arc
/// to make it Clone + Send + Sync, matching Value's requirements.
pub struct DeviceBufferInner {
    bv: *mut iree_hal_buffer_view_t,
    device: Arc<IreeDeviceHandle>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

unsafe impl Send for DeviceBufferInner {}
unsafe impl Sync for DeviceBufferInner {}

impl Drop for DeviceBufferInner {
    fn drop(&mut self) {
        unsafe { iree_hal_buffer_view_release(self.bv); }
    }
}

impl DeviceBufferInner {
    /// Transfer data to host, returning an ndarray of f32.
    /// For bf16 buffers, data is transferred as raw bytes then converted to f32.
    pub fn to_host(&self) -> Result<ArrayD<f32>, SheafError> {
        unsafe {
            let n_elems: usize = self.shape.iter().product::<usize>().max(1);
            let buf = iree_hal_buffer_view_buffer(self.bv);

            let f32_buf = match self.dtype {
                Dtype::F16 => {
                    let byte_len = n_elems * 2;
                    let mut raw: Vec<u16> = vec![0; n_elems];
                    let status = iree_hal_device_transfer_d2h(
                        self.device.device,
                        buf,
                        0,
                        raw.as_mut_ptr() as *mut c_void,
                        byte_len as u64,
                        0,
                        iree_timeout_t::infinite(),
                    );
                    if !iree_status_is_ok(status) {
                        iree_status_fprint(libc_stderr(), status);
                        return Err(super::buffer_convert::iree_err("d2h transfer failed (f16)"));
                    }
                    raw.iter()
                        .map(|&bits| crate::core::dtype::f16_bits_to_f32(bits))
                        .collect()
                }
                Dtype::BF16 => {
                    let byte_len = n_elems * 2;
                    let mut raw: Vec<u16> = vec![0; n_elems];
                    let status = iree_hal_device_transfer_d2h(
                        self.device.device,
                        buf,
                        0,
                        raw.as_mut_ptr() as *mut c_void,
                        byte_len as u64,
                        0,
                        iree_timeout_t::infinite(),
                    );
                    if !iree_status_is_ok(status) {
                        iree_status_fprint(libc_stderr(), status);
                        return Err(super::buffer_convert::iree_err("d2h transfer failed (bf16)"));
                    }
                    raw.iter()
                        .map(|&bits| crate::core::dtype::bf16_bits_to_f32(bits))
                        .collect()
                }
                Dtype::I32 => {
                    let byte_len = n_elems * 4;
                    let mut raw: Vec<i32> = vec![0; n_elems];
                    let status = iree_hal_device_transfer_d2h(
                        self.device.device,
                        buf,
                        0,
                        raw.as_mut_ptr() as *mut c_void,
                        byte_len as u64,
                        0,
                        iree_timeout_t::infinite(),
                    );
                    if !iree_status_is_ok(status) {
                        iree_status_fprint(libc_stderr(), status);
                        return Err(super::buffer_convert::iree_err("d2h transfer failed (i32)"));
                    }
                    raw.into_iter().map(|value| value as f32).collect()
                }
                _ => {
                    let byte_len = n_elems * 4;
                    let mut data: Vec<f32> = vec![0.0; n_elems];
                    let status = iree_hal_device_transfer_d2h(
                        self.device.device,
                        buf,
                        0,
                        data.as_mut_ptr() as *mut c_void,
                        byte_len as u64,
                        0,
                        iree_timeout_t::infinite(),
                    );
                    if !iree_status_is_ok(status) {
                        iree_status_fprint(libc_stderr(), status);
                        return Err(super::buffer_convert::iree_err("d2h transfer failed"));
                    }
                    data
                }
            };

            ArrayD::from_shape_vec(self.shape.clone(), f32_buf)
                .map_err(|e| super::buffer_convert::iree_err(&format!("shape mismatch: {}", e)))
        }
    }

    /// Get the raw buffer view pointer (for passing back to IREE).
    pub fn buffer_view(&self) -> *mut iree_hal_buffer_view_t {
        self.bv
    }

    /// Create a new DeviceBufferInner from raw parts.
    pub(super) fn new(
        bv: *mut iree_hal_buffer_view_t,
        device: Arc<IreeDeviceHandle>,
        shape: Vec<usize>,
        dtype: Dtype,
    ) -> Self {
        Self { bv, device, shape, dtype }
    }
}

pub(super) unsafe fn libc_stderr() -> *mut c_void {
    unsafe {
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                static __stderrp: *mut c_void;
            }
            __stderrp
        }
        #[cfg(not(target_os = "macos"))]
        {
            unsafe extern "C" {
                static stderr: *mut c_void;
            }
            stderr
        }
    }
}

/// Temporarily redirect fd 2 (stderr) to /dev/null, returning the saved fd.
/// Used to suppress IREE C runtime diagnostics in non-verbose mode.
pub(super) fn suppress_stderr() -> Option<i32> {
    unsafe {
        unsafe extern "C" {
            fn open(path: *const std::ffi::c_char, oflag: i32) -> i32;
            fn close(fd: i32) -> i32;
            fn dup(fd: i32) -> i32;
            fn dup2(fd: i32, fd2: i32) -> i32;
        }
        const O_WRONLY: i32 = 1;
        let devnull = open(c"/dev/null".as_ptr(), O_WRONLY);
        if devnull < 0 { return None; }
        let saved = dup(2);
        if saved < 0 { close(devnull); return None; }
        dup2(devnull, 2);
        close(devnull);
        Some(saved)
    }
}

/// Restore stderr from a saved fd (from suppress_stderr).
pub(super) fn restore_stderr(saved_fd: i32) {
    unsafe {
        unsafe extern "C" {
            fn close(fd: i32) -> i32;
            fn dup2(fd: i32, fd2: i32) -> i32;
        }
        dup2(saved_fd, 2);
        close(saved_fd);
    }
}
