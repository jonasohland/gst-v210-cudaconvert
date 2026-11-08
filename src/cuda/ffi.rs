//! Minimal hand-written bindings to libgstcuda-1.0 (gst-plugins-bad), which also
//! re-exports the CUDA driver API it loads at runtime (`Cu*` symbols). Nothing
//! here links to libcuda directly, so the plugin works with any driver version
//! that gst-cuda can load.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use gst::ffi::{
    GstBufferPool, GstContext, GstElement, GstMemory, GstObject, GstObjectClass, GstQuery,
};
use gst::glib::ffi::{gboolean, GType};
use gst_video::ffi::GstVideoInfo;
use std::os::raw::{c_char, c_int, c_uint, c_void};

pub type CUresult = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

pub const CU_MEMORYTYPE_HOST: c_uint = 1;
pub const CU_MEMORYTYPE_DEVICE: c_uint = 2;

/// `GST_MAP_CUDA` = `GST_MAP_FLAG_LAST << 1`
pub const GST_MAP_CUDA: c_uint = gst::ffi::GST_MAP_FLAG_LAST << 1;

pub const CAPS_FEATURE_MEMORY_CUDA: &str = "memory:CUDAMemory";

#[repr(C)]
pub struct CUDA_MEMCPY2D {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: c_uint,
    pub srcHost: *const c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: *mut c_void,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: c_uint,
    pub dstHost: *mut c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: *mut c_void,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}

#[repr(C)]
pub struct GstCudaContext {
    pub object: GstObject,
    pub priv_: *mut c_void,
}

#[repr(C)]
pub struct GstCudaContextClass {
    pub parent_class: GstObjectClass,
}

#[repr(C)]
pub struct GstCudaMemory {
    pub mem: GstMemory,
    pub context: *mut GstCudaContext,
    pub info: GstVideoInfo,
    pub priv_: *mut c_void,
    pub _gst_reserved: [*mut c_void; 4],
}

#[repr(C)]
pub struct GstCudaStream {
    _private: [u8; 0],
}

#[link(name = "gstcuda-1.0")]
extern "C" {
    pub fn gst_cuda_load_library() -> gboolean;
    pub fn gst_cuda_memory_init_once();

    pub fn gst_cuda_context_get_type() -> GType;
    pub fn gst_cuda_context_push(ctx: *mut GstCudaContext) -> gboolean;
    pub fn gst_cuda_context_pop(cuda_ctx: *mut CUcontext) -> gboolean;
    pub fn gst_cuda_context_can_access_peer(
        ctx: *mut GstCudaContext,
        peer: *mut GstCudaContext,
    ) -> gboolean;

    pub fn gst_cuda_ensure_element_context(
        element: *mut GstElement,
        device_id: c_int,
        cuda_ctx: *mut *mut GstCudaContext,
    ) -> gboolean;
    pub fn gst_cuda_handle_set_context(
        element: *mut GstElement,
        context: *mut GstContext,
        device_id: c_int,
        cuda_ctx: *mut *mut GstCudaContext,
    ) -> gboolean;
    pub fn gst_cuda_handle_context_query(
        element: *mut GstElement,
        query: *mut GstQuery,
        cuda_ctx: *mut GstCudaContext,
    ) -> gboolean;

    pub fn gst_cuda_stream_new(context: *mut GstCudaContext) -> *mut GstCudaStream;
    pub fn gst_cuda_stream_get_handle(stream: *mut GstCudaStream) -> CUstream;
    pub fn gst_cuda_stream_unref(stream: *mut GstCudaStream);

    pub fn gst_is_cuda_memory(mem: *mut GstMemory) -> gboolean;
    pub fn gst_cuda_memory_sync(mem: *mut GstCudaMemory);

    pub fn gst_cuda_buffer_pool_new(context: *mut GstCudaContext) -> *mut GstBufferPool;
    pub fn gst_buffer_pool_config_set_cuda_stream(
        config: *mut gst::ffi::GstStructure,
        stream: *mut GstCudaStream,
    );

    // Driver API, loaded at runtime by gst_cuda_load_library().
    pub fn CuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult;
    pub fn CuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn CuModuleUnload(module: CUmodule) -> CUresult;
    pub fn CuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn CuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    pub fn CuMemAllocPitch(
        dptr: *mut CUdeviceptr,
        pPitch: *mut usize,
        WidthInBytes: usize,
        Height: usize,
        ElementSizeBytes: c_uint,
    ) -> CUresult;
    pub fn CuMemFree(dptr: CUdeviceptr) -> CUresult;
    pub fn CuMemcpy2DAsync(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult;
    pub fn CuStreamSynchronize(hStream: CUstream) -> CUresult;
}
