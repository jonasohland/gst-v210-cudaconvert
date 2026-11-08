//! Thin safe-ish wrappers over the gst-cuda library and the CUDA driver API.

pub mod ffi;

use gst::glib;
use gst::glib::translate::*;
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_void;
use std::ptr;

glib::wrapper! {
    /// `GstCudaContext`
    pub struct CudaContext(Object<ffi::GstCudaContext, ffi::GstCudaContextClass>) @extends gst::Object;

    match fn {
        type_ => || ffi::gst_cuda_context_get_type(),
    }
}

// GstCudaContext is internally synchronized.
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    /// Makes the context current on this thread until the guard is dropped.
    pub fn push(&self) -> Result<ContextGuard, Error> {
        if unsafe { ffi::gst_cuda_context_push(self.to_glib_none().0) } == 0 {
            return Err(Error::msg("failed to push CUDA context"));
        }
        Ok(ContextGuard(()))
    }

    /// Whether device memory of `other` can be used from this context.
    pub fn can_access(&self, other: &CudaContext) -> bool {
        self == other
            || unsafe {
                ffi::gst_cuda_context_can_access_peer(self.to_glib_none().0, other.to_glib_none().0)
                    != 0
            }
    }
}

pub struct ContextGuard(());

impl Drop for ContextGuard {
    fn drop(&mut self) {
        unsafe {
            ffi::gst_cuda_context_pop(ptr::null_mut());
        }
    }
}

#[derive(Debug, Clone)]
pub struct Error(String);

impl Error {
    pub fn msg(s: impl Into<String>) -> Self {
        Error(s.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Converts a `CUresult` to a `Result`, naming the failed call.
pub fn check(res: ffi::CUresult, what: &str) -> Result<(), Error> {
    if res == ffi::CUDA_SUCCESS {
        return Ok(());
    }
    let mut name: *const std::os::raw::c_char = ptr::null();
    unsafe { ffi::CuGetErrorName(res, &mut name) };
    let name = if name.is_null() {
        format!("error {res}")
    } else {
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
    };
    Err(Error(format!("{what} failed: {name}")))
}

/// Loads the CUDA driver through gst-cuda. Returns false if no driver is available.
pub fn load_library() -> bool {
    unsafe { ffi::gst_cuda_load_library() != 0 }
}

/// A `GstCudaStream`.
pub struct Stream(ptr::NonNull<ffi::GstCudaStream>);

unsafe impl Send for Stream {}

impl Stream {
    pub fn new(ctx: &CudaContext) -> Result<Self, Error> {
        let s = unsafe { ffi::gst_cuda_stream_new(ctx.to_glib_none().0) };
        ptr::NonNull::new(s)
            .map(Stream)
            .ok_or_else(|| Error::msg("failed to create CUDA stream"))
    }

    pub fn as_ptr(&self) -> *mut ffi::GstCudaStream {
        self.0.as_ptr()
    }

    pub fn handle(&self) -> ffi::CUstream {
        unsafe { ffi::gst_cuda_stream_get_handle(self.0.as_ptr()) }
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        check(
            unsafe { ffi::CuStreamSynchronize(self.handle()) },
            "cuStreamSynchronize",
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { ffi::gst_cuda_stream_unref(self.0.as_ptr()) }
    }
}

/// Kernel parameter block; must match `struct Params` in kernels/v210.cu.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct KernelParams {
    pub plane: [u64; 3],
    pub stride: [i64; 3],
    pub v210: u64,
    pub v210_stride: i64,
    pub width: i32,
    pub height: i32,
}

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/v210.fatbin"));

/// The kernel module loaded into one context. Must be dropped while the context
/// can still be pushed (it holds a reference to it).
pub struct Module {
    ctx: CudaContext,
    module: ffi::CUmodule,
}

unsafe impl Send for Module {}

/// A kernel function in a [`Module`].
#[derive(Clone, Copy)]
pub struct Function(ffi::CUfunction);

unsafe impl Send for Function {}

impl Module {
    pub fn load(ctx: &CudaContext) -> Result<Self, Error> {
        let _g = ctx.push()?;
        let mut module = ptr::null_mut();
        check(
            unsafe { ffi::CuModuleLoadData(&mut module, FATBIN.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(Module {
            ctx: ctx.clone(),
            module,
        })
    }

    pub fn function(&self, name: &str) -> Result<Function, Error> {
        let cname = CString::new(name).unwrap();
        let mut f = ptr::null_mut();
        check(
            unsafe { ffi::CuModuleGetFunction(&mut f, self.module, cname.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        Ok(Function(f))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(_g) = self.ctx.push() {
            unsafe { ffi::CuModuleUnload(self.module) };
        }
    }
}

impl Function {
    /// Launches the kernel with a 2D grid covering `nx` × `ny` threads.
    /// The context must be current.
    pub fn launch(
        &self,
        stream: &Stream,
        nx: u32,
        ny: u32,
        params: &KernelParams,
    ) -> Result<(), Error> {
        const BX: u32 = 64;
        const BY: u32 = 4;
        if nx == 0 || ny == 0 {
            return Ok(());
        }
        let mut p = *params;
        let mut args = [&mut p as *mut KernelParams as *mut c_void];
        check(
            unsafe {
                ffi::CuLaunchKernel(
                    self.0,
                    nx.div_ceil(BX),
                    ny.div_ceil(BY),
                    1,
                    BX,
                    BY,
                    1,
                    0,
                    stream.handle(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

/// A pitched 2D device allocation, used to stage system-memory frames.
pub struct DeviceBuffer {
    ctx: CudaContext,
    pub ptr: ffi::CUdeviceptr,
    pub pitch: usize,
    pub width_bytes: usize,
    pub height: usize,
}

unsafe impl Send for DeviceBuffer {}

impl DeviceBuffer {
    /// The context must be current.
    pub fn new(ctx: &CudaContext, width_bytes: usize, height: usize) -> Result<Self, Error> {
        let mut ptr = 0;
        let mut pitch = 0;
        check(
            unsafe {
                ffi::CuMemAllocPitch(&mut ptr, &mut pitch, width_bytes.max(1), height.max(1), 16)
            },
            "cuMemAllocPitch",
        )?;
        Ok(DeviceBuffer {
            ctx: ctx.clone(),
            ptr,
            pitch,
            width_bytes,
            height,
        })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Ok(_g) = self.ctx.push() {
            unsafe { ffi::CuMemFree(self.ptr) };
        }
    }
}

/// Copies a host plane to a device buffer, asynchronously on `stream`.
/// At most `src_pitch` bytes per row are copied.
pub fn upload(
    stream: &Stream,
    src: *const u8,
    src_pitch: usize,
    dst: &DeviceBuffer,
) -> Result<(), Error> {
    let c = ffi::CUDA_MEMCPY2D {
        srcMemoryType: ffi::CU_MEMORYTYPE_HOST,
        srcHost: src as *const c_void,
        srcPitch: src_pitch,
        dstMemoryType: ffi::CU_MEMORYTYPE_DEVICE,
        dstDevice: dst.ptr,
        dstPitch: dst.pitch,
        WidthInBytes: dst.width_bytes.min(src_pitch),
        Height: dst.height,
        ..memcpy2d_zeroed()
    };
    check(
        unsafe { ffi::CuMemcpy2DAsync(&c, stream.handle()) },
        "cuMemcpy2DAsync(HtoD)",
    )
}

/// Copies a device buffer to a host plane, asynchronously on `stream`.
/// At most `dst_pitch` bytes per row are copied.
pub fn download(
    stream: &Stream,
    src: &DeviceBuffer,
    dst: *mut u8,
    dst_pitch: usize,
) -> Result<(), Error> {
    let c = ffi::CUDA_MEMCPY2D {
        srcMemoryType: ffi::CU_MEMORYTYPE_DEVICE,
        srcDevice: src.ptr,
        srcPitch: src.pitch,
        dstMemoryType: ffi::CU_MEMORYTYPE_HOST,
        dstHost: dst as *mut c_void,
        dstPitch: dst_pitch,
        WidthInBytes: src.width_bytes.min(dst_pitch),
        Height: src.height,
        ..memcpy2d_zeroed()
    };
    check(
        unsafe { ffi::CuMemcpy2DAsync(&c, stream.handle()) },
        "cuMemcpy2DAsync(DtoH)",
    )
}

fn memcpy2d_zeroed() -> ffi::CUDA_MEMCPY2D {
    // All-zero is a valid (empty) CUDA_MEMCPY2D.
    unsafe { std::mem::zeroed() }
}

/// Creates a `GstCudaBufferPool` for `ctx`.
pub fn new_buffer_pool(ctx: &CudaContext) -> gst::BufferPool {
    unsafe { from_glib_full(ffi::gst_cuda_buffer_pool_new(ctx.to_glib_none().0)) }
}

pub fn pool_config_set_stream(config: &mut gst::BufferPoolConfigRef, stream: &Stream) {
    unsafe { ffi::gst_buffer_pool_config_set_cuda_stream(config.as_mut_ptr(), stream.as_ptr()) }
}

/// A video frame mapped with arbitrary map flags (e.g. `GST_MAP_CUDA`, which
/// yields device pointers as plane data).
pub struct MappedFrame {
    frame: gst_video::ffi::GstVideoFrame,
}

impl MappedFrame {
    pub fn map(
        info: &gst_video::VideoInfo,
        buffer: &gst::BufferRef,
        flags: std::os::raw::c_uint,
    ) -> Option<Self> {
        unsafe {
            let mut frame = std::mem::MaybeUninit::<gst_video::ffi::GstVideoFrame>::zeroed();
            let ok = gst_video::ffi::gst_video_frame_map(
                frame.as_mut_ptr(),
                info.to_glib_none().0 as *mut _,
                buffer.as_mut_ptr(),
                flags,
            );
            (ok != 0).then(|| MappedFrame {
                frame: frame.assume_init(),
            })
        }
    }

    pub fn plane(&self, i: usize) -> *mut u8 {
        self.frame.data[i] as *mut u8
    }

    pub fn stride(&self, i: usize) -> usize {
        self.frame.info.stride[i] as usize
    }
}

impl Drop for MappedFrame {
    fn drop(&mut self) {
        unsafe { gst_video::ffi::gst_video_frame_unmap(&mut self.frame) }
    }
}

/// If the buffer's first memory is CUDA memory, returns its context.
pub fn buffer_cuda_context(buffer: &gst::BufferRef) -> Option<CudaContext> {
    if buffer.n_memory() == 0 {
        return None;
    }
    let mem = buffer.peek_memory(0);
    unsafe {
        let raw = mem.as_mut_ptr();
        if ffi::gst_is_cuda_memory(raw) == 0 {
            return None;
        }
        Some(from_glib_none((*(raw as *mut ffi::GstCudaMemory)).context))
    }
}

/// Waits for pending work on every CUDA memory in `buffer`.
pub fn sync_buffer(buffer: &gst::BufferRef) {
    for mem in buffer.iter_memories() {
        unsafe {
            let raw = mem.as_mut_ptr();
            if ffi::gst_is_cuda_memory(raw) != 0 {
                ffi::gst_cuda_memory_sync(raw as *mut ffi::GstCudaMemory);
            }
        }
    }
}
