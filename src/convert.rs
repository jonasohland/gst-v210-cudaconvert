//! Logic shared by `cudav210pack` and `cudav210unpack`. Each element's
//! `BaseTransformImpl` delegates to a [`Converter`].

use crate::cuda::{
    self, ffi, CudaContext, DeviceBuffer, Function, KernelParams, MappedFrame, Module, Stream,
};
use crate::format;
use gst::glib::translate::*;
use gst::prelude::*;
use gst_video::VideoFormat;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{LazyLock, Mutex};

pub static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "cudav210",
        gst::DebugColorFlags::empty(),
        Some("CUDA v210 converter"),
    )
});

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// other format -> v210
    Pack,
    /// v210 -> other format
    Unpack,
}

pub const DEFAULT_DEVICE_ID: i32 = -1;

/// Caps for one pad: the given formats in CUDA memory, then in system memory.
pub fn pad_caps(formats: &[VideoFormat]) -> gst::Caps {
    let mut caps = gst_video::VideoCapsBuilder::new()
        .features([ffi::CAPS_FEATURE_MEMORY_CUDA])
        .format_list(formats.iter().copied())
        .build();
    caps.merge(
        gst_video::VideoCapsBuilder::new()
            .format_list(formats.iter().copied())
            .build(),
    );
    caps
}

/// v210 is system memory only: gst-cuda's allocator has no v210 layout, so
/// v210 can't be carried in `GstCudaMemory` (and no GStreamer element would
/// accept it there). The element uploads/downloads it internally instead.
pub fn v210_caps() -> gst::Caps {
    gst_video::VideoCapsBuilder::new()
        .format(VideoFormat::V210)
        .build()
}

pub fn sink_caps(dir: Direction) -> gst::Caps {
    match dir {
        Direction::Pack => pad_caps(format::OTHER_FORMATS),
        Direction::Unpack => v210_caps(),
    }
}

pub fn src_caps(dir: Direction) -> gst::Caps {
    match dir {
        Direction::Pack => v210_caps(),
        Direction::Unpack => pad_caps(format::OTHER_FORMATS),
    }
}

fn is_cuda_caps(caps: &gst::CapsRef) -> bool {
    caps.features(0)
        .is_some_and(|f| f.contains(ffi::CAPS_FEATURE_MEMORY_CUDA))
}

/// Owned reference to a `GstCudaContext` as managed by the gst-cuda helpers,
/// which want a `GstCudaContext **` to fill in. Not behind a mutex: the helpers
/// may re-enter the element (need-context message -> set_context) on the same
/// thread, exactly like the upstream C elements' plain struct field.
struct RawContext(AtomicPtr<ffi::GstCudaContext>);

impl RawContext {
    fn slot(&self) -> *mut *mut ffi::GstCudaContext {
        self.0.as_ptr()
    }

    fn raw(&self) -> *mut ffi::GstCudaContext {
        self.0.load(Ordering::Acquire)
    }

    fn get(&self) -> Option<CudaContext> {
        let p = self.raw();
        (!p.is_null()).then(|| unsafe { from_glib_none(p) })
    }

    fn clear(&self) {
        let p = self.0.swap(ptr::null_mut(), Ordering::AcqRel);
        if !p.is_null() {
            unsafe { gst::ffi::gst_object_unref(p as *mut _) };
        }
    }
}

/// Per-context resources, created in `start`.
struct Resources {
    ctx: CudaContext,
    stream: Stream,
    module: Module,
}

/// Per-caps state, created in `set_caps`.
struct State {
    in_info: gst_video::VideoInfo,
    out_info: gst_video::VideoInfo,
    other: VideoFormat,
    kernel: Function,
    staging_in: Vec<DeviceBuffer>,
    staging_out: Vec<DeviceBuffer>,
}

pub struct Converter {
    dir: Direction,
    device_id: Mutex<i32>,
    ctx: RawContext,
    res: Mutex<Option<Resources>>,
    state: Mutex<Option<State>>,
}

fn err(e: cuda::Error) -> gst::FlowError {
    gst::error!(CAT, "{e}");
    gst::FlowError::Error
}

impl Converter {
    pub fn new(dir: Direction) -> Self {
        Converter {
            dir,
            device_id: Mutex::new(DEFAULT_DEVICE_ID),
            ctx: RawContext(AtomicPtr::new(ptr::null_mut())),
            res: Mutex::new(None),
            state: Mutex::new(None),
        }
    }

    pub fn device_id(&self) -> i32 {
        *self.device_id.lock().unwrap()
    }

    pub fn set_device_id(&self, id: i32) {
        *self.device_id.lock().unwrap() = id;
    }

    fn element_ptr(elem: &gst_base::BaseTransform) -> *mut gst::ffi::GstElement {
        elem.upcast_ref::<gst::Element>().to_glib_none().0
    }

    pub fn set_context(&self, elem: &gst_base::BaseTransform, context: &gst::Context) {
        unsafe {
            ffi::gst_cuda_handle_set_context(
                Self::element_ptr(elem),
                context.to_glib_none().0,
                self.device_id(),
                self.ctx.slot(),
            );
        }
    }

    /// Answers CONTEXT queries with our CUDA context. Returns true if handled.
    pub fn handle_context_query(
        &self,
        elem: &gst_base::BaseTransform,
        query: &mut gst::QueryRef,
    ) -> bool {
        if !matches!(query.view(), gst::QueryView::Context(_)) {
            return false;
        }
        unsafe {
            ffi::gst_cuda_handle_context_query(
                Self::element_ptr(elem),
                query.as_mut_ptr(),
                self.ctx.raw(),
            ) != 0
        }
    }

    pub fn start(&self, elem: &gst_base::BaseTransform) -> Result<(), gst::ErrorMessage> {
        if !cuda::load_library() {
            return Err(gst::error_msg!(
                gst::ResourceError::Failed,
                ["Could not load the CUDA driver library"]
            ));
        }
        let ctx = {
            let ok = unsafe {
                ffi::gst_cuda_ensure_element_context(
                    Self::element_ptr(elem),
                    self.device_id(),
                    self.ctx.slot(),
                )
            };
            match self.ctx.get() {
                Some(c) if ok != 0 => c,
                _ => {
                    return Err(gst::error_msg!(
                        gst::ResourceError::Failed,
                        [
                            "Could not get a CUDA context for device {}",
                            self.device_id()
                        ]
                    ))
                }
            }
        };
        let make = || -> Result<Resources, cuda::Error> {
            Ok(Resources {
                stream: Stream::new(&ctx)?,
                module: Module::load(&ctx)?,
                ctx: ctx.clone(),
            })
        };
        let res = make()
            .map_err(|e| gst::error_msg!(gst::ResourceError::Failed, ["CUDA setup failed: {e}"]))?;
        *self.res.lock().unwrap() = Some(res);
        Ok(())
    }

    pub fn stop(&self) {
        // Free device buffers and the module before releasing the context.
        *self.state.lock().unwrap() = None;
        *self.res.lock().unwrap() = None;
        self.ctx.clear();
    }

    /// Formats on the opposite pad of a pad with direction `pad_dir`.
    fn target_formats(&self, pad_dir: gst::PadDirection) -> &'static [VideoFormat] {
        let other_is_v210 = matches!(
            (self.dir, pad_dir),
            (Direction::Pack, gst::PadDirection::Sink)
                | (Direction::Unpack, gst::PadDirection::Src)
        );
        if other_is_v210 {
            &[VideoFormat::V210]
        } else {
            format::OTHER_FORMATS
        }
    }

    pub fn transform_caps(
        &self,
        pad_dir: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> gst::Caps {
        let targets = self.target_formats(pad_dir);
        let cuda = gst::CapsFeatures::new([ffi::CAPS_FEATURE_MEMORY_CUDA]);
        let sys = gst::CapsFeatures::new_empty();

        // v210 side: system memory. Other side: CUDA memory preferred, then
        // system memory.
        let mut first = gst::Caps::new_empty();
        let mut second = gst::Caps::new_empty();
        for s in caps.iter() {
            let interlaced = s
                .get::<&str>("interlace-mode")
                .is_ok_and(|m| m != "progressive");
            let formats: Vec<VideoFormat> = targets
                .iter()
                .copied()
                .filter(|&t| !(interlaced && format::is_420(t)))
                .collect();
            if formats.is_empty() {
                continue;
            }
            let mut s = s.to_owned();
            s.remove_field("chroma-site");
            if formats.len() == 1 {
                s.set("format", formats[0].to_str());
            } else {
                s.set("format", gst::List::new(formats.iter().map(|f| f.to_str())));
            }
            if targets == [VideoFormat::V210] {
                first
                    .get_mut()
                    .unwrap()
                    .append_structure_full(s, Some(sys.clone()));
                continue;
            }
            first
                .get_mut()
                .unwrap()
                .append_structure_full(s.clone(), Some(cuda.clone()));
            second
                .get_mut()
                .unwrap()
                .append_structure_full(s, Some(sys.clone()));
        }
        first.merge(second);

        match filter {
            Some(f) => f.intersect_with_mode(&first, gst::CapsIntersectMode::First),
            None => first,
        }
    }

    pub fn set_caps(
        &self,
        incaps: &gst::Caps,
        outcaps: &gst::Caps,
    ) -> Result<(), gst::LoggableError> {
        let in_info = gst_video::VideoInfo::from_caps(incaps)
            .map_err(|_| gst::loggable_error!(CAT, "invalid input caps {incaps}"))?;
        let out_info = gst_video::VideoInfo::from_caps(outcaps)
            .map_err(|_| gst::loggable_error!(CAT, "invalid output caps {outcaps}"))?;
        if in_info.width() != out_info.width() || in_info.height() != out_info.height() {
            return Err(gst::loggable_error!(CAT, "resolution must not change"));
        }
        let (v210, other) = match self.dir {
            Direction::Pack => (&out_info, &in_info),
            Direction::Unpack => (&in_info, &out_info),
        };
        if v210.format() != VideoFormat::V210 {
            return Err(gst::loggable_error!(
                CAT,
                "expected v210, got {:?}",
                v210.format()
            ));
        }
        if format::is_420(other.format())
            && other.interlace_mode() != gst_video::VideoInterlaceMode::Progressive
        {
            return Err(gst::loggable_error!(
                CAT,
                "4:2:0 formats must be progressive"
            ));
        }
        let name = format::kernel_name(other.format(), self.dir == Direction::Pack)
            .ok_or_else(|| gst::loggable_error!(CAT, "unsupported format {:?}", other.format()))?;

        let res = self.res.lock().unwrap();
        let res = res
            .as_ref()
            .ok_or_else(|| gst::loggable_error!(CAT, "set_caps before start"))?;
        let kernel = res
            .module
            .function(&name)
            .map_err(|e| gst::loggable_error!(CAT, "{e}"))?;

        gst::debug!(CAT, "configured {name}: {incaps} -> {outcaps}");
        *self.state.lock().unwrap() = Some(State {
            other: other.format(),
            in_info,
            out_info,
            kernel,
            staging_in: Vec::new(),
            staging_out: Vec::new(),
        });
        Ok(())
    }

    pub fn unit_size(&self, caps: &gst::Caps) -> Option<usize> {
        gst_video::VideoInfo::from_caps(caps).ok().map(|i| i.size())
    }

    fn make_cuda_pool(
        res: &Resources,
        caps: &gst::Caps,
        min: u32,
        max: u32,
    ) -> Option<(gst::BufferPool, u32)> {
        let info = gst_video::VideoInfo::from_caps(caps).ok()?;
        let pool = cuda::new_buffer_pool(&res.ctx);
        let mut config = pool.config();
        config.add_option(gst_video::BUFFER_POOL_OPTION_VIDEO_META);
        config.set_params(Some(caps), info.size() as u32, min, max);
        cuda::pool_config_set_stream(&mut config, &res.stream);
        pool.set_config(config).ok()?;
        // The pool may have adjusted the size for its alignment.
        let size = pool.config().params().map(|(_, s, _, _)| s)?;
        Some((pool, size))
    }

    pub fn propose_allocation(&self, query: &mut gst::query::Allocation) -> bool {
        let (caps, need_pool) = query.get_owned();
        let Some(caps) = caps else { return false };
        query.add_allocation_meta::<gst_video::VideoMeta>(None);
        if !need_pool || !is_cuda_caps(&caps) {
            return true;
        }
        let res = self.res.lock().unwrap();
        if let Some((pool, size)) = res
            .as_ref()
            .and_then(|r| Self::make_cuda_pool(r, &caps, 0, 0))
        {
            query.add_allocation_pool(Some(&pool), size, 0, 0);
        }
        true
    }

    /// Replaces the downstream pool with our own CUDA pool when outputting CUDA memory.
    pub fn decide_allocation(&self, query: &mut gst::query::Allocation) -> bool {
        let (caps, _) = query.get_owned();
        let Some(caps) = caps else { return false };
        if !is_cuda_caps(&caps) {
            return true;
        }
        let (min, max) = query
            .allocation_pools()
            .next()
            .map(|(_, _, min, max)| (min, max))
            .unwrap_or((0, 0));
        let res = self.res.lock().unwrap();
        let Some((pool, size)) = res
            .as_ref()
            .and_then(|r| Self::make_cuda_pool(r, &caps, min, max))
        else {
            gst::error!(CAT, "failed to create CUDA buffer pool");
            return false;
        };
        if query.allocation_pools().len() == 0 {
            query.add_allocation_pool(Some(&pool), size, min, max);
        } else {
            query.set_nth_allocation_pool(0, Some(&pool), size, min, max);
        }
        true
    }

    pub fn transform(
        &self,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let res_guard = self.res.lock().unwrap();
        let res = res_guard.as_ref().ok_or(gst::FlowError::NotNegotiated)?;
        let mut state_guard = self.state.lock().unwrap();
        let state = state_guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;

        let _ctx = res.ctx.push().map_err(err)?;

        // Input: map as device memory if possible, otherwise upload via staging.
        let in_ctx = cuda::buffer_cuda_context(inbuf);
        gst::trace!(
            CAT,
            "input memory context {:?}, ours {:?}",
            in_ctx,
            res.ctx.as_ptr()
        );
        let in_dev = in_ctx.as_ref().is_some_and(|c| res.ctx.can_access(c));
        let in_map = if in_dev {
            cuda::sync_buffer(inbuf);
            MappedFrame::map(
                &state.in_info,
                inbuf,
                gst::ffi::GST_MAP_READ | ffi::GST_MAP_CUDA,
            )
        } else {
            MappedFrame::map(&state.in_info, inbuf, gst::ffi::GST_MAP_READ)
        }
        .ok_or_else(|| {
            gst::error!(CAT, "failed to map input buffer");
            gst::FlowError::Error
        })?;

        let n_in = state.in_info.n_planes() as usize;
        let mut in_planes = [(0u64, 0i64); 3];
        if in_dev {
            for (i, p) in in_planes.iter_mut().enumerate().take(n_in) {
                *p = (in_map.plane(i) as u64, in_map.stride(i) as i64);
            }
        } else {
            ensure_staging(&res.ctx, &mut state.staging_in, &state.in_info).map_err(err)?;
            for (i, p) in in_planes.iter_mut().enumerate().take(n_in) {
                let dev = &state.staging_in[i];
                cuda::upload(&res.stream, in_map.plane(i), in_map.stride(i), dev).map_err(err)?;
                *p = (dev.ptr, dev.pitch as i64);
            }
        }

        // Output: same, the other way around.
        let out_dev = cuda::buffer_cuda_context(outbuf).is_some_and(|c| res.ctx.can_access(&c));
        gst::log!(
            CAT,
            "transform: input {}, output {}",
            if in_dev { "device" } else { "staged" },
            if out_dev { "device" } else { "staged" }
        );
        let n_out = state.out_info.n_planes() as usize;
        let mut out_planes = [(0u64, 0i64); 3];
        let out_map = if out_dev {
            let m = MappedFrame::map(
                &state.out_info,
                outbuf,
                gst::ffi::GST_MAP_WRITE | ffi::GST_MAP_CUDA,
            )
            .ok_or_else(|| {
                gst::error!(CAT, "failed to map output buffer");
                gst::FlowError::Error
            })?;
            for (i, p) in out_planes.iter_mut().enumerate().take(n_out) {
                *p = (m.plane(i) as u64, m.stride(i) as i64);
            }
            Some(m)
        } else {
            ensure_staging(&res.ctx, &mut state.staging_out, &state.out_info).map_err(err)?;
            for (i, p) in out_planes.iter_mut().enumerate().take(n_out) {
                let dev = &state.staging_out[i];
                *p = (dev.ptr, dev.pitch as i64);
            }
            None
        };

        let (v210, other, n_other) = match self.dir {
            Direction::Pack => (out_planes[0], in_planes, n_in),
            Direction::Unpack => (in_planes[0], out_planes, n_out),
        };
        let mut params = KernelParams {
            v210: v210.0,
            v210_stride: v210.1,
            width: state.in_info.width() as i32,
            height: state.in_info.height() as i32,
            ..Default::default()
        };
        for (i, (ptr, stride)) in other.iter().take(n_other).enumerate() {
            params.plane[i] = *ptr;
            params.stride[i] = *stride;
        }

        let w = state.in_info.width();
        let h = state.in_info.height();
        let (nx, ny) = match self.dir {
            // Every 16-byte group of the stride, so padding is zeroed.
            Direction::Pack => ((v210.1 / 16) as u32, h),
            Direction::Unpack if format::is_420(state.other) => (w.div_ceil(6), h.div_ceil(2)),
            Direction::Unpack => (w.div_ceil(6), h),
        };
        state
            .kernel
            .launch(&res.stream, nx, ny, &params)
            .map_err(err)?;

        if out_map.is_none() {
            let m = MappedFrame::map(&state.out_info, outbuf, gst::ffi::GST_MAP_WRITE).ok_or_else(
                || {
                    gst::error!(CAT, "failed to map output buffer");
                    gst::FlowError::Error
                },
            )?;
            for i in 0..n_out {
                cuda::download(&res.stream, &state.staging_out[i], m.plane(i), m.stride(i))
                    .map_err(err)?;
            }
            res.stream.synchronize().map_err(err)?;
            drop(m);
        } else {
            res.stream.synchronize().map_err(err)?;
        }
        drop(out_map);
        drop(in_map);
        Ok(gst::FlowSuccess::Ok)
    }
}

/// (Re)allocates one device buffer per plane of `info` if needed.
fn ensure_staging(
    ctx: &CudaContext,
    bufs: &mut Vec<DeviceBuffer>,
    info: &gst_video::VideoInfo,
) -> Result<(), cuda::Error> {
    let n = info.n_planes() as usize;
    let fits = bufs.len() == n
        && (0..n).all(|i| {
            let (w, h) = format::plane_extent(info, i);
            bufs[i].width_bytes == w && bufs[i].height == h
        });
    if fits {
        return Ok(());
    }
    bufs.clear();
    for i in 0..n {
        let (w, h) = format::plane_extent(info, i);
        bufs.push(DeviceBuffer::new(ctx, w, h)?);
    }
    Ok(())
}

/// Implements an element type around a [`Converter`]. Used by the pack and
/// unpack modules, which only differ in direction and metadata.
#[macro_export]
macro_rules! cuda_v210_element {
    ($imp:ident, $type_name:literal, $dir:expr, $long_name:literal, $description:literal) => {
        use gst::glib;
        use gst::prelude::*;
        use gst::subclass::prelude::*;
        use gst_base::subclass::prelude::*;
        use $crate::convert::{Converter, CAT};

        pub struct $imp {
            conv: Converter,
        }

        impl Default for $imp {
            fn default() -> Self {
                Self {
                    conv: Converter::new($dir),
                }
            }
        }

        #[glib::object_subclass]
        impl ObjectSubclass for $imp {
            const NAME: &'static str = $type_name;
            type Type = super::Element;
            type ParentType = gst_base::BaseTransform;
        }

        impl ObjectImpl for $imp {
            fn properties() -> &'static [glib::ParamSpec] {
                static PROPS: std::sync::LazyLock<Vec<glib::ParamSpec>> =
                    std::sync::LazyLock::new(|| {
                        vec![glib::ParamSpecInt::builder("device-id")
                            .nick("Device ID")
                            .blurb("CUDA device ID to use (-1 = auto)")
                            .minimum(-1)
                            .default_value($crate::convert::DEFAULT_DEVICE_ID)
                            .mutable_ready()
                            .build()]
                    });
                PROPS.as_ref()
            }

            fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
                if pspec.name() == "device-id" {
                    self.conv.set_device_id(value.get().unwrap());
                }
            }

            fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
                match pspec.name() {
                    "device-id" => self.conv.device_id().to_value(),
                    _ => unreachable!(),
                }
            }
        }

        impl GstObjectImpl for $imp {}

        impl ElementImpl for $imp {
            fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
                static META: std::sync::LazyLock<gst::subclass::ElementMetadata> =
                    std::sync::LazyLock::new(|| {
                        gst::subclass::ElementMetadata::new(
                            $long_name,
                            "Filter/Converter/Video/Hardware",
                            $description,
                            "gst-cudav210 contributors",
                        )
                    });
                Some(&*META)
            }

            fn pad_templates() -> &'static [gst::PadTemplate] {
                static TEMPLATES: std::sync::LazyLock<Vec<gst::PadTemplate>> =
                    std::sync::LazyLock::new(|| {
                        vec![
                            gst::PadTemplate::new(
                                "sink",
                                gst::PadDirection::Sink,
                                gst::PadPresence::Always,
                                &$crate::convert::sink_caps($dir),
                            )
                            .unwrap(),
                            gst::PadTemplate::new(
                                "src",
                                gst::PadDirection::Src,
                                gst::PadPresence::Always,
                                &$crate::convert::src_caps($dir),
                            )
                            .unwrap(),
                        ]
                    });
                TEMPLATES.as_ref()
            }

            fn set_context(&self, context: &gst::Context) {
                self.conv.set_context(self.obj().upcast_ref(), context);
                self.parent_set_context(context)
            }
        }

        impl BaseTransformImpl for $imp {
            const MODE: gst_base::subclass::BaseTransformMode =
                gst_base::subclass::BaseTransformMode::NeverInPlace;
            const PASSTHROUGH_ON_SAME_CAPS: bool = false;
            const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

            fn start(&self) -> Result<(), gst::ErrorMessage> {
                self.conv.start(self.obj().upcast_ref())
            }

            fn stop(&self) -> Result<(), gst::ErrorMessage> {
                self.conv.stop();
                Ok(())
            }

            fn query(&self, direction: gst::PadDirection, query: &mut gst::QueryRef) -> bool {
                if self
                    .conv
                    .handle_context_query(self.obj().upcast_ref(), query)
                {
                    return true;
                }
                BaseTransformImplExt::parent_query(self, direction, query)
            }

            fn transform_caps(
                &self,
                direction: gst::PadDirection,
                caps: &gst::Caps,
                filter: Option<&gst::Caps>,
            ) -> Option<gst::Caps> {
                let out = self.conv.transform_caps(direction, caps, filter);
                gst::trace!(
                    CAT,
                    imp = self,
                    "transform_caps {direction:?}: {caps} -> {out}"
                );
                Some(out)
            }

            fn set_caps(
                &self,
                incaps: &gst::Caps,
                outcaps: &gst::Caps,
            ) -> Result<(), gst::LoggableError> {
                self.conv.set_caps(incaps, outcaps)
            }

            fn unit_size(&self, caps: &gst::Caps) -> Option<usize> {
                self.conv.unit_size(caps)
            }

            // One frame in is one frame out; input size varies with padding
            // (e.g. pitched CUDA memory), so don't derive it from unit sizes.
            fn transform_size(
                &self,
                _direction: gst::PadDirection,
                _caps: &gst::Caps,
                _size: usize,
                othercaps: &gst::Caps,
            ) -> Option<usize> {
                self.conv.unit_size(othercaps)
            }

            fn propose_allocation(
                &self,
                decide_query: Option<&gst::query::Allocation>,
                query: &mut gst::query::Allocation,
            ) -> Result<(), gst::LoggableError> {
                self.parent_propose_allocation(decide_query, query)?;
                if self.conv.propose_allocation(query) {
                    Ok(())
                } else {
                    Err(gst::loggable_error!(CAT, "propose_allocation failed"))
                }
            }

            fn decide_allocation(
                &self,
                query: &mut gst::query::Allocation,
            ) -> Result<(), gst::LoggableError> {
                if !self.conv.decide_allocation(query) {
                    return Err(gst::loggable_error!(CAT, "decide_allocation failed"));
                }
                self.parent_decide_allocation(query)
            }

            fn transform(
                &self,
                inbuf: &gst::Buffer,
                outbuf: &mut gst::BufferRef,
            ) -> Result<gst::FlowSuccess, gst::FlowError> {
                self.conv.transform(inbuf, outbuf)
            }
        }
    };
}
