//! Integration tests. They need an NVIDIA GPU and the nvcodec plugin
//! (`cudaupload`/`cudadownload`); run them from `nix develop`.

mod reference;

use gst::prelude::*;
use gst_video::{VideoFormat, VideoFrameExt, VideoFrameRef};
use reference::*;
use std::sync::Once;

const SIZES: &[(usize, usize)] = &[
    (1, 1),
    (7, 5),
    (48, 4),
    (1918, 1081),
    (1920, 1080),
    (1922, 3),
];

/// cudaupload/cudadownload (gst_cuda_buffer_copy) copy only width*2 bytes per
/// row for packed 4:2:2, losing the last macropixel's second half at odd
/// widths. That's an upstream bug, so avoid odd widths there.
fn sizes_through_cudaupload(f: VideoFormat) -> Vec<(usize, usize)> {
    SIZES
        .iter()
        .copied()
        .filter(|&(w, _)| !(is_8bit(f) && w % 2 == 1))
        .collect()
}

fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Use the statically registered plugin, not a copy found on GST_PLUGIN_PATH.
        std::env::remove_var("GST_PLUGIN_PATH");
        gst::init().unwrap();
        gstcudav210::plugin_register_static().unwrap();
    });
}

fn info(f: VideoFormat, w: usize, h: usize) -> gst_video::VideoInfo {
    init();
    gst_video::VideoInfo::builder(f, w as u32, h as u32)
        .fps(gst::Fraction::new(30, 1))
        .build()
        .unwrap()
}

fn make_buffer(
    info: &gst_video::VideoInfo,
    fill: impl FnOnce(&mut VideoFrameRef<&mut gst::BufferRef>),
) -> gst::Buffer {
    let mut buf = gst::Buffer::with_size(info.size()).unwrap();
    {
        let b = buf.get_mut().unwrap();
        // Garbage in all padding, so tests catch reads/writes of it.
        b.map_writable().unwrap().fill(0xa5);
        let mut frame = VideoFrameRef::from_buffer_ref_writable(b, info).unwrap();
        fill(&mut frame);
    }
    buf
}

/// Runs one buffer through `appsrc ! {mid} ! appsink` and returns the output.
fn run(
    mid: &str,
    in_info: &gst_video::VideoInfo,
    buf: gst::Buffer,
) -> (gst::Buffer, gst_video::VideoInfo) {
    init();
    let desc = format!("appsrc name=src format=time ! {mid} ! appsink name=sink sync=false");
    let pipeline = gst::parse::launch(&desc)
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
    let src = pipeline
        .by_name("src")
        .unwrap()
        .downcast::<gst_app::AppSrc>()
        .unwrap();
    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<gst_app::AppSink>()
        .unwrap();
    src.set_caps(Some(&in_info.to_caps().unwrap()));
    pipeline.set_state(gst::State::Playing).unwrap();
    src.push_buffer(buf).unwrap();
    src.end_of_stream().unwrap();

    let sample = sink.pull_sample();
    if sample.is_err() {
        let bus = pipeline.bus().unwrap();
        while let Some(msg) = bus.pop_filtered(&[gst::MessageType::Error]) {
            if let gst::MessageView::Error(e) = msg.view() {
                eprintln!("pipeline error: {} ({:?})", e.error(), e.debug());
            }
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();
    let sample = sample.unwrap_or_else(|_| panic!("no output from `{desc}`"));
    let out_info = gst_video::VideoInfo::from_caps(sample.caps().unwrap()).unwrap();
    (sample.buffer_owned().unwrap(), out_info)
}

fn out_frame<'a>(
    buf: &'a gst::Buffer,
    info: &gst_video::VideoInfo,
) -> VideoFrameRef<&'a gst::BufferRef> {
    VideoFrameRef::from_buffer_ref_readable(buf.as_ref(), info).unwrap()
}

fn v210_bytes(frame: &VideoFrameRef<&gst::BufferRef>) -> Vec<u8> {
    // Full rows of the (default) v210 stride, which the packer zero-pads.
    let row = (frame.width() as usize).div_ceil(48) * 128;
    let stride = frame.plane_stride()[0] as usize;
    let data = frame.plane_data(0).unwrap();
    (0..frame.height() as usize)
        .flat_map(|y| data[y * stride..y * stride + row].iter().copied())
        .collect()
}

fn check_pack(f: VideoFormat, mid: &str, sizes: &[(usize, usize)]) {
    for (i, &(w, h)) in sizes.iter().enumerate() {
        let src = random_native(f, w, h, i as u64);
        let in_info = info(f, w, h);
        let buf = make_buffer(&in_info, |fr| store(&src, fr));
        let (out, out_info) = run(mid, &in_info, buf);
        assert_eq!(out_info.format(), VideoFormat::V210);

        let v210_info = info(VideoFormat::V210, w, h);
        let expected = make_buffer(&v210_info, |fr| v210_encode(&to_422_10(&src, f), fr));
        let got = v210_bytes(&out_frame(&out, &out_info));
        let want = v210_bytes(&out_frame(&expected, &v210_info));
        assert!(
            got == want,
            "{f:?} {w}x{h} via `{mid}`: v210 output differs from reference"
        );
    }
}

fn check_unpack(f: VideoFormat, mid: &str, sizes: &[(usize, usize)]) {
    for (i, &(w, h)) in sizes.iter().enumerate() {
        let p = random_native(VideoFormat::I42210le, w, h, 100 + i as u64);
        let in_info = info(VideoFormat::V210, w, h);
        let buf = make_buffer(&in_info, |fr| v210_encode(&p, fr));
        let (out, out_info) = run(mid, &in_info, buf);
        assert_eq!(out_info.format(), f);
        let got = load(&out_frame(&out, &out_info));
        assert!(
            got == from_422_10(&p, f),
            "{f:?} {w}x{h} via `{mid}`: output differs from reference"
        );
    }
}

#[test]
fn pack_sysmem() {
    for &f in FORMATS {
        check_pack(f, "cudav210pack ! video/x-raw,format=v210", SIZES);
    }
}

#[test]
fn pack_cuda_in() {
    for &f in FORMATS {
        check_pack(
            f,
            "cudaupload ! video/x-raw(memory:CUDAMemory) ! cudav210pack ! video/x-raw,format=v210",
            &sizes_through_cudaupload(f),
        );
    }
}

#[test]
fn unpack_sysmem() {
    for &f in FORMATS {
        check_unpack(
            f,
            &format!("cudav210unpack ! video/x-raw,format={}", f.to_str()),
            SIZES,
        );
    }
}

#[test]
fn unpack_cuda_out() {
    for &f in FORMATS {
        check_unpack(
            f,
            &format!(
                "cudav210unpack ! video/x-raw(memory:CUDAMemory),format={} ! cudadownload ! video/x-raw",
                f.to_str()
            ),
            &sizes_through_cudaupload(f),
        );
    }
}

/// CUDA memory in, v210 (system memory) between the elements, CUDA memory out.
#[test]
fn cuda_chain() {
    for &f in FORMATS {
        for (i, &(w, h)) in sizes_through_cudaupload(f).iter().enumerate() {
            let src = random_native(f, w, h, 200 + i as u64);
            let in_info = info(f, w, h);
            let buf = make_buffer(&in_info, |fr| store(&src, fr));
            let mid = format!(
                "cudaupload ! video/x-raw(memory:CUDAMemory) ! cudav210pack ! cudav210unpack ! \
                 video/x-raw(memory:CUDAMemory),format={0} ! cudadownload ! video/x-raw,format={0}",
                f.to_str()
            );
            let (out, out_info) = run(&mid, &in_info, buf);
            let got = load(&out_frame(&out, &out_info));
            assert!(
                got == from_422_10(&to_422_10(&src, f), f),
                "{f:?} {w}x{h} via `{mid}`"
            );
        }
    }
}

#[test]
fn lossless_422_roundtrip() {
    for &(w, h) in SIZES {
        let src = random_native(VideoFormat::I42210le, w, h, 7);
        let in_info = info(VideoFormat::I42210le, w, h);
        let buf = make_buffer(&in_info, |fr| store(&src, fr));
        let (out, out_info) = run(
            "cudav210pack ! cudav210unpack ! video/x-raw,format=I422_10LE",
            &in_info,
            buf,
        );
        assert_eq!(load(&out_frame(&out, &out_info)), src, "{w}x{h}");
    }
}

/// Cross-check against GStreamer's CPU v210 support in videoconvert.
#[test]
fn matches_videoconvert() {
    for &(w, h) in SIZES {
        let src = random_native(VideoFormat::I42210le, w, h, 9);
        let in_info = info(VideoFormat::I42210le, w, h);
        let buf = make_buffer(&in_info, |fr| store(&src, fr));
        let (ours, oi) = run(
            "cudav210pack ! video/x-raw,format=v210",
            &in_info,
            buf.clone(),
        );
        let (theirs, ti) = run("videoconvert ! video/x-raw,format=v210", &in_info, buf);
        assert_eq!(
            v210_decode(&out_frame(&ours, &oi)),
            v210_decode(&out_frame(&theirs, &ti)),
            "pack {w}x{h}"
        );

        let v_info = info(VideoFormat::V210, w, h);
        let vbuf = make_buffer(&v_info, |fr| v210_encode(&src, fr));
        let (ours, oi) = run(
            "cudav210unpack ! video/x-raw,format=I422_10LE",
            &v_info,
            vbuf.clone(),
        );
        let (theirs, ti) = run("videoconvert ! video/x-raw,format=I422_10LE", &v_info, vbuf);
        assert_eq!(
            load(&out_frame(&ours, &oi)),
            load(&out_frame(&theirs, &ti)),
            "unpack {w}x{h}"
        );
    }
}

/// 4:2:0 output is only offered for progressive input.
#[test]
fn interlaced_420_not_offered() {
    init();
    let unpack = gst::ElementFactory::make("cudav210unpack").build().unwrap();
    let mk = |mode: &str| {
        gst::Caps::builder("video/x-raw")
            .field("format", "v210")
            .field("width", 1920)
            .field("height", 1080)
            .field("interlace-mode", mode)
            .build()
    };
    let formats = |caps: gst::Caps| -> Vec<String> {
        let c = transform(&unpack, &caps);
        c.iter()
            .flat_map(|s| match s.get::<gst::List>("format") {
                Ok(l) => l
                    .iter()
                    .map(|v| v.get::<String>().unwrap())
                    .collect::<Vec<_>>(),
                Err(_) => vec![s.get::<String>("format").unwrap()],
            })
            .collect()
    };
    let progressive = formats(mk("progressive"));
    assert!(progressive.iter().any(|f| f == "I420_10LE"));
    let interlaced = formats(mk("interleaved"));
    assert!(interlaced.iter().any(|f| f == "I422_10LE"));
    assert!(!interlaced
        .iter()
        .any(|f| f == "I420_10LE" || f == "P010_10LE"));
}

/// Calls the element's transform_caps (sink -> src) via a caps query on its src pad
/// while the sink pad is linked to a pad returning `caps`.
fn transform(elem: &gst::Element, caps: &gst::Caps) -> gst::Caps {
    let upstream = gst::Pad::builder(gst::PadDirection::Src)
        .query_function({
            let caps = caps.clone();
            move |_, _, q| match q.view_mut() {
                gst::QueryViewMut::Caps(q) => {
                    q.set_result(&caps);
                    true
                }
                _ => false,
            }
        })
        .build();
    upstream.link(&elem.static_pad("sink").unwrap()).unwrap();
    let out = elem.static_pad("src").unwrap().query_caps(None);
    upstream.unlink(&elem.static_pad("sink").unwrap()).unwrap();
    out
}
