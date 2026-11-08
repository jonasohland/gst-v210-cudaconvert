use gst_video::VideoFormat;

/// Formats that can be converted to/from v210.
pub const OTHER_FORMATS: &[VideoFormat] = &[
    VideoFormat::I42210le,
    VideoFormat::Uyvy,
    VideoFormat::Yuy2,
    VideoFormat::I42010le,
    VideoFormat::P01010le,
];

pub fn is_420(f: VideoFormat) -> bool {
    matches!(f, VideoFormat::I42010le | VideoFormat::P01010le)
}

fn suffix(f: VideoFormat) -> Option<&'static str> {
    Some(match f {
        VideoFormat::I42210le => "i422_10",
        VideoFormat::Uyvy => "uyvy",
        VideoFormat::Yuy2 => "yuy2",
        VideoFormat::I42010le => "i420_10",
        VideoFormat::P01010le => "p010",
        _ => return None,
    })
}

/// Name of the kernel in kernels/v210.cu converting `f` to v210 (`pack`) or back.
pub fn kernel_name(f: VideoFormat, pack: bool) -> Option<String> {
    let s = suffix(f)?;
    Some(format!("v210_{}_{s}", if pack { "pack" } else { "unpack" }))
}

/// Bytes per row and number of rows of `plane` for staging copies.
pub fn plane_extent(info: &gst_video::VideoInfo, plane: usize) -> (usize, usize) {
    let finfo = info.format_info();
    let comp = (0..finfo.n_components() as usize)
        .find(|&c| finfo.plane()[c] as usize == plane)
        .unwrap_or(0);
    let h = finfo.scale_height(comp as u8, info.height()) as usize;
    (info.stride()[plane] as usize, h)
}
