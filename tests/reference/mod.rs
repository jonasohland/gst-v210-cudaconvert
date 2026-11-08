//! CPU reference implementation of the conversions, written independently of the
//! kernels and kept simple. It is the ground truth for the element tests.

#![allow(clippy::needless_range_loop)]

use gst_video::{VideoFormat, VideoFrameExt, VideoFrameRef};

pub const FORMATS: &[VideoFormat] = &[
    VideoFormat::I42210le,
    VideoFormat::Uyvy,
    VideoFormat::Yuy2,
    VideoFormat::I42010le,
    VideoFormat::P01010le,
];

pub fn is_420(f: VideoFormat) -> bool {
    matches!(f, VideoFormat::I42010le | VideoFormat::P01010le)
}

pub fn is_8bit(f: VideoFormat) -> bool {
    matches!(f, VideoFormat::Uyvy | VideoFormat::Yuy2)
}

/// Samples of a frame in its own precision (8 or 10 bit) and chroma resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Native {
    pub w: usize,
    pub h: usize,
    /// chroma width/height
    pub cw: usize,
    pub ch: usize,
    pub y: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
}

impl Native {
    pub fn new(f: VideoFormat, w: usize, h: usize) -> Self {
        let cw = w.div_ceil(2);
        let ch = if is_420(f) { h.div_ceil(2) } else { h };
        Native {
            w,
            h,
            cw,
            ch,
            y: vec![0; w * h],
            cb: vec![0; cw * ch],
            cr: vec![0; cw * ch],
        }
    }
}

fn rd16(row: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([row[2 * i], row[2 * i + 1]])
}

fn wr16(row: &mut [u8], i: usize, v: u16) {
    row[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
}

fn row<'a>(f: &'a VideoFrameRef<&gst::BufferRef>, plane: u32, y: usize) -> &'a [u8] {
    let stride = f.plane_stride()[plane as usize] as usize;
    &f.plane_data(plane).unwrap()[y * stride..(y + 1) * stride]
}

fn row_mut<'a>(
    f: &'a mut VideoFrameRef<&mut gst::BufferRef>,
    plane: u32,
    y: usize,
) -> &'a mut [u8] {
    let stride = f.plane_stride()[plane as usize] as usize;
    &mut f.plane_data_mut(plane).unwrap()[y * stride..(y + 1) * stride]
}

/// Packed 8-bit byte offsets of (Y0, U, Y1, V).
fn packed_offsets(f: VideoFormat) -> [usize; 4] {
    match f {
        VideoFormat::Uyvy => [1, 0, 3, 2],
        VideoFormat::Yuy2 => [0, 1, 2, 3],
        _ => unreachable!(),
    }
}

pub fn load(frame: &VideoFrameRef<&gst::BufferRef>) -> Native {
    let f = frame.format();
    let mut n = Native::new(f, frame.width() as usize, frame.height() as usize);
    match f {
        VideoFormat::I42210le | VideoFormat::I42010le => {
            for y in 0..n.h {
                let r = row(frame, 0, y);
                for x in 0..n.w {
                    n.y[y * n.w + x] = rd16(r, x) & 0x3ff;
                }
            }
            for y in 0..n.ch {
                let (u, v) = (row(frame, 1, y), row(frame, 2, y));
                for x in 0..n.cw {
                    n.cb[y * n.cw + x] = rd16(u, x) & 0x3ff;
                    n.cr[y * n.cw + x] = rd16(v, x) & 0x3ff;
                }
            }
        }
        VideoFormat::P01010le => {
            for y in 0..n.h {
                let r = row(frame, 0, y);
                for x in 0..n.w {
                    n.y[y * n.w + x] = rd16(r, x) >> 6;
                }
            }
            for y in 0..n.ch {
                let r = row(frame, 1, y);
                for x in 0..n.cw {
                    n.cb[y * n.cw + x] = rd16(r, 2 * x) >> 6;
                    n.cr[y * n.cw + x] = rd16(r, 2 * x + 1) >> 6;
                }
            }
        }
        VideoFormat::Uyvy | VideoFormat::Yuy2 => {
            let [oy0, ou, oy1, ov] = packed_offsets(f);
            for y in 0..n.h {
                let r = row(frame, 0, y);
                for c in 0..n.cw {
                    let m = &r[4 * c..4 * c + 4];
                    n.y[y * n.w + 2 * c] = m[oy0] as u16;
                    if 2 * c + 1 < n.w {
                        n.y[y * n.w + 2 * c + 1] = m[oy1] as u16;
                    }
                    n.cb[y * n.cw + c] = m[ou] as u16;
                    n.cr[y * n.cw + c] = m[ov] as u16;
                }
            }
        }
        _ => panic!("unsupported {f:?}"),
    }
    n
}

pub fn store(n: &Native, frame: &mut VideoFrameRef<&mut gst::BufferRef>) {
    let f = frame.format();
    match f {
        VideoFormat::I42210le | VideoFormat::I42010le => {
            for y in 0..n.h {
                let r = row_mut(frame, 0, y);
                for x in 0..n.w {
                    wr16(r, x, n.y[y * n.w + x]);
                }
            }
            for y in 0..n.ch {
                for x in 0..n.cw {
                    wr16(row_mut(frame, 1, y), x, n.cb[y * n.cw + x]);
                    wr16(row_mut(frame, 2, y), x, n.cr[y * n.cw + x]);
                }
            }
        }
        VideoFormat::P01010le => {
            for y in 0..n.h {
                let r = row_mut(frame, 0, y);
                for x in 0..n.w {
                    wr16(r, x, n.y[y * n.w + x] << 6);
                }
            }
            for y in 0..n.ch {
                let r = row_mut(frame, 1, y);
                for x in 0..n.cw {
                    wr16(r, 2 * x, n.cb[y * n.cw + x] << 6);
                    wr16(r, 2 * x + 1, n.cr[y * n.cw + x] << 6);
                }
            }
        }
        VideoFormat::Uyvy | VideoFormat::Yuy2 => {
            let [oy0, ou, oy1, ov] = packed_offsets(f);
            for y in 0..n.h {
                let r = row_mut(frame, 0, y);
                for c in 0..n.cw {
                    let y0 = n.y[y * n.w + 2 * c];
                    let y1 = if 2 * c + 1 < n.w {
                        n.y[y * n.w + 2 * c + 1]
                    } else {
                        y0
                    };
                    let m = &mut r[4 * c..4 * c + 4];
                    m[oy0] = y0 as u8;
                    m[oy1] = y1 as u8;
                    m[ou] = n.cb[y * n.cw + c] as u8;
                    m[ov] = n.cr[y * n.cw + c] as u8;
                }
            }
        }
        _ => panic!("unsupported {f:?}"),
    }
}

/// Converts to 4:2:2 10-bit (the v210 sample grid).
pub fn to_422_10(n: &Native, f: VideoFormat) -> Native {
    let mut o = Native::new(VideoFormat::I42210le, n.w, n.h);
    let expand = |v: u16| if is_8bit(f) { (v << 2) | (v >> 6) } else { v };
    o.y = n.y.iter().map(|&v| expand(v)).collect();
    if is_420(f) {
        for y in 0..n.h {
            let k = y / 2;
            let k2 = if y % 2 == 1 {
                (k + 1).min(n.ch - 1)
            } else {
                k.saturating_sub(1)
            };
            for x in 0..n.cw {
                let i = y * o.cw + x;
                let (a, b) = (k * n.cw + x, k2 * n.cw + x);
                o.cb[i] = (3 * n.cb[a] + n.cb[b] + 2) >> 2;
                o.cr[i] = (3 * n.cr[a] + n.cr[b] + 2) >> 2;
            }
        }
    } else {
        o.cb = n.cb.iter().map(|&v| expand(v)).collect();
        o.cr = n.cr.iter().map(|&v| expand(v)).collect();
    }
    o
}

/// Converts from 4:2:2 10-bit to format `f`'s samples.
pub fn from_422_10(p: &Native, f: VideoFormat) -> Native {
    let mut o = Native::new(f, p.w, p.h);
    let reduce = |v: u16| {
        if is_8bit(f) {
            ((v + 2) >> 2).min(255)
        } else {
            v
        }
    };
    o.y = p.y.iter().map(|&v| reduce(v)).collect();
    if is_420(f) {
        for k in 0..o.ch {
            let (r0, r1) = (2 * k, 2 * k + 1);
            for x in 0..o.cw {
                let a = r0 * p.cw + x;
                let i = k * o.cw + x;
                if r1 < p.h {
                    let b = r1 * p.cw + x;
                    o.cb[i] = (p.cb[a] + p.cb[b] + 1) >> 1;
                    o.cr[i] = (p.cr[a] + p.cr[b] + 1) >> 1;
                } else {
                    o.cb[i] = p.cb[a];
                    o.cr[i] = p.cr[a];
                }
            }
        }
    } else {
        o.cb = p.cb.iter().map(|&v| reduce(v)).collect();
        o.cr = p.cr.iter().map(|&v| reduce(v)).collect();
    }
    o
}

/// Writes 4:2:2 10-bit samples as v210; padding (incomplete groups and the
/// rest of the stride) is zero.
pub fn v210_encode(p: &Native, frame: &mut VideoFrameRef<&mut gst::BufferRef>) {
    for y in 0..p.h {
        let r = row_mut(frame, 0, y);
        r.fill(0);
        let yv = |x: usize| if x < p.w { p.y[y * p.w + x] as u32 } else { 0 };
        let cb = |c: usize| {
            if 2 * c < p.w {
                p.cb[y * p.cw + c] as u32
            } else {
                0
            }
        };
        let cr = |c: usize| {
            if 2 * c < p.w {
                p.cr[y * p.cw + c] as u32
            } else {
                0
            }
        };
        for g in 0..p.w.div_ceil(6) {
            let (x, c) = (6 * g, 3 * g);
            let w = [
                cb(c) | yv(x) << 10 | cr(c) << 20,
                yv(x + 1) | cb(c + 1) << 10 | yv(x + 2) << 20,
                cr(c + 1) | yv(x + 3) << 10 | cb(c + 2) << 20,
                yv(x + 4) | cr(c + 2) << 10 | yv(x + 5) << 20,
            ];
            for (i, w) in w.iter().enumerate() {
                r[16 * g + 4 * i..16 * g + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
            }
        }
    }
}

pub fn v210_decode(frame: &VideoFrameRef<&gst::BufferRef>) -> Native {
    let (w, h) = (frame.width() as usize, frame.height() as usize);
    let mut p = Native::new(VideoFormat::I42210le, w, h);
    for y in 0..h {
        let r = row(frame, 0, y);
        for g in 0..w.div_ceil(6) {
            let word = |i: usize| {
                u32::from_le_bytes(r[16 * g + 4 * i..16 * g + 4 * i + 4].try_into().unwrap())
            };
            let s = |i: usize, k: u32| ((word(i) >> (10 * k)) & 0x3ff) as u16;
            let ys = [s(0, 1), s(1, 0), s(1, 2), s(2, 1), s(3, 0), s(3, 2)];
            let cbs = [s(0, 0), s(1, 1), s(2, 2)];
            let crs = [s(0, 2), s(2, 0), s(3, 1)];
            for i in 0..6 {
                if 6 * g + i < w {
                    p.y[y * w + 6 * g + i] = ys[i];
                }
            }
            for i in 0..3 {
                if 2 * (3 * g + i) < w {
                    p.cb[y * p.cw + 3 * g + i] = cbs[i];
                    p.cr[y * p.cw + 3 * g + i] = crs[i];
                }
            }
        }
    }
    p
}

/// Small deterministic PRNG for test data.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }

    pub fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

/// Random samples valid for format `f`.
pub fn random_native(f: VideoFormat, w: usize, h: usize, seed: u64) -> Native {
    let mut rng = Rng::new(seed);
    let mut n = Native::new(f, w, h);
    let max = if is_8bit(f) { 0xff } else { 0x3ff };
    for v in n.y.iter_mut().chain(n.cb.iter_mut()).chain(n.cr.iter_mut()) {
        *v = (rng.next() & max) as u16;
    }
    n
}
