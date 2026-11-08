// CUDA kernels for converting between v210 and other YUV formats.
//
// v210: every 6 pixels of 4:2:2 10-bit video are stored in 4 little-endian
// 32-bit words (16 bytes), 3 components per word, low bits first:
//   w0 = Cb0 | Y0 << 10 | Cr0 << 20
//   w1 = Y1  | Cb1 << 10 | Y2 << 20
//   w2 = Cr1 | Y3 << 10 | Cb2 << 20
//   w3 = Y4  | Cr2 << 10 | Y5 << 20
//
// One thread handles one 6-pixel group of one row (for 4:2:0 unpack: one
// group of two rows). Sticks to features supported by CUDA 12.9 and 13.x.

#include <stdint.h>

#include "v210-params.h"

using Params = V210Params;

struct Group {
  uint32_t y[6];
  uint32_t cb[3];
  uint32_t cr[3];
};

template <typename T>
__device__ __forceinline__ T *row_ptr(unsigned long long base, long long stride, int row) {
  return reinterpret_cast<T *>(base + (unsigned long long)(stride * row));
}

__device__ __forceinline__ uint32_t expand8(uint32_t v) { return (v << 2) | (v >> 6); }
__device__ __forceinline__ uint32_t reduce8(uint32_t v) { return min((v + 2) >> 2, 255u); }

// ---------------------------------------------------------------------------
// Per-format access. `x0` is the first pixel of the group, `n` the number of
// valid pixels in it (1..6), `nc` the number of valid chroma pairs (1..3).
// Readers fill invalid slots with 0.

// Planar 4:2:2 / 4:2:0, 16-bit containers. `Shift` = 0 for LSB-aligned (I4xx_10LE),
// 6 for MSB-aligned (P010 luma).
struct Planar16Luma {
  template <int Shift>
  __device__ static void load(const Params &p, int y, int x0, int n, Group &g) {
    const uint16_t *src = row_ptr<const uint16_t>(p.plane[0], p.stride[0], y) + x0;
    for (int i = 0; i < 6; i++) g.y[i] = i < n ? (uint32_t)(src[i] >> Shift) & 0x3ff : 0;
  }
  template <int Shift>
  __device__ static void store(const Params &p, int y, int x0, int n, const Group &g) {
    uint16_t *dst = row_ptr<uint16_t>(p.plane[0], p.stride[0], y) + x0;
    for (int i = 0; i < n; i++) dst[i] = (uint16_t)(g.y[i] << Shift);
  }
};

// Chroma readers/writers for one chroma row `cy`.
struct ChromaI {  // separate U and V planes, LSB-aligned 10-bit
  __device__ static void load(const Params &p, int cy, int c0, int nc, uint32_t *cb, uint32_t *cr) {
    const uint16_t *u = row_ptr<const uint16_t>(p.plane[1], p.stride[1], cy) + c0;
    const uint16_t *v = row_ptr<const uint16_t>(p.plane[2], p.stride[2], cy) + c0;
    for (int i = 0; i < 3; i++) {
      cb[i] = i < nc ? u[i] & 0x3ff : 0;
      cr[i] = i < nc ? v[i] & 0x3ff : 0;
    }
  }
  __device__ static void store(const Params &p, int cy, int c0, int nc, const uint32_t *cb, const uint32_t *cr) {
    uint16_t *u = row_ptr<uint16_t>(p.plane[1], p.stride[1], cy) + c0;
    uint16_t *v = row_ptr<uint16_t>(p.plane[2], p.stride[2], cy) + c0;
    for (int i = 0; i < nc; i++) {
      u[i] = (uint16_t)cb[i];
      v[i] = (uint16_t)cr[i];
    }
  }
};

struct ChromaP {  // interleaved UV plane, MSB-aligned 10-bit
  __device__ static void load(const Params &p, int cy, int c0, int nc, uint32_t *cb, uint32_t *cr) {
    const uint16_t *uv = row_ptr<const uint16_t>(p.plane[1], p.stride[1], cy) + 2 * c0;
    for (int i = 0; i < 3; i++) {
      cb[i] = i < nc ? uv[2 * i] >> 6 : 0;
      cr[i] = i < nc ? uv[2 * i + 1] >> 6 : 0;
    }
  }
  __device__ static void store(const Params &p, int cy, int c0, int nc, const uint32_t *cb, const uint32_t *cr) {
    uint16_t *uv = row_ptr<uint16_t>(p.plane[1], p.stride[1], cy) + 2 * c0;
    for (int i = 0; i < nc; i++) {
      uv[2 * i] = (uint16_t)(cb[i] << 6);
      uv[2 * i + 1] = (uint16_t)(cr[i] << 6);
    }
  }
};

// 4:2:2 planar formats.
template <typename Chroma, int LumaShift>
struct Fmt422 {
  __device__ static void load(const Params &p, int y, int x0, int n, int nc, Group &g) {
    Planar16Luma::load<LumaShift>(p, y, x0, n, g);
    Chroma::load(p, y, x0 / 2, nc, g.cb, g.cr);
  }
  __device__ static void store(const Params &p, int y, int x0, int n, int nc, const Group &g) {
    Planar16Luma::store<LumaShift>(p, y, x0, n, g);
    Chroma::store(p, y, x0 / 2, nc, g.cb, g.cr);
  }
};

using FmtI422_10 = Fmt422<ChromaI, 0>;

// Packed 8-bit 4:2:2. Byte offsets of Y0, U, Y1, V inside a 4-byte macropixel.
template <int OY0, int OU, int OY1, int OV>
struct FmtPacked8 {
  __device__ static void load(const Params &p, int y, int x0, int n, int nc, Group &g) {
    const uint8_t *src = row_ptr<const uint8_t>(p.plane[0], p.stride[0], y) + x0 * 2;
    for (int i = 0; i < 3; i++) {
      const uint8_t *m = src + 4 * i;
      bool cv = i < nc;
      g.y[2 * i] = cv ? expand8(m[OY0]) : 0;
      g.y[2 * i + 1] = (2 * i + 1) < n ? expand8(m[OY1]) : 0;
      g.cb[i] = cv ? expand8(m[OU]) : 0;
      g.cr[i] = cv ? expand8(m[OV]) : 0;
    }
  }
  __device__ static void store(const Params &p, int y, int x0, int n, int nc, const Group &g) {
    uint8_t *dst = row_ptr<uint8_t>(p.plane[0], p.stride[0], y) + x0 * 2;
    for (int i = 0; i < nc; i++) {
      uint8_t *m = dst + 4 * i;
      m[OY0] = (uint8_t)reduce8(g.y[2 * i]);
      // For odd widths the second luma of the last macropixel is padding;
      // write the duplicate of the first one, like GStreamer does.
      m[OY1] = (uint8_t)reduce8((2 * i + 1) < n ? g.y[2 * i + 1] : g.y[2 * i]);
      m[OU] = (uint8_t)reduce8(g.cb[i]);
      m[OV] = (uint8_t)reduce8(g.cr[i]);
    }
  }
};

using FmtUYVY = FmtPacked8<1, 0, 3, 2>;
using FmtYUY2 = FmtPacked8<0, 1, 2, 3>;

// ---------------------------------------------------------------------------
// v210 group access

__device__ __forceinline__ void v210_store(const Params &p, int y, int gi, const Group &g) {
  uint4 w;
  w.x = g.cb[0] | g.y[0] << 10 | g.cr[0] << 20;
  w.y = g.y[1] | g.cb[1] << 10 | g.y[2] << 20;
  w.z = g.cr[1] | g.y[3] << 10 | g.cb[2] << 20;
  w.w = g.y[4] | g.cr[2] << 10 | g.y[5] << 20;
  row_ptr<uint4>(p.v210, p.v210_stride, y)[gi] = w;
}

__device__ __forceinline__ void v210_load(const Params &p, int y, int gi, Group &g) {
  uint4 w = row_ptr<const uint4>(p.v210, p.v210_stride, y)[gi];
  g.cb[0] = w.x & 0x3ff; g.y[0] = (w.x >> 10) & 0x3ff; g.cr[0] = (w.x >> 20) & 0x3ff;
  g.y[1] = w.y & 0x3ff; g.cb[1] = (w.y >> 10) & 0x3ff; g.y[2] = (w.y >> 20) & 0x3ff;
  g.cr[1] = w.z & 0x3ff; g.y[3] = (w.z >> 10) & 0x3ff; g.cb[2] = (w.z >> 20) & 0x3ff;
  g.y[4] = w.w & 0x3ff; g.cr[2] = (w.w >> 10) & 0x3ff; g.y[5] = (w.w >> 20) & 0x3ff;
}

__device__ __forceinline__ void group_extent(int width, int gi, int &x0, int &n, int &nc) {
  x0 = gi * 6;
  n = min(6, width - x0);
  nc = (n + 1) / 2;
}

// ---------------------------------------------------------------------------
// Kernels. Pack covers every 16-byte group of the v210 stride so that row
// padding is written as zeros (deterministic output).

template <typename Fmt>
__device__ void pack422(const Params &p) {
  int gi = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y >= p.height || gi >= (int)(p.v210_stride / 16)) return;
  Group g = {};
  int x0, n, nc;
  group_extent(p.width, gi, x0, n, nc);
  if (n > 0) Fmt::load(p, y, x0, n, nc, g);
  v210_store(p, y, gi, g);
}

template <typename Fmt>
__device__ void unpack422(const Params &p) {
  int gi = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x0, n, nc;
  group_extent(p.width, gi, x0, n, nc);
  if (y >= p.height || n <= 0) return;
  Group g;
  v210_load(p, y, gi, g);
  Fmt::store(p, y, x0, n, nc, g);
}

// 4:2:0 -> v210: vertical chroma upsampling, MPEG-2 siting (chroma row k sits
// between luma rows 2k and 2k+1): row 2k = (3*C[k] + C[k-1] + 2) / 4,
// row 2k+1 = (3*C[k] + C[k+1] + 2) / 4, clamped at the edges.
template <typename Chroma, int LumaShift>
__device__ void pack420(const Params &p) {
  int gi = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y >= p.height || gi >= (int)(p.v210_stride / 16)) return;
  Group g = {};
  int x0, n, nc;
  group_extent(p.width, gi, x0, n, nc);
  if (n > 0) {
    Planar16Luma::load<LumaShift>(p, y, x0, n, g);
    int ch = (p.height + 1) / 2;
    int k = y / 2;
    int k2 = (y & 1) ? min(k + 1, ch - 1) : max(k - 1, 0);
    uint32_t cb0[3], cr0[3], cb1[3], cr1[3];
    Chroma::load(p, k, x0 / 2, nc, cb0, cr0);
    Chroma::load(p, k2, x0 / 2, nc, cb1, cr1);
    for (int i = 0; i < 3; i++) {
      g.cb[i] = (3 * cb0[i] + cb1[i] + 2) >> 2;
      g.cr[i] = (3 * cr0[i] + cr1[i] + 2) >> 2;
    }
  }
  v210_store(p, y, gi, g);
}

// v210 -> 4:2:0: chroma of rows 2k and 2k+1 is averaged (rounded). For an odd
// height the last chroma row takes the last luma row's chroma as is.
template <typename Chroma, int LumaShift>
__device__ void unpack420(const Params &p) {
  int gi = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int x0, n, nc;
  group_extent(p.width, gi, x0, n, nc);
  int y0 = 2 * k;
  if (y0 >= p.height || n <= 0) return;
  Group a, b;
  v210_load(p, y0, gi, a);
  Planar16Luma::store<LumaShift>(p, y0, x0, n, a);
  uint32_t cb[3], cr[3];
  if (y0 + 1 < p.height) {
    v210_load(p, y0 + 1, gi, b);
    Planar16Luma::store<LumaShift>(p, y0 + 1, x0, n, b);
    for (int i = 0; i < 3; i++) {
      cb[i] = (a.cb[i] + b.cb[i] + 1) >> 1;
      cr[i] = (a.cr[i] + b.cr[i] + 1) >> 1;
    }
  } else {
    for (int i = 0; i < 3; i++) {
      cb[i] = a.cb[i];
      cr[i] = a.cr[i];
    }
  }
  Chroma::store(p, k, x0 / 2, nc, cb, cr);
}

// ---------------------------------------------------------------------------
// Entry points, looked up by name from the host code (src/gstcudav210convert.c).

#define ENTRY(name, body) \
  extern "C" __global__ void __launch_bounds__(256) name(Params p) { body(p); }

ENTRY(v210_pack_i422_10, pack422<FmtI422_10>)
ENTRY(v210_pack_uyvy, pack422<FmtUYVY>)
ENTRY(v210_pack_yuy2, pack422<FmtYUY2>)
ENTRY(v210_pack_i420_10, (pack420<ChromaI, 0>))
ENTRY(v210_pack_p010, (pack420<ChromaP, 6>))

ENTRY(v210_unpack_i422_10, unpack422<FmtI422_10>)
ENTRY(v210_unpack_uyvy, unpack422<FmtUYVY>)
ENTRY(v210_unpack_yuy2, unpack422<FmtYUY2>)
ENTRY(v210_unpack_i420_10, (unpack420<ChromaI, 0>))
ENTRY(v210_unpack_p010, (unpack420<ChromaP, 6>))
