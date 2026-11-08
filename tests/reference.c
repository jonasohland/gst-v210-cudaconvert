#include "reference.h"

#include <string.h>

const GstVideoFormat ref_formats[] = {
  GST_VIDEO_FORMAT_I422_10LE,
  GST_VIDEO_FORMAT_UYVY,
  GST_VIDEO_FORMAT_YUY2,
  GST_VIDEO_FORMAT_I420_10LE,
  GST_VIDEO_FORMAT_P010_10LE,
};
const guint ref_n_formats = G_N_ELEMENTS (ref_formats);

gboolean
ref_is_420 (GstVideoFormat f)
{
  return f == GST_VIDEO_FORMAT_I420_10LE || f == GST_VIDEO_FORMAT_P010_10LE;
}

gboolean
ref_is_8bit (GstVideoFormat f)
{
  return f == GST_VIDEO_FORMAT_UYVY || f == GST_VIDEO_FORMAT_YUY2;
}

Native *
native_new (GstVideoFormat f, guint w, guint h)
{
  Native *n = g_new0 (Native, 1);

  n->w = w;
  n->h = h;
  n->cw = (w + 1) / 2;
  n->ch = ref_is_420 (f) ? (h + 1) / 2 : h;
  n->y = g_new0 (guint16, w * h);
  n->cb = g_new0 (guint16, n->cw * n->ch);
  n->cr = g_new0 (guint16, n->cw * n->ch);
  return n;
}

void
native_free (Native * n)
{
  if (!n)
    return;
  g_free (n->y);
  g_free (n->cb);
  g_free (n->cr);
  g_free (n);
}

gboolean
native_equal (const Native * a, const Native * b)
{
  gsize ny = a->w * a->h * sizeof (guint16);
  gsize nc = a->cw * a->ch * sizeof (guint16);

  return a->w == b->w && a->h == b->h && a->cw == b->cw && a->ch == b->ch &&
      memcmp (a->y, b->y, ny) == 0 && memcmp (a->cb, b->cb, nc) == 0 &&
      memcmp (a->cr, b->cr, nc) == 0;
}

/* xorshift64 */
static guint32
rng_next (guint64 * state)
{
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  return (guint32) (*state >> 16);
}

Native *
native_random (GstVideoFormat f, guint w, guint h, guint64 seed)
{
  Native *n = native_new (f, w, h);
  guint64 state = (seed * G_GUINT64_CONSTANT (0x9E3779B97F4A7C15)) | 1;
  guint32 max = ref_is_8bit (f) ? 0xff : 0x3ff;

  for (guint i = 0; i < w * h; i++)
    n->y[i] = rng_next (&state) & max;
  for (guint i = 0; i < n->cw * n->ch; i++)
    n->cb[i] = rng_next (&state) & max;
  for (guint i = 0; i < n->cw * n->ch; i++)
    n->cr[i] = rng_next (&state) & max;
  return n;
}

static const guint8 *
row (const GstVideoFrame * f, guint plane, guint y)
{
  return (const guint8 *) GST_VIDEO_FRAME_PLANE_DATA (f, plane) +
      (gsize) y *GST_VIDEO_FRAME_PLANE_STRIDE (f, plane);
}

static guint8 *
row_mut (GstVideoFrame * f, guint plane, guint y)
{
  return (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (f, plane) +
      (gsize) y *GST_VIDEO_FRAME_PLANE_STRIDE (f, plane);
}

static guint16
rd16 (const guint8 * r, guint i)
{
  return GST_READ_UINT16_LE (r + 2 * i);
}

static void
wr16 (guint8 * r, guint i, guint16 v)
{
  GST_WRITE_UINT16_LE (r + 2 * i, v);
}

/* Packed 8-bit byte offsets of (Y0, U, Y1, V). */
static const guint *
packed_offsets (GstVideoFormat f)
{
  static const guint uyvy[4] = { 1, 0, 3, 2 };
  static const guint yuy2[4] = { 0, 1, 2, 3 };

  g_assert (ref_is_8bit (f));
  return f == GST_VIDEO_FORMAT_UYVY ? uyvy : yuy2;
}

Native *
native_load (const GstVideoFrame * frame)
{
  GstVideoFormat f = GST_VIDEO_FRAME_FORMAT (frame);
  Native *n = native_new (f, GST_VIDEO_FRAME_WIDTH (frame),
      GST_VIDEO_FRAME_HEIGHT (frame));

  switch (f) {
    case GST_VIDEO_FORMAT_I422_10LE:
    case GST_VIDEO_FORMAT_I420_10LE:
      for (guint y = 0; y < n->h; y++)
        for (guint x = 0; x < n->w; x++)
          n->y[y * n->w + x] = rd16 (row (frame, 0, y), x) & 0x3ff;
      for (guint y = 0; y < n->ch; y++) {
        for (guint x = 0; x < n->cw; x++) {
          n->cb[y * n->cw + x] = rd16 (row (frame, 1, y), x) & 0x3ff;
          n->cr[y * n->cw + x] = rd16 (row (frame, 2, y), x) & 0x3ff;
        }
      }
      break;
    case GST_VIDEO_FORMAT_P010_10LE:
      for (guint y = 0; y < n->h; y++)
        for (guint x = 0; x < n->w; x++)
          n->y[y * n->w + x] = rd16 (row (frame, 0, y), x) >> 6;
      for (guint y = 0; y < n->ch; y++) {
        for (guint x = 0; x < n->cw; x++) {
          n->cb[y * n->cw + x] = rd16 (row (frame, 1, y), 2 * x) >> 6;
          n->cr[y * n->cw + x] = rd16 (row (frame, 1, y), 2 * x + 1) >> 6;
        }
      }
      break;
    case GST_VIDEO_FORMAT_UYVY:
    case GST_VIDEO_FORMAT_YUY2:{
      const guint *o = packed_offsets (f);

      for (guint y = 0; y < n->h; y++) {
        for (guint c = 0; c < n->cw; c++) {
          const guint8 *m = row (frame, 0, y) + 4 * c;

          n->y[y * n->w + 2 * c] = m[o[0]];
          if (2 * c + 1 < n->w)
            n->y[y * n->w + 2 * c + 1] = m[o[2]];
          n->cb[y * n->cw + c] = m[o[1]];
          n->cr[y * n->cw + c] = m[o[3]];
        }
      }
      break;
    }
    default:
      g_error ("unsupported format %s", gst_video_format_to_string (f));
  }
  return n;
}

void
native_store (const Native * n, GstVideoFrame * frame)
{
  GstVideoFormat f = GST_VIDEO_FRAME_FORMAT (frame);

  switch (f) {
    case GST_VIDEO_FORMAT_I422_10LE:
    case GST_VIDEO_FORMAT_I420_10LE:
      for (guint y = 0; y < n->h; y++)
        for (guint x = 0; x < n->w; x++)
          wr16 (row_mut (frame, 0, y), x, n->y[y * n->w + x]);
      for (guint y = 0; y < n->ch; y++) {
        for (guint x = 0; x < n->cw; x++) {
          wr16 (row_mut (frame, 1, y), x, n->cb[y * n->cw + x]);
          wr16 (row_mut (frame, 2, y), x, n->cr[y * n->cw + x]);
        }
      }
      break;
    case GST_VIDEO_FORMAT_P010_10LE:
      for (guint y = 0; y < n->h; y++)
        for (guint x = 0; x < n->w; x++)
          wr16 (row_mut (frame, 0, y), x, n->y[y * n->w + x] << 6);
      for (guint y = 0; y < n->ch; y++) {
        for (guint x = 0; x < n->cw; x++) {
          wr16 (row_mut (frame, 1, y), 2 * x, n->cb[y * n->cw + x] << 6);
          wr16 (row_mut (frame, 1, y), 2 * x + 1, n->cr[y * n->cw + x] << 6);
        }
      }
      break;
    case GST_VIDEO_FORMAT_UYVY:
    case GST_VIDEO_FORMAT_YUY2:{
      const guint *o = packed_offsets (f);

      for (guint y = 0; y < n->h; y++) {
        for (guint c = 0; c < n->cw; c++) {
          guint8 *m = row_mut (frame, 0, y) + 4 * c;
          guint16 y0 = n->y[y * n->w + 2 * c];
          guint16 y1 = 2 * c + 1 < n->w ? n->y[y * n->w + 2 * c + 1] : y0;

          m[o[0]] = y0;
          m[o[2]] = y1;
          m[o[1]] = n->cb[y * n->cw + c];
          m[o[3]] = n->cr[y * n->cw + c];
        }
      }
      break;
    }
    default:
      g_error ("unsupported format %s", gst_video_format_to_string (f));
  }
}

static guint16
expand (guint16 v, GstVideoFormat f)
{
  return ref_is_8bit (f) ? (v << 2) | (v >> 6) : v;
}

static guint16
reduce (guint16 v, GstVideoFormat f)
{
  return ref_is_8bit (f) ? MIN ((v + 2) >> 2, 255) : v;
}

Native *
native_to_422_10 (const Native * n, GstVideoFormat f)
{
  Native *o = native_new (GST_VIDEO_FORMAT_I422_10LE, n->w, n->h);

  for (guint i = 0; i < n->w * n->h; i++)
    o->y[i] = expand (n->y[i], f);
  if (ref_is_420 (f)) {
    /* MPEG-2 siting: chroma row k sits between luma rows 2k and 2k+1. */
    for (guint y = 0; y < n->h; y++) {
      guint k = y / 2;
      guint k2 = (y % 2) ? MIN (k + 1, n->ch - 1) : (k > 0 ? k - 1 : 0);

      for (guint x = 0; x < n->cw; x++) {
        guint i = y * o->cw + x, a = k * n->cw + x, b = k2 * n->cw + x;

        o->cb[i] = (3 * n->cb[a] + n->cb[b] + 2) >> 2;
        o->cr[i] = (3 * n->cr[a] + n->cr[b] + 2) >> 2;
      }
    }
  } else {
    for (guint i = 0; i < n->cw * n->ch; i++) {
      o->cb[i] = expand (n->cb[i], f);
      o->cr[i] = expand (n->cr[i], f);
    }
  }
  return o;
}

Native *
native_from_422_10 (const Native * p, GstVideoFormat f)
{
  Native *o = native_new (f, p->w, p->h);

  for (guint i = 0; i < p->w * p->h; i++)
    o->y[i] = reduce (p->y[i], f);
  if (ref_is_420 (f)) {
    for (guint k = 0; k < o->ch; k++) {
      guint r0 = 2 * k, r1 = 2 * k + 1;

      for (guint x = 0; x < o->cw; x++) {
        guint a = r0 * p->cw + x, i = k * o->cw + x;

        if (r1 < p->h) {
          guint b = r1 * p->cw + x;

          o->cb[i] = (p->cb[a] + p->cb[b] + 1) >> 1;
          o->cr[i] = (p->cr[a] + p->cr[b] + 1) >> 1;
        } else {
          o->cb[i] = p->cb[a];
          o->cr[i] = p->cr[a];
        }
      }
    }
  } else {
    for (guint i = 0; i < p->cw * p->ch; i++) {
      o->cb[i] = reduce (p->cb[i], f);
      o->cr[i] = reduce (p->cr[i], f);
    }
  }
  return o;
}

static guint32
sample_y (const Native * p, guint y, guint x)
{
  return x < p->w ? p->y[y * p->w + x] : 0;
}

static guint32
sample_cb (const Native * p, guint y, guint c)
{
  return 2 * c < p->w ? p->cb[y * p->cw + c] : 0;
}

static guint32
sample_cr (const Native * p, guint y, guint c)
{
  return 2 * c < p->w ? p->cr[y * p->cw + c] : 0;
}

void
v210_encode (const Native * p, GstVideoFrame * frame)
{
  for (guint y = 0; y < p->h; y++) {
    guint8 *r = row_mut (frame, 0, y);

    memset (r, 0, GST_VIDEO_FRAME_PLANE_STRIDE (frame, 0));
    for (guint g = 0; g < (p->w + 5) / 6; g++) {
      guint x = 6 * g, c = 3 * g;
      guint32 w[4] = {
        sample_cb (p, y, c) | sample_y (p, y, x) << 10 |
            sample_cr (p, y, c) << 20,
        sample_y (p, y, x + 1) | sample_cb (p, y, c + 1) << 10 |
            sample_y (p, y, x + 2) << 20,
        sample_cr (p, y, c + 1) | sample_y (p, y, x + 3) << 10 |
            sample_cb (p, y, c + 2) << 20,
        sample_y (p, y, x + 4) | sample_cr (p, y, c + 2) << 10 |
            sample_y (p, y, x + 5) << 20,
      };

      for (guint i = 0; i < 4; i++)
        GST_WRITE_UINT32_LE (r + 16 * g + 4 * i, w[i]);
    }
  }
}

Native *
v210_decode (const GstVideoFrame * frame)
{
  guint w = GST_VIDEO_FRAME_WIDTH (frame), h = GST_VIDEO_FRAME_HEIGHT (frame);
  Native *p = native_new (GST_VIDEO_FORMAT_I422_10LE, w, h);

  for (guint y = 0; y < h; y++) {
    const guint8 *r = row (frame, 0, y);

    for (guint g = 0; g < (w + 5) / 6; g++) {
      guint32 word[4];

      for (guint i = 0; i < 4; i++)
        word[i] = GST_READ_UINT32_LE (r + 16 * g + 4 * i);
#define S(i, k) ((guint16) ((word[i] >> (10 * (k))) & 0x3ff))
      guint16 ys[6] = { S (0, 1), S (1, 0), S (1, 2), S (2, 1), S (3, 0),
        S (3, 2)
      };
      guint16 cbs[3] = { S (0, 0), S (1, 1), S (2, 2) };
      guint16 crs[3] = { S (0, 2), S (2, 0), S (3, 1) };
#undef S
      for (guint i = 0; i < 6; i++)
        if (6 * g + i < w)
          p->y[y * w + 6 * g + i] = ys[i];
      for (guint i = 0; i < 3; i++) {
        if (2 * (3 * g + i) < w) {
          p->cb[y * p->cw + 3 * g + i] = cbs[i];
          p->cr[y * p->cw + 3 * g + i] = crs[i];
        }
      }
    }
  }
  return p;
}
