/* CPU reference implementation of the conversions, written independently of
 * the kernels and kept simple. It is the ground truth for the element tests. */

#pragma once

#include <gst/video/video.h>

G_BEGIN_DECLS

extern const GstVideoFormat ref_formats[];
extern const guint ref_n_formats;

gboolean ref_is_420 (GstVideoFormat f);
gboolean ref_is_8bit (GstVideoFormat f);

/* Samples of a frame in its own precision (8 or 10 bit) and chroma
 * resolution. */
typedef struct
{
  guint w, h;
  guint cw, ch;                 /* chroma width/height */
  guint16 *y, *cb, *cr;
} Native;

Native *native_new (GstVideoFormat f, guint w, guint h);
void native_free (Native * n);
gboolean native_equal (const Native * a, const Native * b);

/* Random samples valid for format `f`. Deterministic for a given seed. */
Native *native_random (GstVideoFormat f, guint w, guint h, guint64 seed);

Native *native_load (const GstVideoFrame * frame);
void native_store (const Native * n, GstVideoFrame * frame);

/* Converts samples of format `f` to 4:2:2 10-bit (the v210 sample grid). */
Native *native_to_422_10 (const Native * n, GstVideoFormat f);
/* Converts 4:2:2 10-bit samples to format `f`'s samples. */
Native *native_from_422_10 (const Native * p, GstVideoFormat f);

/* Writes 4:2:2 10-bit samples as v210; padding (incomplete groups and the
 * rest of the stride) is zero. */
void v210_encode (const Native * p, GstVideoFrame * frame);
Native *v210_decode (const GstVideoFrame * frame);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (Native, native_free)

G_END_DECLS
