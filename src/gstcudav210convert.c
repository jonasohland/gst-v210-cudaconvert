/* cudav210pack (other format -> v210) and cudav210unpack (v210 -> other
 * format). Both are thin subclasses of the abstract GstCudaV210Convert, which
 * holds all the logic and picks the kernel by the class's direction. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstcudav210convert.h"

#include <gst/base/base.h>
#include <gst/video/video.h>
#include <gst/cuda/gstcuda.h>

#include "v210-params.h"

GST_DEBUG_CATEGORY_STATIC (gst_cuda_v210_debug);
#define GST_CAT_DEFAULT gst_cuda_v210_debug

/* Kernels from kernels/v210.cu, embedded at build time. */
extern const unsigned char gst_cuda_v210_fatbin[];

#define OTHER_FORMATS "{ I422_10LE, UYVY, YUY2, I420_10LE, P010_10LE }"

/* Other formats: CUDA memory preferred, then system memory. */
#define OTHER_CAPS \
  GST_VIDEO_CAPS_MAKE_WITH_FEATURES (GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY, \
      OTHER_FORMATS) "; " GST_VIDEO_CAPS_MAKE (OTHER_FORMATS)

/* v210 is system memory only: gst-cuda's allocator has no v210 layout, so
 * v210 can't be carried in GstCudaMemory (and no GStreamer element would
 * accept it there). The elements upload/download it internally instead. */
#define V210_CAPS GST_VIDEO_CAPS_MAKE ("v210")

static GstStaticPadTemplate other_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (OTHER_CAPS));
static GstStaticPadTemplate other_src_template =
GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (OTHER_CAPS));
static GstStaticPadTemplate v210_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (V210_CAPS));
static GstStaticPadTemplate v210_src_template =
GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (V210_CAPS));

typedef enum
{
  DIRECTION_PACK,               /* other format -> v210 */
  DIRECTION_UNPACK,             /* v210 -> other format */
} Direction;

/* Formats that can be converted to/from v210, with the suffix of their
 * kernels in kernels/v210.cu. */
static const struct
{
  GstVideoFormat format;
  const gchar *kernel_suffix;
} other_formats[] = {
  {GST_VIDEO_FORMAT_I422_10LE, "i422_10"},
  {GST_VIDEO_FORMAT_UYVY, "uyvy"},
  {GST_VIDEO_FORMAT_YUY2, "yuy2"},
  {GST_VIDEO_FORMAT_I420_10LE, "i420_10"},
  {GST_VIDEO_FORMAT_P010_10LE, "p010"},
};

static gboolean
is_420 (GstVideoFormat f)
{
  return f == GST_VIDEO_FORMAT_I420_10LE || f == GST_VIDEO_FORMAT_P010_10LE;
}

/* A pitched device allocation, used to stage one system-memory plane. */
typedef struct
{
  CUdeviceptr ptr;
  gsize pitch;
  gsize width_bytes;
  gsize height;
} Staging;

typedef struct
{
  GstBaseTransform parent;

  gint device_id;               /* protected by the object lock */

  /* From start() */
  GstCudaContext *context;
  GstCudaStream *stream;
  CUmodule module;

  /* From set_caps() */
  gboolean configured;
  GstVideoInfo in_info;
  GstVideoInfo out_info;
  CUfunction kernel;
  Staging staging_in[GST_VIDEO_MAX_PLANES];
  Staging staging_out[GST_VIDEO_MAX_PLANES];
} GstCudaV210Convert;

typedef struct
{
  GstBaseTransformClass parent_class;
  Direction direction;
} GstCudaV210ConvertClass;

#define GST_TYPE_CUDA_V210_CONVERT (gst_cuda_v210_convert_get_type ())
#define GST_CUDA_V210_CONVERT(obj) ((GstCudaV210Convert *) (obj))
#define DIRECTION(self) \
  (((GstCudaV210ConvertClass *) G_OBJECT_GET_CLASS (self))->direction)

GType gst_cuda_v210_convert_get_type (void);
G_DEFINE_ABSTRACT_TYPE (GstCudaV210Convert, gst_cuda_v210_convert,
    GST_TYPE_BASE_TRANSFORM);
#define parent_class gst_cuda_v210_convert_parent_class

enum
{
  PROP_0,
  PROP_DEVICE_ID,
};

#define DEFAULT_DEVICE_ID -1

/* Logs a failed driver call. Returns TRUE on success. */
static gboolean
cuda_check (GstCudaV210Convert * self, CUresult res, const gchar * what)
{
  const gchar *name = NULL;

  if (res == CUDA_SUCCESS)
    return TRUE;
  CuGetErrorName (res, &name);
  GST_ERROR_OBJECT (self, "%s failed: %s", what, name ? name : "unknown error");
  return FALSE;
}

static gint
get_device_id (GstCudaV210Convert * self)
{
  gint id;

  GST_OBJECT_LOCK (self);
  id = self->device_id;
  GST_OBJECT_UNLOCK (self);
  return id;
}

/* Bytes per row and number of rows of `plane` for staging copies. */
static void
plane_extent (const GstVideoInfo * info, guint plane, gsize * width_bytes,
    gsize * height)
{
  gint comp[GST_VIDEO_MAX_COMPONENTS];

  gst_video_format_info_component (info->finfo, plane, comp);
  *width_bytes = GST_VIDEO_INFO_PLANE_STRIDE (info, plane);
  *height = GST_VIDEO_FORMAT_INFO_SCALE_HEIGHT (info->finfo, MAX (comp[0], 0),
      GST_VIDEO_INFO_HEIGHT (info));
}

/* The context must be current. */
static void
staging_free (Staging * bufs)
{
  for (guint i = 0; i < GST_VIDEO_MAX_PLANES; i++) {
    if (bufs[i].ptr)
      CuMemFree (bufs[i].ptr);
    bufs[i] = (Staging) {
    0};
  }
}

/* (Re)allocates one device buffer per plane of `info` if needed. The context
 * must be current. */
static gboolean
staging_ensure (GstCudaV210Convert * self, Staging * bufs,
    const GstVideoInfo * info)
{
  for (guint i = 0; i < GST_VIDEO_INFO_N_PLANES (info); i++) {
    Staging *s = &bufs[i];
    gsize w, h;

    plane_extent (info, i, &w, &h);
    if (s->ptr && s->width_bytes == w && s->height == h)
      continue;
    if (s->ptr)
      CuMemFree (s->ptr);
    *s = (Staging) {
    0};
    if (!cuda_check (self, CuMemAllocPitch (&s->ptr, &s->pitch, MAX (w, 1),
                MAX (h, 1), 16), "cuMemAllocPitch")) {
      s->ptr = 0;
      return FALSE;
    }
    s->width_bytes = w;
    s->height = h;
  }
  return TRUE;
}

/* Frees the staging buffers of both directions. */
static void
staging_clear (GstCudaV210Convert * self)
{
  if (!self->context || !gst_cuda_context_push (self->context))
    return;
  staging_free (self->staging_in);
  staging_free (self->staging_out);
  gst_cuda_context_pop (NULL);
}

/* Copies a host plane to a staging buffer. At most `src_pitch` bytes per row
 * are copied. */
static gboolean
upload (GstCudaV210Convert * self, CUstream stream, const guint8 * src,
    gsize src_pitch, const Staging * dst)
{
  CUDA_MEMCPY2D c = { 0, };

  c.srcMemoryType = CU_MEMORYTYPE_HOST;
  c.srcHost = src;
  c.srcPitch = src_pitch;
  c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  c.dstDevice = dst->ptr;
  c.dstPitch = dst->pitch;
  c.WidthInBytes = MIN (dst->width_bytes, src_pitch);
  c.Height = dst->height;
  return cuda_check (self, CuMemcpy2DAsync (&c, stream),
      "cuMemcpy2DAsync(HtoD)");
}

/* Copies a staging buffer to a host plane. At most `dst_pitch` bytes per row
 * are copied. */
static gboolean
download (GstCudaV210Convert * self, CUstream stream, const Staging * src,
    guint8 * dst, gsize dst_pitch)
{
  CUDA_MEMCPY2D c = { 0, };

  c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  c.srcDevice = src->ptr;
  c.srcPitch = src->pitch;
  c.dstMemoryType = CU_MEMORYTYPE_HOST;
  c.dstHost = dst;
  c.dstPitch = dst_pitch;
  c.WidthInBytes = MIN (src->width_bytes, dst_pitch);
  c.Height = src->height;
  return cuda_check (self, CuMemcpy2DAsync (&c, stream),
      "cuMemcpy2DAsync(DtoH)");
}

/* Launches the kernel with a 2D grid covering nx × ny threads. The context
 * must be current. */
static gboolean
launch (GstCudaV210Convert * self, CUstream stream, guint nx, guint ny,
    V210Params * params)
{
  const guint bx = 64, by = 4;
  void *args[] = { params };

  if (nx == 0 || ny == 0)
    return TRUE;
  return cuda_check (self, CuLaunchKernel (self->kernel,
          (nx + bx - 1) / bx, (ny + by - 1) / by, 1, bx, by, 1, 0, stream,
          args, NULL), "cuLaunchKernel");
}

/* If the buffer's first memory is CUDA memory, returns its context (borrowed). */
static GstCudaContext *
buffer_cuda_context (GstBuffer * buffer)
{
  GstMemory *mem;

  if (gst_buffer_n_memory (buffer) == 0)
    return NULL;
  mem = gst_buffer_peek_memory (buffer, 0);
  if (!gst_is_cuda_memory (mem))
    return NULL;
  return GST_CUDA_MEMORY_CAST (mem)->context;
}

/* Whether device memory of `other` can be used from our context. */
static gboolean
can_access (GstCudaV210Convert * self, GstCudaContext * other)
{
  return other && (other == self->context
      || gst_cuda_context_can_access_peer (self->context, other));
}

/* Waits for pending work on every CUDA memory in `buffer`. */
static void
sync_buffer (GstBuffer * buffer)
{
  for (guint i = 0; i < gst_buffer_n_memory (buffer); i++) {
    GstMemory *mem = gst_buffer_peek_memory (buffer, i);
    if (gst_is_cuda_memory (mem))
      gst_cuda_memory_sync (GST_CUDA_MEMORY_CAST (mem));
  }
}

static gboolean
is_cuda_caps (GstCaps * caps)
{
  GstCapsFeatures *f;

  if (gst_caps_get_size (caps) == 0)
    return FALSE;
  f = gst_caps_get_features (caps, 0);
  return f && gst_caps_features_contains (f,
      GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY);
}

/* ------------------------------------------------------------------------- */

static void
gst_cuda_v210_convert_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (object);

  switch (prop_id) {
    case PROP_DEVICE_ID:
      GST_OBJECT_LOCK (self);
      self->device_id = g_value_get_int (value);
      GST_OBJECT_UNLOCK (self);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_cuda_v210_convert_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (object);

  switch (prop_id) {
    case PROP_DEVICE_ID:
      g_value_set_int (value, get_device_id (self));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_cuda_v210_convert_dispose (GObject * object)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (object);

  /* A context may have been set without the element ever starting. */
  gst_clear_object (&self->context);
  G_OBJECT_CLASS (parent_class)->dispose (object);
}

static void
gst_cuda_v210_convert_set_context (GstElement * element, GstContext * context)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (element);

  gst_cuda_handle_set_context (element, context, get_device_id (self),
      &self->context);
  GST_ELEMENT_CLASS (parent_class)->set_context (element, context);
}

static gboolean
gst_cuda_v210_convert_query (GstBaseTransform * trans,
    GstPadDirection direction, GstQuery * query)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (trans);

  if (GST_QUERY_TYPE (query) == GST_QUERY_CONTEXT &&
      gst_cuda_handle_context_query (GST_ELEMENT (trans), query, self->context))
    return TRUE;
  return GST_BASE_TRANSFORM_CLASS (parent_class)->query (trans, direction,
      query);
}

static gboolean
gst_cuda_v210_convert_stop (GstBaseTransform * trans)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (trans);

  /* Free device buffers and the module before releasing the context. */
  if (self->context && gst_cuda_context_push (self->context)) {
    staging_free (self->staging_in);
    staging_free (self->staging_out);
    if (self->module)
      CuModuleUnload (self->module);
    gst_cuda_context_pop (NULL);
  }
  self->module = NULL;
  self->kernel = NULL;
  self->configured = FALSE;
  gst_clear_cuda_stream (&self->stream);
  gst_clear_object (&self->context);
  return TRUE;
}

static gboolean
gst_cuda_v210_convert_start (GstBaseTransform * trans)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (trans);
  gint device_id = get_device_id (self);
  gboolean ok;

  if (!gst_cuda_load_library ()) {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        ("Could not load the CUDA driver library"), (NULL));
    return FALSE;
  }
  if (!gst_cuda_ensure_element_context (GST_ELEMENT (self), device_id,
          &self->context)) {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        ("Could not get a CUDA context for device %d", device_id), (NULL));
    return FALSE;
  }

  self->stream = gst_cuda_stream_new (self->context);
  if (!self->stream) {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        ("CUDA setup failed: could not create a CUDA stream"), (NULL));
    goto error;
  }
  if (!gst_cuda_context_push (self->context)) {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        ("CUDA setup failed: could not push the CUDA context"), (NULL));
    goto error;
  }
  ok = cuda_check (self, CuModuleLoadData (&self->module,
          gst_cuda_v210_fatbin), "cuModuleLoadData");
  gst_cuda_context_pop (NULL);
  if (!ok) {
    self->module = NULL;
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        ("CUDA setup failed: could not load the kernels"), (NULL));
    goto error;
  }
  return TRUE;

error:
  gst_cuda_v210_convert_stop (trans);
  return FALSE;
}

/* v210 side: system memory. Other side: CUDA memory preferred, then system
 * memory. 4:2:0 formats are only offered for progressive video. */
static GstCaps *
gst_cuda_v210_convert_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  Direction dir = DIRECTION (trans);
  gboolean to_v210 = (dir == DIRECTION_PACK && direction == GST_PAD_SINK) ||
      (dir == DIRECTION_UNPACK && direction == GST_PAD_SRC);
  GstCaps *first = gst_caps_new_empty ();
  GstCaps *second = gst_caps_new_empty ();
  GstCaps *result;

  for (guint i = 0; i < gst_caps_get_size (caps); i++) {
    const GstStructure *in = gst_caps_get_structure (caps, i);
    const gchar *mode = gst_structure_get_string (in, "interlace-mode");
    gboolean interlaced = mode && !g_str_equal (mode, "progressive");
    const gchar *names[G_N_ELEMENTS (other_formats)];
    guint n = 0;
    GstStructure *s;

    if (to_v210) {
      names[n++] = "v210";
    } else {
      for (guint f = 0; f < G_N_ELEMENTS (other_formats); f++) {
        if (interlaced && is_420 (other_formats[f].format))
          continue;
        names[n++] = gst_video_format_to_string (other_formats[f].format);
      }
    }
    if (n == 0)
      continue;

    s = gst_structure_copy (in);
    gst_structure_remove_field (s, "chroma-site");
    if (n == 1) {
      gst_structure_set (s, "format", G_TYPE_STRING, names[0], NULL);
    } else {
      GValue list = G_VALUE_INIT;

      gst_value_list_init (&list, n);
      for (guint f = 0; f < n; f++) {
        GValue v = G_VALUE_INIT;

        g_value_init (&v, G_TYPE_STRING);
        g_value_set_static_string (&v, names[f]);
        gst_value_list_append_and_take_value (&list, &v);
      }
      gst_structure_take_value (s, "format", &list);
    }

    if (to_v210) {
      gst_caps_append_structure_full (first, s, NULL);
    } else {
      gst_caps_append_structure_full (first, gst_structure_copy (s),
          gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY, NULL));
      gst_caps_append_structure_full (second, s, NULL);
    }
  }
  first = gst_caps_merge (first, second);

  if (filter) {
    result = gst_caps_intersect_full (filter, first, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (first);
  } else {
    result = first;
  }

  GST_TRACE_OBJECT (trans, "transform_caps %s: %" GST_PTR_FORMAT " -> %"
      GST_PTR_FORMAT, direction == GST_PAD_SINK ? "sink" : "src", caps,
      result);
  return result;
}

static gboolean
gst_cuda_v210_convert_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (trans);
  gboolean pack = DIRECTION (self) == DIRECTION_PACK;
  GstVideoInfo in_info, out_info;
  const GstVideoInfo *v210, *other;
  const gchar *suffix = NULL;
  CUfunction kernel = NULL;
  gchar *name;
  gboolean ok;

  self->configured = FALSE;

  if (!gst_video_info_from_caps (&in_info, incaps)) {
    GST_ERROR_OBJECT (self, "invalid input caps %" GST_PTR_FORMAT, incaps);
    return FALSE;
  }
  if (!gst_video_info_from_caps (&out_info, outcaps)) {
    GST_ERROR_OBJECT (self, "invalid output caps %" GST_PTR_FORMAT, outcaps);
    return FALSE;
  }
  if (GST_VIDEO_INFO_WIDTH (&in_info) != GST_VIDEO_INFO_WIDTH (&out_info) ||
      GST_VIDEO_INFO_HEIGHT (&in_info) != GST_VIDEO_INFO_HEIGHT (&out_info)) {
    GST_ERROR_OBJECT (self, "resolution must not change");
    return FALSE;
  }

  v210 = pack ? &out_info : &in_info;
  other = pack ? &in_info : &out_info;
  if (GST_VIDEO_INFO_FORMAT (v210) != GST_VIDEO_FORMAT_v210) {
    GST_ERROR_OBJECT (self, "expected v210, got %s",
        GST_VIDEO_INFO_NAME (v210));
    return FALSE;
  }
  if (is_420 (GST_VIDEO_INFO_FORMAT (other)) &&
      GST_VIDEO_INFO_INTERLACE_MODE (other) !=
      GST_VIDEO_INTERLACE_MODE_PROGRESSIVE) {
    GST_ERROR_OBJECT (self, "4:2:0 formats must be progressive");
    return FALSE;
  }
  for (guint i = 0; i < G_N_ELEMENTS (other_formats); i++) {
    if (other_formats[i].format == GST_VIDEO_INFO_FORMAT (other))
      suffix = other_formats[i].kernel_suffix;
  }
  if (!suffix) {
    GST_ERROR_OBJECT (self, "unsupported format %s",
        GST_VIDEO_INFO_NAME (other));
    return FALSE;
  }
  if (!self->module) {
    GST_ERROR_OBJECT (self, "set_caps before start");
    return FALSE;
  }

  name = g_strdup_printf ("v210_%s_%s", pack ? "pack" : "unpack", suffix);
  ok = cuda_check (self, CuModuleGetFunction (&kernel, self->module, name),
      "cuModuleGetFunction");
  if (ok) {
    GST_DEBUG_OBJECT (self, "configured %s: %" GST_PTR_FORMAT " -> %"
        GST_PTR_FORMAT, name, incaps, outcaps);
  }
  g_free (name);
  if (!ok)
    return FALSE;

  /* Sizes may have changed; staging buffers are reallocated on demand. */
  staging_clear (self);
  self->in_info = in_info;
  self->out_info = out_info;
  self->kernel = kernel;
  self->configured = TRUE;
  return TRUE;
}

static gboolean
gst_cuda_v210_convert_get_unit_size (GstBaseTransform * trans, GstCaps * caps,
    gsize * size)
{
  GstVideoInfo info;

  if (!gst_video_info_from_caps (&info, caps))
    return FALSE;
  *size = GST_VIDEO_INFO_SIZE (&info);
  return TRUE;
}

/* One frame in is one frame out. Input size varies with padding (e.g. pitched
 * CUDA memory), so don't derive the output size from unit sizes. */
static gboolean
gst_cuda_v210_convert_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size, GstCaps * othercaps,
    gsize * othersize)
{
  return gst_cuda_v210_convert_get_unit_size (trans, othercaps, othersize);
}

static GstBufferPool *
make_cuda_pool (GstCudaV210Convert * self, GstCaps * caps, guint min,
    guint max, guint * size)
{
  GstBufferPool *pool;
  GstStructure *config;
  GstVideoInfo info;

  if (!self->context || !gst_video_info_from_caps (&info, caps))
    return NULL;

  pool = gst_cuda_buffer_pool_new (self->context);
  config = gst_buffer_pool_get_config (pool);
  gst_buffer_pool_config_add_option (config,
      GST_BUFFER_POOL_OPTION_VIDEO_META);
  gst_buffer_pool_config_set_params (config, caps, GST_VIDEO_INFO_SIZE (&info),
      min, max);
  gst_buffer_pool_config_set_cuda_stream (config, self->stream);
  if (!gst_buffer_pool_set_config (pool, config)) {
    gst_object_unref (pool);
    return NULL;
  }

  /* The pool may have adjusted the size for its alignment. */
  config = gst_buffer_pool_get_config (pool);
  gst_buffer_pool_config_get_params (config, NULL, size, NULL, NULL);
  gst_structure_free (config);
  return pool;
}

static gboolean
gst_cuda_v210_convert_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (trans);
  GstCaps *caps;
  gboolean need_pool;
  GstBufferPool *pool;
  guint size;

  if (!GST_BASE_TRANSFORM_CLASS (parent_class)->propose_allocation (trans,
          decide_query, query))
    return FALSE;

  gst_query_parse_allocation (query, &caps, &need_pool);
  if (!caps)
    return FALSE;
  gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);
  if (!need_pool || !is_cuda_caps (caps))
    return TRUE;

  pool = make_cuda_pool (self, caps, 0, 0, &size);
  if (pool) {
    gst_query_add_allocation_pool (query, pool, size, 0, 0);
    gst_object_unref (pool);
  }
  return TRUE;
}

/* Replaces the downstream pool with our own CUDA pool when outputting CUDA
 * memory. */
static gboolean
gst_cuda_v210_convert_decide_allocation (GstBaseTransform * trans,
    GstQuery * query)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (trans);
  GstCaps *caps;

  gst_query_parse_allocation (query, &caps, NULL);
  if (!caps)
    return FALSE;

  if (is_cuda_caps (caps)) {
    guint n = gst_query_get_n_allocation_pools (query);
    guint min = 0, max = 0, size;
    GstBufferPool *pool;

    if (n > 0)
      gst_query_parse_nth_allocation_pool (query, 0, NULL, NULL, &min, &max);
    pool = make_cuda_pool (self, caps, min, max, &size);
    if (!pool) {
      GST_ERROR_OBJECT (self, "failed to create CUDA buffer pool");
      return FALSE;
    }
    if (n == 0)
      gst_query_add_allocation_pool (query, pool, size, min, max);
    else
      gst_query_set_nth_allocation_pool (query, 0, pool, size, min, max);
    gst_object_unref (pool);
  }

  return GST_BASE_TRANSFORM_CLASS (parent_class)->decide_allocation (trans,
      query);
}

static GstFlowReturn
gst_cuda_v210_convert_transform (GstBaseTransform * trans, GstBuffer * inbuf,
    GstBuffer * outbuf)
{
  GstCudaV210Convert *self = GST_CUDA_V210_CONVERT (trans);
  gboolean pack = DIRECTION (self) == DIRECTION_PACK;
  GstVideoFrame in_frame, out_frame;
  gboolean in_mapped = FALSE, out_mapped = FALSE;
  guint64 in_ptr[GST_VIDEO_MAX_PLANES] = { 0, };
  gint64 in_stride[GST_VIDEO_MAX_PLANES] = { 0, };
  guint64 out_ptr[GST_VIDEO_MAX_PLANES] = { 0, };
  gint64 out_stride[GST_VIDEO_MAX_PLANES] = { 0, };
  const guint64 *other_ptr;
  const gint64 *other_stride;
  guint n_in, n_out, n_other, nx, ny, w, h;
  gboolean in_dev, out_dev;
  V210Params params = { 0, };
  GstVideoFormat other;
  CUstream stream;
  GstFlowReturn ret = GST_FLOW_ERROR;

  if (!self->configured || !self->context)
    return GST_FLOW_NOT_NEGOTIATED;
  if (!gst_cuda_context_push (self->context)) {
    GST_ELEMENT_ERROR (self, RESOURCE, FAILED,
        ("Could not push the CUDA context"), (NULL));
    return GST_FLOW_ERROR;
  }
  stream = gst_cuda_stream_get_handle (self->stream);

  /* Input: map as device memory if possible, otherwise upload via staging. */
  in_dev = can_access (self, buffer_cuda_context (inbuf));
  if (in_dev)
    sync_buffer (inbuf);
  if (!gst_video_frame_map (&in_frame, &self->in_info, inbuf,
          GST_MAP_READ | (in_dev ? GST_MAP_CUDA : 0))) {
    GST_ERROR_OBJECT (self, "failed to map input buffer");
    goto out;
  }
  in_mapped = TRUE;

  n_in = GST_VIDEO_INFO_N_PLANES (&self->in_info);
  if (in_dev) {
    for (guint i = 0; i < n_in; i++) {
      in_ptr[i] = (guintptr) GST_VIDEO_FRAME_PLANE_DATA (&in_frame, i);
      in_stride[i] = GST_VIDEO_FRAME_PLANE_STRIDE (&in_frame, i);
    }
  } else {
    if (!staging_ensure (self, self->staging_in, &self->in_info))
      goto out;
    for (guint i = 0; i < n_in; i++) {
      const Staging *dev = &self->staging_in[i];

      if (!upload (self, stream, GST_VIDEO_FRAME_PLANE_DATA (&in_frame, i),
              GST_VIDEO_FRAME_PLANE_STRIDE (&in_frame, i), dev))
        goto out;
      in_ptr[i] = dev->ptr;
      in_stride[i] = dev->pitch;
    }
  }

  /* Output: same, the other way around. */
  out_dev = can_access (self, buffer_cuda_context (outbuf));
  GST_LOG_OBJECT (self, "transform: input %s, output %s",
      in_dev ? "device" : "staged", out_dev ? "device" : "staged");
  n_out = GST_VIDEO_INFO_N_PLANES (&self->out_info);
  if (out_dev) {
    if (!gst_video_frame_map (&out_frame, &self->out_info, outbuf,
            GST_MAP_WRITE | GST_MAP_CUDA)) {
      GST_ERROR_OBJECT (self, "failed to map output buffer");
      goto out;
    }
    out_mapped = TRUE;
    for (guint i = 0; i < n_out; i++) {
      out_ptr[i] = (guintptr) GST_VIDEO_FRAME_PLANE_DATA (&out_frame, i);
      out_stride[i] = GST_VIDEO_FRAME_PLANE_STRIDE (&out_frame, i);
    }
  } else {
    if (!staging_ensure (self, self->staging_out, &self->out_info))
      goto out;
    for (guint i = 0; i < n_out; i++) {
      out_ptr[i] = self->staging_out[i].ptr;
      out_stride[i] = self->staging_out[i].pitch;
    }
  }

  if (pack) {
    params.v210 = out_ptr[0];
    params.v210_stride = out_stride[0];
    other_ptr = in_ptr;
    other_stride = in_stride;
    n_other = n_in;
    other = GST_VIDEO_INFO_FORMAT (&self->in_info);
  } else {
    params.v210 = in_ptr[0];
    params.v210_stride = in_stride[0];
    other_ptr = out_ptr;
    other_stride = out_stride;
    n_other = n_out;
    other = GST_VIDEO_INFO_FORMAT (&self->out_info);
  }
  for (guint i = 0; i < MIN (n_other, 3); i++) {
    params.plane[i] = other_ptr[i];
    params.stride[i] = other_stride[i];
  }
  w = GST_VIDEO_INFO_WIDTH (&self->in_info);
  h = GST_VIDEO_INFO_HEIGHT (&self->in_info);
  params.width = w;
  params.height = h;

  if (pack) {
    /* Every 16-byte group of the stride, so padding is zeroed. */
    nx = params.v210_stride / 16;
    ny = h;
  } else {
    nx = (w + 5) / 6;
    ny = is_420 (other) ? (h + 1) / 2 : h;
  }
  if (!launch (self, stream, nx, ny, &params))
    goto out;

  if (!out_dev) {
    if (!gst_video_frame_map (&out_frame, &self->out_info, outbuf,
            GST_MAP_WRITE)) {
      GST_ERROR_OBJECT (self, "failed to map output buffer");
      goto out;
    }
    out_mapped = TRUE;
    for (guint i = 0; i < n_out; i++) {
      if (!download (self, stream, &self->staging_out[i],
              GST_VIDEO_FRAME_PLANE_DATA (&out_frame, i),
              GST_VIDEO_FRAME_PLANE_STRIDE (&out_frame, i)))
        goto out;
    }
  }
  if (!cuda_check (self, CuStreamSynchronize (stream), "cuStreamSynchronize"))
    goto out;
  ret = GST_FLOW_OK;

out:
  if (ret != GST_FLOW_OK) {
    /* Don't unmap (and maybe free) memory the GPU may still be using. */
    CuStreamSynchronize (stream);
  }
  if (out_mapped)
    gst_video_frame_unmap (&out_frame);
  if (in_mapped)
    gst_video_frame_unmap (&in_frame);
  gst_cuda_context_pop (NULL);

  if (ret != GST_FLOW_OK) {
    GST_ELEMENT_ERROR (self, STREAM, FAILED, ("CUDA conversion failed"),
        ("see the cudav210 debug log for details"));
  }
  return ret;
}

static void
gst_cuda_v210_convert_class_init (GstCudaV210ConvertClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (klass);

  gobject_class->set_property = gst_cuda_v210_convert_set_property;
  gobject_class->get_property = gst_cuda_v210_convert_get_property;
  gobject_class->dispose = gst_cuda_v210_convert_dispose;

  g_object_class_install_property (gobject_class, PROP_DEVICE_ID,
      g_param_spec_int ("device-id", "Device ID",
          "CUDA device ID to use (-1 = auto)", -1, G_MAXINT, DEFAULT_DEVICE_ID,
          G_PARAM_READWRITE | GST_PARAM_MUTABLE_READY |
          G_PARAM_STATIC_STRINGS));

  element_class->set_context =
      GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_set_context);

  trans_class->passthrough_on_same_caps = FALSE;
  trans_class->transform_ip_on_passthrough = FALSE;
  trans_class->start = GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_start);
  trans_class->stop = GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_stop);
  trans_class->query = GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_query);
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_transform_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_set_caps);
  trans_class->get_unit_size =
      GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_get_unit_size);
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_transform_size);
  trans_class->propose_allocation =
      GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_propose_allocation);
  trans_class->decide_allocation =
      GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_decide_allocation);
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_cuda_v210_convert_transform);

  GST_DEBUG_CATEGORY_INIT (gst_cuda_v210_debug, "cudav210", 0,
      "CUDA v210 converter");
  gst_type_mark_as_plugin_api (GST_TYPE_CUDA_V210_CONVERT, 0);
}

static void
gst_cuda_v210_convert_init (GstCudaV210Convert * self)
{
  self->device_id = DEFAULT_DEVICE_ID;
}

/* ------------------------------------------------------------------------- */
/* The two elements only differ in direction, pad templates and metadata. */

typedef GstCudaV210Convert GstCudaV210Pack;
typedef GstCudaV210ConvertClass GstCudaV210PackClass;
GType gst_cuda_v210_pack_get_type (void);
G_DEFINE_TYPE (GstCudaV210Pack, gst_cuda_v210_pack,
    GST_TYPE_CUDA_V210_CONVERT);

static void
gst_cuda_v210_pack_class_init (GstCudaV210PackClass * klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  klass->direction = DIRECTION_PACK;
  gst_element_class_add_static_pad_template (element_class,
      &other_sink_template);
  gst_element_class_add_static_pad_template (element_class,
      &v210_src_template);
  gst_element_class_set_static_metadata (element_class, "CUDA v210 packer",
      "Filter/Converter/Video/Hardware",
      "Converts YUV 4:2:2 / 4:2:0 video to v210 using CUDA",
      "gst-cudav210 contributors");
}

static void
gst_cuda_v210_pack_init (GstCudaV210Pack * self)
{
}

typedef GstCudaV210Convert GstCudaV210Unpack;
typedef GstCudaV210ConvertClass GstCudaV210UnpackClass;
GType gst_cuda_v210_unpack_get_type (void);
G_DEFINE_TYPE (GstCudaV210Unpack, gst_cuda_v210_unpack,
    GST_TYPE_CUDA_V210_CONVERT);

static void
gst_cuda_v210_unpack_class_init (GstCudaV210UnpackClass * klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  klass->direction = DIRECTION_UNPACK;
  gst_element_class_add_static_pad_template (element_class,
      &v210_sink_template);
  gst_element_class_add_static_pad_template (element_class,
      &other_src_template);
  gst_element_class_set_static_metadata (element_class, "CUDA v210 unpacker",
      "Filter/Converter/Video/Hardware",
      "Converts v210 video to YUV 4:2:2 / 4:2:0 using CUDA",
      "gst-cudav210 contributors");
}

static void
gst_cuda_v210_unpack_init (GstCudaV210Unpack * self)
{
}

GST_ELEMENT_REGISTER_DEFINE (cudav210pack, "cudav210pack", GST_RANK_NONE,
    gst_cuda_v210_pack_get_type ());
GST_ELEMENT_REGISTER_DEFINE (cudav210unpack, "cudav210unpack", GST_RANK_NONE,
    gst_cuda_v210_unpack_get_type ());
