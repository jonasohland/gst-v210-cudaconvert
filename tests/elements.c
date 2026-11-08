/* Element tests. They need an NVIDIA GPU and the nvcodec plugin
 * (cudaupload/cudadownload). Run with `meson test -C build`. */

#include <gst/check/gstcheck.h>
#include <gst/app/app.h>
#include <gst/video/video.h>

#include "reference.h"

static const guint sizes[][2] = {
  {1, 1},
  {7, 5},
  {48, 4},
  {1918, 1081},
  {1920, 1080},
  {1922, 3},
};

/* cudaupload/cudadownload (gst_cuda_buffer_copy) copy only width*2 bytes per
 * row for packed 4:2:2, losing the last macropixel's second half at odd
 * widths. That's an upstream bug, so avoid odd widths there. */
static gboolean
works_through_cudaupload (GstVideoFormat f, guint w)
{
  return !(ref_is_8bit (f) && w % 2 == 1);
}

static void
make_info (GstVideoInfo * info, GstVideoFormat f, guint w, guint h)
{
  fail_unless (gst_video_info_set_format (info, f, w, h));
  GST_VIDEO_INFO_FPS_N (info) = 30;
  GST_VIDEO_INFO_FPS_D (info) = 1;
}

typedef void (*FillFunc) (const Native * n, GstVideoFrame * frame);

static GstBuffer *
make_buffer (GstVideoInfo * info, FillFunc fill, const Native * n)
{
  GstBuffer *buf = gst_buffer_new_allocate (NULL, GST_VIDEO_INFO_SIZE (info),
      NULL);
  GstVideoFrame frame;

  /* Garbage in all padding, so tests catch reads/writes of it. */
  gst_buffer_memset (buf, 0, 0xa5, GST_VIDEO_INFO_SIZE (info));
  fail_unless (gst_video_frame_map (&frame, info, buf, GST_MAP_WRITE));
  fill (n, &frame);
  gst_video_frame_unmap (&frame);
  return buf;
}

/* A mapped output frame. */
typedef struct
{
  GstSample *sample;
  GstVideoInfo info;
  GstVideoFrame frame;
} Output;

static void
output_clear (Output * o)
{
  gst_video_frame_unmap (&o->frame);
  gst_sample_unref (o->sample);
}

/* Runs one buffer through `appsrc ! {mid} ! appsink` and maps the output. */
static void
run (const gchar * mid, GstVideoInfo * in_info, GstBuffer * buf, Output * out)
{
  gchar *desc =
      g_strdup_printf ("appsrc name=src format=time ! %s ! appsink name=sink "
      "sync=false", mid);
  GError *err = NULL;
  GstElement *pipeline, *src, *sink;
  GstCaps *caps;
  GstSample *sample;

  pipeline = gst_parse_launch (desc, &err);
  fail_unless (pipeline && !err, "%s: %s", desc, err ? err->message : "");
  src = gst_bin_get_by_name (GST_BIN (pipeline), "src");
  sink = gst_bin_get_by_name (GST_BIN (pipeline), "sink");

  caps = gst_video_info_to_caps (in_info);
  gst_app_src_set_caps (GST_APP_SRC (src), caps);
  gst_caps_unref (caps);
  fail_unless (gst_element_set_state (pipeline, GST_STATE_PLAYING) !=
      GST_STATE_CHANGE_FAILURE);
  fail_unless_equals_int (gst_app_src_push_buffer (GST_APP_SRC (src), buf),
      GST_FLOW_OK);
  fail_unless_equals_int (gst_app_src_end_of_stream (GST_APP_SRC (src)),
      GST_FLOW_OK);

  sample = gst_app_sink_pull_sample (GST_APP_SINK (sink));
  if (!sample) {
    GstBus *bus = gst_element_get_bus (pipeline);
    GstMessage *msg;

    while ((msg = gst_bus_pop_filtered (bus, GST_MESSAGE_ERROR))) {
      GError *e = NULL;
      gchar *dbg = NULL;

      gst_message_parse_error (msg, &e, &dbg);
      g_printerr ("pipeline error: %s (%s)\n", e->message, dbg ? dbg : "");
      g_clear_error (&e);
      g_free (dbg);
      gst_message_unref (msg);
    }
    gst_object_unref (bus);
  }
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (src);
  gst_object_unref (sink);
  gst_object_unref (pipeline);
  fail_unless (sample, "no output from `%s`", desc);
  g_free (desc);

  out->sample = sample;
  fail_unless (gst_video_info_from_caps (&out->info,
          gst_sample_get_caps (sample)));
  fail_unless (gst_video_frame_map (&out->frame, &out->info,
          gst_sample_get_buffer (sample), GST_MAP_READ));
}

/* Compares full rows of the (default) v210 stride, which the packer
 * zero-pads. */
static gboolean
v210_rows_equal (const GstVideoFrame * a, const GstVideoFrame * b)
{
  guint w = GST_VIDEO_FRAME_WIDTH (a), h = GST_VIDEO_FRAME_HEIGHT (a);
  gsize row = (w + 47) / 48 * 128;

  if (w != (guint) GST_VIDEO_FRAME_WIDTH (b) ||
      h != (guint) GST_VIDEO_FRAME_HEIGHT (b))
    return FALSE;
  for (guint y = 0; y < h; y++) {
    const guint8 *ra = (const guint8 *) GST_VIDEO_FRAME_PLANE_DATA (a, 0) +
        (gsize) y *GST_VIDEO_FRAME_PLANE_STRIDE (a, 0);
    const guint8 *rb = (const guint8 *) GST_VIDEO_FRAME_PLANE_DATA (b, 0) +
        (gsize) y *GST_VIDEO_FRAME_PLANE_STRIDE (b, 0);

    if (memcmp (ra, rb, row) != 0)
      return FALSE;
  }
  return TRUE;
}

static void
check_pack (GstVideoFormat f, const gchar * mid, gboolean cudaupload)
{
  for (guint i = 0; i < G_N_ELEMENTS (sizes); i++) {
    guint w = sizes[i][0], h = sizes[i][1];
    GstVideoInfo in_info, v210_info;
    GstVideoFrame want;
    GstBuffer *expected;
    Output out;

    if (cudaupload && !works_through_cudaupload (f, w))
      continue;

    g_autoptr (Native) src = native_random (f, w, h, i);
    g_autoptr (Native) p = native_to_422_10 (src, f);

    make_info (&in_info, f, w, h);
    run (mid, &in_info, make_buffer (&in_info, native_store, src), &out);
    fail_unless_equals_string (GST_VIDEO_INFO_NAME (&out.info), "v210");

    make_info (&v210_info, GST_VIDEO_FORMAT_v210, w, h);
    expected = make_buffer (&v210_info, v210_encode, p);
    fail_unless (gst_video_frame_map (&want, &v210_info, expected,
            GST_MAP_READ));
    fail_unless (v210_rows_equal (&out.frame, &want),
        "%s %ux%u via `%s`: v210 output differs from reference",
        gst_video_format_to_string (f), w, h, mid);
    gst_video_frame_unmap (&want);
    gst_buffer_unref (expected);
    output_clear (&out);
  }
}

static void
check_unpack (GstVideoFormat f, const gchar * mid, gboolean cudaupload)
{
  for (guint i = 0; i < G_N_ELEMENTS (sizes); i++) {
    guint w = sizes[i][0], h = sizes[i][1];
    GstVideoInfo in_info;
    Output out;

    if (cudaupload && !works_through_cudaupload (f, w))
      continue;

    g_autoptr (Native) p =
        native_random (GST_VIDEO_FORMAT_I422_10LE, w, h, 100 + i);
    g_autoptr (Native) want = native_from_422_10 (p, f);

    make_info (&in_info, GST_VIDEO_FORMAT_v210, w, h);
    run (mid, &in_info, make_buffer (&in_info, v210_encode, p), &out);
    fail_unless_equals_int (GST_VIDEO_INFO_FORMAT (&out.info), f);

    g_autoptr (Native) got = native_load (&out.frame);
    fail_unless (native_equal (got, want),
        "%s %ux%u via `%s`: output differs from reference",
        gst_video_format_to_string (f), w, h, mid);
    output_clear (&out);
  }
}

GST_START_TEST (pack_sysmem)
{
  for (guint f = 0; f < ref_n_formats; f++)
    check_pack (ref_formats[f], "cudav210pack ! video/x-raw,format=v210",
        FALSE);
}

GST_END_TEST;

GST_START_TEST (pack_cuda_in)
{
  for (guint f = 0; f < ref_n_formats; f++)
    check_pack (ref_formats[f], "cudaupload ! video/x-raw(memory:CUDAMemory) "
        "! cudav210pack ! video/x-raw,format=v210", TRUE);
}

GST_END_TEST;

GST_START_TEST (unpack_sysmem)
{
  for (guint f = 0; f < ref_n_formats; f++) {
    g_autofree gchar *mid =
        g_strdup_printf ("cudav210unpack ! video/x-raw,format=%s",
        gst_video_format_to_string (ref_formats[f]));
    check_unpack (ref_formats[f], mid, FALSE);
  }
}

GST_END_TEST;

GST_START_TEST (unpack_cuda_out)
{
  for (guint f = 0; f < ref_n_formats; f++) {
    g_autofree gchar *mid =
        g_strdup_printf ("cudav210unpack ! video/x-raw(memory:CUDAMemory),"
        "format=%s ! cudadownload ! video/x-raw",
        gst_video_format_to_string (ref_formats[f]));
    check_unpack (ref_formats[f], mid, TRUE);
  }
}

GST_END_TEST;

/* CUDA memory in, v210 (system memory) between the elements, CUDA memory
 * out. */
GST_START_TEST (cuda_chain)
{
  for (guint fi = 0; fi < ref_n_formats; fi++) {
    GstVideoFormat f = ref_formats[fi];
    const gchar *name = gst_video_format_to_string (f);
    g_autofree gchar *mid =
        g_strdup_printf ("cudaupload ! video/x-raw(memory:CUDAMemory) ! "
        "cudav210pack ! cudav210unpack ! video/x-raw(memory:CUDAMemory),"
        "format=%s ! cudadownload ! video/x-raw,format=%s", name, name);

    for (guint i = 0; i < G_N_ELEMENTS (sizes); i++) {
      guint w = sizes[i][0], h = sizes[i][1];
      GstVideoInfo in_info;
      Output out;

      if (!works_through_cudaupload (f, w))
        continue;

      g_autoptr (Native) src = native_random (f, w, h, 200 + i);
      g_autoptr (Native) p = native_to_422_10 (src, f);
      g_autoptr (Native) want = native_from_422_10 (p, f);

      make_info (&in_info, f, w, h);
      run (mid, &in_info, make_buffer (&in_info, native_store, src), &out);
      g_autoptr (Native) got = native_load (&out.frame);
      fail_unless (native_equal (got, want), "%s %ux%u via `%s`", name, w, h,
          mid);
      output_clear (&out);
    }
  }
}

GST_END_TEST;

GST_START_TEST (lossless_422_roundtrip)
{
  for (guint i = 0; i < G_N_ELEMENTS (sizes); i++) {
    guint w = sizes[i][0], h = sizes[i][1];
    GstVideoInfo in_info;
    Output out;
    g_autoptr (Native) src =
        native_random (GST_VIDEO_FORMAT_I422_10LE, w, h, 7);

    make_info (&in_info, GST_VIDEO_FORMAT_I422_10LE, w, h);
    run ("cudav210pack ! cudav210unpack ! video/x-raw,format=I422_10LE",
        &in_info, make_buffer (&in_info, native_store, src), &out);
    g_autoptr (Native) got = native_load (&out.frame);
    fail_unless (native_equal (got, src), "%ux%u", w, h);
    output_clear (&out);
  }
}

GST_END_TEST;

/* Cross-check against GStreamer's CPU v210 support in videoconvert. */
GST_START_TEST (matches_videoconvert)
{
  for (guint i = 0; i < G_N_ELEMENTS (sizes); i++) {
    guint w = sizes[i][0], h = sizes[i][1];
    GstVideoInfo in_info, v_info;
    GstBuffer *buf;
    Output ours, theirs;
    g_autoptr (Native) src =
        native_random (GST_VIDEO_FORMAT_I422_10LE, w, h, 9);

    make_info (&in_info, GST_VIDEO_FORMAT_I422_10LE, w, h);
    buf = make_buffer (&in_info, native_store, src);
    run ("cudav210pack ! video/x-raw,format=v210", &in_info,
        gst_buffer_ref (buf), &ours);
    run ("videoconvert ! video/x-raw,format=v210", &in_info, buf, &theirs);
    {
      g_autoptr (Native) a = v210_decode (&ours.frame);
      g_autoptr (Native) b = v210_decode (&theirs.frame);
      fail_unless (native_equal (a, b), "pack %ux%u", w, h);
    }
    output_clear (&ours);
    output_clear (&theirs);

    make_info (&v_info, GST_VIDEO_FORMAT_v210, w, h);
    buf = make_buffer (&v_info, v210_encode, src);
    run ("cudav210unpack ! video/x-raw,format=I422_10LE", &v_info,
        gst_buffer_ref (buf), &ours);
    run ("videoconvert ! video/x-raw,format=I422_10LE", &v_info, buf, &theirs);
    {
      g_autoptr (Native) a = native_load (&ours.frame);
      g_autoptr (Native) b = native_load (&theirs.frame);
      fail_unless (native_equal (a, b), "unpack %ux%u", w, h);
    }
    output_clear (&ours);
    output_clear (&theirs);
  }
}

GST_END_TEST;

static gboolean
upstream_query (GstPad * pad, GstObject * parent, GstQuery * query)
{
  if (GST_QUERY_TYPE (query) != GST_QUERY_CAPS)
    return FALSE;
  gst_query_set_caps_result (query, g_object_get_data (G_OBJECT (pad),
          "caps"));
  return TRUE;
}

/* Calls the element's transform_caps (sink -> src) via a caps query on its
 * src pad while the sink pad is linked to a pad returning `caps`. */
static GstCaps *
transform (GstElement * elem, GstCaps * caps)
{
  GstPad *up = gst_pad_new ("up", GST_PAD_SRC);
  GstPad *sink = gst_element_get_static_pad (elem, "sink");
  GstPad *src = gst_element_get_static_pad (elem, "src");
  GstCaps *out;

  g_object_set_data_full (G_OBJECT (up), "caps", gst_caps_ref (caps),
      (GDestroyNotify) gst_caps_unref);
  gst_pad_set_query_function (up, upstream_query);
  fail_unless_equals_int (gst_pad_link (up, sink), GST_PAD_LINK_OK);
  out = gst_pad_query_caps (src, NULL);
  gst_pad_unlink (up, sink);
  gst_object_unref (src);
  gst_object_unref (sink);
  gst_object_unref (up);
  return out;
}

static gboolean
caps_has_format (GstCaps * caps, const gchar * format)
{
  for (guint i = 0; i < gst_caps_get_size (caps); i++) {
    const GValue *v =
        gst_structure_get_value (gst_caps_get_structure (caps, i), "format");

    if (!v)
      continue;
    if (GST_VALUE_HOLDS_LIST (v)) {
      for (guint j = 0; j < gst_value_list_get_size (v); j++) {
        if (g_str_equal (g_value_get_string (gst_value_list_get_value (v, j)),
                format))
          return TRUE;
      }
    } else if (G_VALUE_HOLDS_STRING (v)
        && g_str_equal (g_value_get_string (v), format)) {
      return TRUE;
    }
  }
  return FALSE;
}

static GstCaps *
v210_caps (const gchar * mode)
{
  return gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "v210",
      "width", G_TYPE_INT, 1920, "height", G_TYPE_INT, 1080,
      "interlace-mode", G_TYPE_STRING, mode, NULL);
}

/* 4:2:0 output is only offered for progressive input. */
GST_START_TEST (interlaced_420_not_offered)
{
  GstElement *unpack = gst_element_factory_make ("cudav210unpack", NULL);
  GstCaps *in, *out;

  fail_unless (unpack);

  in = v210_caps ("progressive");
  out = transform (unpack, in);
  fail_unless (caps_has_format (out, "I420_10LE"));
  gst_caps_unref (out);
  gst_caps_unref (in);

  in = v210_caps ("interleaved");
  out = transform (unpack, in);
  fail_unless (caps_has_format (out, "I422_10LE"));
  fail_if (caps_has_format (out, "I420_10LE"));
  fail_if (caps_has_format (out, "P010_10LE"));
  gst_caps_unref (out);
  gst_caps_unref (in);

  gst_object_unref (unpack);
}

GST_END_TEST;

static Suite *
cudav210_suite (void)
{
  Suite *s = suite_create ("cudav210");
  TCase *tc = tcase_create ("elements");

  suite_add_tcase (s, tc);
  tcase_add_test (tc, pack_sysmem);
  tcase_add_test (tc, pack_cuda_in);
  tcase_add_test (tc, unpack_sysmem);
  tcase_add_test (tc, unpack_cuda_out);
  tcase_add_test (tc, cuda_chain);
  tcase_add_test (tc, lossless_422_roundtrip);
  tcase_add_test (tc, matches_videoconvert);
  tcase_add_test (tc, interlaced_420_not_offered);
  return s;
}

GST_CHECK_MAIN (cudav210);
