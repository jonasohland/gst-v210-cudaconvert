/* GStreamer plugin with CUDA elements converting to (cudav210pack) and from
 * (cudav210unpack) v210. */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/cuda/gstcuda.h>

#include "gstcudav210convert.h"

static gboolean
plugin_init (GstPlugin * plugin)
{
  /* Register even without a usable driver so the elements can be inspected;
   * start() reports the error. */
  if (gst_cuda_load_library ())
    gst_cuda_memory_init_once ();

  return GST_ELEMENT_REGISTER (cudav210pack, plugin) &&
      GST_ELEMENT_REGISTER (cudav210unpack, plugin);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, cudav210,
    "CUDA accelerated conversion to and from v210", plugin_init, VERSION,
    GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
