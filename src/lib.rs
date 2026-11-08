//! GStreamer plugin with CUDA elements converting to (`cudav210pack`) and from
//! (`cudav210unpack`) v210.

use gst::glib;

mod convert;
pub mod cuda;
pub mod format;
mod pack;
mod unpack;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    // Register even without a usable driver so the elements can be inspected;
    // start() reports the error.
    if cuda::load_library() {
        unsafe { cuda::ffi::gst_cuda_memory_init_once() };
    }
    pack::register(plugin)?;
    unpack::register(plugin)?;
    Ok(())
}

gst::plugin_define!(
    cudav210,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "MPL",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
