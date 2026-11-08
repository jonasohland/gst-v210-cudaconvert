use gst::glib;
use gst::prelude::*;

mod imp;

glib::wrapper! {
    pub struct Element(ObjectSubclass<imp::CudaV210Pack>) @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "cudav210pack",
        gst::Rank::NONE,
        Element::static_type(),
    )
}
