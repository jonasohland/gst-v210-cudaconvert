use std::env;
use std::path::PathBuf;
use std::process::Command;

// Default GPU targets: SASS for Turing..Blackwell consumer/datacenter parts plus
// PTX for the newest one so future GPUs can JIT. All of these are supported by
// both CUDA 12.9 and CUDA 13.x. Override with CUDA_ARCHS="75;86;120".
const DEFAULT_ARCHS: &[&str] = &["75", "80", "86", "89", "90", "120"];

fn main() {
    gst_plugin_version_helper::info();

    // Only the library directory is needed. A full probe would also resolve
    // gstreamer-cuda's GL dependencies (glesv2, egl, ...), which we don't use.
    let libdir = pkg_config::get_variable("gstreamer-cuda-1.0", "libdir")
        .expect("gstreamer-cuda-1.0 not found (part of gst-plugins-bad)");
    println!("cargo:rustc-link-search=native={libdir}");

    let src = "kernels/v210.cu";
    println!("cargo:rerun-if-changed={src}");
    println!("cargo:rerun-if-env-changed=CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=NVCC");

    let archs_env = env::var("CUDA_ARCHS").ok();
    let archs: Vec<&str> = match &archs_env {
        Some(s) => s.split([';', ',', ' ']).filter(|s| !s.is_empty()).collect(),
        None => DEFAULT_ARCHS.to_vec(),
    };
    let newest = archs
        .iter()
        .max_by_key(|a| a.parse::<u32>().unwrap_or(0))
        .expect("CUDA_ARCHS is empty");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("v210.fatbin");
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let mut cmd = Command::new(&nvcc);
    cmd.args(["-fatbin", "-O3", "-std=c++17"]);
    for a in &archs {
        cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
    }
    cmd.arg(format!(
        "-gencode=arch=compute_{newest},code=compute_{newest}"
    ));
    cmd.arg("-o").arg(&out).arg(src);

    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to run {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed: {cmd:?}");
}
