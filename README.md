# gst-cudav210

GStreamer elements that convert to and from **v210** on NVIDIA GPUs.

| Element | Input | Output |
|---|---|---|
| `cudav210pack` | I422_10LE, UYVY, YUY2, I420_10LE, P010_10LE (CUDA or system memory) | v210 (system memory) |
| `cudav210unpack` | v210 (system memory) | I422_10LE, UYVY, YUY2, I420_10LE, P010_10LE (CUDA or system memory) |

- 4:2:2 10-bit is lossless. 8-bit formats round on the way to 8 bits and replicate bits on the way to 10.
- 4:2:0 → v210 upsamples chroma with a ¾/¼ vertical filter (MPEG-2 siting). v210 → 4:2:0 averages row pairs. 4:2:0 formats are progressive only.
- v210 is always in system memory, because gst-cuda's allocator has no v210 layout. Elements upload and download it internally.
- Property `device-id` (default -1, automatic). The CUDA context is shared with other gst-cuda/nvcodec elements.

## Build

```sh
nix develop            # CUDA 13.x toolkit
nix develop .#cuda12   # CUDA 12.9 toolkit (target dir: target/cuda12)
cargo build            # plugin ends up on GST_PLUGIN_PATH set by the shell
cargo test             # needs an NVIDIA GPU

nix build              # CUDA 13 -> result/lib/gstreamer-1.0/libgstcudav210.so
nix build .#cuda12     # CUDA 12.9
```

Kernels are compiled for sm_75/80/86/89/90/120, plus PTX for sm_120. Override this with `CUDA_ARCHS="86;120"`. The plugin doesn't link libcuda: gst-cuda loads the driver at runtime. A CUDA 12.9 build runs on 12.9+ drivers, and a CUDA 13 build needs a 13.x driver.

On non-NixOS hosts the dev shell symlinks the host's NVIDIA driver libraries into `.driver-libs/` and adds that to `LD_LIBRARY_PATH`. It also uses a private GStreamer registry, `.gst-registry.bin`.

## Examples

```sh
# GPU 4:2:0 -> v210, e.g. NVDEC output to an SDI card
gst-launch-1.0 videotestsrc ! video/x-raw,format=P010_10LE,width=3840,height=2160 \
  ! cudaupload ! "video/x-raw(memory:CUDAMemory)" ! cudav210pack ! fakesink

# v210 capture -> CUDA I422_10LE
gst-launch-1.0 videotestsrc ! video/x-raw,format=v210,width=1920,height=1080 \
  ! cudav210unpack ! "video/x-raw(memory:CUDAMemory),format=I422_10LE" ! fakesink
```

`cudaupload` prefers passthrough, so put a `(memory:CUDAMemory)` caps filter after it. Otherwise the conversion still runs on the GPU, but through an internal upload.

