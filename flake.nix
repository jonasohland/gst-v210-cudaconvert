{
  description = "pixel format conversion to/from v210 on cuda";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        gst = pkgs.gst_all_1;

        gstDeps = [
          gst.gstreamer
          gst.gst-plugins-base
          gst.gst-plugins-bad
          # gst-cuda's headers include <GL/gl.h>, and its pkg-config file
          # requires glesv2/opengl.
          pkgs.libGL
        ];

        buildTools = [
          pkgs.meson
          pkgs.ninja
          pkgs.pkg-config
          pkgs.python3
        ];

        cudaDeps = cudaPkgs: [
          cudaPkgs.cuda_nvcc
          cudaPkgs.cuda_cudart
          cudaPkgs.cccl
        ];

        # Build the plugin against one CUDA toolkit.
        mkPlugin =
          cudaPkgs:
          cudaPkgs.backendStdenv.mkDerivation {
            pname = "gst-cudav210";
            version = "0.1.0";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = buildTools ++ [ cudaPkgs.cuda_nvcc ];
            # Headers for nvcc and gst-cuda (cuda.h etc.) must be buildInputs.
            buildInputs = gstDeps ++ [
              cudaPkgs.cuda_cudart
              cudaPkgs.cccl
            ];
            # Tests need a GPU; run them from the dev shell instead.
            mesonFlags = [ "-Dtests=disabled" ];
          };

        # Each CUDA version gets its own meson build dir, since meson records
        # the nvcc it found at setup time.
        mkShell =
          buildDir: cudaPkgs:
          (pkgs.mkShell.override { stdenv = cudaPkgs.backendStdenv; }) {
            packages =
              with pkgs;
              [
                clang-tools
                gst.gst-plugins-good
              ]
              ++ buildTools
              ++ gstDeps
              ++ cudaDeps cudaPkgs;
            shellHook = ''
              # Don't let a host CUDA install (e.g. /opt/cuda on Arch) leak in.
              unset NVCC_CCBIN CUDA_PATH CUDA_HOME
              export GST_PLUGIN_PATH="$PWD/${buildDir}"
              # Separate registry so the host's GStreamer cache doesn't interfere.
              export GST_REGISTRY="$PWD/.gst-registry.bin"

              # On non-NixOS hosts, expose only the NVIDIA driver libraries
              # (not all of /usr/lib, which would clash with Nix's glibc).
              if [ ! -d /run/opengl-driver/lib ] && [ -e /usr/lib/libcuda.so.1 ]; then
                mkdir -p "$PWD/.driver-libs"
                ln -sf /usr/lib/libcuda.so* /usr/lib/libnvcuvid.so* /usr/lib/libnvidia-*.so* "$PWD/.driver-libs/" 2>/dev/null
                export LD_LIBRARY_PATH="$PWD/.driver-libs''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              fi
            '';
          };
      in
      {
        packages = {
          default = mkPlugin pkgs.cudaPackages_13;
          cuda13 = mkPlugin pkgs.cudaPackages_13;
          cuda12 = mkPlugin pkgs.cudaPackages_12_9;
        };
        devShells = {
          default = mkShell "build" pkgs.cudaPackages_13;
          cuda13 = mkShell "build" pkgs.cudaPackages_13;
          cuda12 = mkShell "build-cuda12" pkgs.cudaPackages_12_9;
        };
      }
    );
}
