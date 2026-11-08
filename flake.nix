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
        ];

        cudaDeps = cudaPkgs: [
          cudaPkgs.cuda_nvcc
          cudaPkgs.cuda_cudart
          cudaPkgs.cccl
        ];

        # Build the plugin against one CUDA toolkit.
        mkPlugin =
          cudaPkgs:
          pkgs.rustPlatform.buildRustPackage.override { stdenv = cudaPkgs.backendStdenv; } {
            pname = "gst-cudav210";
            version = "0.1.0";
            src = pkgs.lib.cleanSource ./.;
            cargoLock.lockFile = ./Cargo.lock;
            nativeBuildInputs = [
              pkgs.pkg-config
              cudaPkgs.cuda_nvcc
            ];
            # Headers for nvcc (cuda_runtime.h etc.) must be buildInputs.
            buildInputs = gstDeps ++ [
              cudaPkgs.cuda_cudart
              cudaPkgs.cccl
            ];
            # Tests need a GPU; run them from the dev shell instead.
            doCheck = false;
            postInstall = ''
              mkdir -p $out/lib/gstreamer-1.0
              mv $out/lib/libgstcudav210.so $out/lib/gstreamer-1.0/
              rm -f $out/lib/libgstcudav210.rlib
            '';
          };

        # Each CUDA version gets its own cargo target dir: build.rs can't see
        # which nvcc is on PATH, so a shared one would mix up fatbins.
        mkShell =
          targetDir: cudaPkgs:
          (pkgs.mkShell.override { stdenv = cudaPkgs.backendStdenv; }) {
            packages =
              with pkgs;
              [
                rustc
                cargo
                clippy
                rustfmt
                rust-analyzer
                pkg-config
                gst.gst-plugins-good
              ]
              ++ gstDeps
              ++ cudaDeps cudaPkgs;
            shellHook = ''
              # Don't let a host CUDA install (e.g. /opt/cuda on Arch) leak in.
              unset NVCC_CCBIN CUDA_PATH CUDA_HOME
              export CARGO_TARGET_DIR="$PWD/${targetDir}"
              export GST_PLUGIN_PATH="$CARGO_TARGET_DIR/debug"
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
          default = mkShell "target" pkgs.cudaPackages_13;
          cuda13 = mkShell "target" pkgs.cudaPackages_13;
          cuda12 = mkShell "target/cuda12" pkgs.cudaPackages_12_9;
        };
      }
    );
}
