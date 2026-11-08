project('gst-cudav210', 'c',
  version: '0.1.0',
  license: 'MPL-2.0',
  meson_version: '>= 1.1',
  default_options: ['c_std=gnu11', 'warning_level=2', 'buildtype=debugoptimized'])

fs = import('fs')
cc = meson.get_compiler('c')
# GObject vfuncs often ignore some of their arguments.
add_project_arguments(cc.get_supported_arguments('-Wno-unused-parameter'), language: 'c')

gst_req = '>= 1.22'
gst_dep = dependency('gstreamer-1.0', version: gst_req)
gstbase_dep = dependency('gstreamer-base-1.0', version: gst_req)
gstvideo_dep = dependency('gstreamer-video-1.0', version: gst_req)
gstcuda_dep = dependency('gstreamer-cuda-1.0', version: gst_req)

# The kernels are compiled to a fatbin and embedded in the plugin. Default
# targets: SASS for Turing..Blackwell plus PTX for the newest one, so future
# GPUs can JIT. All of these work with both CUDA 12.9 and 13.x.
nvcc = find_program(get_option('nvcc'))
archs = get_option('gpu_archs')
if archs.length() == 0
  error('gpu_archs must not be empty')
endif
newest = archs[0]
foreach a : archs
  if a.to_int() > newest.to_int()
    newest = a
  endif
endforeach
gencode = []
foreach a : archs
  gencode += '-gencode=arch=compute_@0@,code=sm_@0@'.format(a)
endforeach
gencode += '-gencode=arch=compute_@0@,code=compute_@0@'.format(newest)

fatbin = custom_target('v210.fatbin',
  input: 'kernels/v210.cu',
  output: 'v210.fatbin',
  depend_files: 'kernels/v210-params.h',
  command: [nvcc, '-fatbin', '-O3', '-std=c++17', gencode, '-o', '@OUTPUT@', '@INPUT@'])

python = find_program('python3')
fatbin_c = custom_target('v210-fatbin.c',
  input: fatbin,
  output: 'v210-fatbin.c',
  command: [python, files('scripts/embed.py'), '@INPUT@', '@OUTPUT@', 'gst_cuda_v210_fatbin'])

# gst-cuda's headers include <cuda.h>. Nix puts it on the include path; other
# installs (e.g. /opt/cuda) keep it next to nvcc.
cuda_inc = []
if not cc.has_header('cuda.h')
  inc = fs.parent(fs.parent(nvcc.full_path())) / 'include'
  cuda_inc = include_directories(inc, is_system: true)
  if not cc.has_header('cuda.h', include_directories: cuda_inc)
    error('cuda.h not found (looked in @0@)'.format(inc))
  endif
endif

cdata = configuration_data()
cdata.set_quoted('PACKAGE', meson.project_name())
cdata.set_quoted('VERSION', meson.project_version())
cdata.set_quoted('GST_LICENSE', 'MPL')
cdata.set_quoted('GST_PACKAGE_NAME', meson.project_name())
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://example.invalid/gst-cudav210')
configure_file(output: 'config.h', configuration: cdata)

plugin = shared_library('gstcudav210',
  'src/plugin.c',
  'src/gstcudav210convert.c',
  fatbin_c,
  c_args: ['-DHAVE_CONFIG_H', '-DGST_USE_UNSTABLE_API'],
  include_directories: [include_directories('kernels'), cuda_inc],
  dependencies: [gst_dep, gstbase_dep, gstvideo_dep, gstcuda_dep],
  gnu_symbol_visibility: 'hidden',
  install: true,
  install_dir: get_option('libdir') / 'gstreamer-1.0')

if not get_option('tests').disabled()
  gstcheck_dep = dependency('gstreamer-check-1.0', version: gst_req, required: get_option('tests'))
  gstapp_dep = dependency('gstreamer-app-1.0', version: gst_req, required: get_option('tests'))
  if gstcheck_dep.found() and gstapp_dep.found()
    elements_test = executable('elements',
      'tests/elements.c',
      'tests/reference.c',
      dependencies: [gst_dep, gstvideo_dep, gstcheck_dep, gstapp_dep])
    # Tests need an NVIDIA GPU and the nvcodec plugin (cudaupload/cudadownload).
    test_env = environment()
    test_env.set('GST_PLUGIN_PATH', meson.current_build_dir())
    test_env.set('GST_REGISTRY', meson.current_build_dir() / 'test-registry.bin')
    test_env.set('CK_DEFAULT_TIMEOUT', '600')
    test('elements', elements_test, env: test_env, depends: plugin, timeout: 900)
  endif
endif
