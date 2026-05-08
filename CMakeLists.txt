cmake_minimum_required(VERSION 3.24)
project(thor-bench VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "110a" CACHE STRING "CUDA architecture" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies ──────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(thor-probe REQUIRED)

# ── Video Codec SDK (NVENC/NVDEC) ─────────────────────────────────────
set(NVCODEC_INTERFACE_DIR "/opt/nvidia/video-codec-sdk/13.0.37/Interface")

find_library(NVENCODEAPI_LIB nvidia-encode
    PATHS /usr/lib/aarch64-linux-gnu
    DOC "NVIDIA Video Encoder shared library"
)
find_library(CUVID_LIB nvcuvid
    PATHS /usr/lib/aarch64-linux-gnu
    DOC "NVIDIA Video Decoder shared library"
)

if(NVENCODEAPI_LIB AND CUVID_LIB)
    message(STATUS "Video Codec SDK found: NVENC=${NVENCODEAPI_LIB} NVDEC=${CUVID_LIB}")
else()
    message(WARNING "Video Codec SDK not found — encode/decode suites will fall back to stub")
endif()

# Build clone3 workaround shared library
add_library(clone3_workaround SHARED src/clone3_workaround.c)
target_link_libraries(clone3_workaround PRIVATE dl)
set_target_properties(clone3_workaround PROPERTIES
    C_STANDARD 11
    POSITION_INDEPENDENT_CODE ON
)

# Build nocudaextend dlopen interceptor — LD_PRELOAD shared library.
# Blocks libnvcuextend.so from being loaded by libcuda.so at process startup.
# Usage: LD_PRELOAD=build/nocudaextend.so build/thor_bench
add_library(nocudaextend SHARED src/nocudaextend.c)
target_link_libraries(nocudaextend PRIVATE dl)
set_target_properties(nocudaextend PROPERTIES
    C_STANDARD 11
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    OUTPUT_NAME "nocudaextend"
)

# Build seccomp clone3 wrapper
add_executable(seccomp_clone3 src/seccomp_clone3.c)

# ── Source files ──────────────────────────────────────────────────────────
set(BENCH_SRCS
    src/benchmark_main.cpp
    src/bench_runner.cpp
    src/bench_suites.cpp
    src/power_monitor.cpp
    src/sweep_runner.cpp
    src/output/bench_json_serializer.cpp
    src/output/bench_text_formatter.cpp
    src/output/bench_csv_formatter.cpp
)

set(BENCH_CUDA_SRCS
    bench/suites/compute/memory_bench.cu
    bench/suites/compute/sm_compute_bench.cu
    bench/suites/compute/tensor_bench.cu
    bench/suites/compute/sasp_bench.cu
    bench/suites/compute/fp4_bench.cu
    bench/suites/compute/tmem_bench.cu
    bench/suites/compute/fp8_scalar_bench.cu
    bench/suites/compute/int8_scalar_bench.cu
    bench/suites/compute/cublas_bench.cu
    bench/suites/compute/fp64_tensor_bench.cu
    bench/suites/compute/int8_tensor_bench.cu
    bench/suites/memory/tegra_memory.cu
    bench/suites/memory/unified_memory.cu
    bench/suites/memory/tma_copy.cu
    bench/suites/memory/l2_cache.cu
    bench/suites/memory/shared_carveout_bench.cu
    bench/suites/system/host_device_transfer.cu
    bench/suites/system/thermal_throttle.cu
    bench/suites/system/multi_stream_bench.cu
    bench/suites/system/allocator_latency_bench.cu
    bench/suites/sync/mbarrier.cu
    bench/suites/sync/cluster_sync.cu
    bench/suites/sync/kernel_launch.cu
    bench/suites/sync/warp_primitives.cu
    bench/suites/sync/atomic_bench.cu
)

set(BENCH_CUPTI_SRCS
    src/cupti_profiler.cu
)

set(BENCH_CPP_SRCS
    bench/suites/encode/h264_encode_bench.cpp
    bench/suites/encode/hevc_encode_bench.cpp
    bench/suites/decode/h264_decode_bench.cpp
    bench/suites/decode/hevc_decode_bench.cpp
    bench/suites/decode/av1_decode_bench.cpp
    bench/suites/encode/nvjpeg_bench.cpp
    bench/suites/cpu/arm_compute_bench.cpp
    bench/suites/cpu/arm_neon_bench.cpp
    bench/suites/system/mig_bench.cpp
)

# ── Executable ────────────────────────────────────────────────────────────
# The real benchmark binary is named thor_bench_impl.
# A thin C wrapper (thor_bench) sets LD_PRELOAD for nocudaextend.so and execs thor_bench_impl.
add_executable(thor_bench_impl
    ${BENCH_SRCS}
    ${BENCH_CUDA_SRCS}
    ${BENCH_CPP_SRCS}
)

# ── CUPTI (optional) ──────────────────────────────────────────────────────
find_library(NVJPEG_LIB nvjpeg
    PATHS /usr/local/cuda-13.0/targets/sbsa-linux/lib
          /usr/local/cuda/targets/sbsa-linux/lib
          /usr/local/cuda/lib64
          /usr/local/cuda/lib
          /usr/lib/aarch64-linux-gnu
    PATH_SUFFIXES lib lib64 targets/sbsa-linux/lib
)

find_library(CUPTI_LIB cupti
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/targets/sbsa-linux/lib
          ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          ${CUDA_TOOLKIT_ROOT_DIR}/lib
          /usr/lib/aarch64-linux-gnu
)
find_path(CUPTI_INCLUDE cupti.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/targets/sbsa-linux/include
          ${CUDA_TOOLKIT_ROOT_DIR}/include
          /usr/include
)

if(CUPTI_LIB)
    target_sources(thor_bench_impl PRIVATE ${BENCH_CUPTI_SRCS})
    target_link_libraries(thor_bench_impl PRIVATE ${CUPTI_LIB})
    if(CUPTI_INCLUDE)
        target_include_directories(thor_bench_impl PRIVATE ${CUPTI_INCLUDE})
    endif()
    target_compile_definitions(thor_bench_impl PRIVATE HAVE_CUPTI)
else()
    message(STATUS "CUPTI not found — building with stub profiler")
    target_sources(thor_bench_impl PRIVATE src/cupti_profiler_stub.cpp)
endif()

target_include_directories(thor_bench_impl PRIVATE
    src/include
    ${CMAKE_SOURCE_DIR}/bench/suites
    ${NVCODEC_INTERFACE_DIR}
)

target_link_libraries(thor_bench_impl PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublasLt
    CUDA::cublas
    deusridet::communis
    clone3_workaround
    ${CMAKE_DL_LIBS}
)
if(NVENCODEAPI_LIB AND CUVID_LIB)
    target_link_libraries(thor_bench_impl PRIVATE ${NVENCODEAPI_LIB} ${CUVID_LIB})
    target_compile_definitions(thor_bench_impl PRIVATE HAVE_NVENCODEAPI HAVE_CUVID)
endif()
target_link_options(thor_bench_impl PRIVATE
    "LINKER:-rpath-link=${CMAKE_BINARY_DIR}"
    "LINKER:-rpath=${CMAKE_BINARY_DIR}"
)
if(NVJPEG_LIB)
    target_link_libraries(thor_bench_impl PRIVATE ${NVJPEG_LIB})
endif()

set_target_properties(thor_bench_impl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ── Thin wrapper: thor_bench → sets LD_PRELOAD → execs thor_bench_impl ──
add_executable(thor_bench src/thor_bench_wrapper.c)
target_link_libraries(thor_bench PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(thor_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
add_dependencies(thor_bench thor_bench_impl nocudaextend)

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS thor_bench DESTINATION bin)
install(FILES run_thor_bench.sh DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)
install(TARGETS nocudaextend DESTINATION lib)
install(DIRECTORY src/include/ DESTINATION include/thor-bench)
