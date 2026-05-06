cmake_minimum_required(VERSION 3.24)
project(thor-bench VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "110a" CACHE STRING "CUDA architecture" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies ──────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(thor-probe REQUIRED)

# ── Source files ──────────────────────────────────────────────────────────
set(BENCH_SRCS
    src/benchmark_main.cpp
    src/bench_runner.cpp
    src/bench_suites.cpp
    src/output/bench_json_serializer.cpp
    src/output/bench_text_formatter.cpp
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
add_executable(thor_bench
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
    target_sources(thor_bench PRIVATE ${BENCH_CUPTI_SRCS})
    target_link_libraries(thor_bench PRIVATE ${CUPTI_LIB})
    if(CUPTI_INCLUDE)
        target_include_directories(thor_bench PRIVATE ${CUPTI_INCLUDE})
    endif()
    target_compile_definitions(thor_bench PRIVATE HAVE_CUPTI)
else()
    message(STATUS "CUPTI not found — building with stub profiler")
    target_sources(thor_bench PRIVATE src/cupti_profiler_stub.cpp)
endif()

target_include_directories(thor_bench PRIVATE
    src/include
    ${CMAKE_SOURCE_DIR}/bench/suites
)

target_link_libraries(thor_bench PRIVATE
    deusridet::communis
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublasLt
    CUDA::cublas
)
if(NVJPEG_LIB)
    target_link_libraries(thor_bench PRIVATE ${NVJPEG_LIB})
endif()

set_target_properties(thor_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS thor_bench DESTINATION bin)
install(DIRECTORY src/include/ DESTINATION include/thor-bench)
