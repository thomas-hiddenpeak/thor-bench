cmake_minimum_required(VERSION 3.24)
project(thor-bench VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 110a FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies ──────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(thor-probe REQUIRED)

# ── Source files ──────────────────────────────────────────────────────────
set(BENCH_SRCS
    src/benchmark_main.cpp
    src/bench_runner.cpp
    src/output/bench_json_serializer.cpp
    src/output/bench_text_formatter.cpp
)

set(BENCH_CUDA_SRCS
    bench/suites/compute/memory_bench.cu
    bench/suites/compute/sm_compute_bench.cu
    bench/suites/compute/tensor_bench.cu
    bench/suites/system/pcie_bench.cu
)

set(BENCH_CPP_SRCS
    bench/suites/encode/h264_encode_bench.cpp
    bench/suites/decode/h264_decode_bench.cpp
    bench/suites/cpu/arm_compute_bench.cpp
)

# ── Executable ────────────────────────────────────────────────────────────
add_executable(thor_bench
    ${BENCH_SRCS}
    ${BENCH_CUDA_SRCS}
    ${BENCH_CPP_SRCS}
)

target_include_directories(thor_bench PRIVATE
    src/include
    ${CMAKE_SOURCE_DIR}/bench/suites
)

target_link_libraries(thor_bench PRIVATE
    thor-probe::communis
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(thor_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS thor_bench DESTINATION bin)
install(DIRECTORY src/include/ DESTINATION include/thor-bench)
