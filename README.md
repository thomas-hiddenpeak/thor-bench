# thor-bench

C++20/CUDA hardware benchmark suite for NVIDIA Thor (SM110a, Blackwell).

Built on [thor-probe](https://github.com/thomas-hiddenpeak/thor-probe) for hardware detection and system probing.

## Architecture

thor-bench provides a structured benchmarking framework with statistical analysis:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  benchmark_main  │───→│   BenchRunner     │───→│   Suite functions │
│  (CLI, orchestr.)│     │ (warmup, iter, stats)│ │ (memory, compute, ...)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌─────────────┐
                        │ BenchResult  │
                        │ (mean,med,σ,p95,p99) │
                        └─────────────┘
                              │
                         ┌────┴────┐
                         │         │
                    ┌─────────┐ ┌──────────┐
                    │  JSON   │ │  Text    │
                    │Serializer│ │Formatter │
                    └─────────┘ └──────────┘
```

**Key components:**

| Component | File | Description |
|-----------|------|-------------|
| `BenchRunner` | `src/bench_runner.cpp` | Warmup→iterations→stats pipeline with timeout support |
| `BenchSuiteRegistry` | `src/include/bench_suites.h` | Singleton registry with `BENCH_REGISTER_SUITE` macro |
| `BenchResult` | `src/include/bench_schema.h` | Statistical result: mean, median, stddev, p95, p99, min, max |
| JSON serializer | `src/output/bench_json_serializer.cpp` | Structured JSON output |
| Text formatter | `src/output/bench_text_formatter.cpp` | Human-readable output with ANSI colors |

## Benchmark Suites

All suites report `peak_pct` — percentage of T5000 theoretical maximum. See [AGENTS.md](AGENTS.md) for peak reference values.

| Suite | Domain | Metrics | Status |
|-------|--------|---------|--------|
| `memory` | LPDDR5X bandwidth | GB/s (read/write/copy) + shared mem crossbar | ✅ |
| `sm_compute` | SM FP32/FP64 | GFLOP/s (FMA + register pressure) | ✅ |
| `tensor` | FP16/BF16 WMMA (tcgen05.mma) | TFLOP/s | ✅ |
| `sasp` | FP8 dense + 2:4 sparse | TFLOP/s | ✅ FP8 dense (scalar); sparse stub (needs tcgen05) |
| `tegra_memory` | SoC memory architecture | GB/s (Device/Pinned/Registered/Pageable) | ✅ |
| `tma_copy` | TMA async copy | GB/s (H2D/D2H/D2D) | ✅ Fallback — mempool unsupported, uses cudaMalloc |
| `unified_memory` | Managed memory bandwidth | GB/s (read/write via `cudaMemPrefetchAsync()`) | ✅ |
| `l2_cache` | L2 cache hit/miss bandwidth | GB/s (hit/miss) | ✅ |
| `shared_carveout` | L1/shared memory carveout ratio | GB/s (carveout 0–100) | ✅ |
| `fp4` | NVFP4 dense/sparse GEMM | TFLOP/s (via cublasLt) | ✅ |
| `fp8_scalar` | Scalar FP8 GEMM (no Tensor Core) | TFLOP/s | ⚠️ Scalar fallback; sparse stub |
| `int8_scalar` | Scalar INT8 GEMM (no Tensor Core) | TOP/s | ⚠️ Scalar fallback; sparse stub |
| `tmem` | TCGen05 TMEM bandwidth | GB/s | ⚠️ SMEM proxy (tcgen05 ld/st requires SMEM descriptors) |
| `cublas` | cuBLAS SGEMM/DGEMM | TFLOP/s (strided batched) | ✅ cuBLASLt stub (CUDA 13.0 API changed) |
| `fp64_tensor` | WMMA FP64 | TFLOP/s | ⚠️ Stub (CUDA 13.0 `__CUDA_WmmaSupportDouble__` guard) |
| `int8_tensor` | INT8 Tensor Core WMMA | TOP/s | ⚠️ Stub (CUDA 13.0 `nvcuda::wmma` INT8 incomplete) |
| `mbarrier` | cuda::barrier latency | ns | ✅ 6 tests (64-1024 threads + syncthreads baseline) |
| `cluster_sync` | Cluster sync latency | ns (__syncthreads) | ✅ (cluster_barrier stub) |
| `kernel_launch` | Kernel launch + CUDA Graph | µs | ✅ |
| `warp_primitives` | Warp shuffle, ballot, activemask | ns | ✅ |
| `atomic` | Atomic op latency (Add/CAS/Max/Min) | ns | ✅ |
| `h264_encode` | NVENC H.264 encoding | FPS (1080p/4K) | ✅ |
| `h264_decode` | NVDEC H.264 decoding | FPS (1080p/4K) | ✅ |
| `hevc_encode` | NVENC HEVC encoding | FPS (1080p/4K) | ✅ |
| `hevc_decode` | NVDEC HEVC decoding | FPS (1080p/4K) | ✅ |
| `av1_decode` | NVDEC AV1 decoding | FPS (1080p/4K) | ✅ |
| `nvjpeg` | NVJPEG encode/decode | FPS | ⚠️ Stub (NVJPEG not available on Tegra) |
| `arm_compute` | CPU FP32 baseline | GFLOP/s (NEON/SVE, multi-threaded) | ✅ |
| `arm_sve2` | ARM CPU NEON fallback | GFLOP/s (FP32/FP16/INT8) | ⚠️ NEON fallback (SVE2 intrinsics unavailable) |
| `host_device_transfer` | Host↔Device transfer | GB/s (integrated SoC memory) | ✅ |
| `thermal_throttle` | Sustained FP32 under thermal | GFLOP/s (60s run) | ✅ |
| `multi_stream` | Multi-stream copy | GB/s (concurrent streams) | ✅ |
| `allocator_latency` | cudaMalloc/cudaFree latency | µs median + allocs/s | ✅ |
| `mig` | MIG partitioning | GFLOP/s | ⚠️ Full GPU only; MIG partition stub (DevKit) |

## Prerequisites

- NVIDIA Jetson AGX Thor DevKit (aarch64)
- CUDA 13.0+
- GCC 13+
- thor-probe installed (`sudo make install` from thor-probe build)

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

## Usage

```bash
./build/thor_bench                           # run all suites, text output
./build/thor_bench --json                    # JSON output
./build/thor_bench --suites memory,tensor    # run specific suites
./build/thor_bench --iterations 20           # more samples
./build/thor_bench --warmup 5                # warmup runs per test
./build/thor_bench --timeout 60              # per-suite timeout (seconds)
./build/thor_bench --device 1                # target CUDA device
```

## License

MIT
