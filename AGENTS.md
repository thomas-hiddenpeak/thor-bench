# thor-bench — AGENTS.md

C++20/CUDA hardware benchmark suite for NVIDIA Thor (SM110a, Blackwell). MIT-licensed.

## References

- [CUDA for Tegra AppNote](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/) — Primary source for Thor/Tegra platform constraints, memory architecture, supported/unsupported CUDA features. Consult before writing any benchmark that touches memory, system monitoring, or device capabilities.

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

Output binary: `build/thor_bench`
Requires CUDA 13.0+ and GCC 13+ on aarch64 (Jetson AGX Thor DevKit).
Requires `thor-probe` installed (via `find_package(thor-probe)`).

### Critical CMake Details

- **CUDA architecture is forced to `110a`** — same as thor-probe.
- **Depends on `find_package(thor-probe REQUIRED)` — install thor-probe first.**
- Links `thor-probe::communis` for shared utilities.
- `CMAKE_CUDA_ARCHITECTURES` is set with `FORCE`.
- Feature flags (NVENC, NVDEC, etc.) follow thor-probe config.
- Style: `.h` extension for all headers (not `.hpp`).
- Namespaces: `deusridet::bench` (bench core), `deusridet::probe` (from thor-probe).

## Run

```bash
./build/thor_bench                    # run all suites, text output
./build/thor_bench --json             # JSON output
./build/thor_bench --suites memory,compute  # run specific suites
./build/thor_bench --iterations 20    # more samples
./build/thor_bench 1                  # target CUDA device 1
```

## Development Plan

See [`docs/PLAN.md`](docs/PLAN.md) for detailed roadmap, known issues, and pending work.

## Architecture

```
src/
├── benchmark_main.cpp                # Entry point, CLI, orchestration
├── bench_runner.cpp                  # Generic runner (warmup + iterations + stats)
├── bench_suites.cpp                  # Suite registry implementation
├── cupti_profiler.{cu,h}             # CUPTI v1 Activity API profiler
├── cupti_profiler_stub.cpp           # CUPTI fallback stub
├── include/
│   ├── bench_schema.h                # BenchResult (with peak_pct), BenchReport
│   ├── bench_runner.h                # Runner interface
│   ├── bench_suites.h                # Registry + BENCH_REGISTER_SUITE macro
│   ├── bench_json_serializer.h
│   ├── bench_text_formatter.h
│   ├── bench_peaks.h                 # T5000 theoretical peaks
│   └── cupti_profiler.h              # CUPTI profiler header
└── output/
    ├── bench_json_serializer.cpp
    └── bench_text_formatter.cpp

bench/suites/
├── compute/
│   ├── memory_bench.{cu,h}           # LPDDR5X read/write/copy + shared mem crossbar
│   ├── sm_compute_bench.{cu,h}       # FP32 FMA + register spill sweep
│   ├── tensor_bench.{cu,h}           # FP16/BF16 WMMA
│   ├── sasp_bench.{cu,h}             # FP8 dense + 2:4 sparse matmul
│   ├── fp4_bench.{cu,h}              # NVFP4 dense/sparse GEMM via cublasLt
│   ├── tmem_bench.{cu,h}             # TCGen05 TMEM bandwidth (SMEM proxy)
│   ├── tcgen05_fp16_bench.{cu,h}     # TCGen05 FP16/BF16 GEMM
│   ├── tcgen05_fp8_bench.{cu,h}      # TCGen05 FP8 GEMM (scalar fallback)
│   └── int8_tensor_bench.{cu,h}      # INT8 tensor core throughput
├── memory/
│   ├── tegra_memory.{cu,h}           # Device/Pinned/Registered/Pageable
│   ├── tma_copy.{cu,h}               # TMA async copy via mempool
│   ├── unified_memory.{cu,h}         # Managed memory bandwidth
│   ├── l2_cache.{cu,h}               # L2 cache hit/miss bandwidth
│   └── shared_carveout_bench.{cu,h}  # L1/shared memory carveout
├── sync/
│   ├── mbarrier.{cu,h}               # cuda::barrier latency
│   ├── cluster_sync.{cu,h}           # __syncwarps + cluster_barrier
│   ├── kernel_launch.{cu,h}          # Kernel launch + CUDA Graph
│   └── warp_primitives.{cu,h}        # Warp shuffle, ballot, activemask
├── system/
│   ├── host_device_transfer.{cu,h}   # H2D/D2H (integrated memory)
│   ├── thermal_throttle.{cu,h}       # Sustained perf under thermal
│   ├── multi_stream_bench.{cu,h}     # Multi-stream copy benchmark
│   ├── allocator_latency_bench.{cu,h} # cudaMalloc/cudaFree latency
│   └── mig_bench.{cpp,h}             # MIG partitioning overhead
├── encode/
│   ├── h264_encode_bench.{cpp,h}     # NVENC H.264
│   └── hevc_encode_bench.{cpp,h}     # NVENC HEVC encoding
├── decode/
│   ├── h264_decode_bench.{cpp,h}     # NVDEC H.264
│   ├── hevc_decode_bench.{cpp,h}     # NVDEC HEVC decoding
│   └── av1_decode_bench.{cpp,h}      # NVDEC AV1 decoding
└── cpu/
    ├── arm_compute_bench.{cpp,h}     # ARM CPU FP32 baseline
    └── arm_sve2_bench.{cpp,h}        # ARM NEON fallback benchmark
```

## Design Principles

- **Statistical rigor**: Median + p95/p99 + stddev (no averages — outliers skew too much).
- **Warmup**: Fixed warmup rounds before actual measurements.
- **Scoring**: Raw values + normalized score (reference device T5000 = 1000).
- **Output**: JSON (machine-readable) + text (human-readable), consistent with thor-probe.
- **Suite registration**: Static registration via macro, no manual list management.

## No Test Infrastructure

No tests, CI, linters, or formatters yet. Greenfield.

## Thor Platform Constraints

Source: [CUDA for Tegra AppNote](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/)

- **`cudaMemGetInfo()` is unreliable** — does not account for SWAP. Use `/proc/meminfo` to estimate available memory.
- **P2P not supported** — do not use `cudaPeerGetStatus`, `cudaDeviceEnablePeerAccess`, etc.
- **NVML not available** — use `tegrastats` for system monitoring / thermal data.
- **JIT compilation not supported** — no `cudaModuleLoadDataEx` with JIT options.
- **nvGRAPH, cuSOLVER not supported.**
- **CUB is experimental only** — avoid production reliance.

### Memory Architecture (Tegra SoC)

- Device, Host, and Unified memory all reside on the same physical SoC DRAM.
- **Sysmem Full Coherency (Thor-only)**: `cudaHostRegister()` enables GPU L2 caching. `pageableMemoryAccess=1` allows direct GPU access to pageable memory.
- **Unified Memory on Thor**: `concurrentManagedAccess=1`, UVM selects GPU uncached + IO coherent. Use `cudaMemPrefetchAsync()` to prefetch pages.

### Memory Selection Guidance for Benchmarks

| Use case | Recommended type |
|----------|-----------------|
| GPU-only workloads | Device memory |
| Small buffers | Pinned memory |
| CPU + GPU shared | Unified / Registered Host memory |

## T5000 Theoretical Peak Reference

Source: [NVIDIA Jetson T5000 Module Data Sheet](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/thor/docs/jetson_t5000_modules.pdf), [CUDA for Tegra AppNote](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/), [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/), PTXAS/CICC reverse engineering references.

| Metric | Formula / Source | Peak Value |
|---|---|---|
| FP32 (CUDA Cores) | `2560 cores × 2 FLOP/cycle (FMA) × 1.575 GHz` | **8.064 TFLOPS** |
| FP4 Dense | Data Sheet | **1035 TFLOPS** |
| FP4 Sparse (2:4) | Data Sheet | **2070 TFLOPS** |
| FP8 Dense | Data Sheet | **517 TFLOPS** |
| FP8 Sparse (2:4) | Data Sheet | **1035 TFLOPS** |
| FP16/BF16 Tensor Core | 5th-gen TC, `wgmma.mma_async.sp` | Benchmark empirical |
| LPDDR5X Bandwidth | 256-bit @ 4266 MHz | **273 GB/s** |
| Shared Memory per SM | 228 KB max configurable (L1+Shared=256 KB total) | Benchmark empirical |
| L2 Cache | ~50 MB estimated | Benchmark empirical |
| Registers per SM | 65,536 × 32-bit | — |
| Max regs/thread | 255 | — |
| Max warps/SM | 64 | — |
| Max threads/CTA | 1024 | — |
| Max CTAs/SM | 32 | — |
| GPUs | 3 GPC, 10 TPC, 20 SM (2 SM/TPC) | — |
| Tensor Cores | 96 total (4/SM), 5th-gen with tcgen05/TMEM | — |

> ⚠️ **FP4/FP8 peaks are in TFLOPS, not GFLOPS**. The original spec circulated with ~7.5 TFLOPS for FP4 was off by ~2 orders of magnitude. All `% of peak` calculations must use the values above.
>
> FP16/BF16 Tensor Core peak is not explicitly stated in the datasheet — benchmark empirically and report `% of peak` against FP32 CUDA cores as fallback.

## Style Notes

- Headers use `.h` extension (not `.hpp`).
- Everything lives in `deusridet::bench` namespace.
- CMake uses explicit source lists (no GLOB).
- Follows thor-probe conventions exactly.
