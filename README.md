# thor-bench

C++20/CUDA hardware benchmark suite for NVIDIA Thor (SM110a, Blackwell).

Built on [thor-probe](https://github.com/thomas-hiddenpeak/thor-probe) for hardware detection and system probing.

## Architecture

thor-bench provides a structured benchmarking framework with statistical analysis:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  benchmark_main  │───→│   BenchRunner     │───→│   Suite functions │
│  (CLI, orchestr.)│     │ (warmup, iter, stats)│ │ (memory, compute, ...)│
│                  │     │                  │     │                  │
│  sweep_runner    │───→│   SweepRunner     │───→│   Sweep suites   │
│  (param matrix)  │     │ (param sweep)    │     │ (tile size, etc) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                         ┌─────────────┐
                         │ BenchResult  │
                         │ (med,p95,p99,σ,peak_pct) │
                         └─────────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
              ┌───────────┐        ┌───────────┐
              │  JSON     │        │  Text     │
              │Serializer │        │Formatter  │
              └───────────┘        └───────────┘
                     │
              ┌───────────┐
              │  CSV      │
              │Serializer │
              └───────────┘
```

**Key components:**

| Component | File | Description |
|-----------|------|-------------|
| `BenchRunner` | `src/bench_runner.cpp` | Warmup→iterations→stats pipeline with timeout support |
| `BenchSuiteRegistry` | `src/include/bench_suites.h` | Singleton registry with `BENCH_REGISTER_SUITE` macro |
| `BenchResult` | `src/include/bench_schema.h` | Statistical result: median, stddev, p95, p99, peak_pct |
| `SweepRunner` | `src/sweep_runner.cpp` | Parameter matrix sweep runner (tile size, etc.) |
| `PowerMonitor` | `src/power_monitor.cpp` | Real-time power/thermal monitoring via tegrastats |
| JSON serializer | `src/output/bench_json_serializer.cpp` | Structured JSON output |
| Text formatter | `src/output/bench_text_formatter.cpp` | Human-readable output with ANSI colors |
| CSV serializer | `src/output/bench_csv_formatter.cpp` | CSV output for sweep results |
| CUPTI profiler | `src/cupti_profiler.cu` | CUPTI v1 Activity API (optional, `--cupti`) |

**Context Recovery:** `benchmark_main.cpp` includes a `cudaDeviceReset()` + retry loop after `cudaErrorIllegalInstruction` to prevent tcgen05 probe failures from poisoning the entire benchmark run.

## Benchmark Suites

All suites report `peak_pct` — percentage of T5000 theoretical maximum. See [AGENTS.md](AGENTS.md) for peak reference values.

### GPU Compute

| Suite | Domain | Metrics | Status |
|-------|--------|---------|--------|
| `memory` | LPDDR5X bandwidth | GB/s (read/write/copy) + shared mem crossbar | ✅ |
| `sm_compute` | SM FP32/FP64 | GFLOP/s (FMA + register pressure) | ✅ |
| `tensor` | FP16/BF16 WMMA (tcgen05.mma) | TFLOP/s | ✅ |
| `sasp` | FP8 dense + 2:4 sparse | TFLOP/s | ✅ FP8 dense (scalar); ⛔ FP8 sparse (stub) |
| `fp4` | NVFP4 dense/sparse GEMM | TFLOP/s (via tcgen05.mma inline PTX) | ✅ Dense; ⛔ Sparse (stub) |
| `fp8_scalar` | Scalar FP8 GEMM (no Tensor Core) | TFLOP/s | ✅ Dense; ⛔ Sparse (stub) |
| `int8_scalar` | Scalar INT8 GEMM (no Tensor Core) | TOP/s | ✅ Dense; ⛔ Sparse (stub) |
| `tmem` | TCGen05 TMEM bandwidth | GB/s | ⛔ stub (IllegalInstruction on driver 595.58.03) |
| `cublas` | cuBLAS SGEMM/DGEMM | TFLOP/s | ✅ SGEMM/DGEMM/strided-batched |
| `fp64_tensor` | DMMA FP64 (mma.sync.aligned) | TFLOP/s | ✅ |
| `int8_tensor` | INT8 Tensor Core (tcgen05.mma.kind::i8) | TOP/s | ⛔ stub (IllegalInstruction on driver 595.58.03) |
| `mig` | MIG partitioning | TFLOP/s | ✅ Full GPU; ⛔ MIG partition (stub, DevKit) |

### Memory

| Suite | Domain | Metrics | Status |
|-------|--------|---------|--------|
| `tegra_memory` | SoC memory architecture | GB/s (Device/Pinned/Registered/Pageable) | ✅ |
| `tma_copy` | TMA async copy | GB/s (H2D/D2H/D2D) | ✅ Fallback — mempool unsupported, uses cudaMalloc |
| `unified_memory` | Managed memory bandwidth | GB/s (read/write via `cudaMemPrefetchAsync()`) | ✅ |
| `l2_cache` | L2 cache hit/miss bandwidth | GB/s (hit/miss) | ✅ |
| `shared_carveout` | L1/shared memory carveout ratio | GB/s (carveout 0–100) | ✅ |

### Sync

| Suite | Domain | Metrics | Status |
|-------|--------|---------|--------|
| `mbarrier` | cuda::barrier latency | ns | ✅ 6 tests (64-1024 threads + syncthreads baseline) |
| `cluster_sync` | Cluster sync latency | ns (__syncthreads) | ✅ __syncthreads + cluster_barrier |
| `kernel_launch` | Kernel launch + CUDA Graph | µs | ✅ |
| `warp_primitives` | Warp shuffle, ballot, activemask | ns | ✅ |
| `atomic` | Atomic op latency (Add/CAS/Max/Min) | ns | ✅ |

### Encode/Decode

| Suite | Domain | Metrics | Status |
|-------|--------|---------|--------|
| `h264_encode` | NVENC H.264 encoding | FPS (720p/1080p/4K) | ✅ |
| `h264_decode` | NVDEC H.264 decoding | FPS (1080p/4K) | ⛔ stub (needs bitstream files) |
| `hevc_encode` | NVENC HEVC encoding | FPS (720p/1080p/4K) | ✅ |
| `hevc_decode` | NVDEC HEVC decoding | FPS (1080p/4K) | ⛔ stub (needs bitstream files) |
| `av1_decode` | NVDEC AV1 decoding | FPS (1080p/4K) | ⛔ stub (needs bitstream files) |
| `nvjpeg` | NVJPEG encode/decode | FPS | ⛔ stub (NVJPEG not available on Tegra) |

### CPU

| Suite | Domain | Metrics | Status |
|-------|--------|---------|--------|
| `arm_compute` | CPU FP32 baseline | GFLOP/s (NEON/SVE, multi-threaded) | ✅ |
| `arm_neon` | ARM CPU NEON vector | GFLOP/s (FP32/FP16/INT8) | ✅ |

### System

| Suite | Domain | Metrics | Status |
|-------|--------|---------|--------|
| `host_device_transfer` | Host↔Device transfer | GB/s (integrated SoC memory) | ✅ |
| `thermal_throttle` | Sustained FP32 under thermal | GFLOP/s (15s run) | ✅ |
| `multi_stream` | Multi-stream copy | GB/s (concurrent streams) | ✅ |
| `allocator_latency` | cudaMalloc/cudaFree latency | µs median + allocs/s | ✅ |

## Prerequisites

- NVIDIA Jetson AGX Thor DevKit (aarch64)
- CUDA 13.0+
- GCC 13+
- thor-probe installed (`sudo make install` from thor-probe build)
- NVIDIA Video Codec SDK 13.0+ (for NVENC/NVDEC suites)

### Known Driver Issue

NVIDIA 595.58.03 `libnvcuextend.so` has a `.init` constructor that SEGVs on startup. A workaround `nocudaextend.so` dlopen interceptor is shipped. To use:

```bash
export LD_PRELOAD=$(dirname $(readlink -f $0))/nocudaextend.so
./build/thor_bench
```

Or use the `run_thor_bench.sh` wrapper script which sets `LD_PRELOAD` automatically.

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

Output binary: `build/thor_bench`

## Usage

```bash
./build/thor_bench                               # run all suites, text output
./build/thor_bench --json                        # JSON output
./build/thor_bench --suites memory,tensor        # run specific suites
./build/thor_bench --iterations 20               # more samples
./build/thor_bench --warmup 5                    # warmup runs per test
./build/thor_bench --timeout 60                  # per-suite timeout (seconds)
./build/thor_bench --device 1                    # target CUDA device
./build/thor_bench --cupti                       # enable CUPTI activity profiling
./build/thor_bench --sweep                       # run in sweep mode (parameter matrix)
./build/thor_bench --sweep --csv                 # sweep + CSV output
./build/thor_bench --help                        # show help
```

## Live Benchmark Results

Captured on Jetson AGX Thor DevKit (T5000), driver 595.58.03, CUDA 13.0, Video Codec SDK 13.0.37. **95/95 tests pass.**

### GPU Compute

| Test | Median | % Peak |
|------|--------|--------|
| FP32 GEMM (MIG Full GPU) | **5.64 TFLOP/s** | 70.0% |
| FP64 WMMA Dense | 0.09 TFLOP/s | 70.0% |
| FP8 Scalar Dense | 0.22 TFLOP/s | 0.04% |
| INT8 Scalar Dense | 0.22 TOP/s | 2.7% |
| FP32 Sustained (15s thermal) | 3.66 TFLOP/s | 45.4% |
| FP32 Thermal sustain rate | **101.5%** | No throttling |

### Memory Bandwidth

| Test | Median | % Peak |
|------|--------|--------|
| LPDDR5X L2 sequential read | 128.8 GB/s | 52.3% |
| LPDDR5X L2 sequential write | 165.8 GB/s | 67.4% |
| LPDDR5X L2 strided read | 96.2 GB/s | 39.1% |
| LPDDR5X L2 random read | 4.8 GB/s | 2.0% |
| Shared Memory (carveout 100) | ~268 GB/s | — |
| Unified Memory read | 131.2 GB/s | 48.1% |
| Unified Memory write | 215.9 GB/s | 79.1% |
| H2D transfer | 99.0 GB/s | 36.3% |
| D2H transfer | 98.0 GB/s | 35.9% |
| Multi-stream (8 stream) | 211.7 GB/s | 77.5% |

### NVENC Encoding

| Test | Median (fps) | Stddev |
|------|--------|--------|
| H.264 720p | 1,560 | 3.6 |
| H.264 1080p | 726 | 0.95 |
| H.264 4K | 189 | 0.07 |
| HEVC 720p | 1,122 | 2.0 |
| HEVC 1080p | 528 | 0.36 |
| HEVC 4K | 265 | 0.11 |

### Sync & Launch

| Test | Median |
|------|--------|
| __syncthreads (1024 threads) | 0.26 ns |
| cluster_barrier (2SM) | 0.54 ns |
| cuda::barrier (1024 threads) | 0.78 ns |
| Kernel launch (empty, 1 thread) | 4.77 µs |
| CUDA Graph replay (warm) | 4.80 µs |
| Warp shfl | 0.08 ns |
| cudaMalloc | 73.0 µs |

### CPU (ARM)

| Test | Median |
|------|--------|
| ARM FP32 matmul 512 | 2.05 GFLOP/s |
| ARM NEON FP32 | 9.99 GFLOP/s |
| ARM NEON FP16 | 13.53 GFLOP/s |

## Stubs & Limitations

| Suite | Reason |
|-------|--------|
| `tmem` (4 tests) | `tcgen05.alloc/mma` causes IllegalInstruction on driver 595.58.03. TMEM block_scale requires full CUTLASS-style scale factor pipeline. |
| `fp4` sparse | `kind::mxf4nvf4.block_scale.block16` sparse — requires scale factor infrastructure not yet implemented. |
| `sasp` FP8 sparse | `kind::f8f6f4` IllegalInstruction — poisons CUDA context. |
| `fp8_scalar` sparse | `tcgen05.mma.sp` IllegalInstruction — poisons CUDA context. |
| `int8_scalar` sparse | `kind::i8` IllegalInstruction — poisons CUDA context. |
| `int8_tensor` | `kind::i8` IllegalInstruction — poisons CUDA context. |
| NVDEC (H.264/HEVC/AV1) | Requires real bitstream files for decoding. Hardware capability detected but no reference streams included. |
| NVJPEG (encode/decode) | NVJPEG is not available on Tegra. |
| MIG partitioning | Requires `nvidia-smi mig` setup — not available on DevKit firmware. |

## License

MIT
