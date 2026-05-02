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

| Suite | Domain | Metrics | Implementation |
|-------|--------|---------|----------------|
| `memory` | HBM bandwidth | GB/s (read/write/copy) | CUDA kernels + events |
| `compute` | SM FP32/FP64 | GFLOP/s | FMA kernels, parameterized block sizes |
| `tensor` | Tensor Core WMMA | TFLOP/s (FP16/BF16) | WMMA 16\u00d716\u00d716 tiles |
| `pcie` | Host\u2194Device transfer | GB/s | Pinned memory + async copies |
| `h264_encode` | NVENC encoding | FPS | CUDA YUV simulation (NVENC API stub) |
| `h264_decode` | NVDEC decoding | FPS | CUDA YUV simulation (NVDEC API stub) |
| `arm_compute` | CPU FP32 | GFLOP/s | NEON/SVE intrinsics, multi-threaded |

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
