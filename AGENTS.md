# thor-bench — AGENTS.md

C++20/CUDA hardware benchmark suite for NVIDIA Thor (SM110a, Blackwell). MIT-licensed.

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
- `CMAKE_CUDA_ARCHITECTURES is set with `FORCE`.
- Feature flags (NVENC, NVDEC, etc. follow thor-probe config.
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

## Architecture

```
src/
├── benchmark_main.cpp       # Entry point, CLI parsing, orchestration
├── include/
│   ├── bench_schema.h       # Result structures (BenchResult, BenchReport)
│   ├── bench_runner.h       # Generic runner (warmup + iterations + stats)
│   ├── bench_suites.h       # Suite registry + registration macro
│   ├── bench_json_serializer.h
│   └── bench_text_formatter.h
├── bench_runner.cpp
└── output/
    ├── bench_json_serializer.cpp
    └── bench_text_formatter.cpp

bench/suites/
├── compute/     # memory_bench, sm_compute_bench, tensor_bench (.cu)
├── encode/      # h264_encode_bench (.cpp)
├── decode/      # h264_decode_bench (.cpp)
├── cpu/         # arm_compute_bench (.cpp)
└── system/      # pcie_bench (.cu)
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

## Style Notes

- Headers use `.h` extension (not `.hpp`).
- Everything lives in `deusridet::bench` namespace.
- CMake uses explicit source lists (no GLOB).
- Follows thor-probe conventions exactly.
