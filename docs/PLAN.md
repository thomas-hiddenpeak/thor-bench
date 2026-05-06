# thor-bench — Development Plan

Last updated: 2026-05-06

## Live Benchmark Results (Thor DevKit, CUDA 13.0.48)

### GPU Compute
| Suite | Result | % Peak |
|-------|--------|--------|
| `sm_compute` — FP32 FMA | 3134 GFLOPS | 38.9% |
| `sm_compute` — FP64 | 123 GFLOPS | — |
| `sm_compute` — RegPressure | 4492 GFLOPS | 55.8% |
| `tensor` — FP16 WMMA | 402 TFLOPS | — |
| `tensor` — BF16 WMMA | 400 TFLOPS | — |
| `fp8_scalar` — FP8 dense (scalar) | 0.22 TFLOP/s | 0.04% |
| `int8_scalar` — INT8 dense (scalar) | 0.22 TOP/s | — |
| `fp4` — NVFP4 dense (m2048) | 595 TFLOPS | 57.5% |
| `fp4` — NVFP4 sparse (m2048) | 480 TFLOPS | 23.2% |
| `cublas` — SGEMM FP32 | — | — |
| `fp64_tensor` — WMMA FP64 | — | — |
| `int8_tensor` — INT8 TC | — | — |
| `mig` — full GPU FP32 GEMM | 5.65 TFLOPS | 70.1% |

### Memory
| Suite | Result | % Peak |
|-------|--------|--------|
| `memory` — LPDDR5X read | 145 GB/s | 53.3% |
| `memory` — LPDDR5X write | 170 GB/s | 62.3% |
| `memory` — LPDDR5X copy | 212 GB/s | 77.8% |
| `tegra_memory` — Registered write | 183 GB/s | 66.9% |
| `tegra_memory` — Pageable write | 85 GB/s | 31.3% |
| `tma_copy` — H2D (fallback) | 147 GB/s | 53.9% |
| `tma_copy` — D2D (fallback) | 198 GB/s | 72.5% |
| `host_device_transfer` — H2D/D2H | ~110 GB/s | — |
| `unified_memory` — Managed read | 128 GB/s | 46.9% |
| `unified_memory` — Managed write | 132 GB/s | 48.3% |
| `l2_cache` — L2 read | 204 GB/s | — |
| `l2_cache` — L2 write | 209 GB/s | — |
| `shared_carveout` — carveout 100 | 126 GB/s | — |
| `shared_carveout` — carveout 0 | 132 GB/s | — |

### Sync
| Suite | Result |
|-------|--------|
| `mbarrier` — 64-1024 threads | 0.70-0.78 ns |
| `cluster_sync` — __syncthreads | 0.15-0.28 ns |
| `cluster_sync` — cluster_barrier (2SM) | 0.55 ns |
| `kernel_launch` — empty | 4.6 µs |
| `kernel_launch` — CUDA Graph replay | 4.7 µs |
| `warp_primitives` — shfl | 0.07 ns |
| `warp_primitives` — ballot | 0.08 ns |
| `atomic` — Add int | — |
| `atomic` — Add float | — |
| `atomic` — CAS | — |
| `atomic` — Max | — |
| `atomic` — Min | — |

### Encoder/Decoder
| Suite | Result |
|-------|--------|
| `h264_encode` — 1080p | 32K fps |
| `h264_decode` — 1080p | 40K fps |
| `hevc_encode` — 1080p | — |
| `hevc_decode` — 1080p | — |
| `av1_decode` — 1080p | — |
| `nvjpeg` — encode/decode | — (stub, not on Tegra) |

### CPU
| Suite | Result |
|-------|--------|
| `arm_compute` — FP32 matmul_512 | 1.24 GFLOP/s |
| `arm_sve2` — FP32 NEON | 1.38 GFLOP/s |
| `arm_sve2` — FP16 NEON | 1.65 GFLOP/s |
| `arm_sve2` — INT8 NEON | 1.04 GOP/s |

### System
| Suite | Result |
|-------|--------|
| `thermal_throttle` — sustained FP32 | — |
| `multi_stream` — 8-stream copy | — |
| `allocator_latency` — malloc | 83 µs median |
| `allocator_latency` — free | 87 µs median |
| `allocator_latency` — concurrent | 17964 allocs/s |
| `tmem` — SMEM proxy read | 1.53 GB/s |
| `tmem` — SMEM proxy write | 1.53 GB/s |

## Completed

### Phase 1 — Memory & Sync ✅
| Suite | Status | Notes |
|-------|--------|-------|
| `memory` | ✅ Done | LPDDR5X read/write/copy + shared mem crossbar, peak_pct vs 273 GB/s |
| `tegra_memory` | ✅ Done | Device/Pinned/Registered/Pageable (4 types × 2 directions = 8 tests) |
| `tma_copy` | ✅ Fallback | cudaMemPoolCreate unsupported → falls back to cudaMalloc + cudaHostAlloc |
| `unified_memory` | ✅ Done | cudaMemPrefetchAsync managed memory benchmarks |
| `l2_cache` | ✅ Done | L2 cache hit/miss bandwidth |
| `shared_carveout` | ✅ Done | L1/shared memory carveout ratio benchmarks |
| `mbarrier` | ✅ Done | `cuda::barrier<thread_scope_block>` (6 tests) |
| `cluster_sync` | ✅ Done | __syncthreads (5 sizes) + cluster_barrier (`cg::cluster_group::sync()`) |
| `kernel_launch` | ✅ Done | Empty/small launch + CUDA Graph capture/replay/warm (5 tests) |
| `warp_primitives` | ✅ Done | shfl, ballot, activemask |
| `atomic` | ✅ Done | Atomic op latency (Add int/Add float/CAS/Max/Min, 5 tests) |
| `host_device_transfer` | ✅ Done | H2D/D2H with metadata["integrated"]="true" |

### Phase 2 — Compute ✅
| Suite | Status | Notes |
|-------|--------|-------|
| `sm_compute` | ✅ Done | FP32 FMA + FP64 FMA + register pressure kernel |
| `tensor` | ✅ Done | FP16 WMMA ✅; BF16 via FP16 reinterpret (400 TFLOP/s) |
| `sasp` | ✅ Partial | FP8 dense ✅ (scalar); sparse stub — requires tcgen05.mma.sp |
| `fp8_scalar` | ✅ Partial | Scalar FP8 dense (0.22 TFLOP/s); sparse stub |
| `int8_scalar` | ✅ Partial | Scalar INT8 dense (0.22 TOP/s); sparse stub |
| `fp4` | ✅ Done | NVFP4 dense/sparse via cublasLt |
| `cublas` | ✅ Partial | SGEMM/DGEMM/strided-batched SGEMM working; cuBLASLt stub (CUDA 13.0 API) |
| `fp64_tensor` | ⚠️ Stub | WMMA FP64 guarded on `__CUDA_WmmaSupportDouble__` |
| `int8_tensor` | ⚠️ Stub | INT8 TC (CUDA 13.0 `nvcuda::wmma` INT8 incomplete) |

### Phase 3 — Sustained ✅
| Suite | Status | Notes |
|-------|--------|-------|
| `thermal_throttle` | ✅ Done | Sustained FP32 FMA (60s run), tegrastats GPU temp/clock |
| `multi_stream` | ✅ Done | Multi-stream copy benchmark |

### Encode/Decode/CPU ✅
| Suite | Status | Notes |
|-------|--------|-------|
| `h264_encode` | ✅ Done | NVENC 1920×1080 |
| `h264_decode` | ✅ Done | NVDEC 1920×1080 |
| `hevc_encode` | ✅ Done | NVENC HEVC 1920×1080 |
| `hevc_decode` | ✅ Done | NVDEC HEVC 1920×1080 |
| `av1_decode` | ✅ Done | NVDEC AV1 1920×1080 |
| `nvjpeg` | ⚠️ Stub | NVJPEG not available on Tegra → stub |
| `arm_compute` | ✅ Done | ARM FP32 NEON/SVE, 4 threads |
| `arm_sve2` | ✅ Done | NEON fallback (SVE2 intrinsics unavailable), FP32/FP16/INT8 |

### System ✅
| Suite | Status | Notes |
|-------|--------|-------|
| `allocator_latency` | ✅ Done | cudaMalloc/Free latency + concurrent alloc throughput |
| `mig` | ✅ Partial | Full GPU FP32 GEMM (5.65 TFLOPS, 70.1% peak); MIG partition stub (DevKit) |
| `tmem` | ⚠️ Proxy | TMEM benchmarks use SMEM proxy (tcgen05.alloc/ld/st requires SMEM descriptors) |

### Infrastructure ✅
| Feature | Status | Notes |
|---------|--------|-------|
| `BenchResult` schema | ✅ | median/p95/p99/stddev + `peak_pct` + `metadata` map |
| `bench_peaks.h` | ✅ | T5000 theoretical peaks (FP32/FP4/FP8/LPDDR5X) |
| JSON + text serializers | ✅ | metadata, peak_pct support |
| Static suite registration | ✅ | `BENCH_REGISTER_SUITE` macro |
| CUPTI v1 profiler | ✅ | `--cupti` flag, Activity API, buffer callbacks, per-suite ranges |

## Known Issues

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | `sasp_fp8_sparse`: 2:4 sparse requires tcgen05.mma.sp with sparsity metadata descriptor → stub | Medium | Resolved (stub) |
| 2 | `fp8_scalar_sparse`: Same as above → stub | Medium | Resolved (stub) |
| 3 | `int8_scalar_sparse`: INT8 2:4 sparse requires tcgen05.mma.sp → stub | Medium | Resolved (stub) |
| 4 | `mig_0_4tpc`, `mig_1_6tpc`: MIG partitioning requires nvidia-smi setup → stub | Low | Resolved (stub, DevKit limitation) |
| 5 | `tmem`: TMEM benchmarks use SMEM proxy (tcgen05.alloc/ld/st requires SMEM descriptors) | Low | Resolved (proxy) |
| 6 | `fp8_scalar` / `int8_scalar`: Scalar kernels (tcgen05.mma PTX requires descriptor-based layout) | Low | Resolved (scalar) |
| 7 | `arm_sve2`: SVE2 intrinsics (`<arm_sve.h>`) unavailable → NEON fallback | Low | Resolved (fallback) |
| 8 | `cublas_lt`: cuBLASLt API changed in CUDA 13.0 → stub | Low | Resolved (stub) |
| 9 | `fp64_tensor`: WMMA FP64 guarded on `__CUDA_WmmaSupportDouble__` → stub when unavailable | Low | Resolved (stub) |
| 10 | `int8_tensor`: `nvcuda::wmma` INT8 incomplete in CUDA 13.0 → stub | Low | Resolved (stub) |
| 11 | `nvjpeg`: NVJPEG not available on Tegra → stub | Low | Resolved (stub) |

## T5000 Theoretical Peaks

| Metric | Peak | Benchmark |
|--------|------|-----------|
| FP32 | 8.064 TFLOPS | sm_compute, mig, cublas |
| FP64 | 0.126 TFLOPS | sm_compute, fp64_tensor |
| FP4 Dense | 1035 TFLOPS | fp4 |
| FP4 Sparse | 2070 TFLOPS | fp4 |
| FP8 Dense | 517 TFLOPS | sasp, fp8_scalar |
| FP8 Sparse | 1035 TFLOPS | sasp, fp8_scalar |
| INT8 (scalar fallback) | 8.064 TFLOPS | int8_scalar, int8_tensor |
| LPDDR5X | 273 GB/s | memory, tegra_memory, tma_copy |
| L2 Cache | 246 GB/s | l2_cache |
| Shared Mem | 228 KB/SM | memory (shared crossbar) |
| Registers | 65,536/SM | sm_compute (reg spill) |

## Suite Count

**Total: 34 suites, 185 tests**

| Category | Count |
|----------|-------|
| GPU Compute | 11 |
| Memory | 5 |
| Sync | 5 |
| Encode/Decode | 6 |
| CPU | 2 |
| System | 5 |
