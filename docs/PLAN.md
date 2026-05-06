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

---

## Phase 4 — Improvement Plan (Pending)

This section catalogs all remaining stubs/partials and proposes actionable improvement paths. Items are categorized by feasibility.

### Category A — CUDA 13.0 API Gaps (Driver/Toolchain-Blocked)

These require either a newer CUDA version on Tegra, updated driver/firmware, or inline PTX workarounds.

| # | Suite | Current State | Root Cause | Improvement Path | Est. Effort |
|---|-------|--------------|------------|-----------------|-------------|
| A1 | `fp64_tensor` | Stub (`__CUDA_WmmaSupportDouble__` guard) | CUDA 13.0 header does not define `__CUDA_WmmaSupportDouble__`; no `nvcuda::wmma::fragment` for FP64 | Implement inline PTX `tcgen05.mma.async` for FP64 (same pattern as FP4/FP8). Need `wgmma.mma_async.z` descriptor setup + `wgmma.sync`. Benchmark: 128×128×128 tile. Peak reference: 0.126 TFLOPS. | Medium |
| A2 | `int8_tensor` | Dense: tcgen05 inline PTX (driver/firmware wall); Sparse: stub | `kind::i8` not supported by current Tegra driver/firmware. CUDA 13.0 `nvcuda::wmma` lacks INT8 fragment types. | Monitor CUDA 13.1+ Tegra driver updates. Fallback: emit `wgmma.mma_async.b16` with INT8→BF16 cast. Requires validation that `kind::i8` PTX is accepted by PTXAS but rejected at runtime vs. PTXAS rejection. | High (driver-dependent) |
| A3 | `cublas` — cuBLASLt | Stub | `cublasLtMatmul` signatures changed significantly in CUDA 13.0; Tegra `libcublasLt.so` availability uncertain | Verify `libcublasLt.so` exists on Tegra. If available, update to CUDA 13.0 `cublasLtMatmulAlgSearch_t` + `cublasLtMatrixLayout_t` API. If unavailable, keep stub with clear "not on Tegra" marker. | Low (verification) |

### Category B — Sparse Tensor Support (API-Blocked)

All sparse stubs share the same root cause: 2:4 structured sparsity requires `tcgen05.mma.sp` with sparsity metadata descriptors, which are not yet supported.

| # | Suite | Current State | Root Cause | Improvement Path | Est. Effort |
|---|-------|--------------|------------|-----------------|-------------|
| B1 | `sasp` — FP8 sparse | Stub | `cuSPARSELt` only supports INT8X4, not FP8 E4M3 | Two options: (1) Wait for cuSPARSELt FP8 support; (2) Implement manual 2:4 sparsity encoding + `tcgen05.mma.sp` inline PTX with sparsity descriptor. Option 2 is high-risk without NVIDIA guidance. | High |
| B2 | `fp8_scalar` — sparse | Stub | `tcgen05.mma.sp` requires sparsity metadata + descriptor-based layout | Same root cause as B1. Requires NVIDIA-provided sparsity encoding example. | High |
| B3 | `int8_scalar` — sparse | Stub | Same as B1/B2 | Same root cause. INT8X4 is supported by cuSPARSELt — try `cuSPARSELt` path instead of inline PTX. | Medium |

### Category C — Tegra Platform Constraints (Unresolvable on DevKit)

These are hardware/platform limitations. Stubs are appropriate and permanent on DevKit.

| # | Suite | Current State | Root Cause | Verdict |
|---|-------|--------------|------------|---------|
| C1 | `nvjpeg` | Stub | NVJPEG library not shipped on Tegra Jetson | **Permanent stub.** Consider replacing with custom CUDA JPEG kernel (libjpeg-turbo GPU port) or marking suite as "not applicable." |
| C2 | `mig` — MIG partitioning | Stub | DevKit does not support MIG; requires `nvidia-smi mig` setup | **Permanent stub on DevKit.** Works on production Thor modules with MIG-enabled firmware. |
| C3 | `tma_copy` | Fallback (cudaMalloc + cudaHostAlloc) | `cudaMemPoolCreate` unsupported on Tegra | **Permanent fallback.** TMA requires memory pools which are not supported. Current fallback measures effective bandwidth. |
| C4 | `tmem` | SMEM proxy | `tcgen05.alloc` / `tcgen05.ld` / `tcgen05.st` require SMEM descriptors | **Permanent proxy.** TMEM is internal to the Tensor Core and not directly accessible. SMEM proxy is the best available measurement. |

### Category D — Toolchain / Naming Issues (Low-Effort Fixes)

| # | Suite | Current State | Issue | Improvement Path | Est. Effort |
|---|-------|--------------|-------|-----------------|-------------|
| D1 | `arm_sve2` | NEON fallback | SVE2 intrinsics (`<arm_sve.h>`) unavailable; name is misleading | (1) Rename to `arm_neon` or `arm_cpu_vector`. (2) Add `#ifdef` for SVE2 when toolchain supports it. (3) Update README status to clarify NEON fallback. | Low |
| D2 | `cluster_sync` | ✅ (README says "cluster_barrier stub") | README line 64 says "cluster_barrier stub" but actual code implements working cluster_barrier with `__cluster_dims__(2,1,1)` annotation. PLAN.md shows 0.55ns result. | Fix README line 64: `"✅ (cluster_barrier stub)"` → `"✅ __syncthreads + cluster_barrier"`. | Trivial |
| D3 | `fp4` — sparse | Working (480 TFLOPS, 23.2%) | Sparse result is only 23.2% peak (vs 57.5% dense). May indicate suboptimal sparsity encoding or tile size. | Investigate tile sizing (M/N/K), sparsity ratio, and occupancy. Try larger matrices (m4096+). | Medium |

### Category E — Missing Metrics & Coverage Gaps

| # | Suite | Gap | Improvement Path | Est. Effort |
|---|-------|-----|-----------------|-------------|
| E1 | `fp8_scalar` | 0.04% peak — extremely low | Scalar FP8 GEMM is expected to be slow (no Tensor Core). Add context: compare against FP32 scalar baseline. Consider adding `peak_pct` reference vs FP32 scalar (not Tensor Core peak). | Low |
| E2 | `sm_compute` — FP64 | 123 GFLOPS, no `% Peak` | FP64 peak reference (0.126 TFLOPS = 126 GFLOPS) exists in PLAN.md but not computed. Add `peak_pct` calculation. | Low |
| E3 | `sm_compute` — RegPressure | 4492 GFLOPS > FP32 peak (8.064 TFLOPS) | 4492 GFLOPS is listed without `% Peak`. 4492/8064 = 55.8% — this is fine, just needs `peak_pct` added. | Low |
| E4 | `hevc_encode` / `hevc_decode` / `av1_decode` | PLAN.md shows `—` (no results) | Verify these suites produce results on live hardware. May need test bitstream files or higher resolution inputs. | Low (verification) |
| E5 | `atomic` — All ops | PLAN.md shows `—` (no results) | Verify atomic benchmark produces results. May need synchronization fix or larger workload. | Low (verification) |
| E6 | `thermal_throttle` | PLAN.md shows `—` (no result) | 60s sustained run may need tegrastats output. Verify suite completes without timeout. | Low (verification) |
| E7 | `multi_stream` | PLAN.md shows `—` (no result) | Verify multi-stream benchmark produces results on live hardware. | Low (verification) |

### Priority Recommendations

| Priority | Items | Rationale |
|----------|-------|-----------|
| **P0 — Trivial fixes** | D2 (README cluster_sync), E2/E3 (add `peak_pct`) | One-line changes, high documentation value |
| **P1 — Low effort** | D1 (rename `arm_sve2`), A3 (verify cuBLASLt), E1 (FP8 scalar context) | <1 day each |
| **P2 — Medium effort** | A1 (FP64 inline PTX), B3 (INT8 sparse via cuSPARSELt), D3 (FP4 sparse tuning) | 1-3 days each |
| **P3 — Driver-dependent** | A2 (INT8 Tensor Core), B1/B2 (FP8 sparse) | Blocked until NVIDIA updates driver/firmware |
| **P4 — Permanent** | C1-C4 | Platform constraints, stubs are appropriate |
