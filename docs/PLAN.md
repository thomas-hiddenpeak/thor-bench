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
| `arm_neon` — FP32 NEON | 1.38 GFLOP/s |
| `arm_neon` — FP16 NEON | 1.65 GFLOP/s |
| `arm_neon` — INT8 NEON | 1.04 GOP/s |

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
| `tma_copy` | ✅ Fallback | cudaMemPoolCreate NOT in unsupported list; TMA PTX supported. Upgrade to runtime check (`cudaDevAttrMemoryPoolsSupported`). |
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
| `cublas` | ✅ Partial | SGEMM/DGEMM/strided-batched SGEMM working; cuBLASLt stub (should be replaceable — API unchanged) |
| `fp64_tensor` | ⚠️ Stub | FP64 uses DMMA (`mma.sync.aligned`), NOT tcgen05. Implementable via inline PTX. |
| `int8_tensor` | ⚠️ Stub | INT8 `kind::i8` supported on `sm_110a` via `tcgen05.mma` inline PTX (arch-conditional) |

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
| `arm_neon` | ✅ Done | NEON fallback (SVE2 intrinsics unavailable), FP32/FP16/INT8 |

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
| 1 | `sasp_fp8_sparse`: 2:4 sparse via `tcgen05.mma.sp` — sparsity metadata (Matrix E) needed | Medium | **Revised**: cuSPARSELt supports FP8 E4M3 sparse on SM11.0. Can implement via library path. |
| 2 | `fp8_scalar_sparse`: Same as #1 | Medium | **Revised**: `tcgen05.mma.sp` IS supported. Requires compressor kernel + TMEM metadata load. |
| 3 | `int8_scalar_sparse`: Same as #1 | Medium | **Revised**: cuSPARSELt supports INT8 sparse on SM11.0. Can implement via library path. |
| 4 | `mig_0_4tpc`, `mig_1_6tpc`: MIG partitioning requires nvidia-smi setup → stub | Low | Resolved (stub, DevKit limitation) |
| 5 | `tmem`: TMEM benchmarks use SMEM proxy | Low | **Revised**: TMEM IS directly accessible via `tcgen05.alloc/ld/st/cp/dealloc` PTX. Should be rewritten. |
| 6 | `fp8_scalar` / `int8_scalar`: Scalar kernels (tcgen05.mma PTX requires descriptor-based layout) | Low | Resolved (scalar) |
| 7 | `arm_neon`: SVE2 intrinsics (`<arm_sve.h>`) unavailable → NEON fallback | Low | Resolved (fallback) |
| 8 | `cublas_lt`: cuBLASLt API changed in CUDA 13.0 → stub | Low | **Revised**: API unchanged since CUDA 11.5. cuBLASLt ships on Thor. Stub should be replaced with real implementation. |
| 9 | `fp64_tensor`: WMMA FP64 guarded on `__CUDA_WmmaSupportDouble__` → stub | Low | **Revised**: FP64 uses DMMA (`mma.sync.aligned`), not WMMA/tcgen05. Implementable via inline PTX. |
| 10 | `int8_tensor`: `nvcuda::wmma` INT8 incomplete in CUDA 13.0 → stub | Low | **Revised**: `kind::i8` IS supported on `sm_110a` via `tcgen05.mma.cta_group::1.kind::i8` inline PTX. Arch-conditional only. |
| 11 | `nvjpeg`: NVJPEG not available on Tegra → stub | Low | Resolved (stub) |
| 12 | `tma_copy`: Fallback (cudaMalloc + cudaHostAlloc) | Low | **Revised**: `cudaMemPoolCreate` NOT in unsupported list. TMA PTX supported on SM110a. Upgrade to runtime `cudaDevAttrMemoryPoolsSupported` check. |

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

## Phase 4 — Improvement Plan (Revised 2026-05-06 with Web Research)

This section catalogs all remaining stubs/partials and proposes actionable improvement paths. Items are categorized by feasibility.

**Research Methodology**: Each claim was verified via web search (NVIDIA docs, GitHub CUTLASS/CCCL/Triton, arXiv papers, PTXAS reverse engineering references). See inline source citations.

### Research Summary Table

| Claim (Before) | Research Finding | Status |
|---|---|---|
| A1: FP64 WMMA via `tcgen05.mma.kind::f64` | ❌ WRONG — FP64 uses DMMA (`mma.sync.aligned`), NOT tcgen05. CUTLASS already demonstrates inline PTX path. | **CORRECTED** |
| A2: INT8 `kind::i8` blocked by driver/firmware | ❌ WRONG — `kind::i8` IS supported on `sm_110a`, arch-conditional only. CUTLASS/CCCL have working implementations. Requires `-arch=sm_110a`. | **CORRECTED** |
| A3: cuBLASLt API changed in CUDA 13.0; Tegra uncertain | ❌ WRONG — `cublasLtMatmul()` signature unchanged. cuBLASLt ships in unified arm64-sbsa toolkit. NVIDIA explicitly improved `cublasLtMatmul` for Thor in 13.0 Update 1. | **CORRECTED** |
| B1-B3: 2:4 sparse requires sparsity metadata, "not yet supported" | ❌ WRONG — `tcgen05.mma.sp` IS supported. Sparsity metadata (Matrix E) is documented, CUTLASS example 83 works, cuSPARSELt supports FP8/INT8 on SM11.0. | **CORRECTED** |
| C3: `cudaMemPoolCreate` unsupported on Tegra | ⚠️ PARTIAL — NOT in unsupported list. TMA PTX IS supported on SM110a. MemPool availability uncertain at runtime. Use `cudaDevAttrMemoryPoolsSupported` check. | **CORRECTED** |
| C4: TMEM not directly accessible | ❌ WRONG — TMEM IS directly accessible via `tcgen05.alloc/ld/st/cp/dealloc` inline PTX. CUTLASS `copy_sm100.hpp` has working implementation. | **CORRECTED** |

### Category A — Previously Misidentified (Now Actionable)

These were previously labeled as "API gaps / driver-blocked" but research shows they are implementable with CUDA 13.0 on Thor.

| # | Suite | Previous Claim | Research Finding | Improvement Path | Est. Effort |
|---|-------|--------------|-----------------|-----------------|-------------|
| A1 | `fp64_tensor` | "CUDA 13.0 does not support FP64 WMMA, need inline PTX" | FP64 Tensor Core exists but uses **DMMA** (`mma.sync.aligned`), NOT WMMA/WGMMA/tcgen05. `__CUDA_WmmaSupportDouble__` never existed. CUTLASS demonstrates `mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64` inline PTX. Warp-synchronous, SM80+ floor. **Sources**: [PTXAS RE Reference](https://gh.evko.io/nvopen-tools/ptxas/intrinsics/tensor.html), [Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/13.0.0/ampere-tuning-guide/index.html), [CUTLASS mma_sm90.h](https://github.com/NVIDIA/cutlass/blob/d4e16f5d/include/cutlass/arch/mma_sm90.h), [arXiv 2512.02189](https://arxiv.org/html/2512.02189v2) | Replace `__CUDA_WmmaSupportDouble__` guard with `__CUDA_ARCH__ >= 800`. Implement inline PTX `mma.sync.aligned` DMMA with m16n8k4/m8n8k4 shapes. Follow CUTLASS pattern. | Medium |
| A2 | `int8_tensor` | "`kind::i8` not supported by current Tegra driver/firmware" | INT8 `kind::i8` IS supported on `sm_110a` (arch-conditional only). CCCL has generated PTX wrapper, CUTLASS has `MmaI8Op` with inline PTX `tcgen05.mma.cta_group::1.kind::i8`. Requires `-arch=sm_110a` compilation. TMEM management + SMEM descriptors needed. **Sources**: [CICC SM100 RE](https://gh.evko.io/nvopen-tools/cicc/targets/sm100-blackwell.html), [CCCL tcgen05_mma.h](https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__ptx/instructions/generated/tcgen05_mma.h#L277), [CUTLASS mma_sm100_umma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_umma.hpp#L988) | Implement using CUTLASS CuTe `MmaI8Op` abstraction or inline PTX `tcgen05.mma.cta_group::1.kind::i8`. Need: TMEM alloc → tcgen05.mma → tcgen05.commit → tcgen05.ld. SMEM descriptors for A/B matrices. | High |
| A3 | `cublas` — cuBLASLt | "API changed significantly in CUDA 13.0; Tegra availability uncertain" | `cublasLtMatmul()` signature unchanged since CUDA 11.5. cuBLASLt ships in unified arm64-sbsa toolkit. NVIDIA improved `cublasLtMatmul` for Thor in 13.0 Update 1. `libcublasLt.so` is part of `libcublas.so`. **Sources**: [CUDA 13.0 Release Notes](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html), [NVIDIA Blog](https://developer.nvidia.com/blog/whats-new-in-cuda-toolkit-13-0-for-jetson-thor-unified-arm-ecosystem-and-more), [CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt) | Remove stub. Implement real cuBLASLt benchmark following [LtMatmulCustomFind.h](https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/Common/LtMatmulCustomFind.h) pattern. | Low |

### Category B — Sparse Tensor Support (Now Actionable)

2:4 structured sparsity IS supported on Blackwell SM110a via `tcgen05.mma.sp`. Sparsity metadata (Matrix E) must be pre-computed and loaded into TMEM.

| # | Suite | Previous Claim | Research Finding | Improvement Path | Est. Effort |
|---|-------|--------------|-----------------|-----------------|-------------|
| B1 | `sasp` — FP8 sparse | "cuSPARSELt only supports INT8X4, not FP8 E4M3" | cuSPARSELt DOES support FP8 E4M3 structured sparsity on SM11.0. Support matrix explicitly lists E4M3→FP16/BF16/FP32 output with SM11.0. | Use cuSPARSELt `cusparseLtStructuredDescriptorInit` + `cusparseLtSpMMAPrune` + `cusparseLtSpMMACompress` + `cusparseLtMatmul`. FP8 E4M3 sparse is fully supported. | Medium |
| B2 | `fp8_scalar` — sparse | "tcgen05.mma.sp requires sparsity metadata, not yet supported" | `tcgen05.mma.sp` IS supported. Matrix E (sparsity metadata) must be loaded into TMEM via `tcgen05.ld`. CUTLASS example 83 + `mma_sm100_umma.hpp` provide complete inline PTX template. | For a scalar sparse benchmark: implement compressor kernel → load Matrix E into TMEM → `tcgen05.mma.sp` inline PTX. Or use cuSPARSELt. | High |
| B3 | `int8_scalar` — sparse | Same as B2 | cuSPARSELt supports INT8 structured sparsity on SM11.0. `cusparseLtMatmul` with `CUSPARSELT_SPARSITY_50_PERCENT` handles all compression internally. | Use cuSPARSELt INT8 sparse path. Simpler than FP8 since no block-scaled. | Medium |

**Sources for Category B**: [CUTLASS Blackwell Functionality](https://docs.nvidia.com/cutlass/4.3.0/media/docs/cpp/blackwell_functionality.html), [cuSPARSELt Key Features](https://docs.nvidia.com/cuda/cusparselt/index.html), [CUTLASS example 83](https://github.com/NVIDIA/cutlass/blob/main/examples/83_blackwell_sparse_gemm/83_blackwell_sparse_gemm.cu), [CUTLASS mma_sm100_umma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_umma.hpp#L451-L506), [PTXAS RE Reference](https://gh.evko.io/nvopen-tools/ptxas/targets/tcgen05.html)

### Category C — Tegra Platform Constraints (Revised)

| # | Suite | Previous Claim | Research Finding | Verdict |
|---|-------|--------------|-----------------|---------|
| C1 | `nvjpeg` | NVJPEG not shipped on Tegra | Unchanged — NVJPEG is not part of the Tegra CUDA toolkit. | **Permanent stub.** |
| C2 | `mig` — MIG partitioning | DevKit does not support MIG | Unchanged — requires production firmware. | **Permanent stub on DevKit.** |
| C3 | `tma_copy` | `cudaMemPoolCreate` unsupported on Tegra | `cudaMemPoolCreate` is NOT in the unsupported list (§4.6 CUDA for Tegra AppNote). TMA PTX IS supported on SM110a (confirmed by PTXAS RE, CUTLASS SM110, Flash-Attention PRs). MemPool runtime availability uncertain. **Sources**: [CUDA for Tegra AppNote](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/), [PTXAS RE](https://gh.evko.io/nvopen-tools/ptxas/targets/tcgen05.html) | **Upgrade fallback to runtime check.** Use `cudaDeviceGetAttribute(&val, cudaDevAttrMemoryPoolsSupported, device)` before deciding to fallback. If memPools are supported, use real TMA. |
| C4 | `tmem` | TMEM not directly accessible, SMEM proxy only | TMEM IS directly accessible via `tcgen05.alloc/ld/st/cp/dealloc` inline PTX. TMEM is 256KB/SM with ~16TB/s read, ~8TB/s write. CUTLASS `copy_sm100.hpp` provides working implementation. **Sources**: [CUTLASS copy_sm100.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm100.hpp), [PTXAS RE TMEM](https://gh.evko.io/nvopen-tools/ptxas/targets/tcgen05.html) | **Rewrite — NOT permanent proxy.** Implement real TMEM benchmark using `tcgen05.alloc` → `tcgen05.ld` → `tcgen05.st` → `tcgen05.dealloc` inline PTX. Follow CUTLASS `copy_sm100.hpp` pattern. |

### Category D — Toolchain / Naming Issues (Low-Effort Fixes)

| # | Suite | Current State | Issue | Improvement Path | Est. Effort |
|---|-------|--------------|-------|-----------------|-------------|
| D1 | `arm_neon` | NEON fallback | SVE2 intrinsics (`<arm_sve.h>`) unavailable; name is misleading | (1) Rename to `arm_neon` or `arm_cpu_vector`. (2) Add `#ifdef` for SVE2 when toolchain supports it. (3) Update README status. | Low |
| D2 | `cluster_sync` | ✅ (README says "cluster_barrier stub") | README says "stub" but code implements working cluster_barrier with `__cluster_dims__(2,1,1)`. PLAN.md shows 0.55ns result. | Fix README: `"✅ (cluster_barrier stub)"` → `"✅ __syncthreads + cluster_barrier"`. | Trivial |
| D3 | `fp4` — sparse | Working (480 TFLOPS, 23.2%) | Sparse result is only 23.2% peak (vs 57.5% dense). May indicate suboptimal sparsity encoding or tile size. | Investigate tile sizing (M/N/K), sparsity ratio, and occupancy. Try larger matrices (m4096+). | Medium |

### Category E — Missing Metrics & Coverage Gaps

| # | Suite | Gap | Improvement Path | Est. Effort |
|---|-------|-----|-----------------|-------------|
| E1 | `fp8_scalar` | 0.04% peak — extremely low | Scalar FP8 GEMM is expected to be slow (no Tensor Core). Add context: compare against FP32 scalar baseline. | Low |
| E2 | `sm_compute` — FP64 | 123 GFLOPS, no `% Peak` | FP64 peak reference (0.126 TFLOPS = 126 GFLOPS) exists but not computed. Add `peak_pct`. | Low |
| E3 | `sm_compute` — RegPressure | 4492 GFLOPS without `% Peak` | 4492/8064 = 55.8% — needs `peak_pct` added. | Low |
| E4 | `hevc_encode` / `hevc_decode` / `av1_decode` | PLAN.md shows `—` (no results) | Verify on live hardware. May need test bitstream files. | Low |
| E5 | `atomic` — All ops | PLAN.md shows `—` (no results) | Verify benchmark produces results. May need sync fix. | Low |
| E6 | `thermal_throttle` | PLAN.md shows `—` (no result) | 60s sustained run may need tegrastats output. | Low |
| E7 | `multi_stream` | PLAN.md shows `—` (no result) | Verify on live hardware. | Low |

### Revised Priority Recommendations

| Priority | Items | Rationale |
|----------|-------|-----------|
| **P0 — Trivial fixes** | D2 (README cluster_sync), E2/E3 (add `peak_pct`) | One-line changes |
| **P1 — Low effort, high impact** | A3 (cuBLASLt — remove stub, implement real benchmark), C3 (TMA — upgrade to runtime check), D1 (rename `arm_neon`) | <1 day each, eliminates incorrect stubs |
| **P2 — Medium effort** | A1 (FP64 DMMA inline PTX), B1 (FP8 sparse via cuSPARSELt), D3 (FP4 sparse tuning), C4 (TMEM rewrite with tcgen05 PTX) | 1-3 days each |
| **P3 — High effort** | A2 (INT8 tcgen05 inline PTX), B2 (FP8 scalar sparse), B3 (INT8 sparse via cuSPARSELt) | 3-7 days, complex TMEM/descriptor management |
| **P4 — Permanent** | C1 (NVJPEG), C2 (MIG) | Platform constraints |
