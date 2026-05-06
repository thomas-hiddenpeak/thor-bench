#include "compute/tmem_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp4.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace deusridet::bench {

namespace {

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

// Valid tcgen05 PTX instructions on SM110a:
//   alloc, dealloc, relinquish_alloc_permit, mma, ld, fence
// NOTE: st and cp DO NOT exist in PTX ISA. TMEM is written ONLY via mma.

constexpr int WARP_M = 128;
constexpr int WARP_N =   8;

__device__ static uint32_t buildIdesc(int M, int N,
                                       int scaleFormat = 0,  // 0=UE4M3
                                       int kSize = 0)        // 0=K64 dense
{
    constexpr uint8_t E2M1 = 1;
    uint32_t idesc = 0;
    idesc  = E2M1;                              // a_format_ [7:10)
    idesc |= (E2M1 << 10);                      // b_format_ [10:13)
    idesc |= (0 << 15);                         // a_major_ [15:16) row-major
    idesc |= (1 << 16);                         // b_major_ [16:17) col-major
    idesc |= ((N >> 3) & 0x3F) << 17;           // n_dim_ [17:23)
    idesc |= (scaleFormat & 1) << 23;            // scale_format_ [23:24)
    idesc |= ((M >> 4) & 0x1F) << 24;            // m_dim_ [24:29)
    idesc |= (0 << 29);                         // a_sf_id_ [29:31)
    idesc |= (kSize & 1) << 31;                  // k_size_ [31:32)
    return idesc;
}

// ── Matrix descriptor builder (64-bit) — CUTLASS SmemDescriptor format ──────
// Bit  [0:14): start_address_     (base >> 4, 16-byte aligned)
// Bit  [16:30): leading_byte_offset_ (leadingDim*elemBytes >> 4)
// Bit  [32:46): stride_byte_offset_  (strideDim*elemBytes >> 4)
// Bit  [46:48): version_            (1 for Blackwell)
// Bit  [49:52): base_offset_        (0)
// Bit  [52:53): lbo_mode_           (0 = legacy)
// Bit  [61:64): layout_type_        (0=row, 1=col)
__device__ static uint64_t buildMdesc(const void* ptr, int leadingDim, int strideDim,
                                       int elemBytes, int layoutType) {
    uint64_t base = static_cast<uint64_t>(__cvta_generic_to_shared(ptr));

    uint64_t leadingBytes = static_cast<uint64_t>(leadingDim) * elemBytes;
    uint64_t strideBytes  = static_cast<uint64_t>(strideDim) * elemBytes;

    uint64_t desc = 0;
    desc  = (base >> 4) & 0x3FFF;                              // [0:14)
    desc |= (leadingBytes & 0x3FFF) << 16;                     // [16:30)
    desc |= (strideBytes & 0x3FFF) << 32;                      // [32:46)
    desc |= (1ULL) << 46;                                       // [46:48) version_=1 (Blackwell)
    desc |= (0ULL) << 49;                                       // [49:52) base_offset_=0
    desc |= (0ULL) << 52;                                       // [52:53) lbo_mode_=0 (legacy)
    desc |= (static_cast<uint64_t>(layoutType & 0x7)) << 61;   // [61:64) layout_type
    return desc;
}

constexpr int WARP_K = 64;
constexpr int FP4_SCALE_BLOCK = 16;
// SMEM layout: A(4096) + B(256) + scaleA(512) + scaleB(32) + tmemHandle(4) = 4900
constexpr int TMEM_BENCH_SMEM_BYTES = 4096 + 256 + 512 + 32 + 4;



// ── TMEM Read Bandwidth kernel ──────────────────────────────────────────────
// mma primes TMEM, then repeatedly reads via tcgen05.ld
__global__ void tmemReadBwKernel(float* out, int loops) {
    // Shared memory layout (matches FP4 kernel pattern)
    extern __shared__ char smem[];

    __nv_fp4x2_storage_t* sA       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem);
    __nv_fp4x2_storage_t* sB       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem + 4096);
    __nv_fp8_storage_t*   sScaleA  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 4096 + 256);
    __nv_fp8_storage_t*   sScaleB  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 4096 + 256 + 512);
    uint32_t* tmemHandlePtr        = reinterpret_cast<uint32_t*>(smem + 4096 + 256 + 512 + 32);

    int tid = threadIdx.x;

    // Initialize shared memory with valid data
    int aElems = WARP_M * (WARP_K / 2);
    for (int i = tid; i < aElems; i += 32) {
        sA[i] = static_cast<__nv_fp4x2_storage_t>(0x33);
    }

    int bElems = (WARP_K / 2) * WARP_N;
    for (int i = tid; i < bElems; i += 32) {
        sB[i] = static_cast<__nv_fp4x2_storage_t>(0x33);
    }

    int saElems = WARP_M * (WARP_K / FP4_SCALE_BLOCK);
    for (int i = tid; i < saElems; i += 32) {
        sScaleA[i] = static_cast<__nv_fp8_storage_t>(0x3F);  // 1.0 in FP8 E4M3
    }

    int sbElems = (WARP_K / FP4_SCALE_BLOCK) * WARP_N;
    for (int i = tid; i < sbElems; i += 32) {
        sScaleB[i] = static_cast<__nv_fp8_storage_t>(0x3F);  // 1.0 in FP8 E4M3
    }

    __syncthreads();

    uint32_t nCols = WARP_N;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(tmemHandlePtr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    uint32_t tmemHandle = tmemHandlePtr[0];

    // Build valid descriptors (same as dense since we're measuring TMEM, not INT8)
    uint64_t aDesc = buildMdesc(sA, WARP_K / 2, WARP_K / 2, 2, 0);  // row-major, FP4-pair (2B)
    uint64_t bDesc = buildMdesc(sB, WARP_N, WARP_N, 2, 1);          // col-major, FP4-pair (2B)
    uint32_t idesc = buildIdesc(WARP_M, WARP_N);
    uint32_t saPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleA));
    uint32_t sbPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleB));

    // Prime TMEM with mma
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.ne.b32 p, 0, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "  [%0], %1, %2, %3, [%4], [%5], p;}\n"
        :
        : "r"(tmemHandle), "l"(aDesc), "l"(bDesc), "r"(idesc),
          "r"(saPtr), "r"(sbPtr)
        : "memory"
    );

    // Repeatedly read from TMEM via bulk ld
    uint32_t accum = 0;
    int cnt = loops;
    while (cnt--) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile(
            "tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
              "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
            : "r"(tmemHandle)
            : "memory");
        accum += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    }
    asm volatile("" : "+r"(accum));

    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : : "r"(tmemHandle), "r"(nCols) : "memory");

    if (tid == 0) *out = __int_as_float(accum);
}

// ── TMEM Write Bandwidth kernel ──────────────────────────────────────────────
// Repeatedly mma to write to TMEM
__global__ void tmemWriteBwKernel(float* out, int loops) {
    extern __shared__ char smem[];

    __nv_fp4x2_storage_t* sA       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem);
    __nv_fp4x2_storage_t* sB       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem + 4096);
    __nv_fp8_storage_t*   sScaleA  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 4096 + 256);
    __nv_fp8_storage_t*   sScaleB  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 4096 + 256 + 512);
    uint32_t* tmemHandlePtr        = reinterpret_cast<uint32_t*>(smem + 4096 + 256 + 512 + 32);

    int tid = threadIdx.x;

    // Initialize shared memory
    int aElems = WARP_M * (WARP_K / 2);
    for (int i = tid; i < aElems; i += 32) {
        sA[i] = static_cast<__nv_fp4x2_storage_t>(0x33);
    }

    int bElems = (WARP_K / 2) * WARP_N;
    for (int i = tid; i < bElems; i += 32) {
        sB[i] = static_cast<__nv_fp4x2_storage_t>(0x33);
    }

    int saElems = WARP_M * (WARP_K / FP4_SCALE_BLOCK);
    for (int i = tid; i < saElems; i += 32) {
        sScaleA[i] = static_cast<__nv_fp8_storage_t>(0x3F);  // 1.0 in FP8 E4M3
    }

    int sbElems = (WARP_K / FP4_SCALE_BLOCK) * WARP_N;
    for (int i = tid; i < sbElems; i += 32) {
        sScaleB[i] = static_cast<__nv_fp8_storage_t>(0x3F);  // 1.0 in FP8 E4M3
    }

    __syncthreads();

    uint32_t nCols = WARP_N;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(tmemHandlePtr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    uint32_t tmemHandle = tmemHandlePtr[0];

    uint64_t aDesc = buildMdesc(sA, WARP_K / 2, WARP_K / 2, 2, 0);
    uint64_t bDesc = buildMdesc(sB, WARP_N, WARP_N, 2, 1);
    uint32_t idesc = buildIdesc(WARP_M, WARP_N);
    uint32_t saPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleA));
    uint32_t sbPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleB));

    uint32_t accum = 0;
    int cnt = loops;
    while (cnt--) {
        accum++;
        uint32_t enableD = (accum == 1) ? 0 : 1;
        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
            "  [%0], %1, %2, %3, [%4], [%5], p;}\n"
            :
            : "r"(tmemHandle), "l"(aDesc), "l"(bDesc), "r"(idesc),
              "r"(saPtr), "r"(sbPtr), "r"(enableD)
            : "memory"
        );
    }
    asm volatile("" : "+r"(accum));

    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : : "r"(tmemHandle), "r"(nCols) : "memory");

    if (tid == 0) *out = __int_as_float(accum);
}

// ── TMEM Latency kernel ─────────────────────────────────────────────────────
// Single-thread: alloc -> mma -> ld -> dealloc -> relinquish per iteration
__global__ void tmemLatencyKernel(float* out, int loops) {
    int tid = threadIdx.x;

    extern __shared__ char smem[];

    __nv_fp4x2_storage_t* sA       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem);
    __nv_fp4x2_storage_t* sB       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem + 4096);
    __nv_fp8_storage_t*   sScaleA  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 4096 + 256);
    __nv_fp8_storage_t*   sScaleB  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 4096 + 256 + 512);
    uint32_t* tmemHandlePtr        = reinterpret_cast<uint32_t*>(smem + 4096 + 256 + 512 + 32);

    // Initialize shared memory (single thread does all work)
    if (tid == 0) {
        int aElems = WARP_M * (WARP_K / 2);
        for (int i = 0; i < aElems; ++i) {
        sA[i] = static_cast<__nv_fp4x2_storage_t>(0x33);
        }
        int bElems = (WARP_K / 2) * WARP_N;
        for (int i = 0; i < bElems; ++i) {
sB[i] = static_cast<__nv_fp4x2_storage_t>(0x33);
        }
        int saElems = WARP_M * (WARP_K / FP4_SCALE_BLOCK);
        for (int i = 0; i < saElems; ++i) {
            sScaleA[i] = static_cast<__nv_fp8_storage_t>(0x3F);
        }
        int sbElems = (WARP_K / FP4_SCALE_BLOCK) * WARP_N;
        for (int i = 0; i < sbElems; ++i) {
            sScaleB[i] = static_cast<__nv_fp8_storage_t>(0x3F);
        }
    }
    __syncthreads();

    if (tid != 0) return;

    uint64_t aDesc = buildMdesc(sA, WARP_K / 2, WARP_K / 2, 2, 0);
    uint64_t bDesc = buildMdesc(sB, WARP_N, WARP_N, 2, 1);
    uint32_t idesc = buildIdesc(WARP_M, WARP_N);
    uint32_t saPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleA));
    uint32_t sbPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleB));

    uint32_t accum = 0;
    int cnt = loops;

    while (cnt--) {
        uint32_t nCols = WARP_N;
        uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(tmemHandlePtr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            : : "r"(smemTmemPtr), "r"(nCols) : "memory");
        uint32_t tmemHandle = tmemHandlePtr[0];

        uint32_t enableD = 0;
        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
            "  [%0], %1, %2, %3, [%4], [%5], p;}\n"
            :
            : "r"(tmemHandle), "l"(aDesc), "l"(bDesc), "r"(idesc),
              "r"(saPtr), "r"(sbPtr), "r"(enableD)
            : "memory"
        );

        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile(
            "tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
              "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
            : "r"(tmemHandle)
            : "memory");
        (void)r1; (void)r2; (void)r3; (void)r4; (void)r5; (void)r6; (void)r7;
        accum += r0;

        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : : "r"(tmemHandle), "r"(nCols) : "memory");

        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }

    asm volatile("" : "+r"(accum));
    *out = __int_as_float(accum);
}

// ── SMEM Read Bandwidth kernel (for comparison) ─────────────────────────────
template <int TileBytes>
__global__ void smemReadBwKernel(float* out, int loops) {
    __shared__ float sdata[TileBytes / 4];
    int tid = threadIdx.x;
    for (int i = 0; i < blockDim.x; i += blockDim.x) {
        if (tid + i < TileBytes / 4)
            sdata[tid + i] = static_cast<float>(tid + i);
    }
    __syncthreads();

    float accum = 0.0f;
    int cnt = loops;
    while (cnt--) {
        for (int i = 0; i < TileBytes / 4; ++i) {
            accum += sdata[i];
        }
    }
    asm volatile("" : "+f"(accum));
    if (tid == 0) *out = accum;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Test 1: TMEM Read Bandwidth ──────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult measureTMEMReadBW(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int loops = 4096;
    int tpb = 32;

    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    for (int w = 0; w < 3; ++w) {
        tmemReadBwKernel<<<1, tpb, TMEM_BENCH_SMEM_BYTES, str>>>(dOut, loops);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        tmemReadBwKernel<<<1, tpb, TMEM_BENCH_SMEM_BYTES, str>>>(dOut, loops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        // Each ld reads 8*4=32 bytes per thread, 32 threads
        double totalBytes = static_cast<double>(loops) * 32 * sizeof(uint32_t);
        double gbs = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
        vals.push_back(gbs);
    }

    std::ostringstream p;
    p << "{\"loops\":" << loops << ",\"tpb\":" << tpb
      << ",\"bytes_per_ld\":" << (32 * sizeof(uint32_t))
      << ",\"type\":\"tcgen05_tmem\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "tmem";
    res.test_name  = "tmem_read_bandwidth";
    res.unit       = "GB/s";
    res.params_json = p.str();
    res.metadata["proxy"] = "false";
    res.metadata["tcgen05"] = "true";
    res.metadata["note"] = "Real TMEM read via tcgen05.ld (primed by tcgen05.mma)";
    res.peak_pct = 0.0;

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Test 2: TMEM Write Bandwidth (via mma) ───────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult measureTMEMWriteBW(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int loops = 2048;
    int tpb = 32;

    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    for (int w = 0; w < 3; ++w) {
        tmemWriteBwKernel<<<1, tpb, TMEM_BENCH_SMEM_BYTES, str>>>(dOut, loops);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        tmemWriteBwKernel<<<1, tpb, TMEM_BENCH_SMEM_BYTES, str>>>(dOut, loops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        // Each mma writes 8 FP32 = 32 bytes to TMEM per thread
        double totalBytes = static_cast<double>(loops) * 32 * sizeof(uint32_t);
        double gbs = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
        vals.push_back(gbs);
    }

    std::ostringstream p;
    p << "{\"loops\":" << loops << ",\"tpb\":" << tpb
      << ",\"bytes_per_mma\":" << (32 * sizeof(uint32_t))
      << ",\"type\":\"tcgen05_tmem\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "tmem";
    res.test_name  = "tmem_write_bandwidth";
    res.unit       = "GB/s";
    res.params_json = p.str();
    res.metadata["proxy"] = "false";
    res.metadata["tcgen05"] = "true";
    res.metadata["note"] = "Real TMEM write via tcgen05.mma";
    res.peak_pct = 0.0;

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Test 3: TMEM Latency ─────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult measureTMEMLatency(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int loops = 256;
    int tpb = 1;

    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    for (int w = 0; w < 3; ++w) {
        tmemLatencyKernel<<<1, tpb, TMEM_BENCH_SMEM_BYTES, str>>>(dOut, loops);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        tmemLatencyKernel<<<1, tpb, TMEM_BENCH_SMEM_BYTES, str>>>(dOut, loops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double totalNs = ms * 1e6;
        double nsPerRoundTrip = totalNs / loops;
        vals.push_back(nsPerRoundTrip);
    }

    std::ostringstream p;
    p << "{\"loops\":" << loops << ",\"tpb\":" << tpb
      << ",\"type\":\"tcgen05_tmem\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "tmem";
    res.test_name  = "tmem_latency";
    res.unit       = "ns";
    res.params_json = p.str();
    res.metadata["proxy"] = "false";
    res.metadata["tcgen05"] = "true";
    res.metadata["note"] = "TMEM round-trip: alloc->mma->ld->dealloc->relinquish";
    res.peak_pct = 0.0;

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Test 4: TMEM vs SMEM Bandwidth Comparison ────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult measureTMEMvsSMEM(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    // TMEM bandwidth
    constexpr int TmemLoops = 4096;
    int tpb = 32;

    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    double tmemBw = 0.0;
    {
        for (int w = 0; w < 3; ++w) {
tmemReadBwKernel<<<1, tpb, TMEM_BENCH_SMEM_BYTES, str>>>(dOut, TmemLoops);
            chk(cudaStreamSynchronize(str), "warmup");
        }

        chk(cudaEventRecord(evS, str), "rs");
        tmemReadBwKernel<<<1, tpb, TMEM_BENCH_SMEM_BYTES, str>>>(dOut, TmemLoops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double totalBytes = static_cast<double>(TmemLoops) * 32 * sizeof(uint32_t);
        tmemBw = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
    }

    // SMEM bandwidth
    constexpr int TileBytes = 128;
    constexpr int SmemLoops = 4096;
    double smemBw = 0.0;
    {
        for (int w = 0; w < 3; ++w) {
            smemReadBwKernel<TileBytes><<<1, tpb, 0, str>>>(dOut, SmemLoops);
            chk(cudaStreamSynchronize(str), "warmup");
        }

        chk(cudaEventRecord(evS, str), "rs");
        smemReadBwKernel<TileBytes><<<1, tpb, 0, str>>>(dOut, SmemLoops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double totalBytes = static_cast<double>(SmemLoops) * TileBytes;
        smemBw = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
    }

    std::vector<double> vals;
    vals.push_back(tmemBw);

    std::ostringstream p;
    p << "{\"tmem_loops\":" << TmemLoops
      << ",\"smem_tile_bytes\":" << TileBytes << ",\"smem_loops\":" << SmemLoops
      << ",\"tpb\":" << tpb
      << ",\"tmem_bw_gb_s\":" << tmemBw
      << ",\"smem_bw_gb_s\":" << smemBw
      << ",\"tmem_smem_ratio\":" << (smemBw > 0 ? tmemBw / smemBw : 0.0)
      << ",\"type\":\"tcgen05_tmem\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 0);
    res.suite_name = "tmem";
    res.test_name  = "tmem_vs_smem";
    res.unit       = "GB/s";
    res.params_json = p.str();
    res.metadata["proxy"] = "false";
    res.metadata["tcgen05"] = "true";
    res.metadata["note"] = "TMEM vs SMEM read bandwidth (tcgen05 vs __shared__)";
    res.peak_pct = 0.0;

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Fallback stub results ────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult makeStubResult(const char* test_name, const char* unit, const char* reason) {
    BenchResult r{};
    r.suite_name = "tmem";
    r.test_name  = test_name;
    r.unit       = unit;
    r.peak_pct   = 0.0;
    std::string err = "{\"error\":\"";
    err += reason;
    err += "\",\"fallback\":\"smem_proxy\"}";
    r.params_json = err;
    r.metadata["proxy"] = "true";
    r.metadata["tcgen05"] = "false";
    r.metadata["stub"] = "true";
    r.metadata["stub_reason"] = reason;
    return r;
}

} // anonymous namespace

std::vector<BenchResult> runTMEMBench(int device, int matDim, int iterations) {
    (void)matDim;
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    bool failed = false;
    try {
        results.push_back(measureTMEMReadBW(device, iterations));
    } catch (const std::exception& ex) {
        cudaGetLastError();
        results.push_back(makeStubResult("tmem_read_bandwidth", "GB/s",
            (std::string("tcgen05 failed: ") + ex.what()).c_str()));
        failed = true;
    }

    try {
        results.push_back(measureTMEMWriteBW(device, iterations));
    } catch (const std::exception& ex) {
        cudaGetLastError();
        results.push_back(makeStubResult("tmem_write_bandwidth", "GB/s",
            (std::string("tcgen05 failed: ") + ex.what()).c_str()));
        failed = true;
    }

    try {
        results.push_back(measureTMEMLatency(device, iterations));
    } catch (const std::exception& ex) {
        cudaGetLastError();
        results.push_back(makeStubResult("tmem_latency", "ns",
            (std::string("tcgen05 failed: ") + ex.what()).c_str()));
        failed = true;
    }

    try {
        results.push_back(measureTMEMvsSMEM(device, iterations));
    } catch (const std::exception& ex) {
        cudaGetLastError();
        results.push_back(makeStubResult("tmem_vs_smem", "GB/s",
            (std::string("tcgen05 failed: ") + ex.what()).c_str()));
        failed = true;
    }

    (void)failed;
    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(tmem, "TCGen05 TMEM bandwidth and latency (tcgen05.alloc/ld/mma)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runTMEMBench(0, 512, 10);
    });
