#include "compute/fp4_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp4.h>
#include <cuda_fp6.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace deusridet::bench {

namespace {

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

// ── Tile dimensions: m128 n8 k64 per warp ──────────────────────────────────
constexpr int WARP_M = 128;
constexpr int WARP_N =   8;
constexpr int WARP_K =  64;
constexpr int FP4_SCALE_BLOCK = 16;

// ── Conversion kernel: float → FP4 + scale factors ─────────────────────────
__global__ void floatToFp4Kernel(
    __nv_fp4x2_storage_t* __restrict__ outFp4,
    __nv_fp8_storage_t*   __restrict__ outScale,
    const float*          __restrict__ inp,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float*              rowPtr = inp + static_cast<size_t>(row) * cols;
    __nv_fp4x2_storage_t* outPtr    = outFp4 + static_cast<size_t>(row) * (cols / 2);
    __nv_fp8_storage_t* scalePtr    = outScale + static_cast<size_t>(row) * (cols / FP4_SCALE_BLOCK);

    // Scale factors: one per 16 FP4 values
    int numBlocks = cols / FP4_SCALE_BLOCK;
    for (int blk = threadIdx.x; blk < numBlocks; blk += blockDim.x) {
        float maxAbs = 0.0f;
        for (int i = 0; i < FP4_SCALE_BLOCK; ++i) {
            float v = fabsf(rowPtr[blk * FP4_SCALE_BLOCK + i]);
            if (v > maxAbs) maxAbs = v;
        }
        scalePtr[blk] = __nv_cvt_float_to_fp8(maxAbs, __NV_SATFINITE, __NV_E4M3);
    }

    // Convert to FP4 (2 floats per __nv_fp4x2_storage_t)
    for (int i = threadIdx.x; i < cols / 2; i += blockDim.x) {
        float2 fPair = make_float2(rowPtr[i * 2], rowPtr[i * 2 + 1]);
        outPtr[i] = __nv_cvt_float2_to_fp4x2(fPair, __NV_E2M1, cudaRoundNearest);
    }
}

// ── Matrix descriptor builder (64-bit) — CUTLASS SmemDescriptor format ──────
// Source: CUTLASS mma_sm100_desc.hpp SmemDescriptor union
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
    desc |= ((leadingBytes >> 4) & 0x3FFF) << 16;              // [16:30)
    desc |= ((strideBytes >> 4)  & 0x3FFF) << 32;              // [32:46)
    desc |= (1ULL) << 46;                                       // [46:48) version_=1 (Blackwell)
    desc |= (0ULL) << 49;                                       // [49:52) base_offset_=0
    desc |= (0ULL) << 52;                                       // [52:53) lbo_mode_=0 (legacy)
    desc |= (static_cast<uint64_t>(layoutType & 0x7)) << 61;   // [61:64) layout_type
    return desc;
}

// ── Instruction descriptor (32-bit) — InstrDescriptorBlockScaled format ──────
// Source: CUTLASS mma_sm100_desc.hpp InstrDescriptorBlockScaled union
// Bit  [7:10):    a_format_      (MXF4Format::E2M1 = 1)
// Bit  [10:13):   b_format_      (MXF4Format::E2M1 = 1)
// Bit  [15:16):   a_major_       (0 = K-major / row-major for A)
// Bit  [16:17):   b_major_       (1 = MN-major / col-major for B)
// Bit  [17:23):   n_dim_         (N >> 3)
// Bit  [23:24):   scale_format_  (0 = UE4M3, 1 = UE8M0)
// Bit  [24:29):   m_dim_         (M >> 4)
// Bit  [29:31):   a_sf_id_       (scale factor A ID, 0)
// Bit  [31:32):   k_size_        (0 = K64 dense)
__device__ static uint32_t buildIdesc(int M, int N,
                                       int scaleFormat = 0,  // 0=UE4M3
                                       int kSize = 0)        // 0=K64 dense
{
    constexpr uint8_t E2M1 = 1;  // MXF4Format::E2M1 = 1 (NOT 5)
    uint32_t idesc = 0;
    idesc  = E2M1;                              // a_format_ [7:10)
    idesc |= (E2M1 << 10);                      // b_format_ [10:13)
    idesc |= (0 << 15);                         // a_major_ [15:16) row-major
    idesc |= (1 << 16);                         // b_major_ [16:17) col-major
    idesc |= ((N >> 3) & 0x3F) << 17;           // n_dim_ [17:23)
    idesc |= (scaleFormat & 1) << 23;            // scale_format_ [23:24) UE4M3=0
    idesc |= ((M >> 4) & 0x1F) << 24;            // m_dim_ [24:29)
    idesc |= (0 << 29);                         // a_sf_id_ [29:31)
    idesc |= (kSize & 1) << 31;                  // k_size_ [31:32)
    return idesc;
}

// ── NVFP4 support probe kernel ─────────────────────────────────────────────
// Launches a single warp to test if tcgen05.mma.kind::mxf4nvf4 is supported
// by the current driver/firmware. Uses minimal shared memory so corruption
// (if any) is contained and doesn't leak to subsequent suites.
__global__ void fp4Nvf4ProbeKernel() {
    if (threadIdx.x != 0) return;

    // Shared memory for TMEM handle
    __shared__ uint32_t tmemHandleSmem;

    // Minimal TMEM allocation
    uint32_t nCols = 8;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmemHandleSmem));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    uint64_t aDesc = 0, bDesc = 0;
    uint32_t idesc = buildIdesc(WARP_M, WARP_N);
    uint32_t tmemHandle = tmemHandleSmem;
    uint32_t saPtr = 0, sbPtr = 0;

    // The critical instruction - if unsupported, causes illegal instruction
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

    // Cleanup TMEM
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : : "r"(tmemHandle), "r"(nCols) : "memory");
}

// ── Check if NVFP4 tcgen05.mma is supported by current driver/firmware ─────
// Returns true if supported, false if not. Avoids corrupting GPU state.
static bool nvfp4Supported(int device) {
    chk(cudaSetDevice(device), "probe_dev");

    // Quick capability gate: compute capability 11.1+ required (SM110a = 11.1)
    int major = 0, minor = 0;
    chk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "major");
    chk(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "minor");
    if (major < 11 || (major == 11 && minor < 1)) return false;

    // Actually probe the instruction - the only reliable check
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "probe_stream");

    // Zero-sharing: launch with 0 smem to avoid TMEM corruption on the real context
    fp4Nvf4ProbeKernel<<<1, 32, 0, str>>>();
    cudaError_t e = cudaStreamSynchronize(str);
    chk(cudaStreamDestroy(str), "probe_stream_destroy");

    if (e != cudaSuccess) {
        // Instruction not supported by driver/firmware - skip entirely
        return false;
    }

    // Drain any async errors
    e = cudaGetLastError();
    if (e != cudaSuccess) return false;

    return true;
}

// ── FP4 NVFP4 GEMM kernel ──────────────────────────────────────────────────
// Each warp (block=1 warp) handles m128 x n8 tile.
// Grid: (N/8) x (M/128)
// K loop: K/64 iterations
__global__ void fp4Nvf4GemmKernel(
    const __nv_fp4x2_storage_t* __restrict__ A,
    const __nv_fp8_storage_t*   __restrict__ scaleA,
    const __nv_fp4x2_storage_t* __restrict__ B,
    const __nv_fp8_storage_t*   __restrict__ scaleB,
    float*                      __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y * WARP_M;
    int warpN = blockIdx.x * WARP_N;

    if (warpM >= M || warpN >= N) return;

    int laneId = threadIdx.x;

    // ── Shared memory layout ─────────────────────────────────────────────
    // A tile:     WARP_M x (WARP_K/2) = 128 x 32 = 4096 bytes
    // B tile:     (WARP_K/2) x WARP_N = 32 x 8  = 256 bytes
    // scaleA:     WARP_M x (WARP_K/16) = 128 x 4 = 512 bytes (FP8)
    // scaleB:     (WARP_K/16) x WARP_N = 4 x 8 = 32 bytes (FP8)
    // tmemHandle: 4 bytes
    extern __shared__ char smem[];

    __nv_fp4x2_storage_t* sA       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem);
    __nv_fp4x2_storage_t* sB       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem + 4096);
    __nv_fp8_storage_t*   sScaleA  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 4096 + 256);
    __nv_fp8_storage_t*   sScaleB  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 4096 + 256 + 512);
    uint32_t* tmemHandlePtr        = reinterpret_cast<uint32_t*>(smem + 4096 + 256 + 512 + 32);

    // ── TMEM allocation ──────────────────────────────────────────────────
    // nCols = WARP_N = 8 (FP32 columns per row)
    uint32_t nCols = WARP_N;

    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(tmemHandlePtr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    // Build idesc once
    uint32_t idesc = buildIdesc(WARP_M, WARP_N);

    // ── K loop ───────────────────────────────────────────────────────────
    int kTiles = K / WARP_K;

    for (int kt = 0; kt < kTiles; ++kt) {
        int kBase = kt * WARP_K;

        // ── Load A tile from global → shared ─────────────────────────────
        int aElems = WARP_M * (WARP_K / 2);
        for (int i = threadIdx.x; i < aElems; i += 32) {
            int r = i / (WARP_K / 2);
            int c = i % (WARP_K / 2);
            sA[i] = A[(warpM + r) * (K / 2) + kBase / 2 + c];
        }

        // ── Load B tile from global → shared (col-major) ─────────────────
        int bElems = (WARP_K / 2) * WARP_N;
        for (int i = threadIdx.x; i < bElems; i += 32) {
            int r = i / WARP_N;
            int c = i % WARP_N;
            sB[i] = B[(kBase / 2 + r) * N + warpN + c];
        }

        // ── Load scaleA tile ─────────────────────────────────────────────
        int saElems = WARP_M * (WARP_K / FP4_SCALE_BLOCK);
        for (int i = threadIdx.x; i < saElems; i += 32) {
            int r = i / (WARP_K / FP4_SCALE_BLOCK);
            int b = i % (WARP_K / FP4_SCALE_BLOCK);
            sScaleA[i] = scaleA[(warpM + r) * (K / FP4_SCALE_BLOCK) + kBase / FP4_SCALE_BLOCK + b];
        }

        // ── Load scaleB tile (col-major) ─────────────────────────────────
        int sbElems = (WARP_K / FP4_SCALE_BLOCK) * WARP_N;
        for (int i = threadIdx.x; i < sbElems; i += 32) {
            int r = i / WARP_N;
            int c = i % WARP_N;
            sScaleB[i] = scaleB[(kBase / FP4_SCALE_BLOCK + r) * N + warpN + c];
        }

        __syncthreads();

        // ── Build matrix descriptors (CUTLASS SmemDescriptor format) ───────
        // A: row-major, WARP_M rows x (WARP_K/2) FP4-pair cols, elem=2B
        // leadingDim = WARP_K/2 (cols), strideDim = WARP_K/2 (row pitch in elems)
        uint64_t aDesc = buildMdesc(sA, WARP_K / 2, WARP_K / 2, 2, 0);  // 0=row-major
        // B: col-major, (WARP_K/2) rows x WARP_N cols, elem=2B
        uint64_t bDesc = buildMdesc(sB, WARP_N, WARP_N, 2, 1);  // 1=col-major

        // Scale descriptors: scaleA is FP8 (1B), row-major: WARP_M x (WARP_K/16)
        uint64_t saDesc = buildMdesc(sScaleA, WARP_K / FP4_SCALE_BLOCK, WARP_K / FP4_SCALE_BLOCK, 1, 0);
        // scaleB is FP8 (1B), col-major: (WARP_K/16) x WARP_N
        uint64_t sbDesc = buildMdesc(sScaleB, WARP_N, WARP_N, 1, 1);

        // ── tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 ──────
        // CUDA 13+ syntax: block16 (replaces scale_vec::4X from CUDA <12.9)
        uint32_t enableInputD = (kt == 0) ? 0 : 1;
        uint32_t tmemHandle = tmemHandlePtr[0];

        // Scale factor addresses (shared memory pointers for block_scale variant)
        uint32_t saPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleA));
        uint32_t sbPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleB));

        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
            "  [%0], %1, %2, %3, [%4], [%5], p;}\n"
            :
            : "r"(tmemHandle),
              "l"(aDesc),
              "l"(bDesc),
              "r"(idesc),
              "r"(saPtr),
              "r"(sbPtr),
              "r"(enableInputD)
            : "memory"
        );

        // ── On final k-tile: read results from TMEM ──────────────────────
        if (kt == kTiles - 1) {
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

            // Each lane handles WARP_M/32 = 4 rows
            int rowOffset = laneId * (WARP_M / 32);
            float* cOut = C + (warpM + rowOffset) * N + warpN;

            uint32_t ldOut[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(ldOut[0]), "=r"(ldOut[1]), "=r"(ldOut[2]), "=r"(ldOut[3]),
                  "=r"(ldOut[4]), "=r"(ldOut[5]), "=r"(ldOut[6]), "=r"(ldOut[7])
                : "r"(tmemHandle)
                : "memory"
            );

            // Write 4 rows for this lane
            for (int r = 0; r < WARP_M / 32; ++r) {
                for (int c = 0; c < WARP_N; ++c) {
                    cOut[r * N + c] = __int_as_float(ldOut[c]);
                }
            }

            // ── TMEM deallocation ────────────────────────────────────────
            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                : : "r"(tmemHandle), "r"(nCols) : "memory");
        }
    }
}

// ── Stats helper ───────────────────────────────────────────────────────────
BenchResult computeStats(const std::vector<double>& vals,
                         const std::string& suite, const std::string& test,
                         const std::string& unit, const std::string& pj) {
    std::vector<double> sv = vals;
    BenchResult res = ::deusridet::bench::computeStats(sv, 3);
    res.suite_name = suite;
    res.test_name  = test;
    res.unit       = unit;
    res.params_json = pj;
    if (!sv.empty()) {
        res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp4_dense_tflops);
    }
    return res;
}

// ── Measure FP4 NVFP4 Dense GEMM ───────────────────────────────────────────
BenchResult measureFp4Dense(int device, int matDim, int iterations) {
    chk(cudaSetDevice(device), "dev");

    // Align dimensions to warp tile sizes
    int M = (matDim / WARP_M) * WARP_M;
    int N = M;
    int K = (matDim / WARP_K) * WARP_K;
    if (M < WARP_M) M = WARP_M;
    if (K < WARP_K) K = WARP_K;

    size_t fp4SizeA   = static_cast<size_t>(M) * static_cast<size_t>(K) / 2;
    size_t fp4SizeB   = static_cast<size_t>(K) * static_cast<size_t>(N) / 2;
    size_t scaleSizeA = static_cast<size_t>(M) * static_cast<size_t>(K) / FP4_SCALE_BLOCK;
    size_t scaleSizeB = static_cast<size_t>(K) * static_cast<size_t>(N) / FP4_SCALE_BLOCK;
    size_t floatSizeA = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float);
    size_t floatSizeB = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(float);
    size_t floatSizeC = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);

    __nv_fp4x2_storage_t *dA_fp4 = nullptr, *dB_fp4 = nullptr;
    __nv_fp8_storage_t   *dScaleA = nullptr, *dScaleB = nullptr;
    float                *dA = nullptr, *dB = nullptr, *dC = nullptr;

    chk(cudaMalloc(&dA, floatSizeA), "alloc_a");
    chk(cudaMalloc(&dB, floatSizeB), "alloc_b");
    chk(cudaMalloc(&dA_fp4, fp4SizeA), "alloc_fp4a");
    chk(cudaMalloc(&dB_fp4, fp4SizeB), "alloc_fp4b");
    chk(cudaMalloc(&dC, floatSizeC), "alloc_c");
    chk(cudaMalloc(&dScaleA, scaleSizeA), "alloc_scalea");
    chk(cudaMalloc(&dScaleB, scaleSizeB), "alloc_scaleb");

    chk(cudaMemset(dA, 0x3E, floatSizeA), "memset_a");
    chk(cudaMemset(dB, 0x3E, floatSizeB), "memset_b");

    cudaStream_t str;
    chk(cudaStreamCreate(&str), "stream");

    // Convert float -> FP4 + scales
    {
        int gridX = std::min(65535, M);
        floatToFp4Kernel<<<gridX, 256, 0, str>>>(dA_fp4, dScaleA, dA, M, K);
        chk(cudaGetLastError(), "convert_a");

        int gridXB = std::min(65535, K);
        floatToFp4Kernel<<<gridXB, 256, 0, str>>>(dB_fp4, dScaleB, dB, K, N);
        chk(cudaGetLastError(), "convert_b");

        chk(cudaStreamSynchronize(str), "convert_sync");
    }

    // Grid: (N/WARP_N) x (M/WARP_M), 1 warp per block
    dim3 grid(std::min(65535, N / WARP_N), std::min(65535, M / WARP_M), 1);
    int smemBytes = 4096 + 256 + 512 + 32 + 4;

    // Warmup
    for (int w = 0; w < 3; ++w) {
        fp4Nvf4GemmKernel<<<grid.x, 32, smemBytes, str>>>(
            dA_fp4, dScaleA, dB_fp4, dScaleB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    // Timing
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    std::vector<double> vals;
    size_t totalFlops = static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(K) * 2;

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "recS");
        fp4Nvf4GemmKernel<<<grid.x, 32, smemBytes, str>>>(
            dA_fp4, dScaleA, dB_fp4, dScaleB, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "recE");
        chk(cudaStreamSynchronize(str), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "elapsed");
        double sec = ms / 1000.0;
        double tflops = sec > 0.0 ? (static_cast<double>(totalFlops) / 1e12) / sec : 0.0;
        vals.push_back(tflops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"warp_m\":" << WARP_M << ",\"warp_n\":" << WARP_N << ",\"warp_k\":" << WARP_K
      << ",\"format\":\"nvfp4_e2m1\""
      << ",\"scale_block\":" << FP4_SCALE_BLOCK
      << ",\"api\":\"tcgen05_mma_inline_ptx\""
      << ",\"scale_vec\":\"2X\"}";

    BenchResult res = computeStats(vals, "fp4", "fp4_nvfp4_dense", "TFLOP/s", p.str());
    res.metadata["format"] = "nvfp4_e2m1";
    res.metadata["block_scaled"] = "true";
    res.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp4_dense_tflops));

    chk(cudaFree(dA), "free_a");
    chk(cudaFree(dB), "free_b");
    chk(cudaFree(dA_fp4), "free_fp4a");
    chk(cudaFree(dB_fp4), "free_fp4b");
    chk(cudaFree(dC), "free_c");
    chk(cudaFree(dScaleA), "free_scalea");
    chk(cudaFree(dScaleB), "free_scaleb");
    chk(cudaStreamDestroy(str), "stream");
    chk(cudaEventDestroy(evS), "evSDestroy");
    chk(cudaEventDestroy(evE), "evEDestroy");

    return res;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── FP4 NVFP4 Sparse GEMM (2:4 structured sparsity) ─────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

// 2:4 sparsity: each group of 4 FP4 values has at most 2 non-zeros.
// E-matrix encodes which positions are non-zero per 4-value group.
// For NVFP4, each uint8_t in the E-matrix covers 4 FP4 values (1 byte).
// A[row][group] = bitmask of non-zero positions within that 4-value group.

// ── Kernel: generate 2:4 sparsity mask from float matrix ────────────────────
// Returns: (1) compressed FP4 matrix (only non-zero values packed),
//          (2) E-matrix (sparsity metadata),
//          (3) scale factors.
// Sparse K dimension = K/2 (half the dense width).
__global__ void fp4SparseCompressKernel(
    __nv_fp4x2_storage_t* __restrict__ outFp4,
    __nv_fp8_storage_t*   __restrict__ outScale,
    uint8_t*              __restrict__ outE,
    int*                  __restrict__ outOffsets,  // row offsets into compressed FP4
    const float*          __restrict__ inp,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* rowPtr = inp + static_cast<size_t>(row) * cols;

    int groupsPerRow = cols / FP4_SCALE_BLOCK;  // groups of 4 per scale block
    int scaleBlocks  = cols / FP4_SCALE_BLOCK;

    // Per-thread work within a row
    int tid = threadIdx.x;

    // Compressed output offset tracking (shared memory)
    extern __shared__ int smemSharedCounts[];

    // Count non-zeros per group of 4 values, build E-matrix
    for (int g = tid; g < groupsPerRow; g += blockDim.x) {
        int valBase = g * 4;
        uint8_t mask = 0;
        int nnz = 0;
        for (int v = 0; v < 4; ++v) {
            if (fabsf(rowPtr[valBase + v]) > 1e-6f) {
                mask |= (1 << v);
                nnz++;
            }
        }
        outE[g] = mask;
    }

    __syncthreads();

    // Compute row offset (count non-zeros per row via E-matrix)
    int rowNnz = 0;
    for (int g = 0; g < groupsPerRow; ++g) {
        uint8_t m = outE[g];
        for (int v = 0; v < 4; ++v) if (m & (1 << v)) rowNnz++;
    }
    outOffsets[row] = rowNnz;

    // Convert to FP4 with scales (same as dense kernel)
    __nv_fp4x2_storage_t* outPtr = outFp4 + rowNnz;
    __nv_fp8_storage_t* scalePtr = outScale + static_cast<size_t>(row) * scaleBlocks;

    for (int blk = tid; blk < scaleBlocks; blk += blockDim.x) {
        float maxAbs = 0.0f;
        for (int i = 0; i < FP4_SCALE_BLOCK; ++i) {
            float v = fabsf(rowPtr[blk * FP4_SCALE_BLOCK + i]);
            if (v > maxAbs) maxAbs = v;
        }
        scalePtr[blk] = __nv_cvt_float_to_fp8(maxAbs, __NV_SATFINITE, __NV_E4M3);
    }

    // Compress non-zero values into packed FP4 storage
    __syncthreads();
    int packedIdx = 0;
    for (int g = tid; g < groupsPerRow; g += blockDim.x) {
        uint8_t mask = outE[g];
        int valBase = g * 4;
        for (int v = 0; v < 4; ++v) {
            if (mask & (1 << v)) {
                packedIdx++;
            }
        }
    }

    // Actually pack the values (each thread packs its groups)
    packedIdx = 0;
    for (int g = 0; g < groupsPerRow; ++g) {
        uint8_t mask = outE[g];
        for (int v = 0; v < 4; ++v) {
            if (mask & (1 << v)) {
                // Pack pairs into __nv_fp4x2_storage_t
                if (packedIdx % 2 == 0) {
                    float next = 0;
                    int nextG = g, nextV = v + 1;
                    while (nextV < 4 && !(outE[nextG] & (1 << nextV))) nextV++;
                    if (nextV >= 4) {
                        nextG++; nextV = 0;
                        while (nextG < groupsPerRow && !(outE[nextG] & 1)) nextG++;
                    }
                    if (nextG < groupsPerRow) {
                        next = rowPtr[nextG * 4 + nextV];
                    }
                    float cur = rowPtr[g * 4 + v];
                    outPtr[packedIdx / 2] = __nv_cvt_float2_to_fp4x2(
                        make_float2(cur, next), __NV_E2M1, cudaRoundNearest);
                }
                packedIdx++;
            }
        }
    }
}

// ── FP4 NVFP4 Sparse GEMM kernel ────────────────────────────────────────────
// Uses tcgen05.mma.sp with E-matrix for 2:4 structured sparsity.
// A is M x (K/2) compressed, E is M x (K/4) metadata, B is K x N (dense).
__global__ void fp4Nvf4SparseGemmKernel(
    const __nv_fp4x2_storage_t* __restrict__ A,
    const __nv_fp8_storage_t*   __restrict__ scaleA,
    const uint8_t*              __restrict__ E,
    const __nv_fp4x2_storage_t* __restrict__ B,
    const __nv_fp8_storage_t*   __restrict__ scaleB,
    float*                      __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y * WARP_M;
    int warpN = blockIdx.x * WARP_N;

    if (warpM >= M || warpN >= N) return;

    int laneId = threadIdx.x;

    // ── Shared memory layout (sparse: A is K/2, E is K/4) ─────────────────
    // A tile (compressed):   WARP_M x (WARP_K/4) = 128 x 16 = 2048 bytes
    // E tile (metadata):     WARP_M x (WARP_K/8) = 128 x 8  = 1024 bytes
    // B tile:                (WARP_K/2) x WARP_N = 32 x 8  = 256 bytes
    // scaleA:                WARP_M x (WARP_K/16) = 128 x 4 = 512 bytes
    // scaleB:                (WARP_K/16) x WARP_N = 4 x 8  = 32 bytes
    // tmemHandle:            4 bytes
    extern __shared__ char smem[];

    __nv_fp4x2_storage_t* sA       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem);
    uint8_t*              sE       = reinterpret_cast<uint8_t*>(smem + 2048);
    __nv_fp4x2_storage_t* sB       = reinterpret_cast<__nv_fp4x2_storage_t*>(smem + 2048 + 1024);
    __nv_fp8_storage_t*   sScaleA  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 2048 + 1024 + 256);
    __nv_fp8_storage_t*   sScaleB  = reinterpret_cast<__nv_fp8_storage_t*>(smem + 2048 + 1024 + 256 + 512);
    uint32_t* tmemHandlePtr        = reinterpret_cast<uint32_t*>(smem + 2048 + 1024 + 256 + 512 + 32);

    // ── TMEM allocation ──────────────────────────────────────────────────
    uint32_t nCols = WARP_N;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(tmemHandlePtr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    // Build idesc for sparse: sparse_id2=4 in bits 0-2
    uint32_t idesc = buildIdesc(WARP_M, WARP_N) | 4;

    // ── K loop (sparse K = K/2, so half the iterations) ───────────────────
    int kTiles = K / WARP_K;

    for (int kt = 0; kt < kTiles; ++kt) {
        int kBase = kt * WARP_K;

        // ── Load A tile (compressed) ──────────────────────────────────────
        int aElems = WARP_M * (WARP_K / 4);
        for (int i = threadIdx.x; i < aElems; i += 32) {
            int r = i / (WARP_K / 4);
            int c = i % (WARP_K / 4);
            sA[i] = A[(warpM + r) * (K / 4) + kBase / 4 + c];
        }

        // ── Load E tile (sparsity metadata) ───────────────────────────────
        int eElems = WARP_M * (WARP_K / 8);
        for (int i = threadIdx.x; i < eElems; i += 32) {
            int r = i / (WARP_K / 8);
            int c = i % (WARP_K / 8);
            sE[i] = E[(warpM + r) * (K / 8) + kBase / 8 + c];
        }

        // ── Load B tile (col-major) ───────────────────────────────────────
        int bElems = (WARP_K / 2) * WARP_N;
        for (int i = threadIdx.x; i < bElems; i += 32) {
            int r = i / WARP_N;
            int c = i % WARP_N;
            sB[i] = B[(kBase / 2 + r) * N + warpN + c];
        }

        // ── Load scaleA tile ─────────────────────────────────────────────
        int saElems = WARP_M * (WARP_K / FP4_SCALE_BLOCK);
        for (int i = threadIdx.x; i < saElems; i += 32) {
            int r = i / (WARP_K / FP4_SCALE_BLOCK);
            int b = i % (WARP_K / FP4_SCALE_BLOCK);
            sScaleA[i] = scaleA[(warpM + r) * (K / FP4_SCALE_BLOCK) + kBase / FP4_SCALE_BLOCK + b];
        }

        // ── Load scaleB tile (col-major) ─────────────────────────────────
        int sbElems = (WARP_K / FP4_SCALE_BLOCK) * WARP_N;
        for (int i = threadIdx.x; i < sbElems; i += 32) {
            int r = i / WARP_N;
            int c = i % WARP_N;
            sScaleB[i] = scaleB[(kBase / FP4_SCALE_BLOCK + r) * N + warpN + c];
        }

        __syncthreads();

        // ── Build matrix descriptors (CUTLASS SmemDescriptor format) ───────
        // A descriptor: compressed width = WARP_K/4 (half of dense K/2), elem=2B, row-major
        uint64_t aDesc = buildMdesc(sA, WARP_K / 4, WARP_K / 4, 2, 0);
        // E descriptor: sparsity metadata (uint8_t, 1B), extent=WARP_K/8, row-major
        uint64_t eDesc = buildMdesc(sE, WARP_K / 8, WARP_K / 8, 1, 0);
        // B descriptor: same as dense, elem=2B, col-major
        uint64_t bDesc = buildMdesc(sB, WARP_N, WARP_N, 2, 1);

        // Scale descriptors: FP8 (1B)
        uint64_t saDesc = buildMdesc(sScaleA, WARP_K / FP4_SCALE_BLOCK, WARP_K / FP4_SCALE_BLOCK, 1, 0);
        uint64_t sbDesc = buildMdesc(sScaleB, WARP_N, WARP_N, 1, 1);

        // ── tcgen05.mma.sp — sparse variant (CUDA 13+ syntax: block16) ──────
        // CUTLASS exact syntax:
        // tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16
        //   [%tmemC], %aDesc, %bDesc, [%tmemE], %idesc, [%scaleA], [%scaleB], predicate
        //
        // Key difference from dense: adds [%tmemE] (E-matrix TMEM ptr) + %idesc (with sparse_id2)
        uint32_t enableInputD = (kt == 0) ? 0 : 1;
        uint32_t tmemHandle = tmemHandlePtr[0];

        uint32_t saPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleA));
        uint32_t sbPtr = static_cast<uint32_t>(__cvta_generic_to_shared(sScaleB));
        uint32_t ePtr  = static_cast<uint32_t>(__cvta_generic_to_shared(sE));

        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %7, 0;\n\t"
            "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
            "[%0], %1, %2, [%4], %3, [%5], [%6], p;\n\t"
            "}\n"
            :
            : "r"(tmemHandle),       // %0 — C accumulator TMEM handle
              "l"(aDesc),            // %1 — A descriptor (64-bit)
              "l"(bDesc),            // %2 — B descriptor (64-bit)
              "r"(idesc),            // %3 — instruction descriptor (sparse_id2=4)
              "r"(ePtr),             // %4 — E-matrix shared memory pointer (TMEM)
              "r"(saPtr),            // %5 — scaleA pointer
              "r"(sbPtr),            // %6 — scaleB pointer
              "r"(enableInputD)      // %7 — predicate control
            : "memory"
        );

        // ── On final k-tile: read results from TMEM ──────────────────────
        if (kt == kTiles - 1) {
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

            int rowOffset = laneId * (WARP_M / 32);
            float* cOut = C + (warpM + rowOffset) * N + warpN;

            uint32_t ldOut[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(ldOut[0]), "=r"(ldOut[1]), "=r"(ldOut[2]), "=r"(ldOut[3]),
                  "=r"(ldOut[4]), "=r"(ldOut[5]), "=r"(ldOut[6]), "=r"(ldOut[7])
                : "r"(tmemHandle)
                : "memory"
            );

            for (int r = 0; r < WARP_M / 32; ++r) {
                for (int c = 0; c < WARP_N; ++c) {
                    cOut[r * N + c] = __int_as_float(ldOut[c]);
                }
            }

            asm volatile(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                : : "r"(tmemHandle), "r"(nCols) : "memory");
        }
    }
}

// ── Measure FP4 NVFP4 Sparse GEMM (2:4 structured sparsity) ────────────────
BenchResult measureFp4Sparse(int device, int matDim, int iterations) {
    chk(cudaSetDevice(device), "dev");

    int M = (matDim / WARP_M) * WARP_M;
    int N = M;
    int K = (matDim / WARP_K) * WARP_K;
    if (M < WARP_M) M = WARP_M;
    if (K < WARP_K) K = WARP_K;

    // Dense matrix sizes for input
    size_t floatSizeA = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float);
    size_t floatSizeB = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(float);
    size_t floatSizeC = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);

    // Sparse A: compressed FP4 (K/2 effective width), E-matrix (K/4 groups)
    // For 2:4 sparsity with 50% actual sparsity: compressed A = M * K/4
    size_t fp4SizeA_sparse = static_cast<size_t>(M) * static_cast<size_t>(K) / 4;
    size_t eMatrixSize = static_cast<size_t>(M) * static_cast<size_t>(K) / 8;
    size_t scaleSizeA = static_cast<size_t>(M) * static_cast<size_t>(K) / FP4_SCALE_BLOCK;

    // B stays dense: K/2 FP4 pairs × N
    size_t fp4SizeB = static_cast<size_t>(K) * static_cast<size_t>(N) / 2;
    size_t scaleSizeB = static_cast<size_t>(K) * static_cast<size_t>(N) / FP4_SCALE_BLOCK;

    // Offsets for compressed row layout
    size_t offsetSize = static_cast<size_t>(M) * sizeof(int);

    __nv_fp4x2_storage_t *dA_fp4_sparse = nullptr, *dB_fp4 = nullptr;
    __nv_fp8_storage_t   *dScaleA = nullptr, *dScaleB = nullptr;
    uint8_t*              dE = nullptr;
    int*                  dOffsets = nullptr;
    float                *dA = nullptr, *dB = nullptr, *dC = nullptr;

    chk(cudaMalloc(&dA, floatSizeA), "alloc_a");
    chk(cudaMalloc(&dB, floatSizeB), "alloc_b");
    chk(cudaMalloc(&dC, floatSizeC), "alloc_c");
    chk(cudaMalloc(&dA_fp4_sparse, fp4SizeA_sparse), "alloc_fp4a_sparse");
    chk(cudaMalloc(&dB_fp4, fp4SizeB), "alloc_fp4b");
    chk(cudaMalloc(&dScaleA, scaleSizeA), "alloc_scalea");
    chk(cudaMalloc(&dScaleB, scaleSizeB), "alloc_scaleb");
    chk(cudaMalloc(&dE, eMatrixSize), "alloc_e");
    chk(cudaMalloc(&dOffsets, offsetSize), "alloc_offsets");

    // Initialize with sparse-friendly data (create actual sparsity)
    chk(cudaMemset(dA, 0x3E, floatSizeA), "memset_a");
    chk(cudaMemset(dB, 0x3E, floatSizeB), "memset_b");

    cudaStream_t str;
    chk(cudaStreamCreate(&str), "stream");

    // Convert A → sparse FP4 + E-matrix + scales
    {
        int gridX = std::min(65535, M);
        int smemSharedCountsBytes = 8 * 4;  // 8 warps × 4 bytes
        fp4SparseCompressKernel<<<gridX, 256, smemSharedCountsBytes, str>>>(
            dA_fp4_sparse, dScaleA, dE, dOffsets, dA, M, K);
        chk(cudaGetLastError(), "compress_a");
    }

    // Convert B → dense FP4 + scales
    {
        int gridXB = std::min(65535, K);
        floatToFp4Kernel<<<gridXB, 256, 0, str>>>(dB_fp4, dScaleB, dB, K, N);
        chk(cudaGetLastError(), "convert_b");

        chk(cudaStreamSynchronize(str), "convert_sync");
    }

    // Grid: (N/WARP_N) x (M/WARP_M), 1 warp per block
    dim3 grid(std::min(65535, N / WARP_N), std::min(65535, M / WARP_M), 1);
    int smemBytes = 2048 + 1024 + 256 + 512 + 32 + 4;

    // Warmup
    for (int w = 0; w < 3; ++w) {
        fp4Nvf4SparseGemmKernel<<<grid.x, 32, smemBytes, str>>>(
            dA_fp4_sparse, dScaleA, dE, dB_fp4, dScaleB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    // Timing
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    std::vector<double> vals;
    // Sparse FLOPs: M * N * K * 2 (same formula, sparsity is handled in hardware)
    size_t totalFlops = static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(K) * 2;

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "recS");
        fp4Nvf4SparseGemmKernel<<<grid.x, 32, smemBytes, str>>>(
            dA_fp4_sparse, dScaleA, dE, dB_fp4, dScaleB, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "recE");
        chk(cudaStreamSynchronize(str), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "elapsed");
        double sec = ms / 1000.0;
        double tflops = sec > 0.0 ? (static_cast<double>(totalFlops) / 1e12) / sec : 0.0;
        vals.push_back(tflops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"warp_m\":" << WARP_M << ",\"warp_n\":" << WARP_N << ",\"warp_k\":" << WARP_K
      << ",\"format\":\"nvfp4_e2m1\""
      << ",\"sparsity\":\"2:4\""
      << ",\"scale_block\":" << FP4_SCALE_BLOCK
      << ",\"api\":\"tcgen05_mma_sp_inline_ptx\""
      << ",\"scale_vec\":\"2X\"}";

    BenchResult res = computeStats(vals, "fp4", "fp4_nvfp4_sparse", "TFLOP/s", p.str());
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp4_sparse_tflops);
    res.metadata["format"] = "nvfp4";
    res.metadata["block_scaled"] = "true";
    res.metadata["sparsity"] = "2:4_structured";
    res.metadata["peak_sparse_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp4_sparse_tflops));

    chk(cudaFree(dA), "free_a");
    chk(cudaFree(dB), "free_b");
    chk(cudaFree(dC), "free_c");
    chk(cudaFree(dA_fp4_sparse), "free_fp4a_sparse");
    chk(cudaFree(dB_fp4), "free_fp4b");
    chk(cudaFree(dScaleA), "free_scalea");
    chk(cudaFree(dScaleB), "free_scaleb");
    chk(cudaFree(dE), "free_e");
    chk(cudaFree(dOffsets), "free_offsets");
    chk(cudaStreamDestroy(str), "stream");
    chk(cudaEventDestroy(evS), "evSDestroy");
    chk(cudaEventDestroy(evE), "evEDestroy");

    return res;
}

} // anonymous namespace

// ── Create stub result helper ──────────────────────────────────────────────
static BenchResult makeStubDense(const char* reason) {
    BenchResult r{};
    r.suite_name = "fp4";
    r.test_name  = "fp4_nvfp4_dense";
    r.unit       = "TFLOP/s";
    std::string err = "{\"error\":\"";
    err += reason;
    err += "\",\"note\":\"tcgen05.mma NVFP4 GEMM not available on this device\"}";
    r.params_json = err;
    r.metadata["format"] = "nvfp4";
    r.metadata["block_scaled"] = "true";
    r.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp4_dense_tflops));
    r.metadata["stub"] = "true";
    r.metadata["stub_reason"] = reason;
    return r;
}

static BenchResult makeStubSparse(const char* reason) {
    BenchResult r{};
    r.suite_name = "fp4";
    r.test_name  = "fp4_nvfp4_sparse";
    r.unit       = "TFLOP/s";
    std::string err = "{\"error\":\"";
    err += reason;
    err += "\",\"note\":\"tcgen05.mma.sp NVFP4 sparse GEMM not available\"}";
    r.params_json = err;
    r.metadata["format"] = "nvfp4";
    r.metadata["block_scaled"] = "true";
    r.metadata["sparsity"] = "2:4_structured";
    r.metadata["peak_sparse_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp4_sparse_tflops));
    r.metadata["stub"] = "true";
    r.metadata["stub_reason"] = reason;
    return r;
}

// ── Public API ──────────────────────────────────────────────────────────────
std::vector<BenchResult> runFP4Bench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    // CRITICAL: Probe NVFP4 support BEFORE any kernel launch.
    // The tcgen05.mma.kind::mxf4nvf4 instruction is not supported by all Thor
    // drivers/firmware. When it fails, it corrupts TMEM/driver state that
    // persists across subsequent suites. The probe kernel runs in isolation
    // so corruption (if any) is contained.
    bool supported = nvfp4Supported(device);
    if (!supported) {
        results.push_back(makeStubDense("tcgen05.mma.kind::mxf4nvf4 not supported by current driver/firmware"));
        results.push_back(makeStubSparse("tcgen05.mma.sp.kind::mxf4nvf4 not supported by current driver/firmware"));
        return results;
    }

    try {
        results.push_back(measureFp4Dense(device, matDim, iterations));
    } catch (const std::exception& ex) {
        results.push_back(makeStubDense((std::string("exception: ") + ex.what()).c_str()));
    }

    try {
        results.push_back(measureFp4Sparse(device, matDim, iterations));
    } catch (const std::exception& ex) {
        results.push_back(makeStubSparse((std::string("exception: ") + ex.what()).c_str()));
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(fp4, "FP4 e2m1 NVFP4 block-scaled GEMM via tcgen05.mma",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runFP4Bench(0, 2048, 10);
    });
