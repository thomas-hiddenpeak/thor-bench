#include "compute/fp8_scalar_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace deusridet::bench {

namespace {

constexpr int kTpb = 256;
constexpr int kTile = 8;

// ── FP8 dense matmul kernel ────────────────────────────────────────────────
// C[M,N] = A[M,K] @ B[K,N], tiled scalar kernel.
// Inputs stored as FP8 (__nv_fp8_storage_t), accumulated in float.
__global__ void fp8MatmulKernel(
    const __nv_fp8_storage_t* __restrict__ A,
    const __nv_fp8_storage_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.x * kTile + threadIdx.x;
    int col = blockIdx.y * kTile + threadIdx.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = __half2float(static_cast<__half>(
            __nv_cvt_fp8_to_halfraw(A[row * K + k], __NV_E4M3)));
        float b = __half2float(static_cast<__half>(
            __nv_cvt_fp8_to_halfraw(B[k * N + col], __NV_E4M3)));
        sum = fmaf(a, b, sum);
    }
    C[row * N + col] = sum;
}

// ── Float → FP8 conversion kernel ──────────────────────────────────────────
__global__ void floatToFp8Kernel(
    __nv_fp8_storage_t* __restrict__ out,
    const float* __restrict__ inp,
    size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        out[idx] = __nv_cvt_float_to_fp8(inp[idx], __NV_SATFINITE, __NV_E4M3);
}

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}
// ── Measure FP8 Dense ──────────────────────────────────────────────────────
BenchResult measureFP8Dense(int device, int matDim, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    chk(cudaSetDevice(device), "dev");

    matDim = std::min(matDim, 8192);
    int M = matDim, N = matDim, K = matDim;

    size_t sizeF32 = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t sizeC   = static_cast<size_t>(M) * static_cast<size_t>(N);

    float *dA = nullptr, *dB = nullptr;
    __nv_fp8_storage_t *dA8 = nullptr, *dB8 = nullptr;
    float *dC = nullptr;

    chk(cudaMalloc(&dA, sizeF32 * sizeof(float)), "a");
    chk(cudaMalloc(&dB, sizeF32 * sizeof(float)), "b");
    chk(cudaMalloc(&dA8, sizeF32 * sizeof(__nv_fp8_storage_t)), "a8");
    chk(cudaMalloc(&dB8, sizeF32 * sizeof(__nv_fp8_storage_t)), "b8");
    chk(cudaMalloc(&dC, sizeC * sizeof(float)), "c");

    chk(cudaMemset(dA, 0x3E, sizeF32 * sizeof(float)), "a");
    chk(cudaMemset(dB, 0x3E, sizeF32 * sizeof(float)), "b");

    int gridConv = std::max(1, static_cast<int>((sizeF32 + kTpb - 1) / kTpb));
    floatToFp8Kernel<<<gridConv, kTpb, 0, str>>>(dA8, dA, M * K);
    floatToFp8Kernel<<<gridConv, kTpb, 0, str>>>(dB8, dB, K * N);
    chk(cudaStreamSynchronize(str), "sync");

    dim3 grid(std::min(65535, (M + kTile - 1) / kTile),
              std::min(65535, (N + kTile - 1) / kTile), 1);
    dim3 block(kTile, kTile, 1);

    for (int w = 0; w < 3; ++w) {
        fp8MatmulKernel<<<grid, block, 0, str>>>(dA8, dB8, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        fp8MatmulKernel<<<grid, block, 0, str>>>(dA8, dB8, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        size_t totalFlops = static_cast<size_t>(M) * N * K * 2;
        double tflops = sec > 0.0 ? (totalFlops / 1e12) / sec : 0.0;
        vals.push_back(tflops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"tile\":" << kTile << ",\"format\":\"e4m3\""
      << ",\"type\":\"scalar_fp8_kernel\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "fp8_scalar";
    res.test_name  = "fp8_scalar_dense";
    res.unit       = "TFLOP/s";
    res.params_json = p.str();
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp8_dense_tflops);
    res.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_dense_tflops));
    res.metadata["note"] = "scalar FP8 kernel; tcgen05.mma kind::f8f6f4 PTX requires SMEM descriptors + TMEM alloc";

    chk(cudaFree(dA), "fa");
    chk(cudaFree(dB), "fb");
    chk(cudaFree(dA8), "fa8");
    chk(cudaFree(dB8), "fb8");
    chk(cudaFree(dC), "fc");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ════════════════════════════════════════════════════════════════════════════
// ── FP8 2:4 Sparse GEMM via tcgen05.mma.sp ─────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════

// Probe kernel: test tcgen05.mma.sp support
__global__ void fp8ScalarSparseProbeKernel() {
    uint32_t tmemHandleSmem;
    uint32_t nCols = 8;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmemHandleSmem));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    uint64_t aDesc = 0, bDesc = 0;
    uint32_t idesc = 0;
    uint32_t tmemHandle = tmemHandleSmem;
    uint32_t ePtrSmem = 0;
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.ne.b32 p, 0, 0;\n\t"
        "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 "
        "[%0], %1, %2, [%4], %3, p;}\n"
        : : "r"(tmemHandle), "l"(aDesc), "l"(bDesc), "r"(idesc), "r"(ePtrSmem)
        : "memory"
    );
}

static bool fp8ScalarSparseSupported(int device) {
    chk(cudaSetDevice(device), "probe_dev");
    int major = 0;
    chk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "major");
    // HARD LESSON: tcgen05.mma.sp kind::f8f6f4 IllegalInstruction permanently
    // poisons the CUDA context on driver 595.58.03. cudaDeviceReset() fails on Tegra.
    // Never attempt to run this kernel — return stub immediately.
    return false;
}

constexpr int spFp8ScalarTileM = 16;
constexpr int spFp8ScalarTileN = 16;
constexpr int spFp8ScalarTileK = 16;

// ── Matrix descriptor builder (64-bit) ─────────────────────────────────────
__device__ static uint64_t buildFp8ScalarSmemDesc(const void* ptr, int leadingDim, int strideDim,
                                                    int elemBytes, int layoutType) {
    uint64_t base = static_cast<uint64_t>(__cvta_generic_to_shared(ptr));
    uint64_t leadingBytes = static_cast<uint64_t>(leadingDim) * elemBytes;
    uint64_t strideBytes  = static_cast<uint64_t>(strideDim) * elemBytes;
    uint64_t desc = 0;
    desc  = (base >> 4) & 0x3FFF;
    desc |= ((leadingBytes >> 4) & 0x3FFF) << 16;
    desc |= ((strideBytes >> 4)  & 0x3FFF) << 32;
    desc |= (1ULL) << 46;
    desc |= (0ULL) << 49;
    desc |= (0ULL) << 52;
    desc |= (static_cast<uint64_t>(layoutType & 0x7)) << 61;
    return desc;
}

// ── Instruction descriptor for kind::f8f6f4 (FP8 sparse) ──────────────────
__device__ static uint32_t buildFp8ScalarIdesc(int M, int N) {
    constexpr uint8_t F8F6F4 = 0x3;
    uint32_t idesc = 0;
    idesc  = F8F6F4;
    idesc |= (F8F6F4 << 10);
    idesc |= (0 << 15);
    idesc |= (1 << 16);
    idesc |= ((N >> 4) & 0x3F) << 17;
    idesc |= ((M >> 4) & 0x1F) << 24;
    return idesc;
}

// ── FP8 Sparse Compressor kernel ───────────────────────────────────────────
__global__ void fp8ScalarSparseCompressKernel(
    __nv_fp8_storage_t* __restrict__ outCompressed,
    uint8_t*            __restrict__ outE,
    int*                __restrict__ outOffsets,
    const __nv_fp8_storage_t* __restrict__ inp,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_fp8_storage_t* rowPtr = inp + static_cast<size_t>(row) * cols;
    uint8_t* ePtr = outE + static_cast<size_t>(row) * (cols / 4);

    int groupsPerRow = cols / 4;
    int tid = threadIdx.x;

    int rowNnz = 0;
    for (int g = tid; g < groupsPerRow; g += blockDim.x) {
        int valBase = g * 4;
        uint8_t mask = 0;
        int nnz = 0;
        for (int v = 0; v < 4; ++v) {
            uint8_t raw = static_cast<uint8_t>(rowPtr[valBase + v]);
            if (raw != 0) {
                mask |= (1 << v);
                nnz++;
            }
        }
        ePtr[g] = mask;
        rowNnz += nnz;
    }

    extern __shared__ int sSum[];
    sSum[tid] = rowNnz;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) sSum[tid] += sSum[tid + stride];
    }
    if (tid == 0) outOffsets[row] = sSum[0];

    __syncthreads();
    __syncthreads();
    sSum[0] = 0;
    __syncthreads();

    for (int g = 0; g < groupsPerRow; ++g) {
        uint8_t mask = ePtr[g];
        int valBase = g * 4;
        for (int v = 0; v < 4; ++v) {
            if (mask & (1 << v)) {
                int atomicIdx = atomicAdd(&sSum[0], 1);
                outCompressed[static_cast<size_t>(row) * (cols / 2) + atomicIdx] = rowPtr[valBase + v];
            }
        }
    }
}

// ── FP8 2:4 Sparse GEMM kernel via tcgen05.mma.sp ─────────────────────────
__global__ void fp8ScalarSparseMmaKernel(
    const __nv_fp8_storage_t* __restrict__ A,
    const uint8_t*            __restrict__ E,
    const __nv_fp8_storage_t* __restrict__ B,
    float*                    __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y * spFp8ScalarTileM;
    int warpN = blockIdx.x * spFp8ScalarTileN;

    if (warpM >= M || warpN >= N) return;

    int laneId = threadIdx.x;

    extern __shared__ char smem[];

    __nv_fp8_storage_t* sA            = reinterpret_cast<__nv_fp8_storage_t*>(smem);
    uint8_t*            sE            = reinterpret_cast<uint8_t*>(smem + spFp8ScalarTileM * (spFp8ScalarTileK / 2));
    __nv_fp8_storage_t* sB            = reinterpret_cast<__nv_fp8_storage_t*>(smem + spFp8ScalarTileM * (spFp8ScalarTileK / 2) + spFp8ScalarTileM * (spFp8ScalarTileK / 4));
    uint32_t*           tmemHandlePtr = reinterpret_cast<uint32_t*>(
        smem + spFp8ScalarTileM * (spFp8ScalarTileK / 2) + spFp8ScalarTileM * (spFp8ScalarTileK / 4) + spFp8ScalarTileK * spFp8ScalarTileN);

    uint32_t nCols = spFp8ScalarTileN;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(tmemHandlePtr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    uint32_t idesc = buildFp8ScalarIdesc(spFp8ScalarTileM, spFp8ScalarTileN) | 4;

    int kTiles = K / spFp8ScalarTileK;

    for (int kt = 0; kt < kTiles; ++kt) {
        int kBase = kt * spFp8ScalarTileK;

        int aElems = spFp8ScalarTileM * (spFp8ScalarTileK / 2);
        for (int i = threadIdx.x; i < aElems; i += 32) {
            int r = i / (spFp8ScalarTileK / 2);
            int c = i % (spFp8ScalarTileK / 2);
            sA[i] = A[(warpM + r) * (K / 2) + kBase / 2 + c];
        }

        int eElems = spFp8ScalarTileM * (spFp8ScalarTileK / 4);
        for (int i = threadIdx.x; i < eElems; i += 32) {
            int r = i / (spFp8ScalarTileK / 4);
            int c = i % (spFp8ScalarTileK / 4);
            sE[i] = E[(warpM + r) * (K / 4) + kBase / 4 + c];
        }

        int bElems = spFp8ScalarTileK * spFp8ScalarTileN;
        for (int i = threadIdx.x; i < bElems; i += 32) {
            int r = i / spFp8ScalarTileN;
            int c = i % spFp8ScalarTileN;
            sB[i] = B[(kBase + r) * N + warpN + c];
        }

        __syncthreads();

        uint64_t aDesc = buildFp8ScalarSmemDesc(sA, spFp8ScalarTileK / 2, spFp8ScalarTileK / 2, 1, 0);
        uint64_t bDesc = buildFp8ScalarSmemDesc(sB, spFp8ScalarTileN, spFp8ScalarTileN, 1, 1);

        uint32_t enableInputD = (kt == 0) ? 0 : 1;
        uint32_t tmemHandle = tmemHandlePtr[0];
        uint32_t ePtrSmem = static_cast<uint32_t>(__cvta_generic_to_shared(sE));

        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.ne.b32 p, %5, 0;\n\t"
            "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 "
            "[%0], %1, %2, [%4], %3, p;}\n"
            :
            : "r"(tmemHandle), "l"(aDesc), "l"(bDesc),
              "r"(idesc), "r"(ePtrSmem), "r"(enableInputD)
            : "memory"
        );

        if (kt == kTiles - 1) {
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

            int row    = (laneId / 2);
            int colOff = (laneId % 2) * 8;
            float* cRow = C + (warpM + row) * N + warpN + colOff;

            uint32_t ldOut[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(ldOut[0]), "=r"(ldOut[1]), "=r"(ldOut[2]), "=r"(ldOut[3]),
                  "=r"(ldOut[4]), "=r"(ldOut[5]), "=r"(ldOut[6]), "=r"(ldOut[7])
                : "r"(tmemHandle)
                : "memory"
            );

            for (int c = 0; c < 8; ++c) {
                cRow[c] = __int_as_float(ldOut[c]);
            }

            if (laneId == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                             : : "r"(tmemHandle), "r"(nCols) : "memory");
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
            }
        }
    }
}

BenchResult measureFP8Sparse(int device, int matDim, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    chk(cudaSetDevice(device), "dev");

    int M = (matDim / spFp8ScalarTileM) * spFp8ScalarTileM;
    int N = M;
    int K = (matDim / spFp8ScalarTileK) * spFp8ScalarTileK;
    if (M < spFp8ScalarTileM) M = spFp8ScalarTileM;
    if (K < spFp8ScalarTileK) K = spFp8ScalarTileK;

    size_t sizeA_dense  = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t sizeB        = static_cast<size_t>(K) * static_cast<size_t>(N);
    size_t sizeC        = static_cast<size_t>(M) * static_cast<size_t>(N);
    size_t sizeA_sparse = static_cast<size_t>(M) * static_cast<size_t>(K) / 2;
    size_t sizeE = static_cast<size_t>(M) * static_cast<size_t>(K) / 4;
    size_t sizeOffsets = static_cast<size_t>(M) * sizeof(int);

    __nv_fp8_storage_t* dA_dense  = nullptr;
    __nv_fp8_storage_t* dA_sparse = nullptr;
    __nv_fp8_storage_t* dB        = nullptr;
    float*              dC        = nullptr;
    uint8_t*            dE        = nullptr;
    int*                dOffsets  = nullptr;

    chk(cudaMalloc(&dA_dense,  sizeA_dense),           "dense_a");
    chk(cudaMalloc(&dA_sparse, sizeA_sparse),          "sparse_a");
    chk(cudaMalloc(&dB,        sizeB),                 "b");
    chk(cudaMalloc(&dC,        sizeC * sizeof(float)), "c");
    chk(cudaMalloc(&dE,        sizeE),                 "e");
    chk(cudaMalloc(&dOffsets,  sizeOffsets),           "offsets");

    chk(cudaMemset(dA_dense, 0x3E, sizeA_dense), "dense_a");
    chk(cudaMemset(dB,       0x3E, sizeB),       "b");

    {
        int gridX = std::min(65535, M);
        int smemBytes = sizeof(int) * 256;
        fp8ScalarSparseCompressKernel<<<gridX, 256, smemBytes, str>>>(
            dA_sparse, dE, dOffsets, dA_dense, M, K);
        chk(cudaStreamSynchronize(str), "compress_sync");
    }

    dim3 grid(std::min(65535, N / spFp8ScalarTileN), std::min(65535, M / spFp8ScalarTileM), 1);
    int smemBytes = spFp8ScalarTileM * (spFp8ScalarTileK / 2) + spFp8ScalarTileM * (spFp8ScalarTileK / 4)
                  + spFp8ScalarTileK * spFp8ScalarTileN + 4;

    for (int w = 0; w < 3; ++w) {
        fp8ScalarSparseMmaKernel<<<grid, 32, smemBytes, str>>>(
            dA_sparse, dE, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    size_t totalFlops = static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(K) * 2;

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        fp8ScalarSparseMmaKernel<<<grid, 32, smemBytes, str>>>(
            dA_sparse, dE, dB, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double tflops = sec > 0.0 ? (static_cast<double>(totalFlops) / 1e12) / sec : 0.0;
        vals.push_back(tflops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"tile_m\":" << spFp8ScalarTileM << ",\"tile_n\":" << spFp8ScalarTileN
      << ",\"tile_k\":" << spFp8ScalarTileK << ",\"sparsity\":\"2:4\""
      << ",\"api\":\"tcgen05_mma_sp_inline_ptx\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "fp8_scalar";
    res.test_name  = "fp8_scalar_sparse";
    res.unit       = "TFLOP/s";
    res.peak_pct   = computePeakPctFromT(res.median, T5000Peaks::fp8_sparse_tflops);
    res.params_json = p.str();
    res.metadata["tcgen05"] = "true";
    res.metadata["precision"] = "fp8_e4m3";
    res.metadata["sparsity"] = "2:4_structured";
    res.metadata["sparse_instruction"] = "tcgen05.mma.sp.kind::f8f6f4";
    res.metadata["peak_sparse_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_sparse_tflops));

    chk(cudaFree(dA_dense),  "free_dense_a");
    chk(cudaFree(dA_sparse), "free_sparse_a");
    chk(cudaFree(dB),        "free_b");
    chk(cudaFree(dC),        "free_c");
    chk(cudaFree(dE),        "free_e");
    chk(cudaFree(dOffsets),  "free_offsets");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

} // anonymous namespace

std::vector<BenchResult> runFP8ScalarBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    // --- FP8 Dense ---
    try {
        results.push_back(measureFP8Dense(device, matDim, iterations));
    } catch (const std::exception& ex) {
        cudaDeviceSynchronize();
        cudaGetLastError();
        BenchResult r{};
        r.suite_name = "fp8_scalar";
        r.test_name  = "fp8_scalar_dense";
        r.unit       = "TFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        r.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_dense_tflops));
        results.push_back(r);
    }

    // --- FP8 Sparse ---
    // Probe tcgen05 support first; fallback to stub if unsupported
    try {
        if (!fp8ScalarSparseSupported(device)) {
            BenchResult r{};
            r.suite_name = "fp8_scalar";
            r.test_name  = "fp8_scalar_sparse";
            r.unit       = "TFLOP/s";
            r.sample_count = 0;
            r.warmup_count = 0;
            r.median = 0.0;
            r.params_json = "{\"note\":\"tcgen05.mma.sp.kind::f8f6f4 not supported by current driver/firmware\"}";
            r.metadata["stub"] = "true";
            r.metadata["stub_reason"] = "tcgen05.mma.sp.kind::f8f6f4 not supported by current driver/firmware";
            r.metadata["peak_sparse_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_sparse_tflops));
            results.push_back(r);
        } else {
            results.push_back(measureFP8Sparse(device, matDim, iterations));
        }
    } catch (const std::exception& ex) {
        // CRITICAL: tcgen05 IllegalInstruction poisons device context.
        // MUST synchronize to drain error BEFORE any subsequent CUDA call.
        cudaDeviceSynchronize();
        cudaGetLastError();
        BenchResult r{};
        r.suite_name = "fp8_scalar";
        r.test_name  = "fp8_scalar_sparse";
        r.unit       = "TFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        r.metadata["peak_sparse_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_sparse_tflops));
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(fp8_scalar, "Scalar FP8 GEMM (no Tensor Core — ~0.04% of peak)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runFP8ScalarBench(0, 512, 10);
    });
