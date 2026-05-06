#include "compute/sasp_bench.h"
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
constexpr int kTile = 8;   // tile dimension per thread-block row/col

// --- FP8 dense matmul kernel ---
// C[M,N] = A[M,K] @ B[K,N], tiled.
// Inputs stored as FP8 (__nv_fp8_storage_t, 1 byte each), accumulated in float.
// Uses fmaf for FP8→float→float multiply-add per element.
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

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

__global__ void floatToFp8Kernel(
    __nv_fp8_storage_t* __restrict__ out,
    const float* __restrict__ inp,
    size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        out[idx] = __nv_cvt_float_to_fp8(inp[idx], __NV_SATFINITE, __NV_E4M3);
}

BenchResult measureFP8Dense(int device, int matDim, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    chk(cudaSetDevice(device), "dev");

    matDim = std::min(matDim, 8192); // upper bound guard
    int M = matDim, N = matDim, K = matDim;
    size_t sizeFp8 = static_cast<size_t>(M) * static_cast<size_t>(K) + static_cast<size_t>(K) * static_cast<size_t>(N); // A + B in fp8
    size_t sizeF32 = static_cast<size_t>(M) * static_cast<size_t>(K) + static_cast<size_t>(K) * static_cast<size_t>(N); // float inputs
    size_t sizeC   = static_cast<size_t>(M) * static_cast<size_t>(N);                                                   // float output

    float *dA = nullptr, *dB = nullptr;
    __nv_fp8_storage_t *dA8 = nullptr, *dB8 = nullptr;
    float *dC = nullptr;

    chk(cudaMalloc(&dA, sizeF32 * sizeof(float)), "a");
    chk(cudaMalloc(&dB, sizeF32 * sizeof(float)), "b");
    chk(cudaMalloc(&dA8, sizeFp8 * sizeof(__nv_fp8_storage_t)), "a8");
    chk(cudaMalloc(&dB8, sizeFp8 * sizeof(__nv_fp8_storage_t)), "b8");
    chk(cudaMalloc(&dC, sizeC * sizeof(float)), "c");

    // Initialize with small non-zero values to avoid NaN/Inf in FP8
    chk(cudaMemset(dA, 0x3E, sizeF32 * sizeof(float)), "a"); // ~0.23
    chk(cudaMemset(dB, 0x3E, sizeF32 * sizeof(float)), "b");

    // Convert to FP8 once (outside timing)
    int gridConv = std::max(1, static_cast<int>((static_cast<size_t>(M) * static_cast<size_t>(K) + kTpb - 1) / kTpb));
    floatToFp8Kernel<<<gridConv, kTpb, 0, str>>>(dA8, dA, M * K);
    floatToFp8Kernel<<<gridConv, kTpb, 0, str>>>(dB8, dB, K * N);
    chk(cudaStreamSynchronize(str), "sync");

    // Grid for matmul: one thread per output element in kTile x kTile blocks
    dim3 grid(std::min(65535, (M + kTile - 1) / kTile), std::min(65535, (N + kTile - 1) / kTile), 1);
    dim3 block(kTile, kTile, 1);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        fp8MatmulKernel<<<grid, block, 0, str>>>(dA8, dB8, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    // Measure
    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        fp8MatmulKernel<<<grid, block, 0, str>>>(dA8, dB8, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;

        // Each element of C requires K multiply + K add = 2K FLOPs
        // Total = M * N * 2 * K FLOPs
        size_t totalFlops = static_cast<size_t>(M) * N * K * 2;
        double tflops = sec > 0.0 ? (totalFlops / 1e12) / sec : 0.0;
        vals.push_back(tflops);
    }

    // Compute stats
    BenchResult res;
    res.suite_name = "sasp";
    res.test_name  = "sasp_fp8_dense";
    res.unit       = "TFLOP/s";
    res.warmup_count = 3;

    int n = static_cast<int>(vals.size());
    res.sample_count = n;
    if (!vals.empty()) {
        std::sort(vals.begin(), vals.end());
        double sum = 0;
        for (double v : vals) sum += v;
        double mean = sum / n;

        res.min_val  = vals.front();
        res.max_val  = vals.back();
        res.mean     = mean;
        res.median   = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;

        double sq = 0;
        for (double v : vals) { double d = v - mean; sq += d * d; }
        res.stddev = std::sqrt(sq / n);

        auto pct = [&](double p) -> double {
            if (n <= 1) return vals[0];
            double r = p * (n - 1);
            int lo = static_cast<int>(std::floor(r));
            int hi = static_cast<int>(std::ceil(r));
            if (hi >= n) return vals.back();
            return vals[lo] * (1.0 - (r - lo)) + vals[hi] * (r - lo);
        };
        res.p95 = pct(0.95);
        res.p99 = pct(0.99);
        res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp8_dense_tflops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"tile\":" << kTile
      << ",\"format\":\"e4m3\""
      << ",\"type\":\"scalar_fp8_kernel\"}";
    res.params_json = p.str();
    res.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_dense_tflops));
    res.metadata["note"] = "scalar kernel with FP8 storage + float accumulate; no WMMA/tcgen05";

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

// Probe kernel: test tcgen05.mma.sp support (same pattern as fp4_bench.cu)
__global__ void fp8SparseProbeKernel() {
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

static bool fp8SparseSupported(int device) {
    chk(cudaSetDevice(device), "probe_dev");
    int major = 0, minor = 0;
    chk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "major");
    chk(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "minor");
    if (major < 11) return false;

    cudaStream_t str;
    chk(cudaStreamCreate(&str), "probe_stream");
    fp8SparseProbeKernel<<<1, 32, 0, str>>>();
    cudaError_t e = cudaStreamSynchronize(str);
    chk(cudaStreamDestroy(str), "probe_stream_destroy");
    if (e != cudaSuccess) {
        cudaGetLastError(); // drain IllegalInstruction from probe
        return false;
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaGetLastError(); // drain
        return false;
    }
    return true;
}

// Tile dimensions: 16x16x16 per warp (same as INT8 TC sparse kernel).
constexpr int spFp8TileM = 16;
constexpr int spFp8TileN = 16;
constexpr int spFp8TileK = 16;

// ── Matrix descriptor builder (64-bit) — CUTLASS SmemDescriptor format ────
__device__ static uint64_t buildFp8SmemDesc(const void* ptr, int leadingDim, int strideDim,
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

// ── Instruction descriptor (32-bit) for kind::f8f6f4 (FP8 sparse) ─────────
// Bit  [7:10):    a_format_      (F8F6F4 = 0x3)
// Bit  [10:13):   b_format_      (F8F6F4 = 0x3)
// Bit  [15:16):   a_major_       (0 = row-major)
// Bit  [16:17):   b_major_       (1 = col-major)
// Bit  [17:23):   n_dim_         (N >> 4)
// Bit  [24:29):   m_dim_         (M >> 4)
// Bits [0:2]:     sparse_id2     (4 = 2:4 sparsity for .sp variant)
__device__ static uint32_t buildFp8Idesc(int M, int N) {
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
// 2:4 sparsity: each group of 4 FP8 values has at most 2 non-zeros.
// E-matrix encodes which positions are non-zero (uint8_t bitmask per group).
// Compressed A stores only non-zero values packed row-wise.
__global__ void fp8SparseCompressKernel(
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

    // Build E-matrix and count non-zeros per row
    int rowNnz = 0;
    for (int g = tid; g < groupsPerRow; g += blockDim.x) {
        int valBase = g * 4;
        uint8_t mask = 0;
        int nnz = 0;
        for (int v = 0; v < 4; ++v) {
            // Check if FP8 value is non-zero (compare raw byte)
            uint8_t raw = static_cast<uint8_t>(rowPtr[valBase + v]);
            if (raw != 0) {
                mask |= (1 << v);
                nnz++;
            }
        }
        ePtr[g] = mask;
        rowNnz += nnz;
    }

    // Sum rowNnz across threads
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

    // Pack non-zero values into compressed storage
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
// Each warp (block=32 threads) handles spFp8TileM x spFp8TileN tile.
// A is row-major compressed: M x (K/2) __nv_fp8_storage_t
// E is row-major:             M x (K/4) uint8_t
// B is row-major dense:       K x N __nv_fp8_storage_t
// C is row-major:             M x N float
__global__ void fp8SparseMmaKernel(
    const __nv_fp8_storage_t* __restrict__ A,
    const uint8_t*            __restrict__ E,
    const __nv_fp8_storage_t* __restrict__ B,
    float*                    __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y * spFp8TileM;
    int warpN = blockIdx.x * spFp8TileN;

    if (warpM >= M || warpN >= N) return;

    int laneId = threadIdx.x;

    // ── Shared memory layout ─────────────────────────────────────────────
    // A tile (compressed): spFp8TileM x (spFp8TileK/2) = 16 x 8  = 128 bytes
    // E tile (metadata):   spFp8TileM x (spFp8TileK/4) = 16 x 4  =  64 bytes
    // B tile:              spFp8TileK x spFp8TileN     = 16 x 16 = 256 bytes
    // tmemHandle:                          4 bytes
    extern __shared__ char smem[];

    __nv_fp8_storage_t* sA            = reinterpret_cast<__nv_fp8_storage_t*>(smem);
    uint8_t*            sE            = reinterpret_cast<uint8_t*>(smem + spFp8TileM * (spFp8TileK / 2));
    __nv_fp8_storage_t* sB            = reinterpret_cast<__nv_fp8_storage_t*>(smem + spFp8TileM * (spFp8TileK / 2) + spFp8TileM * (spFp8TileK / 4));
    uint32_t*           tmemHandlePtr = reinterpret_cast<uint32_t*>(
        smem + spFp8TileM * (spFp8TileK / 2) + spFp8TileM * (spFp8TileK / 4) + spFp8TileK * spFp8TileN);

    // ── TMEM allocation ──────────────────────────────────────────────────
    uint32_t nCols = spFp8TileN;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(tmemHandlePtr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    // Build idesc: base fp8 idesc | sparse_id2=4
    uint32_t idesc = buildFp8Idesc(spFp8TileM, spFp8TileN) | 4;

    // ── K loop ───────────────────────────────────────────────────────────
    int kTiles = K / spFp8TileK;

    for (int kt = 0; kt < kTiles; ++kt) {
        int kBase = kt * spFp8TileK;

        // ── Load A tile (compressed) ─────────────────────────────────────
        int aElems = spFp8TileM * (spFp8TileK / 2);
        for (int i = threadIdx.x; i < aElems; i += 32) {
            int r = i / (spFp8TileK / 2);
            int c = i % (spFp8TileK / 2);
            sA[i] = A[(warpM + r) * (K / 2) + kBase / 2 + c];
        }

        // ── Load E tile (sparsity metadata) ──────────────────────────────
        int eElems = spFp8TileM * (spFp8TileK / 4);
        for (int i = threadIdx.x; i < eElems; i += 32) {
            int r = i / (spFp8TileK / 4);
            int c = i % (spFp8TileK / 4);
            sE[i] = E[(warpM + r) * (K / 4) + kBase / 4 + c];
        }

        // ── Load B tile (dense, row-major -> col-major in shared) ───────
        int bElems = spFp8TileK * spFp8TileN;
        for (int i = threadIdx.x; i < bElems; i += 32) {
            int r = i / spFp8TileN;
            int c = i % spFp8TileN;
            sB[i] = B[(kBase + r) * N + warpN + c];
        }

        __syncthreads();

        // ── Build matrix descriptors ─────────────────────────────────────
        uint64_t aDesc = buildFp8SmemDesc(sA, spFp8TileK / 2, spFp8TileK / 2, 1, 0);
        uint64_t bDesc = buildFp8SmemDesc(sB, spFp8TileN, spFp8TileN, 1, 1);

        // ── tcgen05.mma.sp.cta_group::1.kind::f8f6f4 ────────────────────
        uint32_t enableInputD = (kt == 0) ? 0 : 1;
        uint32_t tmemHandle = tmemHandlePtr[0];
        uint32_t ePtrSmem = static_cast<uint32_t>(__cvta_generic_to_shared(sE));

        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.ne.b32 p, %5, 0;\n\t"
            "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 "
            "[%0], %1, %2, [%4], %3, p;}\n"
            :
            : "r"(tmemHandle),  // %0 — C accumulator TMEM handle
              "l"(aDesc),       // %1 — A descriptor (64-bit, compressed)
              "l"(bDesc),       // %2 — B descriptor (64-bit, dense)
              "r"(idesc),       // %3 — instruction descriptor (sparse)
              "r"(ePtrSmem),    // %4 — E-matrix shared memory pointer
              "r"(enableInputD) // %5 — predicate control
            : "memory"
        );

        // ── On final k-tile: read results from TMEM ──────────────────────
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

            // ── TMEM deallocation ────────────────────────────────────────
            if (laneId == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                             : : "r"(tmemHandle), "r"(nCols) : "memory");
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
            }
        }
    }
}

// ── Measure FP8 2:4 Sparse GEMM ────────────────────────────────────────────
BenchResult measureFP8Sparse(int device, int matDim, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    chk(cudaSetDevice(device), "dev");

    // Align dimensions to tile sizes
    int M = (matDim / spFp8TileM) * spFp8TileM;
    int N = M;
    int K = (matDim / spFp8TileK) * spFp8TileK;
    if (M < spFp8TileM) M = spFp8TileM;
    if (K < spFp8TileK) K = spFp8TileK;

    // Dense input for compression
    size_t sizeA_dense  = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t sizeB        = static_cast<size_t>(K) * static_cast<size_t>(N);
    size_t sizeC        = static_cast<size_t>(M) * static_cast<size_t>(N);

    // Compressed A: M x (K/2) __nv_fp8_storage_t
    size_t sizeA_sparse = static_cast<size_t>(M) * static_cast<size_t>(K) / 2;
    // E-matrix: M x (K/4) uint8_t
    size_t sizeE = static_cast<size_t>(M) * static_cast<size_t>(K) / 4;
    size_t sizeOffsets = static_cast<size_t>(M) * sizeof(int);

    __nv_fp8_storage_t* dA_dense   = nullptr;
    __nv_fp8_storage_t* dA_sparse  = nullptr;
    __nv_fp8_storage_t* dB         = nullptr;
    float*              dC         = nullptr;
    uint8_t*            dE         = nullptr;
    int*                dOffsets   = nullptr;

    chk(cudaMalloc(&dA_dense,  sizeA_dense),           "dense_a");
    chk(cudaMalloc(&dA_sparse, sizeA_sparse),          "sparse_a");
    chk(cudaMalloc(&dB,        sizeB),                 "b");
    chk(cudaMalloc(&dC,        sizeC * sizeof(float)), "c");
    chk(cudaMalloc(&dE,        sizeE),                 "e");
    chk(cudaMalloc(&dOffsets,  sizeOffsets),           "offsets");

    // Initialize dense A with small non-zero values
    chk(cudaMemset(dA_dense, 0x3E, sizeA_dense), "dense_a"); // ~0.23 in FP8
    chk(cudaMemset(dB,       0x3E, sizeB),       "b");

    // Compress A: dense -> compressed + E-matrix
    {
        int gridX = std::min(65535, M);
        int smemBytes = sizeof(int) * 256;
        fp8SparseCompressKernel<<<gridX, 256, smemBytes, str>>>(
            dA_sparse, dE, dOffsets, dA_dense, M, K);
        chk(cudaStreamSynchronize(str), "compress_sync");
    }

    // Grid: (N/spFp8TileN) x (M/spFp8TileM), 1 warp per block
    dim3 grid(std::min(65535, N / spFp8TileN), std::min(65535, M / spFp8TileM), 1);
    int smemBytes = spFp8TileM * (spFp8TileK / 2) + spFp8TileM * (spFp8TileK / 4)
                  + spFp8TileK * spFp8TileN + 4;

    // Warmup
    for (int w = 0; w < 3; ++w) {
        fp8SparseMmaKernel<<<grid, 32, smemBytes, str>>>(
            dA_sparse, dE, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    // Measure
    std::vector<double> vals;
    size_t totalFlops = static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(K) * 2;

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        fp8SparseMmaKernel<<<grid, 32, smemBytes, str>>>(
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
      << ",\"tile_m\":" << spFp8TileM << ",\"tile_n\":" << spFp8TileN
      << ",\"tile_k\":" << spFp8TileK << ",\"sparsity\":\"2:4\""
      << ",\"api\":\"tcgen05_mma_sp_inline_ptx\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "sasp";
    res.test_name  = "sasp_fp8_sparse";
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

std::vector<BenchResult> runSASPBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    // --- FP8 Dense ---
    try {
        results.push_back(measureFP8Dense(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "sasp";
        r.test_name  = "sasp_fp8_dense";
        r.unit       = "TFLOP/s";
        r.sample_count = 0;
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        r.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_dense_tflops));
        results.push_back(r);
    }

    // --- FP8 Sparse (2:4) via tcgen05.mma.sp ---
    // Probe tcgen05 support first; fallback to stub if unsupported
    try {
        if (!fp8SparseSupported(device)) {
            BenchResult r{};
            r.suite_name = "sasp";
            r.test_name  = "sasp_fp8_sparse";
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
        BenchResult r{};
        r.suite_name = "sasp";
        r.test_name  = "sasp_fp8_sparse";
        r.unit       = "TFLOP/s";
        r.sample_count = 0;
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

BENCH_REGISTER_SUITE(sasp, "FP8 dense matmul + 2:4 structured sparse (scalar kernel, no Tensor Core)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runSASPBench(0, 512, 10);
    });
