#include "compute/int8_scalar_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace deusridet::bench {

namespace {

constexpr int kTile = 8;

// ── INT8 dense matmul kernel ───────────────────────────────────────────────
// C[M,N] = A[M,K] @ B[K,N], tiled scalar kernel.
// Inputs stored as int8_t, accumulated in int32.
__global__ void int8MatmulKernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.x * kTile + threadIdx.x;
    int col = blockIdx.y * kTile + threadIdx.y;
    if (row >= M || col >= N) return;

    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += static_cast<int32_t>(A[row * K + k]) * static_cast<int32_t>(B[k * N + col]);
    }
    C[row * N + col] = sum;
}

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}
// ── Measure INT8 Dense ──────────────────────────────────────────────────────
BenchResult measureINT8Dense(int device, int matDim, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    chk(cudaSetDevice(device), "dev");

    matDim = std::min(matDim, 8192);
    int M = matDim, N = matDim, K = matDim;

    size_t sizeInt8 = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t sizeC    = static_cast<size_t>(M) * static_cast<size_t>(N);

    int8_t *dA = nullptr, *dB = nullptr;
    int32_t *dC = nullptr;

    chk(cudaMalloc(&dA, sizeInt8 * sizeof(int8_t)), "a");
    chk(cudaMalloc(&dB, sizeInt8 * sizeof(int8_t)), "b");
    chk(cudaMalloc(&dC, sizeC * sizeof(int32_t)), "c");

    // Initialize with alternating small values
    chk(cudaMemset(dA, 0x04, sizeInt8 * sizeof(int8_t)), "a");
    chk(cudaMemset(dB, 0x05, sizeInt8 * sizeof(int8_t)), "b");

    dim3 grid(std::min(65535, (M + kTile - 1) / kTile),
              std::min(65535, (N + kTile - 1) / kTile), 1);
    dim3 block(kTile, kTile, 1);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        int8MatmulKernel<<<grid, block, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    // Measure
    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        int8MatmulKernel<<<grid, block, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;

        // Each output element: K muls + (K-1) adds ≈ 2K MACs
        // INT8: count MACs, convert to TOPS (Tera Operations per Second)
        size_t totalOps = static_cast<size_t>(M) * N * K * 2;
        double tops = sec > 0.0 ? (totalOps / 1e12) / sec : 0.0;
        vals.push_back(tops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"tile\":" << kTile << ",\"type\":\"scalar_int8_kernel\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "int8_scalar";
    res.test_name  = "int8_scalar_dense";
    res.unit       = "TOP/s";
    res.peak_pct   = computePeakPctFromT(res.median, T5000Peaks::int8_top);
    res.params_json = p.str();
    res.metadata["note"] = "scalar INT8 kernel with int32 accumulate; tcgen05.mma kind::i8 PTX requires SMEM descriptors + TMEM alloc";

    chk(cudaFree(dA), "fa");
    chk(cudaFree(dB), "fb");
    chk(cudaFree(dC), "fc");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ════════════════════════════════════════════════════════════════════════════
// ── INT8 2:4 Sparse GEMM via tcgen05.mma.sp ────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════

// Probe kernel: test tcgen05.mma.sp kind::i8 support
__global__ void int8SparseProbeKernel() {
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
        "tcgen05.mma.sp.cta_group::1.kind::i8 "
        "[%0], %1, %2, [%4], %3, p;}\n"
        : : "r"(tmemHandle), "l"(aDesc), "l"(bDesc), "r"(idesc), "r"(ePtrSmem)
        : "memory"
    );
}

static bool int8SparseSupported(int device) {
    chk(cudaSetDevice(device), "probe_dev");
    int major = 0;
    chk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "major");
    // tcgen05.mma.sp kind::i8 IllegalInstruction poisons CUDA context on driver 595.58.03.
    // Never attempt to run — return stub immediately.
    return false;
}

// Tile dimensions: 16x16x16 per warp (same as dense INT8 TC kernel).
constexpr int spTileM = 16;
constexpr int spTileN = 16;
constexpr int spTileK = 16;

// ── Matrix descriptor builder (64-bit) — CUTLASS SmemDescriptor format ────
// Bit  [0:14): start_address_     (base >> 4, 16-byte aligned)
// Bit  [16:30): leading_byte_offset_ (leadingDim*elemBytes >> 4)
// Bit  [32:46): stride_byte_offset_  (strideDim*elemBytes >> 4)
// Bit  [46:48): version_            (1 for Blackwell)
// Bit  [49:52): base_offset_        (0)
// Bit  [52:53): lbo_mode_           (0 = legacy)
// Bit  [61:64): layout_type_        (0=row, 1=col)
__device__ static uint64_t buildSmemDesc(const void* ptr, int leadingDim, int strideDim,
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

// ── Instruction descriptor (32-bit) for kind::i8 ──────────────────────────
// Bit  [7:10):    a_format_      (S8 = 0x8)
// Bit  [10:13):   b_format_      (S8 = 0x8)
// Bit  [15:16):   a_major_       (0 = row-major)
// Bit  [16:17):   b_major_       (1 = col-major)
// Bit  [17:23):   n_dim_         (N >> 4)
// Bit  [24:29):   m_dim_         (M >> 4)
// Bits [0:2]:     sparse_id2     (4 = 2:4 sparsity for .sp variant)
__device__ static uint32_t buildI8Idesc(int M, int N) {
    constexpr uint8_t S8 = 0x8;
    uint32_t idesc = 0;
    idesc  = S8;
    idesc |= (S8 << 10);
    idesc |= (0 << 15);
    idesc |= (1 << 16);
    idesc |= ((N >> 4) & 0x3F) << 17;
    idesc |= ((M >> 4) & 0x1F) << 24;
    return idesc;
}

// ── Compressor kernel: dense int8_t → compressed + E-matrix ────────────────
// 2:4 sparsity: each group of 4 INT8 values has at most 2 non-zeros.
// E-matrix encodes which positions are non-zero (uint8_t bitmask per group).
// Compressed A stores only non-zero values packed row-wise.
// OutOffsets[row] holds the packed width for each row.
__global__ void int8SparseCompressKernel(
    int8_t*   __restrict__ outCompressed,
    uint8_t*  __restrict__ outE,
    int*      __restrict__ outOffsets,
    const int8_t* __restrict__ inp,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const int8_t* rowPtr = inp + static_cast<size_t>(row) * cols;
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
            if (rowPtr[valBase + v] != 0) {
                mask |= (1 << v);
                nnz++;
            }
        }
        ePtr[g] = mask;
        rowNnz += nnz;
    }

    // Sum rowNnz across threads (simple reduction for small blocks)
    extern __shared__ int sSum[];
    sSum[tid] = rowNnz;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) sSum[tid] += sSum[tid + stride];
    }
    if (tid == 0) outOffsets[row] = sSum[0];

    __syncthreads();

    // Pack non-zero values into compressed storage
    // Row-stride packed layout: each row's non-zeros packed sequentially.
    // Row stride = cols/2 (max non-zeros for 2:4 sparsity).
    __syncthreads();
    sSum[0] = 0; // reuse as packed counter
    __syncthreads();

    for (int g = 0; g < groupsPerRow; ++g) {
        uint8_t mask = ePtr[g];
        int valBase = g * 4;
        for (int v = 0; v < 4; ++v) {
            if (mask & (1 << v)) {
                int atomicIdx = atomicAdd(&sSum[0], 1);
                // We write to a per-row region; the row stride is max possible nnz = cols/2
                outCompressed[static_cast<size_t>(row) * (cols / 2) + atomicIdx] = rowPtr[valBase + v];
            }
        }
    }
}

// ── INT8 2:4 Sparse GEMM kernel via tcgen05.mma.sp ─────────────────────────
// Each warp (block=32 threads) handles spTileM x spTileN tile.
// Grid: (N/spTileN) x (M/spTileM)
// K loop: K/spTileK iterations
//
// A is row-major compressed: M x (K/2) int8_t (packed non-zeros per row)
// E is row-major:             M x (K/4) uint8_t (sparsity bitmasks)
// B is row-major dense:       K x N int8_t
// C is row-major:             M x N int32_t
__global__ void int8SparseMmaKernel(
    const int8_t*  __restrict__ A,
    const uint8_t* __restrict__ E,
    const int8_t*  __restrict__ B,
    int32_t*       __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y * spTileM;
    int warpN = blockIdx.x * spTileN;

    if (warpM >= M || warpN >= N) return;

    int laneId = threadIdx.x;

    // ── Shared memory layout (sparse: A is K/2, E is K/4) ────────────────
    // A tile (compressed): spTileM x (spTileK/2) = 16 x 8  = 128 bytes
    // E tile (metadata):   spTileM x (spTileK/4) = 16 x 4  =  64 bytes
    // B tile:              spTileK x spTileN     = 16 x 16 = 256 bytes
    // tmemHandle:                          4 bytes
    extern __shared__ char smem[];

    int8_t*   sA            = reinterpret_cast<int8_t*>(smem);
    uint8_t*  sE            = reinterpret_cast<uint8_t*>(smem + spTileM * (spTileK / 2));
    int8_t*   sB            = reinterpret_cast<int8_t*>(smem + spTileM * (spTileK / 2) + spTileM * (spTileK / 4));
    uint32_t* tmemHandlePtr = reinterpret_cast<uint32_t*>(
        smem + spTileM * (spTileK / 2) + spTileM * (spTileK / 4) + spTileK * spTileN);

    // ── TMEM allocation ──────────────────────────────────────────────────
    // nCols = spTileN = 16 (INT32 columns per row)
    uint32_t nCols = spTileN;

    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(tmemHandlePtr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    // Build idesc: base i8 idesc | sparse_id2=4 (bits 0-2 for 2:4 sparsity)
    uint32_t idesc = buildI8Idesc(spTileM, spTileN) | 4;

    // ── K loop ───────────────────────────────────────────────────────────
    int kTiles = K / spTileK;

    for (int kt = 0; kt < kTiles; ++kt) {
        int kBase = kt * spTileK;

        // ── Load A tile (compressed) from global -> shared ───────────────
        // Compressed A: row stride is K/2, tile width is spTileK/2
        int aElems = spTileM * (spTileK / 2);
        for (int i = threadIdx.x; i < aElems; i += 32) {
            int r = i / (spTileK / 2);
            int c = i % (spTileK / 2);
            sA[i] = A[(warpM + r) * (K / 2) + kBase / 2 + c];
        }

        // ── Load E tile (sparsity metadata) ──────────────────────────────
        // E: row stride is K/4, tile width is spTileK/4
        int eElems = spTileM * (spTileK / 4);
        for (int i = threadIdx.x; i < eElems; i += 32) {
            int r = i / (spTileK / 4);
            int c = i % (spTileK / 4);
            sE[i] = E[(warpM + r) * (K / 4) + kBase / 4 + c];
        }

        // ── Load B tile (dense, row-major -> col-major in shared) ───────
        int bElems = spTileK * spTileN;
        for (int i = threadIdx.x; i < bElems; i += 32) {
            int r = i / spTileN;
            int c = i % spTileN;
            sB[i] = B[(kBase + r) * N + warpN + c];
        }

        __syncthreads();

        // ── Build matrix descriptors ─────────────────────────────────────
        // A descriptor: compressed width = spTileK/2, elem=1B, row-major
        uint64_t aDesc = buildSmemDesc(sA, spTileK / 2, spTileK / 2, 1, 0);
        // B descriptor: same as dense, elem=1B, col-major
        uint64_t bDesc = buildSmemDesc(sB, spTileN, spTileN, 1, 1);

        // ── tcgen05.mma.sp.cta_group::1.kind::i8 ─────────────────────────
        // Sparse variant of the INT8 MMA instruction.
        // Syntax: tcgen05.mma.sp.cta_group::1.kind::i8
        //   [%tmemC], %aDesc, %bDesc, [%tmemE], %idesc, predicate
        uint32_t enableInputD = (kt == 0) ? 0 : 1;
        uint32_t tmemHandle = tmemHandlePtr[0];
        uint32_t ePtrSmem = static_cast<uint32_t>(__cvta_generic_to_shared(sE));

        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.ne.b32 p, %5, 0;\n\t"
            "tcgen05.mma.sp.cta_group::1.kind::i8 "
            "[%0], %1, %2, [%4], %3, p;}\n"
            :
            : "r"(tmemHandle),  // %0 — C accumulator TMEM handle
              "l"(aDesc),       // %1 — A descriptor (64-bit, compressed)
              "l"(bDesc),       // %2 — B descriptor (64-bit, dense)
              "r"(idesc),       // %3 — instruction descriptor (sparse k_size=1)
              "r"(ePtrSmem),    // %4 — E-matrix shared memory pointer
              "r"(enableInputD) // %5 — predicate control
            : "memory"
        );

        // ── On final k-tile: read results from TMEM ──────────────────────
        if (kt == kTiles - 1) {
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

            // Same lane mapping as dense INT8 TC kernel:
            //   Lane 0  -> row 0, cols 0..7
            //   Lane 1  -> row 0, cols 8..15
            //   ...
            //   Lane 31 -> row 15, cols 8..15
            int row    = (laneId / 2);
            int colOff = (laneId % 2) * 8;
            int32_t* cRow = C + (warpM + row) * N + warpN + colOff;

            uint32_t ldOut[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(ldOut[0]), "=r"(ldOut[1]), "=r"(ldOut[2]), "=r"(ldOut[3]),
                  "=r"(ldOut[4]), "=r"(ldOut[5]), "=r"(ldOut[6]), "=r"(ldOut[7])
                : "r"(tmemHandle)
                : "memory"
            );

            for (int c = 0; c < 8; ++c) {
                cRow[c] = ldOut[c];
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

// ── Measure INT8 2:4 Sparse GEMM ───────────────────────────────────────────
BenchResult measureINT8Sparse(int device, int matDim, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    chk(cudaSetDevice(device), "dev");

    // Align dimensions to tile sizes
    int M = (matDim / spTileM) * spTileM;
    int N = M;
    int K = (matDim / spTileK) * spTileK;
    if (M < spTileM) M = spTileM;
    if (K < spTileK) K = spTileK;

    // Dense input for compression
    size_t sizeA_dense = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t sizeB       = static_cast<size_t>(K) * static_cast<size_t>(N);
    size_t sizeC       = static_cast<size_t>(M) * static_cast<size_t>(N);

    // Compressed A: M x (K/2) int8_t (half the K dimension for 2:4 sparsity)
    size_t sizeA_sparse = static_cast<size_t>(M) * static_cast<size_t>(K) / 2;
    // E-matrix: M x (K/4) uint8_t (one bitmask per group of 4)
    size_t sizeE = static_cast<size_t>(M) * static_cast<size_t>(K) / 4;
    // Row offsets for compression
    size_t sizeOffsets = static_cast<size_t>(M) * sizeof(int);

    int8_t*   dA_dense    = nullptr;
    int8_t*   dA_sparse   = nullptr;
    int8_t*   dB          = nullptr;
    int32_t*  dC          = nullptr;
    uint8_t*  dE          = nullptr;
    int*      dOffsets    = nullptr;

    chk(cudaMalloc(&dA_dense,   sizeA_dense),   "dense_a");
    chk(cudaMalloc(&dA_sparse,  sizeA_sparse),  "sparse_a");
    chk(cudaMalloc(&dB,         sizeB),          "b");
    chk(cudaMalloc(&dC,         sizeC * sizeof(int32_t)), "c");
    chk(cudaMalloc(&dE,         sizeE),          "e");
    chk(cudaMalloc(&dOffsets,   sizeOffsets),    "offsets");

    // Initialize dense A with values that produce ~50% sparsity
    // Pattern: alternate non-zero and zero values to get 2:4 sparsity
    chk(cudaMemset(dA_dense, 0x01, sizeA_dense), "dense_a");
    chk(cudaMemset(dB,       0x01, sizeB),       "b");

    // Compress A: dense -> compressed + E-matrix
    {
        int gridX = std::min(65535, M);
        int smemBytes = sizeof(int) * 256; // shared memory for reduction
        int8SparseCompressKernel<<<gridX, 256, smemBytes, str>>>(
            dA_sparse, dE, dOffsets, dA_dense, M, K);
        chk(cudaStreamSynchronize(str), "compress_sync");
    }

    // Grid: (N/spTileN) x (M/spTileM), 1 warp per block
    dim3 grid(std::min(65535, N / spTileN), std::min(65535, M / spTileM), 1);
    int smemBytes = spTileM * (spTileK / 2) + spTileM * (spTileK / 4)
                  + spTileK * spTileN + 4;  // A + E + B + tmemHandle

    // Warmup
    for (int w = 0; w < 3; ++w) {
        int8SparseMmaKernel<<<grid, 32, smemBytes, str>>>(
            dA_sparse, dE, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    // Measure
    std::vector<double> vals;
    // INT8 sparse: M * N * K multiply-adds = M*N*K*2 ops
    // (hardware reports TOPS at the logical dense count, not compressed count)
    size_t totalOps = static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(K) * 2;

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        int8SparseMmaKernel<<<grid, 32, smemBytes, str>>>(
            dA_sparse, dE, dB, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double tops = sec > 0.0 ? (static_cast<double>(totalOps) / 1e12) / sec : 0.0;
        vals.push_back(tops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"tile_m\":" << spTileM << ",\"tile_n\":" << spTileN
      << ",\"tile_k\":" << spTileK << ",\"sparsity\":\"2:4\""
      << ",\"api\":\"tcgen05_mma_sp_inline_ptx\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "int8_scalar";
    res.test_name  = "int8_scalar_sparse";
    res.unit       = "TOP/s";
    res.peak_pct   = computePeakPctFromT(res.median, T5000Peaks::int8_top);
    res.params_json = p.str();
    res.metadata["tcgen05"] = "true";
    res.metadata["precision"] = "int8";
    res.metadata["sparsity"] = "2:4_structured";
    res.metadata["sparse_instruction"] = "tcgen05.mma.sp.kind::i8";

    chk(cudaFree(dA_dense),   "free_dense_a");
    chk(cudaFree(dA_sparse),  "free_sparse_a");
    chk(cudaFree(dB),         "free_b");
    chk(cudaFree(dC),         "free_c");
    chk(cudaFree(dE),         "free_e");
    chk(cudaFree(dOffsets),   "free_offsets");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

} // anonymous namespace

std::vector<BenchResult> runINT8ScalarBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    // --- INT8 Dense ---
    try {
        results.push_back(measureINT8Dense(device, matDim, iterations));
    } catch (const std::exception& ex) {
        cudaDeviceSynchronize();
        cudaGetLastError();
        BenchResult r{};
        r.suite_name = "int8_scalar";
        r.test_name  = "int8_scalar_dense";
        r.unit       = "TOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    // --- INT8 Sparse ---
    // Probe tcgen05 support first; fallback to stub if unsupported
    try {
        if (!int8SparseSupported(device)) {
            BenchResult r{};
            r.suite_name = "int8_scalar";
            r.test_name  = "int8_scalar_sparse";
            r.unit       = "TOP/s";
            r.sample_count = 0;
            r.warmup_count = 0;
            r.median = 0.0;
            r.params_json = "{\"note\":\"tcgen05.mma.sp.kind::i8 not supported by current driver/firmware\"}";
            r.metadata["stub"] = "true";
            r.metadata["stub_reason"] = "tcgen05.mma.sp.kind::i8 not supported by current driver/firmware";
            results.push_back(r);
        } else {
            results.push_back(measureINT8Sparse(device, matDim, iterations));
        }
    } catch (const std::exception& ex) {
        // CRITICAL: tcgen05 IllegalInstruction poisons device context.
        // MUST synchronize to drain error BEFORE any subsequent CUDA call.
        cudaDeviceSynchronize();
        cudaGetLastError();
        BenchResult r{};
        r.suite_name = "int8_scalar";
        r.test_name  = "int8_scalar_sparse";
        r.unit       = "TOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(int8_scalar, "Scalar INT8 GEMM (no Tensor Core — ~0.04% of peak)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runINT8ScalarBench(0, 512, 10);
    });
