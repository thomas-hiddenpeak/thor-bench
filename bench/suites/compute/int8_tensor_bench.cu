#include "compute/int8_tensor_bench.h"
#include "bench_schema.h"
#include "bench_stats.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

// ── Tile dimensions: 16x16x16 per warp ──────────────────────────────────────
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

// ── Matrix descriptor builder (64-bit) — CUTLASS SmemDescriptor format ──────
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
    desc  = (base >> 4) & 0x3FFF;                              // [0:14)
    desc |= (leadingBytes & 0x3FFF) << 16;                     // [16:30)
    desc |= (strideBytes & 0x3FFF) << 32;                      // [32:46)
    desc |= (1ULL) << 46;                                       // [46:48) version_=1 (Blackwell)
    desc |= (0ULL) << 49;                                       // [49:52) base_offset_=0
    desc |= (0ULL) << 52;                                       // [52:53) lbo_mode_=0 (legacy)
    desc |= (static_cast<uint64_t>(layoutType & 0x7)) << 61;   // [61:64) layout_type
    return desc;
}

// ── Instruction descriptor (32-bit) for kind::i8 ────────────────────────────
// Bit  [7:10):    a_format_      (S8 = 0x8 for signed 8-bit integer)
// Bit  [10:13):   b_format_      (S8 = 0x8 for signed 8-bit integer)
// Bit  [15:16):   a_major_       (0 = K-major / row-major for A)
// Bit  [16:17):   b_major_       (1 = MN-major / col-major for B)
// Bit  [17:23):   n_dim_         (N >> 4)
// Bit  [24:29):   m_dim_         (M >> 4)
// Bit  [31:32):   k_size_        (0 = K16 dense)
__device__ static uint32_t buildI8Idesc(int M, int N) {
    constexpr uint8_t S8 = 0x8;  // Signed 8-bit integer format
    uint32_t idesc = 0;
    idesc  = S8;                               // a_format_ [7:10)
    idesc |= (S8 << 10);                       // b_format_ [10:13)
    idesc |= (0 << 15);                        // a_major_ [15:16) row-major
    idesc |= (1 << 16);                        // b_major_ [16:17) col-major
    idesc |= ((N >> 4) & 0x3F) << 17;          // n_dim_ [17:23)
    idesc |= ((M >> 4) & 0x1F) << 24;          // m_dim_ [24:29)
    idesc |= (0 << 31);                        // k_size_ [31:32) 0=K16 dense
    return idesc;
}

// ── INT8 GEMM kernel via tcgen05.mma.kind::i8 ───────────────────────────────
// Each warp (block=32 threads) handles TILE_M x TILE_N tile.
// Grid: (N/TILE_N) x (M/TILE_M)
// K loop: K/TILE_K iterations
__global__ void int8MmaKernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t*      __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y * TILE_M;
    int warpN = blockIdx.x * TILE_N;

    if (warpM >= M || warpN >= N) return;

    int laneId = threadIdx.x;

    // ── Shared memory layout ─────────────────────────────────────────────
    // A tile:     TILE_M x TILE_K = 16 x 16 = 256 bytes
    // B tile:     TILE_K x TILE_N = 16 x 16 = 256 bytes
    // tmemHandle: 4 bytes
    extern __shared__ char smem[];

    int8_t*   sA            = reinterpret_cast<int8_t*>(smem);
    int8_t*   sB            = reinterpret_cast<int8_t*>(smem + TILE_M * TILE_K);
    uint32_t* tmemHandlePtr = reinterpret_cast<uint32_t*>(smem + TILE_M * TILE_K + TILE_K * TILE_N);

    // ── TMEM allocation ──────────────────────────────────────────────────
    // nCols = TILE_N = 16 (INT32 columns per row)
    uint32_t nCols = TILE_N;

    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(tmemHandlePtr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    // Build idesc once
    uint32_t idesc = buildI8Idesc(TILE_M, TILE_N);

    // ── K loop ───────────────────────────────────────────────────────────
    int kTiles = K / TILE_K;

    for (int kt = 0; kt < kTiles; ++kt) {
        int kBase = kt * TILE_K;

        // ── Load A tile from global -> shared (row-major) ────────────────
        int aElems = TILE_M * TILE_K;
        for (int i = threadIdx.x; i < aElems; i += 32) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            sA[i] = A[(warpM + r) * K + kBase + c];
        }

        // ── Load B tile from global -> shared (col-major) ────────────────
        int bElems = TILE_K * TILE_N;
        for (int i = threadIdx.x; i < bElems; i += 32) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            sB[i] = B[(kBase + r) * N + warpN + c];
        }

        __syncthreads();

        // ── Build matrix descriptors (CUTLASS SmemDescriptor format) ─────
        // A: row-major, TILE_M rows x TILE_K cols, elem=1B
        uint64_t aDesc = buildSmemDesc(sA, TILE_K, TILE_K, 1, 0);  // 0=row-major
        // B: col-major, TILE_K rows x TILE_N cols, elem=1B
        uint64_t bDesc = buildSmemDesc(sB, TILE_N, TILE_N, 1, 1);  // 1=col-major

        // ── tcgen05.mma.cta_group::1.kind::i8 ────────────────────────────
        uint32_t enableInputD = (kt == 0) ? 0 : 1;
        uint32_t tmemHandle = tmemHandlePtr[0];

        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::i8 "
            "  [%0], %1, %2, %3, p;}\n"
            :
            : "r"(tmemHandle),
              "l"(aDesc),
              "l"(bDesc),
              "r"(idesc),
              "r"(enableInputD)
            : "memory"
        );

        // ── On final k-tile: read results from TMEM ──────────────────────
        if (kt == kTiles - 1) {
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

            // TMEM result: 16 rows x 16 cols x 4 bytes (INT32) = 1024 bytes.
            // tcgen05.ld outputs 8 b32 values per call = 8 INT32s.
            // 256 INT32s / 8 per-lane / 32 lanes = 1 ld call per lane. Perfect.
            //
            // Lane mapping:
            //   Lane 0  -> row 0, cols 0..7
            //   Lane 1  -> row 0, cols 8..15
            //   Lane 2  -> row 1, cols 0..7
            //   Lane 3  -> row 1, cols 8..15
            //   ...
            //   Lane 31 -> row 15, cols 8..15

            int row    = (laneId / 2);            // 0..15
            int colOff = (laneId % 2) * 8;        // 0 or 8
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
            // dealloc is CTA-scoped; only one thread needs to call it
            if (laneId == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                             : : "r"(tmemHandle), "r"(nCols) : "memory");
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
            }
        }
    }
}

// ── Stub result builders ────────────────────────────────────────────────────
BenchResult makeStubDense(const std::string& reason) {
    BenchResult res{};
    res.suite_name = "int8_tensor";
    res.test_name  = "int8_tensor_dense";
    res.unit       = "TOP/s";
    res.sample_count = 0;
    res.warmup_count = 0;
    std::ostringstream p;
    p << "{\"error\":\"" << reason
      << "\",\"note\":\"tcgen05.mma kind::i8 not available on this device\"}";
    res.params_json = p.str();
    res.metadata["tcgen05"] = "true";
    res.metadata["precision"] = "int8";
    res.metadata["stub"] = "true";
    res.metadata["stub_reason"] = reason;
    res.peak_pct = 0.0;
    return res;
}

BenchResult makeStubSparse(const std::string& reason) {
    BenchResult res{};
    res.suite_name = "int8_tensor";
    res.test_name  = "int8_tensor_sparse";
    res.unit       = "TOP/s";
    res.sample_count = 0;
    res.warmup_count = 0;
    std::ostringstream p;
    p << "{\"error\":\"" << reason
      << "\",\"note\":\"INT8 2:4 sparse tcgen05.mma.sp not yet implemented\"}";
    res.params_json = p.str();
    res.metadata["tcgen05"] = "true";
    res.metadata["precision"] = "int8";
    res.metadata["stub"] = "true";
    res.metadata["stub_reason"] = reason;
    res.peak_pct = 0.0;
    return res;
}

// ── Measure INT8 Dense GEMM ────────────────────────────────────────────────
BenchResult measureInt8Dense(int device, int matDim, int iterations) {
    chk(cudaSetDevice(device), "dev");

    int M = (matDim / TILE_M) * TILE_M;
    int N = M;
    int K = (matDim / TILE_K) * TILE_K;
    if (M < TILE_M) M = TILE_M;
    if (K < TILE_K) K = TILE_K;

    size_t sizeA = static_cast<size_t>(M) * static_cast<size_t>(K);   // int8_t
    size_t sizeB = static_cast<size_t>(K) * static_cast<size_t>(N);   // int8_t
    size_t sizeC = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t);  // int32_t

    int8_t*  dA = nullptr;
    int8_t*  dB = nullptr;
    int32_t* dC = nullptr;

    chk(cudaMalloc(&dA, sizeA), "alloc_a");
    chk(cudaMalloc(&dB, sizeB), "alloc_b");
    chk(cudaMalloc(&dC, sizeC), "alloc_c");

    // Initialize with small values (INT8 range)
    chk(cudaMemset(dA, 0x01, sizeA), "memset_a");
    chk(cudaMemset(dB, 0x01, sizeB), "memset_b");

    cudaStream_t str;
    chk(cudaStreamCreate(&str), "stream");

    // Grid: (N/TILE_N) x (M/TILE_M), 1 warp per block
    dim3 grid(std::min(65535, N / TILE_N), std::min(65535, M / TILE_M), 1);
    int smemBytes = TILE_M * TILE_K + TILE_K * TILE_N + 4;  // A + B + tmemHandle

    // Warmup
    for (int w = 0; w < 3; ++w) {
        int8MmaKernel<<<grid, 32, smemBytes, str>>>(dA, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    // Timing
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    std::vector<double> vals;
    // INT8: M * N * K multiply-adds = M*N*K*2 ops
    size_t totalOps = static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(K) * 2;

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "recS");
        int8MmaKernel<<<grid, 32, smemBytes, str>>>(dA, dB, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "recE");
        chk(cudaStreamSynchronize(str), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "elapsed");
        double sec = ms / 1000.0;
        double tops = sec > 0.0 ? (static_cast<double>(totalOps) / 1e12) / sec : 0.0;
        vals.push_back(tops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"tile_m\":" << TILE_M << ",\"tile_n\":" << TILE_N << ",\"tile_k\":" << TILE_K
      << ",\"format\":\"int8\""
      << ",\"api\":\"tcgen05_mma_inline_ptx\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "int8_tensor";
    res.test_name  = "int8_tensor_dense";
    res.unit       = "TOP/s";
    res.params_json = p.str();
    res.peak_pct = 0.0;  // no INT8 TC peak reference in T5000Peaks
    res.metadata["tcgen05"] = "true";
    res.metadata["precision"] = "int8";
    res.metadata["stub"] = "false";

    chk(cudaFree(dA), "free_a");
    chk(cudaFree(dB), "free_b");
    chk(cudaFree(dC), "free_c");
    chk(cudaStreamDestroy(str), "stream");
    chk(cudaEventDestroy(evS), "evSDestroy");
    chk(cudaEventDestroy(evE), "evEDestroy");

    return res;
}

} // anonymous namespace

// ── Public API ──────────────────────────────────────────────────────────────
std::vector<BenchResult> runINT8TensorBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;
    chk(cudaSetDevice(device), "dev");

    try {
        results.push_back(measureInt8Dense(device, matDim, iterations));
    } catch (const std::exception& ex) {
        results.push_back(makeStubDense((std::string("exception: ") + ex.what()).c_str()));
        cudaGetLastError(); // drain
        results.push_back(makeStubSparse("tcgen05.mma.sp not supported (dense already failed)"));
        cudaGetLastError(); // final drain
        return results;
    }

    // Sparse INT8 via tcgen05.mma.sp is not yet implemented
    results.push_back(makeStubSparse("tcgen05.mma.sp kind::i8 2:4 sparse not yet implemented"));
    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(int8_tensor, "INT8 Tensor Core GEMM via tcgen05.mma kind::i8 inline PTX",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        try {
            return deusridet::bench::runINT8TensorBench(0, 2048, 10);
        } catch (const std::exception& ex) {
            std::vector<deusridet::bench::BenchResult> fallback;
            deusridet::bench::BenchResult r{};
            r.suite_name = "int8_tensor";
            r.test_name  = "int8_tensor_dense";
            r.unit       = "TOP/s";
            r.params_json = std::string("{\"error\":\"") + ex.what() + "\"}";
            r.metadata["tcgen05"] = "true";
            r.metadata["precision"] = "int8";
            r.metadata["stub"] = "true";
            r.metadata["stub_reason"] = std::string("runtime error: ") + ex.what();
            r.peak_pct = 0.0;
            fallback.push_back(r);
            return fallback;
        }
    });
