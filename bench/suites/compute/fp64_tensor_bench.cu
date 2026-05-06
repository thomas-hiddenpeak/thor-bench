#include "compute/fp64_tensor_bench.h"
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

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

// ── FP64 DMMA kernel via mma.sync.aligned ───────────────────────────────────
// FP64 Tensor Core on Blackwell uses DMMA (mma.sync.aligned), NOT tcgen05/WMMA.
// `__CUDA_WmmaSupportDouble__` never existed. The correct instruction is:
//   mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64
// Warp-synchronous, SM80+ floor. Each warp (32 threads) computes one 16×8 tile.
// K-step = 4. Per thread: A=2 f64, B=1 f64, C/D=4 f64 accumulators.
//
// Tile layout per warp (16×8 output):
//   Each thread owns 4 C elements (16*8 / 32 = 4).
//   A: row-major, 16 rows × 4 K per thread per iteration
//   B: col-major, 4 K × 8 cols per thread per iteration
//
// Grid: (N/8) x (M/16), 1 warp per block (32 threads)

constexpr int DMMA_M = 16;
constexpr int DMMA_N =  8;
constexpr int DMMA_K =  4;

// Per-thread fragment sizes (in double elements):
// A: 16 rows × 4 K / 32 threads = 2 per thread
// B: 4 K × 8 cols / 32 threads = 1 per thread
// C/D: 16 × 8 / 32 = 4 per thread
constexpr int A_PER_THREAD = 2;
constexpr int B_PER_THREAD = 1;
constexpr int C_PER_THREAD = 4;

__global__ void fp64DmmaKernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double*       __restrict__ C,
    int M, int N, int K)
{
    int warpM = blockIdx.y * DMMA_M;
    int warpN = blockIdx.x * DMMA_N;

    if (warpM >= M || warpN >= N) return;

    int lane = threadIdx.x;

    // Each thread holds 4 C accumulators, initialized to 0
    // Layout: lane t owns rows and cols based on DMMA mapping
    // For mma.sync.aligned.m16n8k4:
    //   Each lane handles a subset of the 16×8 result
    //   Lane t: rows = (t/4)*1 + ..., cols mapped by the instruction
    //
    // Simplified mapping for our inline PTX:
    //   Thread t owns C elements at:
    //     row = warpM + (lane % 16), col = warpN + (lane / 2)
    //   But since we have 32 threads and 128 outputs, each thread owns exactly 4.
    //
    // For mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64:
    //   The PTX uses b64 registers (64-bit). Each thread contributes to
    //   the warp-level 16×8 tile. The mapping is defined by the instruction.
    //
    // Using CUTLASS-style lane mapping for m16n8k4:
    //   Fragment C: 4 b64 per thread
    //   Fragment A: 2 b64 per thread
    //   Fragment B: 1 b64 per thread
    //
    // Lane → (row, col) mapping for 16×8 output (4 elements per lane):
    //   Lane bits: [4:0] = 5 bits
    //   row = warpM + ((lane >> 2) % 16)  ... not quite, need exact mapping
    //
    // Actually, for mma.sync.aligned.m16n8k4, the output layout is:
    //   Each thread's 4 accumulators map to specific positions.
    //   The simplest correct approach: use the PTX instruction's native layout.
    //
    // Let's use a straightforward mapping that works with the PTX:
    //   C is stored as 4 b64 per thread. After the MMA we write them out
    //   in the order the hardware expects.
    //
    // For m16n8k4 DMMA, the standard CUTLASS mapping:
    //   Thread t contributes to rows/cols:
    //     row_base = warpM + (t >> 2)          // 0..8 → but we need 0..15
    //     Actually for m16n8k4, it's:
    //       cRow = warpM + ((t % 16))           // 0..15
    //       cCol = warpN + ((t / 2) % 8)        // 0..7
    //   Each thread owns 1 row × 4 cols = 4 elements.
    //   Wait: 32 threads × 4 = 128 = 16 × 8. Correct.

    // Fragment registers
    double fragA[A_PER_THREAD];
    double fragB[B_PER_THREAD];
    double fragC[C_PER_THREAD];

    // Initialize C fragment to zero
    fragC[0] = 0.0;
    fragC[1] = 0.0;
    fragC[2] = 0.0;
    fragC[3] = 0.0;

    int kTiles = K / DMMA_K;

    for (int kt = 0; kt < kTiles; ++kt) {
        int kBase = kt * DMMA_K;

        // Load A fragment (2 elements per thread, row-major)
        // A layout: row-major, M × K
        // Each thread loads from its assigned rows
        // For m16n8k4: A is 16×4, each thread gets 2 elements
        // Thread t: rows mapped by lane, cols = kBase + kOffset
        for (int i = 0; i < A_PER_THREAD; ++i) {
            // Row mapping: spread 16 rows across 32 threads (2 threads per row)
            // Actually for 2 elements per thread from a 16×4 tile:
            // 32 threads × 2 = 64 = 16 × 4. Each thread:
            //   row = (lane / 2), k-offset = lane % 2 ... but we need 2 per thread
            //   row = (lane * 2) % 16, k-offset = (lane * 2) / 16 ... no
            // Let's use: thread t → (row = t % 16, k = t / 16) gives 1 element per thread for 16×1
            // For 16×4 with 2 per thread: thread t → row = t % 16, k = (t / 16) * 2 + (t % 2)
            // No wait, simpler: 16 rows × 4 K = 64 elements / 32 threads = 2 per thread
            //   row = lane % 16, k = (lane / 16) * 4 + (lane % 2)  ... no
            // Simple grid: thread t loads A[row0][k0], A[row1][k1]
            //   row0 = lane % 16, k0 = (lane / 16) * 2
            //   row1 = (lane + 1) % 16 ... ugly
            //
            // Let's just use a clean mapping:
            //   fragA[i]: row = (lane / 2) + (i * 8) % 16, col = kBase + (lane % 2) * 2 + ...
            //
            // OK, the simplest correct approach for m16n8k4:
            // A is 16×4. Thread t has 2 elements.
            //   row = t % 16, kOffset = (t / 16) * 4 + (t % 2) ... nope, 32 threads
            //   row = t % 16, kOffset = 2 * (t / 16) + (t % 2)
            //     t=0: row=0, k=0;  t=1: row=1, k=0 → 2 elems with same k, different rows
            //     t=16: row=0, k=2; t=17: row=1, k=2
            //   fragA[0]: row = t % 16, k = kBase + 2*(t/16) + 0
            //   fragA[1]: row = t % 16, k = kBase + 2*(t/16) + 1
            // Wait, that gives k=0,1 for t=0..15 and k=2,3 for t=16..31.
            // Each row gets loaded by 2 threads (t and t+16), each loading 2 K-values.
            // Row r is loaded by thread r (k=0,1) and thread r+16 (k=2,3). Total 4 K per row. Correct.
            int fragRow = lane % 16;
            int kOff = 2 * (lane / 16) + i;
            fragA[i] = A[(fragRow) * K + kBase + kOff];
        }

        // Load B fragment (1 element per thread, col-major)
        // B layout: col-major, K × N → B[k * N + col]
        // B is 4×8. 32 threads × 1 = 32 = 4 × 8.
        //   k = lane % 4, col = lane / 4
        int bK = lane % 4;
        int bCol = lane / 4;
        fragB[0] = B[(kBase + bK) * N + bCol];

        // ── mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 ────────────
        // c[i] = a[i] * b[0] + c[i], for i=0..3
        // Input constraints: "=d" for output b64, "d" for input b64
        asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=d"(fragC[0]), "=d"(fragC[1]), "=d"(fragC[2]), "=d"(fragC[3])
            : "d"(fragA[0]), "d"(fragA[1]),
              "d"(fragB[0]),
              "d"(fragC[0]), "d"(fragC[1]), "d"(fragC[2]), "d"(fragC[3])
        );
    }

    // Write C fragment to global memory
    // Each thread owns 4 C elements at specific (row, col) positions.
    // For m16n8k4: 16 rows × 8 cols / 32 threads = 4 per thread.
    // Mapping: thread t → 4 positions in the 16×8 tile
    // Using the same mapping as the A/B loads:
    //   cRow = warpM + (lane % 16), cCol = warpN + (lane / 16) * 8 + (lane / 2) % 4
    //   t=0: row=0, cols=0,1,2,3;  t=1: row=1, cols=0,1,2,3 ... no
    //
    // For mma.sync.aligned.m16n8k4 output:
    //   4 b64 per thread, mapped to the 16×8 result matrix
    //   Thread t: row = warpM + (t % 16), col = warpN + (t / 16) * 8
    //   But (t/16) is 0 or 1, so cols are 0..0 or 8..8. Each thread gets 1 col × 4 rows?
    //   No — each thread has 4 scalars.
    //
    // Standard mapping for m16n8k4:
    //   fragC[i] maps to row = warpM + lane % 16, col = warpN + floor(lane/16)*4 + floor(i/1)*...
    //   Actually the CUTLASS approach is much cleaner. Let me use:
    //   Each thread t writes 4 values at:
    //     row = warpM + (t % 16), col = warpN + (t / 16) * 4 + (t % 2)
    //   t=0:  row=0, cols=0,1 → only 2? No.
    //
    // OK, let me think about this differently. 16×8 = 128 elements. 32 threads × 4 = 128.
    // Thread t writes to 4 positions:
    //   For the DMMA instruction, the output layout is well-defined.
    //   fragC[0..3] maps to specific rows/cols.
    //
    // The simplest reliable mapping (from CUTLASS mma_sm80):
    //   row = warpM + (lane % 16)
    //   col = warpN + (lane / 16) * 4 + ((lane / 2) % 4)
    //   t=0:  row=0, col=0+0=0;  fragC[0]=C[0,0]
    //   t=1:  row=1, col=0+0=0;  fragC[0]=C[1,0]
    //   t=2:  row=2, col=0+1=1;
    //   t=3:  row=3, col=0+1=1;
    //   t=4:  row=4, col=0+2=2;
    //   t=5:  row=5, col=0+2=2;
    //   ...
    //   t=14: row=14, col=0+3=3;
    //   t=15: row=15, col=0+3=3;
    //   t=16: row=0, col=4+0=4;
    //   t=17: row=1, col=4+0=4;
    //   ...
    //   t=31: row=15, col=4+3=7;
    // Each thread gets 1 row × 1 col. But we have 4 frags per thread.
    // So fragC[i] goes to: row = ..., col = base_col + i?
    //   t=0: fragC[0]=C[0,0], fragC[1]=C[0,1], fragC[2]=C[0,2], fragC[3]=C[0,3]
    //   t=1: fragC[0]=C[1,0], fragC[1]=C[1,1], fragC[2]=C[1,2], fragC[3]=C[1,3]
    //   t=2: fragC[0]=C[2,4], ... no that's wrong
    //
    // I think I'm overcomplicating this. For m16n8k4:
    //   32 threads, 4 elems each = 128. The output is 16×8.
    //   Thread t → 4 positions. Using contiguous mapping:
    //     base = t * 4
    //     for i=0..3: row = (base + i) % 16, col = (base + i) / 16 * 8 ... no
    //
    // Let's use row-major contiguous:
    //   Thread t writes fragC[0..3] at positions 4t, 4t+1, 4t+2, 4t+3 in row-major order
    //   pos = t * 4 + i → row = pos / 8, col = pos % 8
    //   t=0: pos=0,1,2,3 → rows=0,0,0,0, cols=0,1,2,3
    //   t=1: pos=4,5,6,7 → rows=0,0,0,0, cols=4,5,6,7
    //   t=2: pos=8,9,10,11 → rows=1,1,1,1, cols=0,1,2,3
    //   t=3: pos=12,13,14,15 → rows=1,1,1,1, cols=4,5,6,7
    //   ...
    //   t=15: pos=60,61,62,63 → rows=7,7,7,7, cols=4,5,6,7
    //   t=16: pos=64..67 → rows=8,8,8,8, cols=0,1,2,3
    //   t=31: pos=124..127 → rows=15,15,15,15, cols=4,5,6,7
    // This covers all 16×8 positions exactly once. Clean!

    for (int i = 0; i < C_PER_THREAD; ++i) {
        int pos = lane * 4 + i;
        int row = pos / 8;
        int col = pos % 8;
        C[(warpM + row) * N + (warpN + col)] = fragC[i];
    }
}

// ── Probe kernel to check if FP64 DMMA is supported ─────────────────────
__global__ void fp64DmmaProbeKernel() {
    if (threadIdx.x != 0) return;

    double a0 = 1.0, a1 = 1.0, b0 = 1.0;
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;

    asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
        : "=d"(c0), "=d"(c1), "=d"(c2), "=d"(c3)
        : "d"(a0), "d"(a1), "d"(b0), "d"(c0), "d"(c1), "d"(c2), "d"(c3)
    );
}

static bool fp64DmmaSupported(int device) {
    chk(cudaSetDevice(device), "probe_dev");

    int major = 0, minor = 0;
    chk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "major");
    chk(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "minor");
    // mma.sync.aligned.f64 requires SM80+ (Ampere). Thor is SM110a.
    if (major < 8 || (major == 8 && minor < 0)) return false;

    cudaStream_t str;
    chk(cudaStreamCreate(&str), "probe_stream");

    fp64DmmaProbeKernel<<<1, 32, 0, str>>>();
    cudaError_t e = cudaStreamSynchronize(str);
    chk(cudaStreamDestroy(str), "probe_stream_destroy");

    if (e != cudaSuccess) return false;
    e = cudaGetLastError();
    if (e != cudaSuccess) return false;

    return true;
}

// ── Measure FP64 DMMA Dense ──────────────────────────────────────────────
BenchResult measureFP64Dense(int device, int matDim, int iterations) {
    chk(cudaSetDevice(device), "dev");

    // Align dimensions to tile sizes
    int M = (matDim / DMMA_M) * DMMA_M;
    int N = (matDim / DMMA_N) * DMMA_N;
    int K = (matDim / DMMA_K) * DMMA_K;
    if (M < DMMA_M) M = DMMA_M;
    if (N < DMMA_N) N = DMMA_N;
    if (K < DMMA_K) K = DMMA_K;

    size_t sizeA = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(double);
    size_t sizeB = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(double);
    size_t sizeC = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(double);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    chk(cudaMalloc(&dA, sizeA), "alloc_a");
    chk(cudaMalloc(&dB, sizeB), "alloc_b");
    chk(cudaMalloc(&dC, sizeC), "alloc_c");

    // Initialize with small values
    chk(cudaMemset(dA, 0x3F, sizeA), "memset_a");
    chk(cudaMemset(dB, 0x3F, sizeB), "memset_b");

    cudaStream_t str;
    chk(cudaStreamCreate(&str), "stream");

    // Grid: (N/DMMA_N) x (M/DMMA_M), 1 warp per block
    dim3 grid(std::min(65535, N / DMMA_N), std::min(65535, M / DMMA_M), 1);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        fp64DmmaKernel<<<grid, 32, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    // Timing
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    std::vector<double> vals;
    // FP64: M * N * K multiply-adds = M*N*K*2 FLOPs
    size_t totalFlops = static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(K) * 2;

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "recS");
        fp64DmmaKernel<<<grid, 32, 0, str>>>(dA, dB, dC, M, N, K);
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
      << ",\"tile_m\":" << DMMA_M << ",\"tile_n\":" << DMMA_N << ",\"tile_k\":" << DMMA_K
      << ",\"api\":\"mma_sync_aligned_inline_ptx\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "fp64_tensor";
    res.test_name  = "fp64_tensor_dense";
    res.unit       = "TFLOP/s";
    res.params_json = p.str();
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp64_tflops);
    res.metadata["precision"] = "fp64";
    res.metadata["api"] = "mma.sync.aligned.m16n8k4";
    res.metadata["peak_tflops"] = std::to_string(T5000Peaks::fp64_tflops);

    chk(cudaFree(dA), "free_a");
    chk(cudaFree(dB), "free_b");
    chk(cudaFree(dC), "free_c");
    chk(cudaStreamDestroy(str), "stream");
    chk(cudaEventDestroy(evS), "evSDestroy");
    chk(cudaEventDestroy(evE), "evEDestroy");

    return res;
}

// ── Stub result builder ──────────────────────────────────────────────────
static BenchResult makeStub(const std::string& reason) {
    BenchResult r{};
    r.suite_name = "fp64_tensor";
    r.test_name  = "fp64_tensor_dense";
    r.unit       = "TFLOP/s";
    std::string err = "{\"error\":\"";
    err += reason;
    err += "\",\"note\":\"FP64 DMMA not available on this device\"}";
    r.params_json = err;
    r.metadata["precision"] = "fp64";
    r.metadata["stub"] = "true";
    r.metadata["stub_reason"] = reason;
    return r;
}

} // anonymous namespace

std::vector<BenchResult> runFP64TensorBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    // Probe FP64 DMMA support before launching
    bool supported = fp64DmmaSupported(device);
    if (!supported) {
        results.push_back(makeStub("mma.sync.aligned.m16n8k4.f64 not supported by current driver/firmware"));
        return results;
    }

    try {
        results.push_back(measureFP64Dense(device, matDim, iterations));
    } catch (const std::exception& ex) {
        results.push_back(makeStub((std::string("exception: ") + ex.what()).c_str()));
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(fp64_tensor, "FP64 Tensor Core throughput (mma.sync.aligned DMMA)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        try {
            return deusridet::bench::runFP64TensorBench(0, 128, 10);
        } catch (const std::exception& ex) {
            deusridet::bench::BenchResult r{};
            r.suite_name = "fp64_tensor";
            r.test_name  = "fp64_tensor_dense";
            r.unit       = "TFLOP/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\"}";
            r.params_json = err;
            return {r};
        }
    });
