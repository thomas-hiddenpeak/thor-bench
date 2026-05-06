#include "compute/int8_scalar_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
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

// ── Stats helpers ──────────────────────────────────────────────────────────
BenchResult computeStats(const std::vector<double>& vals,
                          const std::string& suite, const std::string& test,
                          const std::string& unit, const std::string& pj,
                          double peakTflops = 0.0) {
    BenchResult res;
    res.suite_name = suite;
    res.test_name  = test;
    res.unit       = unit;
    res.warmup_count = 3;
    res.params_json = pj;
    int n = static_cast<int>(vals.size());
    res.sample_count = n;
    if (!vals.empty()) {
        std::vector<double> sv = vals;
        std::sort(sv.begin(), sv.end());
        double sum = 0;
        for (double v : sv) sum += v;
        double mean = sum / n;
        res.min_val  = sv.front();
        res.max_val  = sv.back();
        res.mean     = mean;
        res.median   = (n % 2 == 1) ? sv[n / 2] : (sv[n / 2 - 1] + sv[n / 2]) / 2.0;
        double sq = 0;
        for (double v : sv) { double d = v - mean; sq += d * d; }
        res.stddev = std::sqrt(sq / n);
        auto pct = [&](double p) -> double {
            if (n <= 1) return sv[0];
            double r = p * (n - 1);
            int lo = static_cast<int>(std::floor(r));
            int hi = static_cast<int>(std::ceil(r));
            if (hi >= n) return sv.back();
            return sv[lo] * (1.0 - (r - lo)) + sv[hi] * (r - lo);
        };
        res.p95 = pct(0.95);
        res.p99 = pct(0.99);
    }
    if (peakTflops > 0.0) {
        res.peak_pct = computePeakPctFromT(res.median, peakTflops);
    }
    return res;
}

// ── Measure INT8 Dense ─────────────────────────────────────────────────────
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

    BenchResult res = computeStats(vals, "int8_scalar", "int8_scalar_dense",
                                     "TOP/s", p.str());
    res.metadata["note"] = "scalar INT8 kernel with int32 accumulate; tcgen05.mma kind::i8 PTX requires SMEM descriptors + TMEM alloc";

    chk(cudaFree(dA), "fa");
    chk(cudaFree(dB), "fb");
    chk(cudaFree(dC), "fc");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ── INT8 Sparse stub ───────────────────────────────────────────────────────
// INT8 2:4 sparse requires tcgen05.mma.sp inline PTX with sparsity metadata.
BenchResult measureINT8Sparse(int device, int matDim, int iterations) {
    (void)device; (void)matDim; (void)iterations;
    BenchResult res{};
    res.suite_name = "int8_scalar";
    res.test_name  = "int8_scalar_sparse";
    res.unit       = "TOP/s";
    res.sample_count = 0;
    res.warmup_count = 0;
    res.params_json = "{\"note\":\"INT8 2:4 sparse requires tcgen05.mma.sp with sparsity metadata descriptor\"}";
    res.metadata["stub"] = "true";
    res.metadata["stub_reason"] = "INT8 sparse tcgen05.mma.sp requires sparsity metadata + descriptor-based layout";
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runINT8ScalarBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    // --- INT8 Dense ---
    try {
        results.push_back(measureINT8Dense(device, matDim, iterations));
    } catch (const std::exception& ex) {
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
    try {
        results.push_back(measureINT8Sparse(device, matDim, iterations));
    } catch (const std::exception& ex) {
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
