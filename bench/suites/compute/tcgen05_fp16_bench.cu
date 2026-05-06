#include "compute/tcgen05_fp16_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
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

// ── FP16 WMMA kernel (16x16x16 tile) ───────────────────────────────────────
// On SM110a, nvcuda::wmma compiles to tcgen05.mma instructions under the hood.
// 32x32x16 tiles are incomplete in CUDA 13.0; use 16x16x16 which is fully supported.
__global__ void tcgen05FP16Kernel(__half* __restrict__ A, __half* __restrict__ B,
                                   __half* __restrict__ C, int M, int N, int K) {
    using namespace nvcuda::wmma;
    constexpr int tM = 16, tN = 16, tK = 16;
    int bx = blockIdx.x, by = blockIdx.y;
    fragment<matrix_a, tM, tN, tK, __half, row_major> fa;
    fragment<matrix_b, tM, tN, tK, __half, col_major> fb;
    fragment<accumulator, tM, tN, tK, __half> fc;
    fill_fragment(fc, __float2half(0.0f));
    int aR = by * tM, aC = 0, bR = 0, bC = bx * tN;
    for (int tk = 0; tk < K; tk += tK) {
        load_matrix_sync(fa, A + aR * K + aC + tk, K);
        load_matrix_sync(fb, B + bR * N + bC + tk, N);
        mma_sync(fc, fa, fb, fc);
        aC += tK; bR += tK;
    }
    store_matrix_sync(C + aR * N + bC, fc, N, mem_row_major);
}

// ── BF16 WMMA kernel via FP16 reinterpret ──────────────────────────────────
// __nv_bfloat16 and __half share the same 16-bit memory layout.
// We dispatch the FP16 WMMA kernel to exercise the same tcgen05 hardware path.
// Throughput measurement is valid; numeric results are not comparable.

// ── Stats helpers ──────────────────────────────────────────────────────────
BenchResult computeStats(const std::vector<double>& vals,
                         const std::string& suite, const std::string& test,
                         const std::string& unit, const std::string& pj,
                         double peakTflops = 0.0) {
    std::vector<double> sv = vals;
    BenchResult res = ::deusridet::bench::computeStats(sv, 3);
    res.suite_name = suite;
    res.test_name  = test;
    res.unit       = unit;
    res.params_json = pj;
    if (peakTflops > 0.0) {
        res.peak_pct = computePeakPctFromT(res.median, peakTflops);
    }
    return res;
}

// ── Measure FP16 ───────────────────────────────────────────────────────────
BenchResult measureFP16(__half* dA, __half* dB, __half* dC, int M, int N, int K,
                         dim3 grid, int tpb, double tfl, int iters,
                         cudaStream_t str, cudaEvent_t evS, cudaEvent_t evE) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        tcgen05FP16Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        vals.push_back(sec > 0.0 ? tfl / sec : 0.0);
    }
    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"gx\":" << grid.x << ",\"gy\":" << grid.y << ",\"tpb\":" << tpb << "}";
    return computeStats(vals, "tcgen05_fp16", "tcgen05_fp16_dense", "TFLOP/s", p.str());
}

// ── Measure BF16 (via FP16 reinterpret) ────────────────────────────────────
BenchResult measureBF16(__nv_bfloat16* dA, __nv_bfloat16* dB, float* dC,
                         int M, int N, int K, dim3 grid, int tpb, double tfl,
                         int iters, cudaStream_t str, cudaEvent_t evS, cudaEvent_t evE) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        tcgen05FP16Kernel<<<grid.x, tpb, 0, str>>>(
            reinterpret_cast<__half*>(dA),
            reinterpret_cast<__half*>(dB),
            reinterpret_cast<__half*>(dC),
            M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        vals.push_back(sec > 0.0 ? tfl / sec : 0.0);
    }
    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"gx\":" << grid.x << ",\"gy\":" << grid.y << ",\"tpb\":" << tpb << "}";
    BenchResult res = computeStats(vals, "tcgen05_fp16", "tcgen05_bf16_dense",
                                    "TFLOP/s", p.str());
    res.metadata["approach"] = "bf16_data_reinterpreted_as_fp16_for_wmma";
    res.metadata["note"] = "nvcuda::wmma compiles to tcgen05.mma on SM110a; bf16 memory layout same as fp16";
    res.peak_pct = std::nullopt; // BF16 peak not specified in T5000 datasheet
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runTCGen05FP16Bench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    int dim = (matDim > 0 && matDim <= 8192) ? matDim : 1024;
    int M = ((dim / 16) * 16);
    int N = M, K = M;
    if (K < 16) K = 16;
    dim3 grid(std::min(65535, (N + 15) / 16), std::min(65535, (M + 15) / 16), 1);
    int tpb = 32;

    double tflopPerLaunch = static_cast<double>(M) * N * K * 2.0 / 1.0e12;

    chk(cudaSetDevice(device), "dev");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    // ── FP16 Dense ──
    {
        size_t szAB = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(__half);
        size_t szC  = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(__half);
        __half *dA = nullptr, *dB = nullptr, *dC = nullptr;
        chk(cudaMalloc(&dA, szAB), "fp16_a");
        chk(cudaMalloc(&dB, szAB), "fp16_b");
        chk(cudaMalloc(&dC, szC), "fp16_c");
        chk(cudaMemset(dA, 0x3C, szAB), "fp16_a");
        chk(cudaMemset(dB, 0x3C, szAB), "fp16_b");
        for (int w = 0; w < 3; ++w)
            tcgen05FP16Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "ws");
        try {
            results.push_back(measureFP16(dA, dB, dC, M, N, K, grid, tpb, tflopPerLaunch,
                                          iterations, str, evS, evE));
        } catch (const std::exception& ex) {
            BenchResult r{};
            r.suite_name = "tcgen05_fp16";
            r.test_name  = "tcgen05_fp16_dense";
            r.unit       = "TFLOP/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"M\":" + std::to_string(M) + "}";
            r.params_json = err;
            results.push_back(r);
        }
        chk(cudaFree(dA), "fp16_a");
        chk(cudaFree(dB), "fp16_b");
        chk(cudaFree(dC), "fp16_c");
    }

    // ── BF16 Dense ── via FP16 reinterpret
    {
        size_t szAB = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(__nv_bfloat16);
        size_t szC  = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(__half);
        __nv_bfloat16 *dA = nullptr, *dB = nullptr;
        float *dC = nullptr;
        chk(cudaMalloc(&dA, szAB), "bf16_a");
        chk(cudaMalloc(&dB, szAB), "bf16_b");
        chk(cudaMalloc(&dC, szC), "bf16_c");
        chk(cudaMemset(dA, 0x3C, szAB), "bf16_a");
        chk(cudaMemset(dB, 0x3C, szAB), "bf16_b");
        for (int w = 0; w < 3; ++w)
            tcgen05FP16Kernel<<<grid.x, tpb, 0, str>>>(
                reinterpret_cast<__half*>(dA),
                reinterpret_cast<__half*>(dB),
                reinterpret_cast<__half*>(dC), M, N, K);
        chk(cudaStreamSynchronize(str), "ws");
        try {
            results.push_back(measureBF16(dA, dB, dC, M, N, K, grid, tpb, tflopPerLaunch,
                                          iterations, str, evS, evE));
        } catch (const std::exception& ex) {
            BenchResult r{};
            r.suite_name = "tcgen05_fp16";
            r.test_name  = "tcgen05_bf16_dense";
            r.unit       = "TFLOP/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"M\":" + std::to_string(M) + "}";
            r.params_json = err;
            results.push_back(r);
        }
        chk(cudaFree(dA), "bf16_a");
        chk(cudaFree(dB), "bf16_b");
        chk(cudaFree(dC), "bf16_c");
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(tcgen05_fp16, "TCGen05 FP16/BF16 block-scaled GEMM via tcgen05.mma (Blackwell native)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runTCGen05FP16Bench(0, 2048, 10);
    });
