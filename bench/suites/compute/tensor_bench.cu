#include "bench/suites/compute/tensor_bench.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
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
__global__ void mmulFP16Kernel(__half* A, __half* B, __half* C, int M, int N, int K) {
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

// ── BF16 WMMA kernel (16x16x16 tile, float accumulator) ───────────────────
__global__ void mmulBF16Kernel(__nv_bfloat16* A, __nv_bfloat16* B, float* C, int M, int N, int K) {
    using namespace nvcuda::wmma;
    constexpr int tM = 16, tN = 16, tK = 16;
    int bx = blockIdx.x, by = blockIdx.y;
    fragment<matrix_a, tM, tN, tK, __nv_bfloat16, row_major> fa;
    fragment<matrix_b, tM, tN, tK, __nv_bfloat16, col_major> fb;
    fragment<accumulator, tM, tN, tK, float> fc;
    fill_fragment(fc, 0.0f);
    int aR = by * tM, aC = 0, bR = 0, bC = bx * tN;
    for (int tk = 0; tk < K; tk += tK) {
        load_matrix_sync(fa, A + aR * K + aC + tk, K);
        load_matrix_sync(fb, B + bR * N + bC + tk, N);
        mma_sync(fc, fa, fb, fc);
        aC += tK; bR += tK;
    }
    store_matrix_sync(C + aR * N + bC, fc, N, mem_row_major);
}

// ── Stats ──────────────────────────────────────────────────────────────────
BenchResult computeStats(const std::vector<double>& vals,
                         const std::string& suite, const std::string& test,
                         const std::string& unit, const std::string& pj) {
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
    return res;
}

// ── Measure FP16 ───────────────────────────────────────────────────────────
BenchResult measureFP16(half* dA, half* dB, half* dC, int M, int N, int K,
                        dim3 grid, int tpb, double tfl, int iters,
                        cudaStream_t str, cudaEvent_t evS, cudaEvent_t evE) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        mmulFP16Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
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
    return computeStats(vals, "tensor", "fp16_mma", "TFLOP/s", p.str());
}

// ── Measure BF16 ───────────────────────────────────────────────────────────
BenchResult measureBF16(__nv_bfloat16* dA, __nv_bfloat16* dB, float* dC,
                        int M, int N, int K, dim3 grid, int tpb, double tfl, int iters,
                        cudaStream_t str, cudaEvent_t evS, cudaEvent_t evE) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        mmulBF16Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
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
    return computeStats(vals, "tensor", "bf16_mma", "TFLOP/s", p.str());
}

} // anonymous namespace

std::vector<BenchResult> runTensorBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    int dim = (matDim > 0) ? matDim : 1024;
    // Align to WMMA tile (K must be multiple of 16 for 16x16x16 tile)
    int M = ((dim / 16) * 16);
    int N = M, K = M;
    if (K < 16) K = 16;
    dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
    int tpb = 32;

    double tflopPerLaunch = static_cast<double>(M) * N * K * 2.0 / 1.0e12;

    chk(cudaSetDevice(device), "dev");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    // ── FP16 ──
    {
        size_t szAB = M * K * sizeof(half);
        size_t szC  = M * N * sizeof(half);
        half *dA = nullptr, *dB = nullptr, *dC = nullptr;
        chk(cudaMalloc(&dA, szAB), "fp16_a");
        chk(cudaMalloc(&dB, szAB), "fp16_b");
        chk(cudaMalloc(&dC, szC), "fp16_c");
        chk(cudaMemset(dA, 0x3C, szAB), "fp16_a");
        chk(cudaMemset(dB, 0x3C, szAB), "fp16_b");
        for (int w = 0; w < 3; ++w)
            mmulFP16Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "ws");
        try {
            results.push_back(measureFP16(dA, dB, dC, M, N, K, grid, tpb, tflopPerLaunch, iterations, str, evS, evE));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "tensor";
            r.test_name  = "fp16_mma";
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

    // ── BF16 ── (uses float accumulator and output due to WMMA API)
    {
        size_t szAB = M * K * sizeof(__nv_bfloat16);
        size_t szC  = M * N * sizeof(float); // float accumulator output
        __nv_bfloat16 *dA = nullptr, *dB = nullptr;
        float *dC = nullptr;
        chk(cudaMalloc(&dA, szAB), "bf16_a");
        chk(cudaMalloc(&dB, szAB), "bf16_b");
        chk(cudaMalloc(&dC, szC), "bf16_c");
        chk(cudaMemset(dA, 0x3C, szAB), "bf16_a");
        chk(cudaMemset(dB, 0x3C, szAB), "bf16_b");
        for (int w = 0; w < 3; ++w)
            mmulBF16Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "ws");
        try {
            results.push_back(measureBF16(dA, dB, dC, M, N, K, grid, tpb, tflopPerLaunch, iterations, str, evS, evE));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "tensor";
            r.test_name  = "bf16_mma";
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
