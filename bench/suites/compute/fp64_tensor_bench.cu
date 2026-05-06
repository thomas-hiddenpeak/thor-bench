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

// ── FP64 WMMA kernel attempt ───────────────────────────────────────────────
// nvcuda::wmma does not provide double/kind::f64 fragment types in CUDA 13.0.
// FP64 Tensor Core on tcgen05 requires descriptor-based inline PTX
// (tcgen05.mma with kind::f64), which is not exposed via the C++ WMMA API.
// We attempt compilation here and fall back to a stub if the types are missing.

#ifdef __CUDA_WmmaSupportDouble__

__global__ void mmulFP64Kernel(double* A, double* B, double* C, int M, int N, int K) {
    using namespace nvcuda::wmma;
    constexpr int tM = 16, tN = 16, tK = 16;
    int bx = blockIdx.x, by = blockIdx.y;
    fragment<matrix_a, tM, tN, tK, double, row_major> fa;
    fragment<matrix_b, tM, tN, tK, double, col_major> fb;
    fragment<accumulator, tM, tN, tK, double> fc;
    fill_fragment(fc, 0.0);
    int aR = by * tM, aC = 0, bR = 0, bC = bx * tN;
    for (int tk = 0; tk < K; tk += tK) {
        load_matrix_sync(fa, A + aR * K + aC + tk, K);
        load_matrix_sync(fb, B + bR * N + bC + tk, N);
        mma_sync(fc, fa, fb, fc);
        aC += tK; bR += tK;
    }
    store_matrix_sync(C + aR * N + bC, fc, N, mem_row_major);
}

BenchResult measureFP64Dense(double* dA, double* dB, double* dC,
                              int M, int N, int K, dim3 grid, int tpb,
                              double tfl, int iters,
                              cudaStream_t str, cudaEvent_t evS, cudaEvent_t evE) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        mmulFP64Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
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
    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "fp64_tensor";
    res.test_name  = "fp64_tensor_dense";
    res.unit       = "TFLOP/s";
    res.params_json = p.str();
    res.metadata["tcgen05"] = "true";
    res.metadata["precision"] = "fp64";
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp64_tflops);
    return res;
}

#endif // __CUDA_WmmaSupportDouble__

// ── FP64 Dense stub ────────────────────────────────────────────────────────
BenchResult measureFP64DenseStub(int device, int matDim, int iterations) {
    (void)device; (void)iterations;
    BenchResult res{};
    res.suite_name = "fp64_tensor";
    res.test_name  = "fp64_tensor_dense";
    res.unit       = "TFLOP/s";
    res.sample_count = 0;
    res.warmup_count = 0;
    res.params_json = std::string("{\"M\":") + std::to_string(matDim) + ",\"note\":\"FP64 WMMA stub\"}";
    res.metadata["tcgen05"] = "true";
    res.metadata["precision"] = "fp64";
    res.metadata["stub"] = "true";
    res.metadata["stub_reason"] = "nvcuda::wmma FP64 fragments not available in CUDA 13.0; tcgen05.mma kind::f64 requires descriptor-based PTX";
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runFP64TensorBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

#ifdef __CUDA_WmmaSupportDouble__
    // Attempt real FP64 WMMA
    try {
        int dim = (matDim > 0 && matDim <= 8192) ? matDim : 128;
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

        size_t szAB = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(double);
        size_t szC  = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(double);
        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        chk(cudaMalloc(&dA, szAB), "fp64_a");
        chk(cudaMalloc(&dB, szAB), "fp64_b");
        chk(cudaMalloc(&dC, szC), "fp64_c");
        chk(cudaMemset(dA, 0x3F, szAB), "fp64_a");
        chk(cudaMemset(dB, 0x3F, szAB), "fp64_b");

        for (int w = 0; w < 3; ++w)
            mmulFP64Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "ws");

        results.push_back(measureFP64Dense(dA, dB, dC, M, N, K, grid, tpb,
                                           tflopPerLaunch, iterations, str, evS, evE));

        chk(cudaFree(dA), "fp64_a");
        chk(cudaFree(dB), "fp64_b");
        chk(cudaFree(dC), "fp64_c");
        chk(cudaStreamDestroy(str), "ds");
        chk(cudaEventDestroy(evS), "de");
        chk(cudaEventDestroy(evE), "de");
    } catch (const std::exception&) {
        results.push_back(measureFP64DenseStub(device, matDim, iterations));
    }
#else
    // FP64 WMMA not available — return stub
    results.push_back(measureFP64DenseStub(device, matDim, iterations));
#endif

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(fp64_tensor, "FP64 Tensor Core throughput (tcgen05)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
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
