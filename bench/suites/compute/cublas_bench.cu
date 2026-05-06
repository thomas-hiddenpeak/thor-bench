#include "compute/cublas_bench.h"
#include "bench_schema.h"
#include "bench_stats.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
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

inline void chkCublas(cublasStatus_t s, const char* m) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS(") + m + "): status " + std::to_string(s));
}



// ── SGEMM (FP32) ─────────────────────────────────────────────────────────────
BenchResult runSgemm(int device, int matDim, int iterations) {
    BenchResult res;
    res.suite_name = "cublas";
    res.test_name  = "sgemm_fp32";
    res.unit       = "TFLOP/s";
    res.warmup_count = 3;

    chk(cudaSetDevice(device), "dev");

    cublasHandle_t handle;
    chkCublas(cublasCreate(&handle), "create");
    chkCublas(cublasSetStream(handle, 0), "stream");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    int n = std::min(matDim, 4096);
    size_t sz = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    chk(cudaMalloc(&dA, sz), "dA");
    chk(cudaMalloc(&dB, sz), "dB");
    chk(cudaMalloc(&dC, sz), "dC");

    chk(cudaMemset(dA, 0x3F, sz), "initA");
    chk(cudaMemset(dB, 0x3F, sz), "initB");

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int w = 0; w < 3; ++w) {
        chkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                    &alpha, dA, n, dB, n, &beta, dC, n), "warmup");
    }
    chk(cudaDeviceSynchronize(), "warmupSync");

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS), "recS");
        chkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                    &alpha, dA, n, dB, n, &beta, dC, n), "sgemm");
        chk(cudaEventRecord(evE), "recE");
        chk(cudaEventSynchronize(evE), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double tflops = sec > 0.0 ? (2.0 * n * n * n) / (sec * 1e12) : 0.0;
        vals.push_back(tflops);
    }

    std::ostringstream p;
    p << "{\"mat_dim\":" << n << ",\"flops_per_call\":" << (2LL * n * n * n) << "}";

    res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "cublas";
    res.test_name  = "sgemm_fp32";
    res.unit       = "TFLOP/s";
    res.params_json = p.str();
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp32_tflops);
    res.metadata["api"] = "cublas";
    res.metadata["precision"] = "fp32";

    chk(cudaFree(dA), "freeA");
    chk(cudaFree(dB), "freeB");
    chk(cudaFree(dC), "freeC");
    chk(cudaEventDestroy(evS), "destroyEvS");
    chk(cudaEventDestroy(evE), "destroyEvE");
    cublasDestroy(handle);

    return res;
}

// ── DGEMM (FP64) ─────────────────────────────────────────────────────────────
BenchResult runDgemm(int device, int matDim, int iterations) {
    BenchResult res;
    res.suite_name = "cublas";
    res.test_name  = "dgemm_fp64";
    res.unit       = "TFLOP/s";
    res.warmup_count = 3;

    chk(cudaSetDevice(device), "dev");

    cublasHandle_t handle;
    chkCublas(cublasCreate(&handle), "create");
    chkCublas(cublasSetStream(handle, 0), "stream");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    int n = std::min(matDim, 4096);
    size_t sz = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(double);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    chk(cudaMalloc(&dA, sz), "dA");
    chk(cudaMalloc(&dB, sz), "dB");
    chk(cudaMalloc(&dC, sz), "dC");

    chk(cudaMemset(dA, 0x3F, sz), "initA");
    chk(cudaMemset(dB, 0x3F, sz), "initB");

    double alpha = 1.0, beta = 0.0;

    // Warmup
    for (int w = 0; w < 3; ++w) {
        chkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                    &alpha, dA, n, dB, n, &beta, dC, n), "warmup");
    }
    chk(cudaDeviceSynchronize(), "warmupSync");

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS), "recS");
        chkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                    &alpha, dA, n, dB, n, &beta, dC, n), "dgemm");
        chk(cudaEventRecord(evE), "recE");
        chk(cudaEventSynchronize(evE), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double tflops = sec > 0.0 ? (2.0 * n * n * n) / (sec * 1e12) : 0.0;
        vals.push_back(tflops);
    }

    std::ostringstream p;
    p << "{\"mat_dim\":" << n << ",\"flops_per_call\":" << (2LL * n * n * n) << "}";

    res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "cublas";
    res.test_name  = "dgemm_fp64";
    res.unit       = "TFLOP/s";
    res.params_json = p.str();
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp64_tflops);
    res.metadata["api"] = "cublas";
    res.metadata["precision"] = "fp64";

    chk(cudaFree(dA), "freeA");
    chk(cudaFree(dB), "freeB");
    chk(cudaFree(dC), "freeC");
    chk(cudaEventDestroy(evS), "destroyEvS");
    chk(cudaEventDestroy(evE), "destroyEvE");
    cublasDestroy(handle);

    return res;
}

// ── SGEMM Strided Batched (FP32) ─────────────────────────────────────────────
BenchResult runSgemmStridedBatched(int device, int matDim, int iterations) {
    BenchResult res;
    res.suite_name = "cublas";
    res.test_name  = "sgemm_strided_batched_fp32";
    res.unit       = "TFLOP/s";
    res.warmup_count = 3;

    chk(cudaSetDevice(device), "dev");

    cublasHandle_t handle;
    chkCublas(cublasCreate(&handle), "create");
    chkCublas(cublasSetStream(handle, 0), "stream");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    int n = std::min(matDim, 4096);
    int batchCount = 8;
    int64_t stride = static_cast<int64_t>(n) * n;
    size_t sz = static_cast<size_t>(stride * batchCount) * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    chk(cudaMalloc(&dA, sz), "dA");
    chk(cudaMalloc(&dB, sz), "dB");
    chk(cudaMalloc(&dC, sz), "dC");

    chk(cudaMemset(dA, 0x3F, sz), "initA");
    chk(cudaMemset(dB, 0x3F, sz), "initB");

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int w = 0; w < 3; ++w) {
        chkCublas(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n, &alpha, dA, n, stride,
                    dB, n, stride, &beta, dC, n, stride,
                    batchCount), "warmup");
    }
    chk(cudaDeviceSynchronize(), "warmupSync");

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS), "recS");
        chkCublas(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, n, n, &alpha, dA, n, stride,
                    dB, n, stride, &beta, dC, n, stride,
                    batchCount), "sgemmStridedBatched");
        chk(cudaEventRecord(evE), "recE");
        chk(cudaEventSynchronize(evE), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double tflops = sec > 0.0 ? (2.0 * n * n * n * batchCount) / (sec * 1e12) : 0.0;
        vals.push_back(tflops);
    }

    std::ostringstream p;
    p << "{\"mat_dim\":" << n << ",\"batch_count\":" << batchCount
      << ",\"flops_per_call\":" << (2LL * n * n * n * batchCount) << "}";

    res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "cublas";
    res.test_name  = "sgemm_strided_batched_fp32";
    res.unit       = "TFLOP/s";
    res.params_json = p.str();
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp32_tflops);
    res.metadata["api"] = "cublas";
    res.metadata["precision"] = "fp32";
    res.metadata["variant"] = "strided_batched";

    chk(cudaFree(dA), "freeA");
    chk(cudaFree(dB), "freeB");
    chk(cudaFree(dC), "freeC");
    chk(cudaEventDestroy(evS), "destroyEvS");
    chk(cudaEventDestroy(evE), "destroyEvE");
    cublasDestroy(handle);

    return res;
}

// ── cuBLASLt SGEMM (FP32) stub ───────────────────────────────────────────────
// cuBLASLt API changed significantly in CUDA 13.0 (cublasLtMatrixLayoutCreate,
// cublasLtMatmul signatures changed). Tegra availability uncertain. Stub for now.
BenchResult runCublasLtSgemm(int device, int matDim, int iterations) {
    (void)device; (void)matDim; (void)iterations;
    BenchResult res{};
    res.suite_name = "cublas";
    res.test_name  = "lt_sgemm_fp32";
    res.unit       = "TFLOP/s";
    res.sample_count = 0;
    res.warmup_count = 0;
    res.params_json = "{\"note\":\"cuBLASLt API changed in CUDA 13.0; Tegra availability uncertain\"}";
    res.metadata["api"] = "cublasLt";
    res.metadata["precision"] = "fp32";
    res.metadata["stub"] = "true";
    res.metadata["stub_reason"] = "cuBLASLt API changed significantly in CUDA 13.0; Tegra availability uncertain";
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runCublasBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    try {
        results.push_back(runSgemm(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "cublas";
        r.test_name  = "sgemm_fp32";
        r.unit       = "TFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"mat_dim\":" + std::to_string(matDim) + "}";
        r.params_json = err;
        r.metadata["api"] = "cublas";
        r.metadata["precision"] = "fp32";
        results.push_back(r);
    }

    try {
        results.push_back(runDgemm(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "cublas";
        r.test_name  = "dgemm_fp64";
        r.unit       = "TFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"mat_dim\":" + std::to_string(matDim) + "}";
        r.params_json = err;
        r.metadata["api"] = "cublas";
        r.metadata["precision"] = "fp64";
        results.push_back(r);
    }

    try {
        results.push_back(runSgemmStridedBatched(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "cublas";
        r.test_name  = "sgemm_strided_batched_fp32";
        r.unit       = "TFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"mat_dim\":" + std::to_string(matDim) + "}";
        r.params_json = err;
        r.metadata["api"] = "cublas";
        r.metadata["precision"] = "fp32";
        r.metadata["variant"] = "strided_batched";
        results.push_back(r);
    }

    try {
        results.push_back(runCublasLtSgemm(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "cublas";
        r.test_name  = "lt_sgemm_fp32";
        r.unit       = "TFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"mat_dim\":" + std::to_string(matDim) + "}";
        r.params_json = err;
        r.metadata["api"] = "cublasLt";
        r.metadata["precision"] = "fp32";
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(cublas, "cuBLAS/cuBLASLt GEMM throughput (FP32/FP64)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runCublasBench(0, 2048, 10);
    });
