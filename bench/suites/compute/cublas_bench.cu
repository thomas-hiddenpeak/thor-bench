#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "compute/cublas_bench.h"
#include "bench_schema.h"
#include "bench_stats.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include "sweep_schema.h"
#include "power_monitor.h"

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

inline void chkCublasLt(cublasStatus_t s, const char* m) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLASLt(") + m + "): status " + std::to_string(s));
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

// ── cuBLASLt SGEMM (FP32) ────────────────────────────────────────────────────
// Square n×n×n GEMM via cublasLtMatmul. cuBLASLt uses column-major internally;
// for square matrices transA=T, transB=T with uniform layouts produces the correct
// result (equivalent to transA=T, transB=N for square operands).
BenchResult runCublasLtSgemm(int device, int matDim, int iterations) {
    BenchResult res;
    res.suite_name = "cublas";
    res.test_name  = "lt_sgemm_fp32";
    res.unit       = "TFLOP/s";
    res.warmup_count = 3;

    chk(cudaSetDevice(device), "dev");

    cublasLtHandle_t ltHandle;
    chkCublasLt(cublasLtCreate(&ltHandle), "create");

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

    // Operation descriptor: FP32 compute, both operands transposed
    cublasLtMatmulDesc_t operationDesc;
    chkCublasLt(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "opDesc");
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_T;
    chkCublasLt(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                &transA, sizeof(transA)), "transA");
    chkCublasLt(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                &transB, sizeof(transB)), "transB");

    // Matrix layouts: all n×n, leading dim = n
    cudaDataType_t Atype = CUDA_R_32F;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    chkCublasLt(cublasLtMatrixLayoutCreate(&Adesc, Atype, n, n, n), "Adesc");
    chkCublasLt(cublasLtMatrixLayoutCreate(&Bdesc, Atype, n, n, n), "Bdesc");
    chkCublasLt(cublasLtMatrixLayoutCreate(&Cdesc, Atype, n, n, n), "Cdesc");
    chkCublasLt(cublasLtMatrixLayoutCreate(&Ddesc, Atype, n, n, n), "Ddesc");

    // Preference: no workspace
    cublasLtMatmulPreference_t preference;
    chkCublasLt(cublasLtMatmulPreferenceCreate(&preference), "pref");
    size_t workspace = 0;
    chkCublasLt(cublasLtMatmulPreferenceSetAttribute(preference,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)), "prefWS");

    // Find best algorithm
    cublasLtMatmulHeuristicResult_t algo;
    int returned = 0;
    chkCublasLt(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc,
                preference, 1, &algo, &returned), "algo");
    if (returned == 0) {
        throw std::runtime_error("cuBLASLt: no algorithm found for SGEMM");
    }

    // Warmup
    for (int w = 0; w < 3; ++w) {
        chkCublasLt(cublasLtMatmul(ltHandle, operationDesc, &alpha, dA, Adesc, dB, Bdesc,
                    &beta, dC, Cdesc, dC, Ddesc, &algo.algo, nullptr, 0, 0), "warmup");
    }
    chk(cudaDeviceSynchronize(), "warmupSync");

    // Measurement
    long long flopsPerCall = 2LL * n * n * n;
    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS), "recS");
        chkCublasLt(cublasLtMatmul(ltHandle, operationDesc, &alpha, dA, Adesc, dB, Bdesc,
                    &beta, dC, Cdesc, dC, Ddesc, &algo.algo, nullptr, 0, 0), "ltSgemm");
        chk(cudaEventRecord(evE), "recE");
        chk(cudaEventSynchronize(evE), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double tflops = sec > 0.0 ? flopsPerCall / (sec * 1e12) : 0.0;
        vals.push_back(tflops);
    }

    std::ostringstream p;
    p << "{\"mat_dim\":" << n << ",\"flops_per_call\":" << flopsPerCall << "}";

    res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "cublas";
    res.test_name  = "lt_sgemm_fp32";
    res.unit       = "TFLOP/s";
    res.params_json = p.str();
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp32_tflops);
    res.metadata["api"] = "cublasLt";
    res.metadata["precision"] = "fp32";

    // Cleanup
    chk(cudaFree(dA), "freeA");
    chk(cudaFree(dB), "freeB");
    chk(cudaFree(dC), "freeC");
    chk(cudaEventDestroy(evS), "destroyEvS");
    chk(cudaEventDestroy(evE), "destroyEvE");
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);

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

std::string getSweepTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf{};
    localtime_r(&time_t_now, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(cublas, "cuBLAS/cuBLASLt GEMM throughput (FP32/FP64)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runCublasBench(0, 1024, 10);
    });

BENCH_REGISTER_SWEEP_SUITE(cublas, "cuBLAS/cuBLASLt GEMM throughput (FP32/FP64)",
    [](deusridet::bench::BenchRunner& runner, int device) -> std::vector<deusridet::bench::SweepReport> {
        deusridet::bench::SweepReport report;
        report.suite_name = "cublas";
        report.description = "cuBLAS/cuBLASLt GEMM throughput (FP32/FP64)";
        report.param_names.push_back("mat_dim");

        deusridet::bench::PowerMonitor pm;
        pm.init();

        for (int matDim : std::vector<int>{512, 1024, 2048, 4096}) {
            deusridet::bench::SweepResult point;
            point.suite_name = "cublas";
            point.test_name = "sgemm/dgemm/lt_sgemm";
            {
                std::ostringstream p;
                p << "{\"mat_dim\":" << matDim << "}";
                point.params_json = p.str();
            }
            pm.markStart();
            try {
                auto benchResults = deusridet::bench::runCublasBench(device, matDim, 10);
                if (!benchResults.empty()) {
                    point.result = benchResults[0];
                }
            } catch (const std::exception& e) {
                point.error_message = e.what();
            }
            point.power_watts = pm.markEnd();
            point.timestamp = deusridet::bench::getSweepTimestamp();
            report.points.push_back(point);
        }

        pm.shutdown();

        report.total_points   = static_cast<int>(report.points.size());
        report.success_points = 0;
        report.error_points   = 0;
        for (const auto& pt : report.points) {
            if (pt.error_message.has_value()) ++report.error_points;
            else ++report.success_points;
        }

        return {report};
    });
