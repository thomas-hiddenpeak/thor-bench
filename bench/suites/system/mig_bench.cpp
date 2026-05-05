#include "system/mig_bench.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

bool isMIGEnabled(int device) {
    int migMode = 0;
    // cudaDevAttrMigMode = 90; may not be defined in CUDA 13.0/Tegra headers
    cudaError_t err = cudaDeviceGetAttribute(&migMode, cudaDeviceAttr(90), device);
    if (err != cudaSuccess) return false;
    return (migMode != 0);
}

BenchResult runFullGPUFP32(int device, int matDim, int iterations) {
    BenchResult res;
    res.suite_name = "mig";
    res.test_name  = "mig_full_gpu";
    res.unit       = "TFLOP/s";
    res.warmup_count = 3;

    chk(cudaSetDevice(device), "dev");

    cublasHandle_t handle;
    chk(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown, "cublasCreate");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");

    int n = matDim;
    size_t size = static_cast<size_t>(n) * n * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    chk(cudaMalloc(&dA, size), "a");
    chk(cudaMalloc(&dB, size), "b");
    chk(cudaMalloc(&dC, size), "c");

    chk(cudaMemset(dA, 0x3F, size), "initA");
    chk(cudaMemset(dB, 0x3F, size), "initB");

    float alpha = 1.0f, beta = 0.0f;

    for (int w = 0; w < 3; ++w) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                     &alpha, dA, n, dB, n, &beta, dC, n);
    }
    chk(cudaDeviceSynchronize(), "warmup sync");

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS), "rs");
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                     &alpha, dA, n, dB, n, &beta, dC, n);
        chk(cudaEventRecord(evE), "re");
        chk(cudaEventSynchronize(evE), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double tflops = sec > 0.0 ? (2.0 * n * n * n) / (sec * 1e12) : 0.0;
        vals.push_back(tflops);
    }

    int count = static_cast<int>(vals.size());
    res.sample_count = count;
    if (!vals.empty()) {
        std::sort(vals.begin(), vals.end());
        double sum = 0;
        for (double v : vals) sum += v;
        double mean = sum / count;

        res.min_val  = vals.front();
        res.max_val  = vals.back();
        res.mean     = mean;
        res.median   = (count % 2 == 1) ? vals[count / 2]
                                         : (vals[count / 2 - 1] + vals[count / 2]) / 2.0;

        double sq = 0;
        for (double v : vals) { double d = v - mean; sq += d * d; }
        res.stddev = std::sqrt(sq / count);

        auto pct = [&](double p) -> double {
            if (count <= 1) return vals[0];
            double r = p * (count - 1);
            int lo = static_cast<int>(std::floor(r));
            int hi = static_cast<int>(std::ceil(r));
            if (hi >= count) return vals.back();
            return vals[lo] * (1.0 - (r - lo)) + vals[hi] * (r - lo);
        };
        res.p95 = pct(0.95);
        res.p99 = pct(0.99);
    }

    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp32_tflops);

    std::ostringstream p;
    p << "{\"mat_dim\":" << n
      << ",\"flops_per_call\":" << (2LL * n * n * n)
      << ",\"note\":\"Full GPU FP32 GEMM baseline (20 SMs / 3 GPC / 10 TPC)\"}";
    res.params_json = p.str();

    chk(cudaFree(dA), "fa");
    chk(cudaFree(dB), "fb");
    chk(cudaFree(dC), "fc");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    cublasDestroy(handle);

    return res;
}

BenchResult makeMIGStub(const std::string& testName, const std::string& note) {
    BenchResult r;
    r.suite_name = "mig";
    r.test_name  = testName;
    r.unit       = "TFLOP/s";
    r.sample_count = 0;
    r.warmup_count = 0;
    r.median = 0.0;
    std::ostringstream p;
    p << "{\"note\":\"" << note << "\"}";
    r.params_json = p.str();
    r.metadata["stub"] = "true";
    r.metadata["stub_reason"] = "MIG not enabled on this device; requires nvidia-smi mig setup";
    return r;
}

} // anonymous namespace

std::vector<BenchResult> runMIGBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    try {
        results.push_back(runFullGPUFP32(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "mig";
        r.test_name  = "mig_full_gpu";
        r.unit       = "TFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"mat_dim\":" + std::to_string(matDim) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    bool migEnabled = isMIGEnabled(device);

    results.push_back(makeMIGStub("mig_0_4tpc",
        migEnabled ? "MIG mode active but partitioning not yet implemented"
                   : "MIG instance 0 with 4 TPCs"));

    results.push_back(makeMIGStub("mig_1_6tpc",
        migEnabled ? "MIG mode active but partitioning not yet implemented"
                   : "MIG instance 1 with 6 TPCs"));

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(mig, "MIG (Multi-Instance GPU) partitioning overhead",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runMIGBench(0, 2048, 10);
    });
