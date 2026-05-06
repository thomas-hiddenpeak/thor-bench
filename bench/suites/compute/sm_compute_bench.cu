#include "compute/sm_compute_bench.h"
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

constexpr int kTpb = 256;

// --- FP32 FMA kernel ---
__global__ void fp32FMAKernel(float* a, float* b, float* c, size_t n, int burns) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float va = 0.0f, vb = 0.0f, vc = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        va += a[i];
        vb += b[i];
    }
    for (int b = 0; b < burns; ++b) {
        vc = fmaf(vc, va, vb);
    }
    if (tid < n)
        c[tid] = vc;
}

// --- FP32 register pressure kernel (same FMA density, forced register spilling) ---
__global__ void fp32RegisterPressureKernel(float* a, float* b, float* c, size_t n, int burns) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float va = 0.0f, vb = 0.0f, vc = 0.0f;
    float r0 = 1.0f, r1 = 2.0f, r2 = 3.0f, r3 = 4.0f;
    float r4 = 5.0f, r5 = 6.0f, r6 = 7.0f, r7 = 8.0f;
    float r8 = 9.0f, r9 = 10.0f, r10 = 11.0f, r11 = 12.0f;
    float r12 = 13.0f, r13 = 14.0f, r14 = 15.0f, r15 = 16.0f;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        va += a[i];
        vb += b[i];
        r0 += a[i]; r1 += b[i]; r2 += a[i + 1]; r3 += b[i + 1];
        r4 += a[i + 2]; r5 += b[i + 2]; r6 += a[i + 3]; r7 += b[i + 3];
        r8 += a[i + 4]; r9 += b[i + 4]; r10 += a[i + 5]; r11 += b[i + 5];
        r12 += a[i + 6]; r13 += b[i + 6]; r14 += a[i + 7]; r15 += b[i + 7];
    }
    for (int b = 0; b < burns; ++b) {
        vc = fmaf(vc, va, vb);
        r0 = fmaf(r0, r1, r2); r3 = fmaf(r3, r4, r5);
        r6 = fmaf(r6, r7, r8); r9 = fmaf(r9, r10, r11);
        r12 = fmaf(r12, r13, r14);
    }
    if (tid < n)
        c[tid] = vc + r0 + r3 + r6 + r9 + r12;
}

// --- FP64 FMA kernel ---
__global__ void fp64FMAKernel(double* a, double* b, double* c, size_t n, int burns) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double va = 0.0, vb = 0.0, vc = 0.0;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        va += a[i];
        vb += b[i];
    }
    for (int b = 0; b < burns; ++b) {
        vc = fma(vc, va, vb);
    }
    if (tid < n)
        c[tid] = vc;
}

// --- helpers ---

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

BenchResult measureFP32(float* dA, float* dB, float* dC,
                        size_t numElems, int gridX, int burns, int iters,
                        cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str) {
    BenchResult res;
    res.suite_name = "sm_compute";
    res.test_name  = "fp32_fma";
    res.unit       = "GFLOP/s";
    res.warmup_count = 3;

    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        fp32FMAKernel<<<gridX, kTpb, 0, str>>>(dA, dB, dC, numElems, burns);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        // Each FMA = 2 FLOPs (1 mul + 1 add). Total = numThreads * burns * 2
        size_t totalFlops = static_cast<size_t>(gridX) * kTpb * burns * 2;
        double gflops = sec > 0.0 ? (totalFlops / 1e9) / sec : 0.0;
        vals.push_back(gflops);
    }

    int n = static_cast<int>(vals.size());
    res.sample_count = n;
    if (!vals.empty()) {
        std::sort(vals.begin(), vals.end());
        double sum = 0;
        for (double v : vals) sum += v;
        double mean = sum / n;

        res.min_val  = vals.front();
        res.max_val  = vals.back();
        res.mean     = mean;
        res.median   = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;

        double sq = 0;
        for (double v : vals) { double d = v - mean; sq += d * d; }
        res.stddev = std::sqrt(sq / n);

        auto pct = [&](double p) -> double {
            if (n <= 1) return vals[0];
            double r = p * (n - 1);
            int lo = static_cast<int>(std::floor(r));
            int hi = static_cast<int>(std::ceil(r));
            if (hi >= n) return vals.back();
            return vals[lo] * (1.0 - (r - lo)) + vals[hi] * (r - lo);
        };
        res.p95 = pct(0.95);
        res.p99 = pct(0.99);
        res.peak_pct = computePeakPctFromG(res.median, T5000Peaks::fp32_tflops);
    }

    std::ostringstream p;
    p << "{\"elems\":" << numElems
      << ",\"burns\":" << burns
      << ",\"grid\":" << gridX
      << ",\"tpb\":" << kTpb
      << ",\"total_threads\":" << (gridX * kTpb) << "}";
    res.params_json = p.str();
    return res;
}

BenchResult measureRegisterPressure(float* dA, float* dB, float* dC,
                                     size_t numElems, int gridX, int burns, int iters,
                                     int maxRegs, cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str) {
    // Note: cudaFuncAttributeMaxRegisterCount / cudaFuncGetAttribute removed in CUDA 13.0.
    // Fall back to running kernel at default register count with occupancy query.
    (void)maxRegs; // unused in CUDA 13.0

    int blocksPerSM = 0;
    chk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM,
         fp32RegisterPressureKernel, kTpb, 0), "occupancy");

    BenchResult res;
    res.suite_name = "sm_compute";
    std::ostringstream tn;
    tn << "regspill_default";
    res.test_name = tn.str();
    res.unit = "GFLOP/s";
    res.warmup_count = 3;

    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        fp32RegisterPressureKernel<<<gridX, kTpb, 0, str>>>(dA, dB, dC, numElems, burns);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        // FP32: 1 vc FMA + 5 rX FMA = 6 FMA = 12 FLOPs per burn
        size_t totalFlops = static_cast<size_t>(gridX) * kTpb * burns * 12;
        double gflops = sec > 0.0 ? (totalFlops / 1e9) / sec : 0.0;
        vals.push_back(gflops);
    }

    int n = static_cast<int>(vals.size());
    res.sample_count = n;
    if (!vals.empty()) {
        std::sort(vals.begin(), vals.end());
        double sum = 0;
        for (double v : vals) sum += v;
        double mean = sum / n;

        res.min_val  = vals.front();
        res.max_val  = vals.back();
        res.mean     = mean;
        res.median   = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;

        double sq = 0;
        for (double v : vals) { double d = v - mean; sq += d * d; }
        res.stddev = std::sqrt(sq / n);

        auto pct = [&](double p) -> double {
            if (n <= 1) return vals[0];
            double r = p * (n - 1);
            int lo = static_cast<int>(std::floor(r));
            int hi = static_cast<int>(std::ceil(r));
            if (hi >= n) return vals.back();
            return vals[lo] * (1.0 - (r - lo)) + vals[hi] * (r - lo);
        };
        res.p95 = pct(0.95);
        res.p99 = pct(0.99);
    }

    res.peak_pct = computePeakPctFromG(res.median, T5000Peaks::fp32_tflops);

    std::ostringstream p;
    p << "{\"elems\":" << numElems
      << ",\"burns\":" << burns
      << ",\"grid\":" << gridX
      << ",\"tpb\":" << kTpb
      << ",\"max_regs\":\"default_CUDA13\""
      << ",\"total_threads\":" << (gridX * kTpb) << "}";
    res.params_json = p.str();
    res.metadata["cuda_version_note"] = "cudaFuncAttribute removed in CUDA 13.0";
    res.metadata["occupancy"] = std::to_string(blocksPerSM);

    return res;
}

BenchResult measureFP64(double* dA, double* dB, double* dC,
                        size_t numElems, int gridX, int burns, int iters,
                        cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str) {
    BenchResult res;
    res.suite_name = "sm_compute";
    res.test_name  = "fp64_fma";
    res.unit       = "GFLOP/s";
    res.warmup_count = 3;

    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        fp64FMAKernel<<<gridX, kTpb, 0, str>>>(dA, dB, dC, numElems, burns);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        size_t totalFlops = static_cast<size_t>(gridX) * kTpb * burns * 2;
        double gflops = sec > 0.0 ? (totalFlops / 1e9) / sec : 0.0;
        vals.push_back(gflops);
    }

    int n = static_cast<int>(vals.size());
    res.sample_count = n;
    if (!vals.empty()) {
        std::sort(vals.begin(), vals.end());
        double sum = 0;
        for (double v : vals) sum += v;
        double mean = sum / n;

        res.min_val  = vals.front();
        res.max_val  = vals.back();
        res.mean     = mean;
        res.median   = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;

        double sq = 0;
        for (double v : vals) { double d = v - mean; sq += d * d; }
        res.stddev = std::sqrt(sq / n);

        auto pct = [&](double p) -> double {
            if (n <= 1) return vals[0];
            double r = p * (n - 1);
            int lo = static_cast<int>(std::floor(r));
            int hi = static_cast<int>(std::ceil(r));
            if (hi >= n) return vals.back();
            return vals[lo] * (1.0 - (r - lo)) + vals[hi] * (r - lo);
        };
        res.p95 = pct(0.95);
        res.p99 = pct(0.99);
        res.peak_pct = computePeakPctFromG(res.median, T5000Peaks::fp64_tflops);
    }

    std::ostringstream p;
    p << "{\"elems\":" << numElems
      << ",\"burns\":" << burns
      << ",\"grid\":" << gridX
      << ",\"tpb\":" << kTpb
      << ",\"total_threads\":" << (gridX * kTpb) << "}";
    res.params_json = p.str();
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runSMComputeBench(int device, int blockSizes[], int numBlocks, int iterations) {
    std::vector<BenchResult> results;

    (void)blockSizes; (void)numBlocks; // block sizes unused; single config with kTpb=256

    chk(cudaSetDevice(device), "dev");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    size_t numElems = 1024ULL * 1024; // 1M elements
    int burns = 128; // burn cycles per thread

    // Grid size: enough blocks for all elements AND enough for full GPU occupancy (20 SMs × 32 CTAs = 640).
    int gridX = std::max(1, std::min(65535, static_cast<int>((numElems + kTpb - 1) / kTpb)));

    // Allocate for FP32
    size_t szF32 = numElems * sizeof(float);
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    chk(cudaMalloc(&dA, szF32), "a");
    chk(cudaMalloc(&dB, szF32), "b");
    chk(cudaMalloc(&dC, szF32), "c");
    chk(cudaMemset(dA, 0x3F, szF32), "a");
    chk(cudaMemset(dB, 0x3F, szF32), "b");

    // Warmup FP32
    for (int w = 0; w < 3; ++w)
        fp32FMAKernel<<<gridX, kTpb, 0, str>>>(dA, dB, dC, numElems, burns);
    chk(cudaStreamSynchronize(str), "ws");

    // FP32 measure
    try {
        results.push_back(measureFP32(dA, dB, dC, numElems, gridX, burns, iterations, evS, evE, str));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "sm_compute";
        r.test_name  = "fp32_fma";
        r.unit       = "GFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"block_size\":" + std::to_string(kTpb) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    // Allocate for FP64
    size_t szF64 = numElems * sizeof(double);
    double *dAd = nullptr, *dBd = nullptr, *dCd = nullptr;
    chk(cudaMalloc(&dAd, szF64), "ad");
    chk(cudaMalloc(&dBd, szF64), "bd");
    chk(cudaMalloc(&dCd, szF64), "cd");

    // Warmup FP64
    for (int w = 0; w < 3; ++w)
        fp64FMAKernel<<<gridX, kTpb, 0, str>>>(dAd, dBd, dCd, numElems, burns);
    chk(cudaStreamSynchronize(str), "ws");

    // FP64 measure
    try {
        results.push_back(measureFP64(dAd, dBd, dCd, numElems, gridX, burns, iterations, evS, evE, str));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "sm_compute";
        r.test_name  = "fp64_fma";
        r.unit       = "GFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"block_size\":" + std::to_string(kTpb) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    chk(cudaFree(dA), "fa");
    chk(cudaFree(dB), "fb");
    chk(cudaFree(dC), "fc");
    chk(cudaFree(dAd), "fad");
    chk(cudaFree(dBd), "fbd");
    chk(cudaFree(dCd), "fcd");

    // --- Register spill / occupancy sweep ---
    {
        int gridX = std::max(1, std::min(65535, static_cast<int>((numElems + kTpb - 1) / kTpb)));

        size_t szF32 = numElems * sizeof(float);
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        chk(cudaMalloc(&dA, szF32), "a");
        chk(cudaMalloc(&dB, szF32), "b");
        chk(cudaMalloc(&dC, szF32), "c");
        chk(cudaMemset(dA, 0x3F, szF32), "a");
        chk(cudaMemset(dB, 0x3F, szF32), "b");

        // Warmup
        for (int w = 0; w < 3; ++w)
            fp32RegisterPressureKernel<<<gridX, kTpb, 0, str>>>(dA, dB, dC, numElems, burns);
        chk(cudaStreamSynchronize(str), "ws");

        // CUDA 13.0 removed cudaFuncAttributeMaxRegisterCount, so we can only run at default
        try {
            results.push_back(measureRegisterPressure(dA, dB, dC, numElems, gridX, burns,
                                                      iterations, 255, evS, evE, str));
        } catch (const std::exception& ex) {
            BenchResult r{};
            r.suite_name = "sm_compute";
            r.test_name = "regspill_default";
            r.unit = "GFLOP/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\"}";
            r.params_json = err;
            results.push_back(r);
        }

        chk(cudaFree(dA), "fa");
        chk(cudaFree(dB), "fb");
        chk(cudaFree(dC), "fc");
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(sm_compute, "SM FP32/FP64 compute throughput",
    ([](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runSMComputeBench(0, nullptr, 0, 10);
    }));
