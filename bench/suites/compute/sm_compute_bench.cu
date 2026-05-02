#include "bench/suites/compute/sm_compute_bench.h"
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

    // Default block sizes if not provided
    std::vector<int> bszs;
    if (blockSizes && numBlocks > 0) {
        for (int i = 0; i < numBlocks; ++i)
            bszs.push_back(blockSizes[i]);
    } else {
        bszs = {128, 256, 512, 1024};
    }

    // Get device properties for peak GFLOP calculation
    cudaDeviceProp prop;
    chk(cudaGetDeviceProperties(&prop, device), "props");

    // Note: prop.clockRate deprecated in newer CUDA; using limits instead

    chk(cudaSetDevice(device), "dev");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    size_t numElems = 1024ULL * 1024; // 1M elements
    int burns = 128; // burn cycles per thread

    for (int bsz : bszs) {
        int gridX = std::max(1, std::min(65535, static_cast<int>(numBlocks)));
        // Use provided block size, clamped to reasonable range
        int tpb = std::max(64, std::min(1024, bsz));

        // Allocate for FP32
        size_t szF32 = numElems * sizeof(float);
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        chk(cudaMalloc(&dA, szF32), "a");
        chk(cudaMalloc(&dB, szF32), "b");
        chk(cudaMalloc(&dC, szF32), "c");
        chk(cudaMemset(dA, 0x3F, szF32), "a");
        chk(cudaMemset(dB, 0x3F, szF32), "b");

        // Warmup
        for (int w = 0; w < 3; ++w)
            fp32FMAKernel<<<gridX, tpb, 0, str>>>(dA, dB, dC, numElems, burns);
        chk(cudaStreamSynchronize(str), "ws");

        // FP32 measure
        try {
            results.push_back(measureFP32(dA, dB, dC, numElems, gridX, burns, iterations, evS, evE, str));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "sm_compute";
            r.test_name  = "fp32_fma";
            r.unit       = "GFLOP/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"block_size\":" + std::to_string(tpb) + "}";
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
            fp64FMAKernel<<<gridX, tpb, 0, str>>>(dAd, dBd, dCd, numElems, burns);
        chk(cudaStreamSynchronize(str), "ws");

        // FP64 measure
        try {
            results.push_back(measureFP64(dAd, dBd, dCd, numElems, gridX, burns, iterations, evS, evE, str));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "sm_compute";
            r.test_name  = "fp64_fma";
            r.unit       = "GFLOP/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"block_size\":" + std::to_string(tpb) + "}";
            r.params_json = err;
            results.push_back(r);
        }

        chk(cudaFree(dA), "fa");
        chk(cudaFree(dB), "fb");
        chk(cudaFree(dC), "fc");
        chk(cudaFree(dAd), "fad");
        chk(cudaFree(dBd), "fbd");
        chk(cudaFree(dCd), "fcd");
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return results;
}

} // namespace deusridet::bench
