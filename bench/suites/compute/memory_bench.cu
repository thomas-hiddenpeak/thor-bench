#include "bench/suites/compute/memory_bench.h"
#include "bench_suites.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace deusridet::bench {

namespace {

constexpr int kTpb = 256;

__global__ void memReadKernel(const float* data, float* out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x)
        s += data[i];
    out[tid] = s;
}

__global__ void memWriteKernel(float* data, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x)
        data[i] = static_cast<float>(i) * 1.234f;
    __threadfence();
}

__global__ void memCopyKernel(const float* src, float* dst, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x)
        dst[i] = src[i];
}

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

void launchKernel(const char* label, float* dSrc, float* dDst,
                  size_t numElems, int gridX, cudaStream_t str) {
    if (label[0] == 'r')
        memReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
    else if (label[0] == 'w')
        memWriteKernel<<<gridX, kTpb, 0, str>>>(dDst, numElems);
    else
        memCopyKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
}

BenchResult measure(const char* label, double byteMult,
                    float* dSrc, float* dDst, size_t numElems,
                    size_t allocBytes, int gridX, int iters,
                    cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str) {
    BenchResult res;
    res.suite_name = "memory";
    res.test_name  = label;
    res.unit       = "GB/s";
    res.warmup_count = 3;

    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        launchKernel(label, dSrc, dDst, numElems, gridX, str);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes * byteMult / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
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
    p << "{\"bytes\":" << allocBytes
      << ",\"mult\":" << byteMult
      << ",\"grid\":" << gridX
      << ",\"tpb\":" << kTpb << "}";
    res.params_json = p.str();
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runMemoryBench(int device, size_t transferSize, int iterations) {
    constexpr int kTpb = 256;
    std::vector<BenchResult> results;

    size_t numElems   = (transferSize + 3) / 4;
    size_t allocBytes = numElems * 4;
    int gridX = std::max(1, std::min(65535, static_cast<int>((numElems + kTpb - 1) / kTpb)));

    float* dSrc = nullptr;
    float* dDst = nullptr;
    cudaEvent_t evS, evE;
    cudaStream_t str;

    chk(cudaSetDevice(device), "dev");
    chk(cudaMalloc(&dSrc, allocBytes), "ms");
    chk(cudaMalloc(&dDst, allocBytes), "md");
    chk(cudaMemset(dSrc, 0xAA, allocBytes), "ms");
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    chk(cudaStreamCreate(&str), "st");

    // warmup
    for (int w = 0; w < 3; ++w) {
        memReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
        memWriteKernel<<<gridX, kTpb, 0, str>>>(dDst, numElems);
        memCopyKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
    }
    chk(cudaStreamSynchronize(str), "ws");

    struct TD { const char* label; double byteMult; };
    TD tests[] = {
        {"hbm_read",  1.0},
        {"hbm_write", 1.0},
        {"hbm_copy",  2.0},
    };

    for (const auto& td : tests) {
        try {
            results.push_back(measure(td.label, td.byteMult, dSrc, dDst, numElems, allocBytes, gridX, iterations, evS, evE, str));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "memory";
            r.test_name  = td.label;
            r.unit       = "GB/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"bytes\":";
            err += std::to_string(allocBytes);
            err += "}";
            r.params_json = err;
            results.push_back(r);
        }
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    chk(cudaFree(dSrc), "fs");
    chk(cudaFree(dDst), "fd");

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(memory, "HBM memory bandwidth (read/write/copy)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runMemoryBench(0, 256 * 1024 * 1024, 10);
    });
