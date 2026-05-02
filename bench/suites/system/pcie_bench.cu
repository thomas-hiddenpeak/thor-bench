#include "bench/suites/system/pcie_bench.h"
#include <cuda_runtime.h>
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

BenchResult computeStats(const std::vector<double>& vals,
                         const std::string& test, const std::string& pj) {
    BenchResult res;
    res.suite_name = "pcie";
    res.test_name  = test;
    res.unit       = "GB/s";
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

} // anonymous namespace

std::vector<BenchResult> runPCIEBench(int device, size_t transferSize, int iterations) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    // Get device properties
    cudaDeviceProp prop;
    chk(cudaGetDeviceProperties(&prop, device), "props");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    // Allocate pinned host memory and device memory
    unsigned char* hSrc = nullptr;
    unsigned char* hDst = nullptr;
    unsigned char* dBuf = nullptr;

    chk(cudaHostAlloc(&hSrc, transferSize, cudaHostAllocPortable), "hsrc");
    chk(cudaHostAlloc(&hDst, transferSize, cudaHostAllocPortable), "hdst");
    chk(cudaMalloc(&dBuf, transferSize), "dbuf");

    // Fill host source
    for (size_t i = 0; i < transferSize; ++i)
        hSrc[i] = static_cast<unsigned char>(i & 0xFF);

    // Warmup
    chk(cudaMemcpy(dBuf, hSrc, transferSize, cudaMemcpyHostToDevice), "wh2d");
    chk(cudaMemcpy(hDst, dBuf, transferSize, cudaMemcpyDeviceToHost), "wd2h");

    // ── Host → Device ──
    {
        std::vector<double> vals;
        for (int i = 0; i < iterations; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            chk(cudaMemcpyAsync(dBuf, hSrc, transferSize, cudaMemcpyHostToDevice, str), "h2d");
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (transferSize / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        std::ostringstream p;
        p << "{\"transfer_size\":" << transferSize
          << ",\"pinned\":true,"
          << ",\"integrated\":" << (prop.integrated ? "true" : "false") << "}";
        results.push_back(computeStats(vals, "host_to_device", p.str()));
    }

    // ── Device → Host ──
    {
        std::vector<double> vals;
        for (int i = 0; i < iterations; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            chk(cudaMemcpyAsync(hDst, dBuf, transferSize, cudaMemcpyDeviceToHost, str), "d2h");
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (transferSize / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        std::ostringstream p;
        p << "{\"transfer_size\":" << transferSize
          << ",\"pinned\":true,"
          << ",\"integrated\":" << (prop.integrated ? "true" : "false") << "}";
        results.push_back(computeStats(vals, "device_to_host", p.str()));
    }

    // Cleanup
    chk(cudaFree(dBuf), "dbuf");
    chk(cudaFreeHost(hSrc), "hsrc");
    chk(cudaFreeHost(hDst), "hdst");
    chk(cudaStreamDestroy(str), "st");
    chk(cudaEventDestroy(evS), "es");
    chk(cudaEventDestroy(evE), "ee");

    return results;
}

} // namespace deusridet::bench
