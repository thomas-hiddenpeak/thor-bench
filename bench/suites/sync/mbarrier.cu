#include "sync/mbarrier.h"
#include "bench_schema.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/barrier>
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

BenchResult computeStats(std::vector<double>& vals, int warmup) {
    std::sort(vals.begin(), vals.end());
    int n = static_cast<int>(vals.size());
    double sum = 0;
    for (double v : vals) sum += v;
    double mean = sum / n;

    double sq = 0;
    for (double v : vals) { double d = v - mean; sq += d * d; }
    double stddev = std::sqrt(sq / n);

    auto pct = [&](double p) -> double {
        if (n <= 1) return vals[0];
        double r = p * (n - 1);
        int lo = static_cast<int>(std::floor(r));
        int hi = static_cast<int>(std::ceil(r));
        if (hi >= n) return vals.back();
        return vals[lo] * (1.0 - (r - lo)) + vals[hi] * (r - lo);
    };

    BenchResult res;
    res.sample_count = n;
    res.warmup_count = warmup;
    res.min_val  = vals.front();
    res.max_val  = vals.back();
    res.mean     = mean;
    res.median   = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
    res.stddev   = stddev;
    res.p95      = pct(0.95);
    res.p99      = pct(0.99);
    return res;
}

// ---------------------------------------------------------------------------
// cuda::barrier<thread_scope_block> latency kernel
// Uses placement-new construction in dynamically allocated device memory.
// cuda::barrier in CUDA 13.0 maps to C++20 std::barrier semantics.
// ---------------------------------------------------------------------------

__global__ void mbarrierKernel(float* data, int n, int burns, char* barMem) {
    // placement-new construct the barrier with the expected thread count
    cuda::barrier<cuda::thread_scope_block>* bar =
        new (barMem) cuda::barrier<cuda::thread_scope_block>(blockDim.x);
    __syncthreads(); // ensure all threads have constructed the barrier
    for (int i = 0; i < burns; ++i) {
        int idx = (threadIdx.x + i) % n;
        data[idx] += static_cast<float>(i);
        bar->wait(bar->arrive());
    }
}

// ---------------------------------------------------------------------------
// __syncthreads baseline kernel
// ---------------------------------------------------------------------------

__global__ void syncthreadsKernel(float* data, int n, int burns) {
    int tid = threadIdx.x;
    for (int i = 0; i < burns; ++i) {
        int idx = (tid + i) % n;
        data[idx] += static_cast<float>(i);
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

BenchResult benchMbarrier(int threads, int burns, int iterations) {
    size_t barSize = sizeof(cuda::barrier<cuda::thread_scope_block>);
    char* dBarMem = nullptr;
    chk(cudaMalloc(&dBarMem, barSize), "bar_alloc");

    int n = std::max(threads, 256);
    float* dData = nullptr;
    chk(cudaMalloc(&dData, n * sizeof(float)), "md");
    chk(cudaMemset(dData, 0, n * sizeof(float)), "ms");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");

    // Warmup
    for (int w = 0; w < 3; ++w)
        mbarrierKernel<<<1, threads>>>(dData, n, burns, dBarMem);
    chk(cudaGetLastError(), "mw");
    chk(cudaDeviceSynchronize(), "ws");

    std::vector<double> vals;
    vals.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        chk(cudaMemset(dData, 0, n * sizeof(float)), "ms");
        chk(cudaEventRecord(evS), "rs");
        mbarrierKernel<<<1, threads>>>(dData, n, burns, dBarMem);
        chk(cudaEventRecord(evE), "re");
        chk(cudaDeviceSynchronize(), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double ns = ms * 1000.0;
        double latency = ns / burns; // avg per barrier
        vals.push_back(latency);
    }

    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    chk(cudaFree(dData), "fd");
    chk(cudaFree(dBarMem), "fb");

    if (vals.empty()) {
        BenchResult r;
        r.suite_name = "mbarrier";
        r.test_name  = "mbarrier_" + std::to_string(threads) + "threads";
        r.unit       = "ns";
        r.sample_count = 0;
        r.warmup_count = 0;
        return r;
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "mbarrier";
    res.test_name  = "mbarrier_" + std::to_string(threads) + "threads";
    res.unit       = "ns";

    std::ostringstream p;
    p << "{\"threads\":" << threads
      << ",\"burns\":" << burns
      << ",\"barrier_size_bytes\":" << barSize << "}";
    res.params_json = p.str();
    return res;
}

BenchResult benchSyncthreadsBaseline(int threads, int burns, int iterations) {
    int n = std::max(threads, 256);
    float* dData = nullptr;
    chk(cudaMalloc(&dData, n * sizeof(float)), "md");
    chk(cudaMemset(dData, 0, n * sizeof(float)), "ms");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");

    // Warmup
    for (int w = 0; w < 3; ++w)
        syncthreadsKernel<<<1, threads>>>(dData, n, burns);
    chk(cudaGetLastError(), "sw");
    chk(cudaDeviceSynchronize(), "ws");

    std::vector<double> vals;
    vals.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        chk(cudaMemset(dData, 0, n * sizeof(float)), "ms");
        chk(cudaEventRecord(evS), "rs");
        syncthreadsKernel<<<1, threads>>>(dData, n, burns);
        chk(cudaEventRecord(evE), "re");
        chk(cudaDeviceSynchronize(), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double ns = ms * 1000.0;
        double latency = ns / burns; // avg per __syncthreads
        vals.push_back(latency);
    }

    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    chk(cudaFree(dData), "fd");

    if (vals.empty()) {
        BenchResult r;
        r.suite_name = "mbarrier";
        r.test_name  = "syncthreads_" + std::to_string(threads) + "threads";
        r.unit       = "ns";
        r.sample_count = 0;
        r.warmup_count = 0;
        return r;
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "mbarrier";
    res.test_name  = "syncthreads_" + std::to_string(threads) + "threads";
    res.unit       = "ns";

    std::ostringstream p;
    p << "{\"threads\":" << threads
      << ",\"burns\":" << burns << "}";
    res.params_json = p.str();
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runMBarrierBench(int device, int iterations) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    int burns = 128;

    // cuda::barrier at varying thread counts
    constexpr int threadCounts[] = {64, 128, 256, 512, 1024};
    for (int threads : threadCounts) {
        try {
            results.push_back(benchMbarrier(threads, burns, iterations));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "mbarrier";
            r.test_name  = "mbarrier_" + std::to_string(threads) + "threads";
            r.unit       = "ns";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"threads\":" + std::to_string(threads) + "}";
            r.params_json = err;
            results.push_back(r);
        }
    }

    // __syncthreads baseline at 256 threads for comparison
    try {
        results.push_back(benchSyncthreadsBaseline(256, burns, iterations));
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "mbarrier";
        r.test_name  = "syncthreads_256threads";
        r.unit       = "ns";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"threads\":256}";
        r.params_json = err;
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

#include "bench_suites.h"

BENCH_REGISTER_SUITE(mbarrier, "cuda::barrier<thread_scope_block> latency",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runMBarrierBench(0, 10);
    });
