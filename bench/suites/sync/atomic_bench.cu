#include "sync/atomic_bench.h"
#include "bench_schema.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

// ---------------------------------------------------------------------------
// Atomic kernels — multiple threads contend on a small pool of counters.
// Each kernel runs a tight loop of `burns` iterations.
// Host-side cudaEvent_t timing (clock64 is unreliable on SM110a).
// ---------------------------------------------------------------------------

__global__ void atomicAddIntKernel(int* counters, int numCounters, int burns) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % numCounters;
    int val = 1;
    for (int i = 0; i < burns; ++i) {
        atomicAdd(&counters[lane], val);
        // Rotate the addend to prevent compile-time folding.
        val = (val + (i & 3) + 1) & 0xFF;
    }
}

__global__ void atomicAddFloatKernel(float* counters, int numCounters, int burns) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % numCounters;
    float val = 1.0f;
    for (int i = 0; i < burns; ++i) {
        atomicAdd(&counters[lane], val);
        val = val + static_cast<float>((i & 3) + 1) * 0.1f;
    }
}

__global__ void atomicCASKernel(int* counters, int numCounters, int burns) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % numCounters;
    int expected = 0;
    for (int i = 0; i < burns; ++i) {
        int old = atomicCAS(&counters[lane], expected, i + 1);
        expected = old + 1;
    }
}

__global__ void atomicMaxKernel(int* counters, int numCounters, int burns) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % numCounters;
    for (int i = 0; i < burns; ++i) {
        int val = (i + 1) & 0xFFFF;
        atomicMax(&counters[lane], val);
    }
}

__global__ void atomicMinKernel(int* counters, int numCounters, int burns) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % numCounters;
    for (int i = 0; i < burns; ++i) {
        // Walk down from a large value so min keeps decreasing.
        int val = 100000 - i;
        atomicMin(&counters[lane], val);
    }
}

// ---------------------------------------------------------------------------
// Benchmark helpers — generic for integer and float atomic kernels
// ---------------------------------------------------------------------------

template<typename T, typename KernelFn>
BenchResult benchAtomic(const char* testName, KernelFn kernel, int grid,
                        int threads, int numCounters, int burns, int iterations) {
    int totalThreads = grid * threads;
    size_t allocBytes = numCounters * sizeof(T);

    T* dCounters = nullptr;
    chk(cudaMalloc(&dCounters, allocBytes), "malloc");
    chk(cudaMemset(dCounters, 0, allocBytes), "memset");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "evs");
    chk(cudaEventCreate(&evE), "eve");

    // Warmup
    for (int w = 0; w < 3; ++w)
        kernel<<<grid, threads>>>(dCounters, numCounters, burns);
    chk(cudaGetLastError(), "warmup");
    chk(cudaDeviceSynchronize(), "wsync");

    std::vector<double> vals;
    vals.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        chk(cudaMemset(dCounters, 0, allocBytes), "ms");
        chk(cudaEventRecord(evS), "rs");
        kernel<<<grid, threads>>>(dCounters, numCounters, burns);
        chk(cudaEventRecord(evE), "re");
        chk(cudaDeviceSynchronize(), "sync");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "elapsed");
        double ns = ms * 1000.0;
        // Per-atomic latency: total time / (total_threads * burns)
        double latency = ns / (static_cast<double>(totalThreads) * burns);
        vals.push_back(latency);
    }

    chk(cudaEventDestroy(evS), "des");
    chk(cudaEventDestroy(evE), "dee");
    chk(cudaFree(dCounters), "free");

    if (vals.empty()) {
        BenchResult r;
        r.suite_name = "atomic";
        r.test_name  = testName;
        r.unit       = "ns";
        r.sample_count = 0;
        r.warmup_count = 0;
        return r;
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "atomic";
    res.test_name  = testName;
    res.unit       = "ns";

    std::ostringstream p;
    p << "{\"grid\":" << grid
      << ",\"threads\":" << threads
      << ",\"counters\":" << numCounters
      << ",\"burns\":" << burns
      << ",\"total_atoms\":" << (totalThreads * burns) << "}";
    res.params_json = p.str();
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runAtomicBench(int device, int iterations) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    constexpr int grid   = 1;     // 1 block
    constexpr int threads = 1024; // 1024 threads/block
    constexpr int burns   = 10;   // iterations per thread per sample
    // Number of counters: single counter to force contention.
    constexpr int numCounters = 1;

    // -----------------------------------------------------------------------
    // 1. atomicAdd (int)
    // -----------------------------------------------------------------------
    try {
        BenchResult res = benchAtomic<int>(
            "atomicAdd_int", atomicAddIntKernel,
            grid, threads, numCounters, burns, iterations);
        res.metadata["op"] = "atomic_add_int";
        results.push_back(res);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "atomic";
        r.test_name  = "atomicAdd_int";
        r.unit       = "ns";
        r.metadata["op"] = "atomic_add_int";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"grid\":" + std::to_string(grid) + ",\"threads\":" + std::to_string(threads) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // 2. atomicAdd (float)
    // -----------------------------------------------------------------------
    try {
        BenchResult res = benchAtomic<float>(
            "atomicAdd_float", atomicAddFloatKernel,
            grid, threads, numCounters, burns, iterations);
        res.metadata["op"] = "atomic_add_float";
        results.push_back(res);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "atomic";
        r.test_name  = "atomicAdd_float";
        r.unit       = "ns";
        r.metadata["op"] = "atomic_add_float";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"grid\":" + std::to_string(grid) + ",\"threads\":" + std::to_string(threads) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // 3. atomicCAS
    // -----------------------------------------------------------------------
    try {
        BenchResult res = benchAtomic<int>(
            "atomicCAS", atomicCASKernel,
            grid, threads, numCounters, burns, iterations);
        res.metadata["op"] = "atomic_cas";
        results.push_back(res);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "atomic";
        r.test_name  = "atomicCAS";
        r.unit       = "ns";
        r.metadata["op"] = "atomic_cas";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"grid\":" + std::to_string(grid) + ",\"threads\":" + std::to_string(threads) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // 4. atomicMax
    // -----------------------------------------------------------------------
    try {
        BenchResult res = benchAtomic<int>(
            "atomicMax", atomicMaxKernel,
            grid, threads, numCounters, burns, iterations);
        res.metadata["op"] = "atomic_max";
        results.push_back(res);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "atomic";
        r.test_name  = "atomicMax";
        r.unit       = "ns";
        r.metadata["op"] = "atomic_max";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"grid\":" + std::to_string(grid) + ",\"threads\":" + std::to_string(threads) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // 5. atomicMin
    // -----------------------------------------------------------------------
    try {
        BenchResult res = benchAtomic<int>(
            "atomicMin", atomicMinKernel,
            grid, threads, numCounters, burns, iterations);
        res.metadata["op"] = "atomic_min";
        results.push_back(res);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "atomic";
        r.test_name  = "atomicMin";
        r.unit       = "ns";
        r.metadata["op"] = "atomic_min";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"grid\":" + std::to_string(grid) + ",\"threads\":" + std::to_string(threads) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

#include "bench_suites.h"

BENCH_REGISTER_SUITE(atomic, "CUDA atomic operation latency (Add/CAS/Max/Min)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runAtomicBench(0, 10);
    });
