#include "sync/warp_primitives.h"
#include "bench_schema.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

// __reduce_add_sync is SM80+ but not always exposed in CUDA 13.0 headers for device code.
// We use __ballot_sync as the fourth warp primitive test instead.
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
// Warp primitive kernels — one warp per block (32 threads).
// Each kernel runs a tight loop of `burns` iterations.
// Dummy computation before/after prevents nvcc from optimizing the primitive away.
// Host-side cudaEvent_t timing (clock64 is unreliable on SM110a).
// ---------------------------------------------------------------------------

__global__ void shflSyncKernel(float* data, int gridSize, int burns) {
    int tid = threadIdx.x;
    float val = static_cast<float>(tid);
    for (int i = 0; i < burns; ++i) {
        // dummy computation before primitive
        val += static_cast<float>(i);
        // __shfl_sync: broadcast from lane 0
        float result = __shfl_sync(0xFFFFFFFF, val, 0);
        // dummy computation after primitive
        val += result;
    }
    int idx = tid + blockIdx.x * blockDim.x;
    if (idx < gridSize) data[idx] = val;
}

__global__ void voteSyncKernel(float* data, int gridSize, int burns) {
    int tid = threadIdx.x;
    float val = static_cast<float>(tid);
    for (int i = 0; i < burns; ++i) {
        // dummy computation before primitive
        val += static_cast<float>(i);
        // __vote_sync: __all_sync + __any_sync consensus
        int allResult = __all_sync(0xFFFFFFFF, val > 0.0f);
        int anyResult = __any_sync(0xFFFFFFFF, val == 0.0f);
        // dummy computation after primitive
        val += static_cast<float>(allResult + anyResult);
    }
    int idx = tid + blockIdx.x * blockDim.x;
    if (idx < gridSize) data[idx] = val;
}

__global__ void matchAnySyncKernel(float* data, int gridSize, int burns) {
    int tid = threadIdx.x;
    float val = static_cast<float>(tid);
    for (int i = 0; i < burns; ++i) {
        // dummy computation before primitive
        val += static_cast<float>(i);
        // __match_any_sync: find any thread matching val
        unsigned int mask = __match_any_sync(0xFFFFFFFF, val);
        // dummy computation after primitive
        val += static_cast<float>(mask & 0xFF);
    }
    int idx = tid + blockIdx.x * blockDim.x;
    if (idx < gridSize) data[idx] = val;
}

__global__ void ballotSyncKernel(float* data, int gridSize, int burns) {
    int tid = threadIdx.x;
    float val = static_cast<float>(tid);
    for (int i = 0; i < burns; ++i) {
        // dummy computation before primitive
        val += static_cast<float>(i);
        // __ballot_sync: warp-wide predicate ballot
        unsigned int mask = __ballot_sync(0xFFFFFFFF, val > static_cast<float>(tid) * 2.0f);
        // dummy computation after primitive
        val += static_cast<float>(mask & 0xFF);
    }
    int idx = tid + blockIdx.x * blockDim.x;
    if (idx < gridSize) data[idx] = val;
}

// ---------------------------------------------------------------------------
// Benchmark helper — generic for any warp primitive kernel
// ---------------------------------------------------------------------------

template<typename KernelFn>
BenchResult benchWarpPrimitive(const char* testName, KernelFn kernel, int grid,
                                int threads, int burns, int iterations) {
    int gridSize = grid * threads;
    size_t allocBytes = gridSize * sizeof(float);

    // Guard: limit allocation to 256 MB
    if (allocBytes > 256ULL * 1024 * 1024) {
        BenchResult r;
        r.suite_name = "warp_primitives";
        r.test_name  = testName;
        r.unit       = "ns";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.params_json = "{\"error\":\"allocation exceeds 256 MB limit\",\"grid\":" +
            std::to_string(grid) + ",\"threads\":" + std::to_string(threads) + "}";
        return r;
    }

    float* dData = nullptr;
    chk(cudaMalloc(&dData, allocBytes), "md");
    chk(cudaMemset(dData, 0, allocBytes), "ms");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");

    // Warmup
    for (int w = 0; w < 3; ++w)
        kernel<<<grid, threads>>>(dData, gridSize, burns);
    chk(cudaGetLastError(), "warmup");
    chk(cudaDeviceSynchronize(), "ws");

    std::vector<double> vals;
    vals.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        chk(cudaMemset(dData, 0, allocBytes), "ms");
        chk(cudaEventRecord(evS), "rs");
        kernel<<<grid, threads>>>(dData, gridSize, burns);
        chk(cudaEventRecord(evE), "re");
        chk(cudaDeviceSynchronize(), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double ns = ms * 1000.0;
        // Per-invocation latency: total time / (num_blocks * burns)
        double latency = ns / (static_cast<double>(grid) * burns);
        vals.push_back(latency);
    }

    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    chk(cudaFree(dData), "fd");

    if (vals.empty()) {
        BenchResult r;
        r.suite_name = "warp_primitives";
        r.test_name  = testName;
        r.unit       = "ns";
        r.sample_count = 0;
        r.warmup_count = 0;
        return r;
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "warp_primitives";
    res.test_name  = testName;
    res.unit       = "ns";

    std::ostringstream p;
    p << "{\"grid\":" << grid
      << ",\"threads\":" << threads
      << ",\"burns\":" << burns
      << ",\"total_invocations\":" << (grid * burns) << "}";
    res.params_json = p.str();
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runWarpPrimitivesBench(int device, int iterations) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    // Verify device attributes
    int warpSize = 0;
    chk(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device), "warp");

    if (warpSize != 32) {
        BenchResult r;
        r.suite_name = "warp_primitives";
        r.test_name  = "__shfl_sync_broadcast";
        r.unit       = "ns";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.params_json = "{\"error\":\"unsupported warp size\",\"warp_size\":" +
            std::to_string(warpSize) + "}";
        results.push_back(r);
        return results;
    }

    constexpr int threads = 32;  // one warp per block
    constexpr int burns   = 128;

    // Single warp per benchmark (one block × 32 threads) to measure per-warp primitive latency.
    // High burn count to accumulate measurable wall-clock time.
    int maxGrid = 1;

    // -----------------------------------------------------------------------
    // 1. __shfl_sync — broadcast
    // -----------------------------------------------------------------------
    try {
        results.push_back(benchWarpPrimitive(
            "__shfl_sync_broadcast", shflSyncKernel,
            maxGrid, threads, burns, iterations));
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "warp_primitives";
        r.test_name  = "__shfl_sync_broadcast";
        r.unit       = "ns";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"grid\":" + std::to_string(maxGrid) + ",\"threads\":" + std::to_string(threads) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // 2. __vote_sync — all/any consensus
    // -----------------------------------------------------------------------
    try {
        results.push_back(benchWarpPrimitive(
            "__vote_sync_all_any", voteSyncKernel,
            maxGrid, threads, burns, iterations));
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "warp_primitives";
        r.test_name  = "__vote_sync_all_any";
        r.unit       = "ns";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"grid\":" + std::to_string(maxGrid) + ",\"threads\":" + std::to_string(threads) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // 3. __match_any_sync — any matching value
    // -----------------------------------------------------------------------
    try {
        results.push_back(benchWarpPrimitive(
            "__match_any_sync", matchAnySyncKernel,
            maxGrid, threads, burns, iterations));
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "warp_primitives";
        r.test_name  = "__match_any_sync";
        r.unit       = "ns";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"grid\":" + std::to_string(maxGrid) + ",\"threads\":" + std::to_string(threads) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    // -----------------------------------------------------------------------
    // 4. __ballot_sync — warp-wide predicate ballot
    // -----------------------------------------------------------------------
    try {
        results.push_back(benchWarpPrimitive(
            "__ballot_sync", ballotSyncKernel,
            maxGrid, threads, burns, iterations));
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "warp_primitives";
        r.test_name  = "__ballot_sync";
        r.unit       = "ns";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"grid\":" + std::to_string(maxGrid) + ",\"threads\":" + std::to_string(threads) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

#include "bench_suites.h"

BENCH_REGISTER_SUITE(warp_primitives, "Warp-level primitives latency (__shfl_sync, __vote_sync, __match_any_sync, __add_sync)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runWarpPrimitivesBench(0, 10);
    });
