#include "sync/cluster_sync.h"
#include "bench_schema.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cg = cooperative_groups;

namespace deusridet::bench {

namespace {

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------
// __syncthreads latency kernel — baseline block sync
// Uses a dummy computation to prevent __syncthreads from being optimized away,
// and relies on host-side cudaEvent_t timing (clock64() is unreliable in
// syncthreads loops on SM110a).
// ---------------------------------------------------------------------------

__global__ void syncthreadsKernel(float* data, int n, int burns) {
    int tid = threadIdx.x;
    for (int i = 0; i < burns; ++i) {
        // Do work before sync to prevent optimization
        int idx = (tid + i) % n;
        data[idx] += static_cast<float>(i);
        __syncthreads();
    }
}

BenchResult benchSyncthreads(int threads, int burns, int iterations, double clockGHz) {
    (void)clockGHz; // host-side timing, not device clock64

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
    chk(cudaGetLastError(), "st");
    chk(cudaDeviceSynchronize(), "ws");

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        // Reset data before each iteration
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
    chk(cudaFree(dData), "fc");

    if (vals.empty()) {
        BenchResult r;
        r.suite_name = "cluster_sync";
        r.test_name = "syncthreads_" + std::to_string(threads) + "threads";
        r.unit = "ns";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.metadata["sync_primitive"] = "syncthreads";
        r.metadata["thread_count"] = std::to_string(threads);
        r.metadata["stub_reason"] = "no samples collected";
        return r;
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "cluster_sync";
    res.test_name  = "syncthreads_" + std::to_string(threads) + "threads";
    res.unit       = "ns";
    res.metadata["sync_primitive"] = "syncthreads";
    res.metadata["thread_count"] = std::to_string(threads);

    std::ostringstream p;
    p << "{\"threads\":" << threads
      << ",\"burns\":" << burns << "}";
    res.params_json = p.str();
    return res;
}

// ---------------------------------------------------------------------------
// Cluster barrier kernel — multi-block cluster sync via cooperative groups
// __cluster_dims__ specifies 2 blocks per cluster. The grid MUST launch exactly
// 2 blocks (matching the annotation). Cluster dim is 1x1x1 (1 cluster).
// ---------------------------------------------------------------------------

__global__ void __cluster_dims__(2, 1, 1) clusterBarrierKernel(
    float* data, int n, int burns) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    for (int i = 0; i < burns; ++i) {
        int idx = (tid + bid * 256 + i) % n;
        data[idx] += static_cast<float>(i);
        cg::cluster_group::sync();
    }
}

BenchResult benchClusterBarrier(int burns, int iterations) {
    int n = 256 * 2; // enough for 2 blocks
    float* dData = nullptr;
    chk(cudaMalloc(&dData, n * sizeof(float)), "md");
    chk(cudaMemset(dData, 0, n * sizeof(float)), "ms");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");

    // __cluster_dims__(2,1,1) requires grid to be exactly 2 blocks.
    // DO NOT use cudaLaunchKernelEx with cluster attributes here — the
    // __cluster_dims__ annotation already defines the cluster layout.
    // Using both causes "cluster misconfiguration" error.
    dim3 grid(2, 1, 1);
    dim3 block(256, 1, 1);

    // Warmup
    for (int w = 0; w < 3; ++w)
        clusterBarrierKernel<<<grid, block>>>(dData, n, burns);
    chk(cudaGetLastError(), "cb");
    chk(cudaDeviceSynchronize(), "ws");

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaMemset(dData, 0, n * sizeof(float)), "ms");
        chk(cudaEventRecord(evS), "rs");
        clusterBarrierKernel<<<grid, block>>>(dData, n, burns);
        chk(cudaEventRecord(evE), "re");
        chk(cudaDeviceSynchronize(), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double ns = ms * 1000.0;
        double latency = ns / burns;
        vals.push_back(latency);
    }

    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    chk(cudaFree(dData), "fc");

    if (vals.empty()) {
        BenchResult r;
        r.suite_name = "cluster_sync";
        r.test_name  = "cluster_barrier_2SM";
        r.unit       = "ns";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.metadata["sync_primitive"] = "cluster_barrier";
        r.metadata["cluster_mode"] = "true";
        r.metadata["stub_reason"] = "no samples collected";
        return r;
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "cluster_sync";
    res.test_name  = "cluster_barrier_2SM";
    res.unit       = "ns";
    res.metadata["sync_primitive"] = "cluster_barrier";
    res.metadata["cluster_mode"] = "true";

    std::ostringstream p;
    p << "{\"burns\":" << burns
      << ",\"blocks\":2"
      << ",\"threadsPerBlock\":256"
      << ",\"clusterDim\":\"1x1x1\"}";
    res.params_json = p.str();
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runClusterSyncBench(int device, int iterations) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    int mhz;
    chk(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, device), "attr");
    double clockGHz = static_cast<double>(mhz) / 1e3;

    int burns = 128;

    // __syncthreads baseline with varying thread counts
    constexpr int threadCounts[] = {64, 128, 256, 512, 1024};
    for (int threads : threadCounts) {
        try {
            results.push_back(benchSyncthreads(threads, burns, iterations, clockGHz));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "cluster_sync";
            r.test_name  = "syncthreads_" + std::to_string(threads) + "threads";
            r.unit       = "ns";
            r.metadata["sync_primitive"] = "syncthreads";
            r.metadata["thread_count"] = std::to_string(threads);
            r.metadata["stub_reason"] = std::string("error: ") + ex.what();
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"threads\":" + std::to_string(threads) + "}";
            r.params_json = err;
            results.push_back(r);
        }
    }

    // Cluster barrier via cooperative_groups::cluster_group::sync()
    {
        try {
            results.push_back(benchClusterBarrier(burns, iterations));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "cluster_sync";
            r.test_name  = "cluster_barrier_2SM";
            r.unit       = "ns";
            r.metadata["sync_primitive"] = "cluster_barrier";
            r.metadata["cluster_mode"] = "true";
            r.metadata["stub_reason"] = std::string("error: ") + ex.what();
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"blocks\":2}";
            r.params_json = err;
            results.push_back(r);
        }
    }

    return results;
}

} // namespace deusridet::bench

#include "bench_suites.h"

BENCH_REGISTER_SUITE(cluster_sync, "Cluster sync latency (__syncthreads baseline; cluster_barrier via cg::cluster_group::sync())",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runClusterSyncBench(0, 10);
    });
