#include "memory/shared_carveout_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include "bench_stats.h"
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

constexpr int kTpb = 256;
constexpr size_t kSharedBytesPerBlock = kTpb * sizeof(float);
constexpr int kRounds = 4096;

// ── Shared memory crossbar kernel (same as memory_bench.cu) ────────────────

__global__ void sharedCrossbarKernel(const float* __restrict__ data, float* __restrict__ out, size_t n, int rounds) {
    extern __shared__ float sdata[];
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    float sum = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        int idx = (tid + r * n) % n;
        sdata[t] = data[idx];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (t < stride)
                sdata[t] += sdata[t + stride];
            __syncthreads();
        }
        if (t == 0)
            sum += sdata[0];
        __syncthreads();
    }
    out[tid] = sum;
}

// ── L1-intense kernel (reads through L1, minimal shared memory) ────────────

__global__ void l1ReadKernel(const float* __restrict__ data, float* __restrict__ out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x)
        s += data[i];
    out[tid] = s;
}

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

// ── Measure shared mem bandwidth at a given carveout level ──────────────────

BenchResult measureSmemBandwidth(int device, int carveoutLevel, size_t numElems,
                                  int gridX, int iterations, int warmup) {
    BenchResult r{};
    r.suite_name = "shared_carveout";
    r.test_name  = "smem_carveout_l" + std::to_string(carveoutLevel);
    r.unit       = "GB/s";
    r.warmup_count = warmup;

    // Map level 0-9 to carveout percentage 0-100
    // Level 0 → 0% (max L1, min shared), Level 9 → 100% (min L1, max shared)
    int carveoutPct = std::min(100, (carveoutLevel * 100) / 9);

    float* dSrc = nullptr;
    float* dOut = nullptr;
    size_t allocBytes = numElems * sizeof(float);

    chk(cudaSetDevice(device), "dev");
    chk(cudaMalloc(&dSrc, allocBytes), "malloc_src");
    chk(cudaMalloc(&dOut, allocBytes), "malloc_out");
    chk(cudaMemset(dSrc, 0xAA, allocBytes), "memset");

    // Set carveout on this kernel
    chk(cudaFuncSetAttribute(sharedCrossbarKernel,
                              cudaFuncAttributePreferredSharedMemoryCarveout,
                              carveoutPct), "carveout");

    cudaStream_t str;
    cudaEvent_t evS, evE;
    chk(cudaStreamCreate(&str), "stream");
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        sharedCrossbarKernel<<<gridX, kTpb, kSharedBytesPerBlock, str>>>(dSrc, dOut, numElems, kRounds);
    }
    chk(cudaStreamSynchronize(str), "warmup");

    // Timing
    std::vector<double> vals;
    double bytesPerRound = static_cast<double>(gridX) * kSharedBytesPerBlock * 3.0;
    double totalBytes = bytesPerRound * kRounds;

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "evS");
        sharedCrossbarKernel<<<gridX, kTpb, kSharedBytesPerBlock, str>>>(dSrc, dOut, numElems, kRounds);
        chk(cudaEventRecord(evE, str), "evE");
        chk(cudaStreamSynchronize(str), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "elapsed");
        double sec = ms / 1000.0;
        double gbs = sec > 0.0 ? (totalBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gbs);
    }

    // Compute stats
    BenchResult st = ::deusridet::bench::computeStats(vals, warmup);
    r.sample_count = st.sample_count;
    r.warmup_count = st.warmup_count;
    r.median = st.median;
    r.mean   = st.mean;
    r.stddev = st.stddev;
    r.p95    = st.p95;
    r.p99    = st.p99;
    r.min_val = st.min_val;
    r.max_val = st.max_val;
    r.peak_pct = 0.0; // no T5000 peak reference for shared mem carveout

    // Compute approximate SMEM/L1 split
    int smemKb = (carveoutPct * 228) / 100;
    int l1Kb = 256 - smemKb;

    std::ostringstream p;
    p << "{\"carveout_pct\":" << carveoutPct
      << ",\"smem_kb\":" << smemKb
      << ",\"l1_kb\":" << l1Kb
      << ",\"total_l1_smem_kb\":256"
      << ",\"tpb\":" << kTpb
      << ",\"rounds\":" << kRounds
      << ",\"grid\":" << gridX << "}";
    r.params_json = p.str();
    r.metadata["carveout_pct"] = std::to_string(carveoutPct);
    r.metadata["smem_kb"] = std::to_string(smemKb);
    r.metadata["l1_kb"] = std::to_string(l1Kb);

    // Cleanup
    chk(cudaEventDestroy(evS), "evD");
    chk(cudaEventDestroy(evE), "evD");
    chk(cudaStreamDestroy(str), "stream");
    chk(cudaFree(dSrc), "free_src");
    chk(cudaFree(dOut), "free_out");

    return r;
}

// ── Measure L1 bandwidth under different cache configs ─────────────────────

BenchResult measureL1Bandwidth(int device, cudaFuncCache cacheConfig,
                                size_t numElems, int gridX,
                                int iterations, int warmup, const char* label) {
    BenchResult r{};
    r.suite_name = "shared_carveout";
    r.test_name  = label;
    r.unit       = "GB/s";
    r.warmup_count = warmup;

    float* dSrc = nullptr;
    float* dOut = nullptr;
    size_t allocBytes = numElems * sizeof(float);

    chk(cudaSetDevice(device), "dev");
    chk(cudaDeviceSetCacheConfig(cacheConfig), "cache_config");
    chk(cudaMalloc(&dSrc, allocBytes), "malloc_src");
    chk(cudaMalloc(&dOut, allocBytes), "malloc_out");
    chk(cudaMemset(dSrc, 0xBB, allocBytes), "memset");

    cudaStream_t str;
    cudaEvent_t evS, evE;
    chk(cudaStreamCreate(&str), "stream");
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        l1ReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dOut, numElems);
    }
    chk(cudaStreamSynchronize(str), "warmup");

    // Timing
    std::vector<double> vals;
    // Each kernel reads numElems * sizeof(float) bytes
    double totalBytes = static_cast<double>(numElems) * sizeof(float);

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "evS");
        l1ReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dOut, numElems);
        chk(cudaEventRecord(evE, str), "evE");
        chk(cudaStreamSynchronize(str), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "elapsed");
        double sec = ms / 1000.0;
        double gbs = sec > 0.0 ? (totalBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gbs);
    }

    BenchResult st = ::deusridet::bench::computeStats(vals, warmup);
    r.sample_count = st.sample_count;
    r.median = st.median;
    r.mean   = st.mean;
    r.stddev = st.stddev;
    r.p95    = st.p95;
    r.p99    = st.p99;
    r.min_val = st.min_val;
    r.max_val = st.max_val;
    r.peak_pct = computePeakPctSame(r.median, T5000Peaks::memory_bandwidth_gbs);

    const char* configName = (cacheConfig == cudaFuncCachePreferShared)
                              ? "prefer_shared" :
                              (cacheConfig == cudaFuncCachePreferL1)
                              ? "prefer_l1" :
                              (cacheConfig == cudaFuncCachePreferEqual)
                              ? "prefer_equal" : "prefer_none";

    std::ostringstream p;
    p << "{\"cache_config\":\"" << configName
      << "\",\"bytes\":" << allocBytes
      << ",\"grid\":" << gridX << ",\"tpb\":" << kTpb << "}";
    r.params_json = p.str();
    r.metadata["cache_config"] = configName;

    // Cleanup
    chk(cudaEventDestroy(evS), "evD");
    chk(cudaEventDestroy(evE), "evD");
    chk(cudaStreamDestroy(str), "stream");
    chk(cudaFree(dSrc), "free_src");
    chk(cudaFree(dOut), "free_out");

    // Reset cache config to default
    chk(cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual), "reset_cache");

    return r;
}

} // anonymous namespace

std::vector<BenchResult> runSharedCarveoutBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;
    constexpr int kWarmup = 3;

    size_t numElems = static_cast<size_t>(matDim) * matDim;
    if (numElems < 65536) numElems = 65536;
    int gridX = std::max(1, std::min(65535, static_cast<int>((numElems + kTpb - 1) / kTpb)));

    // ── 10-level carveout sweep ───────────────────────────────────────────
    for (int level = 0; level < 10; ++level) {
        try {
            results.push_back(measureSmemBandwidth(device, level, numElems, gridX, iterations, kWarmup));
        } catch (const std::exception& ex) {
            BenchResult r{};
            r.suite_name = "shared_carveout";
            r.test_name  = "smem_carveout_l" + std::to_string(level);
            r.unit       = "GB/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"carveout_level\":";
            err += std::to_string(level);
            err += "}";
            r.params_json = err;
            r.metadata["stub"] = "true";
            r.metadata["stub_reason"] = (std::string("carveout failure: ") + ex.what()).c_str();
            results.push_back(r);
        }
    }

    // ── L1 bandwidth at different cache configs ────────────────────────────

    // Prefer shared memory config
    try {
        results.push_back(measureL1Bandwidth(device, cudaFuncCachePreferShared,
                                              numElems, gridX, iterations, kWarmup,
                                              "l1_bw_prefer_shared"));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "shared_carveout";
        r.test_name  = "l1_bw_prefer_shared";
        r.unit       = "GB/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        r.metadata["stub"] = "true";
        results.push_back(r);
    }

    // Prefer cache config
    try {
        results.push_back(measureL1Bandwidth(device, cudaFuncCachePreferL1,
                                              numElems, gridX, iterations, kWarmup,
                                              "l1_bw_prefer_l1"));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "shared_carveout";
        r.test_name  = "l1_bw_prefer_cache";
        r.unit       = "GB/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        r.metadata["stub"] = "true";
        results.push_back(r);
    }

    // L1 bandwidth at max SMEM (carveout=100, i.e., 228KB shared, 28KB L1)
    try {
        results.push_back(measureSmemBandwidth(device, 9, numElems, gridX, iterations, kWarmup));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "shared_carveout";
        r.test_name  = "l1_bw_max_smem";
        r.unit       = "GB/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        r.metadata["stub"] = "true";
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(shared_carveout, "Shared memory vs L1 carveout sensitivity sweep",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runSharedCarveoutBench(0, 512, 10);
    });
