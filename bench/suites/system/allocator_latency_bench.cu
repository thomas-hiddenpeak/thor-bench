#include "system/allocator_latency_bench.h"
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

constexpr size_t SIZES[] = {
    4ULL * 1024,           // 4KB
    64ULL * 1024,          // 64KB
    1ULL * 1024 * 1024,    // 1MB
    4ULL * 1024 * 1024,    // 4MB
    16ULL * 1024 * 1024,   // 16MB
    64ULL * 1024 * 1024,   // 64MB
};
constexpr int NUM_SIZES = 6;

// ---------------------------------------------------------------------------
// cuda_malloc_latency
// ---------------------------------------------------------------------------

std::vector<BenchResult> benchMallocLatency(int device, int iterations) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    cudaEvent_t evS, evE;
    cudaStream_t str;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    chk(cudaStreamCreate(&str), "st");

    // Warmup: allocate/free once per size
    for (int s = 0; s < NUM_SIZES; ++s) {
        void* ptr = nullptr;
        chk(cudaMalloc(&ptr, SIZES[s]), "wu");
        chk(cudaDeviceSynchronize(), "ws");
        chk(cudaFree(ptr), "wf");
    }
    chk(cudaDeviceSynchronize(), "wds");

    std::vector<double> latencies;
    latencies.reserve(static_cast<size_t>(iterations) * NUM_SIZES);

    for (int iter = 0; iter < iterations; ++iter) {
        for (int s = 0; s < NUM_SIZES; ++s) {
            void* ptr = nullptr;
            chk(cudaEventRecord(evS, str), "es");
            chk(cudaMalloc(&ptr, SIZES[s]), "malloc");
            chk(cudaEventRecord(evE, str), "ee");
            chk(cudaStreamSynchronize(str), "sy");

            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double us = ms * 1000.0; // ms -> us
            latencies.push_back(us);

            chk(cudaFree(ptr), "free");
        }
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    if (!latencies.empty()) {
        BenchResult res = computeStats(latencies, NUM_SIZES);
        res.suite_name = "allocator_latency";
        res.test_name  = "cuda_malloc_latency";
        res.unit       = "us";

        std::ostringstream p;
        p << "{\"sizes_b\":[4096,65536,1048576,4194304,16777216,67108864],"
          << "\"num_sizes\":" << NUM_SIZES
          << ",\"per_size_iters\":" << iterations << "}";
        res.params_json = p.str();
        results.push_back(res);
    }

    return results;
}

// ---------------------------------------------------------------------------
// cuda_free_latency
// ---------------------------------------------------------------------------

std::vector<BenchResult> benchFreeLatency(int device, int iterations) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    cudaEvent_t evS, evE;
    cudaStream_t str;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    chk(cudaStreamCreate(&str), "st");

    // Warmup: allocate and free once per size
    for (int s = 0; s < NUM_SIZES; ++s) {
        void* ptr = nullptr;
        chk(cudaMalloc(&ptr, SIZES[s]), "wu");
        chk(cudaDeviceSynchronize(), "ws");
        chk(cudaFree(ptr), "wf");
    }
    chk(cudaDeviceSynchronize(), "wds");

    std::vector<double> latencies;
    latencies.reserve(static_cast<size_t>(iterations) * NUM_SIZES);

    for (int iter = 0; iter < iterations; ++iter) {
        for (int s = 0; s < NUM_SIZES; ++s) {
            // Allocate outside timing window
            void* ptr = nullptr;
            chk(cudaMalloc(&ptr, SIZES[s]), "pre_alloc");
            chk(cudaDeviceSynchronize(), "pre_sync");

            // Time only the free
            chk(cudaEventRecord(evS, str), "es");
            chk(cudaFree(ptr), "free");
            chk(cudaEventRecord(evE, str), "ee");
            chk(cudaStreamSynchronize(str), "sy");

            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double us = ms * 1000.0; // ms -> us
            latencies.push_back(us);
        }
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    if (!latencies.empty()) {
        BenchResult res = computeStats(latencies, NUM_SIZES);
        res.suite_name = "allocator_latency";
        res.test_name  = "cuda_free_latency";
        res.unit       = "us";

        std::ostringstream p;
        p << "{\"sizes_b\":[4096,65536,1048576,4194304,16777216,67108864],"
          << "\"num_sizes\":" << NUM_SIZES
          << ",\"per_size_iters\":" << iterations << "}";
        res.params_json = p.str();
        results.push_back(res);
    }

    return results;
}

// ---------------------------------------------------------------------------
// cuda_malloc_concurrent
// ---------------------------------------------------------------------------

std::vector<BenchResult> benchMallocConcurrent(int device, int iterations) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    constexpr int NUM_STREAMS = 8;
    constexpr size_t BUF_SIZE = 1ULL * 1024 * 1024; // 1MB per alloc

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        chk(cudaStreamCreate(&streams[i]), "stream_create");
    }

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");

    // Warmup: one round of concurrent allocs
    void* warmupPtrs[NUM_STREAMS] = {};
    for (int i = 0; i < NUM_STREAMS; ++i) {
        chk(cudaMallocAsync(&warmupPtrs[i], BUF_SIZE, streams[i]), "wu_alloc");
    }
    chk(cudaDeviceSynchronize(), "wu_sync");
    for (int i = 0; i < NUM_STREAMS; ++i) {
        chk(cudaFreeAsync(warmupPtrs[i], streams[i]), "wu_free");
    }
    chk(cudaDeviceSynchronize(), "wu_sync2");

    std::vector<double> throughputVals;
    throughputVals.reserve(static_cast<size_t>(iterations));

    for (int iter = 0; iter < iterations; ++iter) {
        void* ptrs[NUM_STREAMS] = {};

        // Start timing
        chk(cudaEventRecord(evS, streams[0]), "es");

        // Launch concurrent allocations across all streams
        for (int i = 0; i < NUM_STREAMS; ++i) {
            chk(cudaMallocAsync(&ptrs[i], BUF_SIZE, streams[i]), "c_alloc");
        }

        // End timing (record on first stream, synchronize all)
        chk(cudaEventRecord(evE, streams[0]), "ee");
        for (int i = 0; i < NUM_STREAMS; ++i) {
            chk(cudaStreamSynchronize(streams[i]), "sy");
        }

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double allocsPerSec = sec > 0.0 ? (NUM_STREAMS / sec) : 0.0;
        throughputVals.push_back(allocsPerSec);

        // Cleanup
        for (int i = 0; i < NUM_STREAMS; ++i) {
            chk(cudaFreeAsync(ptrs[i], streams[i]), "c_free");
        }
        chk(cudaDeviceSynchronize(), "cl_sync");
    }

    // Cleanup
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    for (int i = 0; i < NUM_STREAMS; ++i) {
        chk(cudaStreamDestroy(streams[i]), "stream_destroy");
    }

    if (!throughputVals.empty()) {
        BenchResult res = computeStats(throughputVals, 1);
        res.suite_name = "allocator_latency";
        res.test_name  = "cuda_malloc_concurrent";
        res.unit       = "allocs/s";

        std::ostringstream p;
        p << "{\"num_streams\":" << NUM_STREAMS
          << ",\"buf_size_b\":" << BUF_SIZE
          << ",\"total_allocs_per_round\":" << NUM_STREAMS
          << ",\"rounds\":" << iterations << "}";
        res.params_json = p.str();
        results.push_back(res);
    }

    return results;
}

} // anonymous namespace

std::vector<BenchResult> runAllocatorLatencyBench(int device, int iterations) {
    std::vector<BenchResult> results;

    // Clamp iterations to reasonable range
    int iters = std::max(1, std::min(iterations, 100));

    try {
        auto r = benchMallocLatency(device, iters);
        results.insert(results.end(), r.begin(), r.end());
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "allocator_latency";
        r.test_name  = "cuda_malloc_latency";
        r.unit       = "us";
        r.sample_count = 0;
        r.warmup_count = 0;
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        auto r = benchFreeLatency(device, iters);
        results.insert(results.end(), r.begin(), r.end());
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "allocator_latency";
        r.test_name  = "cuda_free_latency";
        r.unit       = "us";
        r.sample_count = 0;
        r.warmup_count = 0;
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        auto r = benchMallocConcurrent(device, iters);
        results.insert(results.end(), r.begin(), r.end());
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "allocator_latency";
        r.test_name  = "cuda_malloc_concurrent";
        r.unit       = "allocs/s";
        r.sample_count = 0;
        r.warmup_count = 0;
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(allocator_latency, "cudaMalloc/cudaFree latency and throughput",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runAllocatorLatencyBench(0, 10);
    });
