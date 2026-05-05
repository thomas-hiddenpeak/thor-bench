#include "system/multi_stream_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
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

// ── Kernels (same pattern as memory_bench.cu) ──────────────────────────────

__global__ void streamCopyKernel(const float* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x)
        dst[i] = src[i];
}

__global__ void streamReadKernel(const float* __restrict__ data, float* __restrict__ out, size_t n) {
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

// ── Stats helpers (same pattern as memory_bench.cu) ────────────────────────

struct Stats {
    double median = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
};

Stats computeStats(std::vector<double>& vals) {
    Stats s;
    int n = static_cast<int>(vals.size());
    if (n == 0) return s;

    std::sort(vals.begin(), vals.end());

    double sum = 0;
    for (double v : vals) sum += v;
    double mean = sum / n;

    s.min_val  = vals.front();
    s.max_val  = vals.back();
    s.mean     = mean;
    s.median   = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;

    double sq = 0;
    for (double v : vals) { double d = v - mean; sq += d * d; }
    s.stddev = std::sqrt(sq / n);

    auto pct = [&](double p) -> double {
        if (n <= 1) return vals[0];
        double r = p * (n - 1);
        int lo = static_cast<int>(std::floor(r));
        int hi = static_cast<int>(std::ceil(r));
        if (hi >= n) return vals.back();
        return vals[lo] * (1.0 - (r - lo)) + vals[hi] * (r - lo);
    };
    s.p95 = pct(0.95);
    s.p99 = pct(0.99);

    return s;
}

// ── Run concurrent streams, return total GB/s per iteration ────────────────

std::vector<double> runMultiStream(int device, int numStreams, size_t numElemsPerStream,
                                    int gridX, int iterations, int warmup) {
    chk(cudaSetDevice(device), "dev");

    // Allocate buffers per stream
    std::vector<float*> dSrc(numStreams), dDst(numStreams);
    std::vector<cudaStream_t> streams(numStreams);
    size_t allocBytes = numElemsPerStream * sizeof(float);

    for (int i = 0; i < numStreams; ++i) {
        chk(cudaMalloc(&dSrc[i], allocBytes), "malloc_src");
        chk(cudaMalloc(&dDst[i], allocBytes), "malloc_dst");
        chk(cudaMemset(dSrc[i], 0xBB, allocBytes), "memset");
        chk(cudaStreamCreate(&streams[i]), "stream_create");
    }

    // Create events for timing the whole batch
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "evS");
    chk(cudaEventCreate(&evE), "evE");

    // Warmup
    for (int w = 0; w < warmup; ++w) {
        for (int i = 0; i < numStreams; ++i) {
            streamCopyKernel<<<gridX, kTpb, 0, streams[i]>>>(dSrc[i], dDst[i], numElemsPerStream);
            streamReadKernel<<<gridX, kTpb, 0, streams[i]>>>(dSrc[i], dDst[i], numElemsPerStream);
        }
    }
    for (int i = 0; i < numStreams; ++i)
        chk(cudaStreamSynchronize(streams[i]), "warmup_sync");

    // Timing
    std::vector<double> totals; // total GB/s per iteration
    // Total bytes per iteration = numStreams * (read_bytes + copy_bytes) = numStreams * allocBytes * 2
    size_t totalBytesPerIter = static_cast<size_t>(numStreams) * allocBytes * 2;

    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, streams[0]), "evS_record");

        // Launch kernels on all streams concurrently
        for (int s = 0; s < numStreams; ++s) {
            streamCopyKernel<<<gridX, kTpb, 0, streams[s]>>>(dSrc[s], dDst[s], numElemsPerStream);
            streamReadKernel<<<gridX, kTpb, 0, streams[s]>>>(dSrc[s], dDst[s], numElemsPerStream);
        }

        // Record end event on the last stream
        chk(cudaEventRecord(evE, streams[numStreams - 1]), "evE_record");

        // Wait for all streams
        for (int s = 0; s < numStreams; ++s)
            chk(cudaStreamSynchronize(streams[s]), "sync");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "elapsed");
        double sec = ms / 1000.0;
        double gbs = sec > 0.0 ? (static_cast<double>(totalBytesPerIter) / 1073741824.0) / sec : 0.0;
        totals.push_back(gbs);
    }

    // Cleanup
    chk(cudaEventDestroy(evS), "evD");
    chk(cudaEventDestroy(evE), "evD");
    for (int i = 0; i < numStreams; ++i) {
        chk(cudaStreamDestroy(streams[i]), "stream_destroy");
        chk(cudaFree(dSrc[i]), "free_src");
        chk(cudaFree(dDst[i]), "free_dst");
    }

    return totals;
}

} // anonymous namespace

std::vector<BenchResult> runMultiStreamBench(int device, int bufferSize, int iterations) {
    std::vector<BenchResult> results;

    constexpr int kWarmup = 3;
    size_t numElems = static_cast<size_t>(bufferSize) / sizeof(float);

    // Ensure numElems is reasonable
    if (numElems < 65536) numElems = 65536;

    int gridX = std::max(1, std::min(65535, static_cast<int>((numElems + kTpb - 1) / kTpb)));

    int streamCounts[] = {1, 2, 4, 8};

    // First, measure single-stream baseline
    std::vector<double> singleStreamVals = runMultiStream(device, 1, numElems, gridX, iterations, kWarmup);
    Stats singleStats = computeStats(singleStreamVals);
    double singleBaseline = singleStats.median;

    // Single-stream baseline result
    {
        BenchResult r{};
        r.suite_name = "multi_stream";
        r.test_name  = "single_stream_baseline";
        r.unit       = "GB/s";
        r.sample_count = static_cast<int>(singleStreamVals.size());
        r.warmup_count = kWarmup;
        r.median = singleStats.median;
        r.mean   = singleStats.mean;
        r.stddev = singleStats.stddev;
        r.p95    = singleStats.p95;
        r.p99    = singleStats.p99;
        r.min_val = singleStats.min_val;
        r.max_val = singleStats.max_val;
        r.peak_pct = computePeakPctSame(r.median, T5000Peaks::memory_bandwidth_gbs);

        std::ostringstream p;
        p << "{\"streams\":1,\"bytes\":" << bufferSize
          << ",\"grid\":" << gridX << ",\"tpb\":" << kTpb
          << ",\"kernels_per_stream\":2}";
        r.params_json = p.str();
        results.push_back(r);
    }

    // Multi-stream configs
    for (int streamIdx = 1; streamIdx < 4; ++streamIdx) {
        int numStreams = streamCounts[streamIdx];

        std::vector<double> vals = runMultiStream(device, numStreams, numElems, gridX, iterations, kWarmup);
        Stats st = computeStats(vals);
        double totalGbs = st.median;
        double perStreamGbs = totalGbs / numStreams;
        double speedup = singleBaseline > 0.0 ? totalGbs / singleBaseline : 0.0;

        BenchResult r{};
        r.suite_name = "multi_stream";
        r.test_name  = "multi_stream_" + std::to_string(numStreams);
        r.unit       = "GB/s";
        r.sample_count = static_cast<int>(vals.size());
        r.warmup_count = kWarmup;
        r.median = totalGbs;
        r.mean   = st.mean;
        r.stddev = st.stddev;
        r.p95    = st.p95;
        r.p99    = st.p99;
        r.min_val = st.min_val;
        r.max_val = st.max_val;
        r.peak_pct = computePeakPctSame(r.median, T5000Peaks::memory_bandwidth_gbs);

        std::ostringstream p;
        p << "{\"streams\":" << numStreams
          << ",\"bytes_per_stream\":" << bufferSize
          << ",\"total_bytes\":" << (static_cast<size_t>(numStreams) * bufferSize * 2)
          << ",\"grid\":" << gridX << ",\"tpb\":" << kTpb
          << ",\"kernels_per_stream\":2"
          << ",\"per_stream_gbs\":" << perStreamGbs
          << ",\"speedup_vs_single\":" << speedup << "}";
        r.params_json = p.str();
        r.metadata["num_streams"] = std::to_string(numStreams);
        r.metadata["per_stream_gbs"] = std::to_string(perStreamGbs);
        r.metadata["speedup_vs_single"] = std::to_string(speedup);
        results.push_back(r);
    }

    // ── Stream independence test ─────────────────────────────────────────
    // Measure per-stream throughput variance to check independence
    {
        int numStreams = 4;
        // Run each stream independently to get per-stream measurement
        std::vector<double> perStreamThroughputs;

        std::vector<float*> dSrc(numStreams), dDst(numStreams);
        std::vector<cudaStream_t> streams(numStreams);
        size_t allocBytes = numElems * sizeof(float);

        for (int i = 0; i < numStreams; ++i) {
            chk(cudaMalloc(&dSrc[i], allocBytes), "malloc_src");
            chk(cudaMalloc(&dDst[i], allocBytes), "malloc_dst");
            chk(cudaMemset(dSrc[i], 0xCC, allocBytes), "memset");
            chk(cudaStreamCreate(&streams[i]), "stream_create");
        }

        // Warmup
        for (int w = 0; w < kWarmup; ++w) {
            for (int i = 0; i < numStreams; ++i) {
                streamCopyKernel<<<gridX, kTpb, 0, streams[i]>>>(dSrc[i], dDst[i], numElems);
            }
        }
        for (int i = 0; i < numStreams; ++i)
            chk(cudaStreamSynchronize(streams[i]), "warmup_sync");

        // Per-stream timing: launch all concurrently, time each independently
        std::vector<std::vector<double>> perStreamMs(numStreams);
        cudaEvent_t evS[4], evE[4];
        for (int i = 0; i < numStreams; ++i) {
            chk(cudaEventCreate(&evS[i]), "evS");
            chk(cudaEventCreate(&evE[i]), "evE");
        }

        for (int iter = 0; iter < iterations; ++iter) {
            for (int s = 0; s < numStreams; ++s) {
                chk(cudaEventRecord(evS[s], streams[s]), "evS");
                streamCopyKernel<<<gridX, kTpb, 0, streams[s]>>>(dSrc[s], dDst[s], numElems);
                streamReadKernel<<<gridX, kTpb, 0, streams[s]>>>(dSrc[s], dDst[s], numElems);
                chk(cudaEventRecord(evE[s], streams[s]), "evE");
            }
            for (int s = 0; s < numStreams; ++s)
                chk(cudaStreamSynchronize(streams[s]), "sync");

            for (int s = 0; s < numStreams; ++s) {
                float ms = 0;
                chk(cudaEventElapsedTime(&ms, evS[s], evE[s]), "elapsed");
                perStreamMs[s].push_back(ms);
            }
        }

        // Compute per-stream GB/s and report the coefficient of variation
        for (int s = 0; s < numStreams; ++s) {
            double sumMs = 0;
            for (double m : perStreamMs[s]) sumMs += m;
            double meanMs = sumMs / perStreamMs[s].size();
            // 2 kernels × allocBytes each = 2 * allocBytes total bytes
            double bytesPerStream = static_cast<double>(allocBytes) * 2.0;
            double gbs = meanMs > 0.0 ? (bytesPerStream / 1073741824.0) / (meanMs / 1000.0) : 0.0;
            perStreamThroughputs.push_back(gbs);
        }

        // Compute CV (coefficient of variation) of per-stream throughput
        double meanThrpt = 0;
        for (double t : perStreamThroughputs) meanThrpt += t;
        meanThrpt /= numStreams;
        double varSum = 0;
        for (double t : perStreamThroughputs) {
            double d = t - meanThrpt;
            varSum += d * d;
        }
        double stddev = std::sqrt(varSum / numStreams);
        double cv = meanThrpt > 0.0 ? stddev / meanThrpt : 0.0;

        BenchResult r{};
        r.suite_name = "multi_stream";
        r.test_name  = "multi_stream_independence";
        r.unit       = "GB/s";
        r.sample_count = iterations;
        r.warmup_count = kWarmup;
        r.median = meanThrpt;
        r.mean   = meanThrpt;
        r.stddev = stddev;
        r.min_val = *std::min_element(perStreamThroughputs.begin(), perStreamThroughputs.end());
        r.max_val = *std::max_element(perStreamThroughputs.begin(), perStreamThroughputs.end());
        r.peak_pct = 0.0; // informational test

        std::ostringstream p;
        p << "{\"streams\":" << numStreams
          << ",\"cv\":" << cv
          << ",\"mean_per_stream\":" << meanThrpt
          << ",\"note\":\"CV < 0.05 indicates good stream independence\"}";
        r.params_json = p.str();
        r.metadata["cv"] = std::to_string(cv);
        r.metadata["interpretation"] = cv < 0.05 ? "streams_well_balanced" : "streams_imbalanced";
        results.push_back(r);

        // Cleanup
        for (int i = 0; i < numStreams; ++i) {
            chk(cudaEventDestroy(evS[i]), "evD");
            chk(cudaEventDestroy(evE[i]), "evD");
            chk(cudaStreamDestroy(streams[i]), "stream_destroy");
            chk(cudaFree(dSrc[i]), "free_src");
            chk(cudaFree(dDst[i]), "free_dst");
        }
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(multi_stream, "Multi-stream memory concurrency and overlap",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runMultiStreamBench(0, 1048576, 10);
    });
