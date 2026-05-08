#include "compute/memory_bench.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include "sweep_schema.h"
#include "power_monitor.h"
#include <chrono>
#include <ctime>
#include <iomanip>

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

__global__ void sharedMemCrossbarKernel(const float* data, float* out, size_t n, int rounds) {
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

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

void launchKernel(const char* label, float* dSrc, float* dDst,
                  size_t numElems, int gridX, cudaStream_t str) {
    if (label[8] == 'r')
        memReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
    else if (label[8] == 'w')
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

    res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);

    std::ostringstream p;
    p << "{\"bytes\":" << allocBytes
      << ",\"mult\":" << byteMult
      << ",\"grid\":" << gridX
      << ",\"tpb\":" << kTpb << "}";
    res.params_json = p.str();
    if (label[8] == 'r') res.metadata["access_pattern"] = "read";
    else if (label[8] == 'w') res.metadata["access_pattern"] = "write";
    else res.metadata["access_pattern"] = "copy";
    res.metadata["memory_type"] = "device";
    return res;
}

BenchResult measureSharedMem(size_t numElems, int gridX, int iters,
                             cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str) {
    BenchResult res;
    res.suite_name = "memory";
    res.test_name  = "shared_mem_crossbar";
    res.unit       = "GB/s";
    res.warmup_count = 3;

    constexpr int kSharedTpb = 256;
    constexpr size_t kSharedBytesPerBlock = kSharedTpb * sizeof(float);
    constexpr int kRounds = 4096;

    float* dSrc = nullptr;
    float* dOut = nullptr;
    size_t allocBytes = numElems * sizeof(float);
    chk(cudaMalloc(&dSrc, allocBytes), "msm");
    chk(cudaMalloc(&dOut, allocBytes), "mout");
    chk(cudaMemset(dSrc, 0xAA, allocBytes), "msm");

    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        sharedMemCrossbarKernel<<<gridX, kSharedTpb, kSharedBytesPerBlock, str>>>(dSrc, dOut, numElems, kRounds);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        // Each round: load + reduce per block. Total reads per round = gridX * kSharedTpb * sizeof(float) * 2 (read + write)
        // Plus reduction traffic. Approximate: 3 * kSharedBytesPerBlock per block per round.
        double bytesPerRound = static_cast<double>(gridX) * kSharedBytesPerBlock * 3.0;
        double totalBytes = bytesPerRound * kRounds;
        double gb = sec > 0.0 ? (totalBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }

    chk(cudaFree(dSrc), "fsm");
    chk(cudaFree(dOut), "fout");

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

    res.peak_pct = 0.0; // shared mem crossbar has no T5000 peak reference

    std::ostringstream p;
    p << "{\"tpb\":" << kSharedTpb
      << ",\"shared_bytes_per_block\":" << kSharedBytesPerBlock
      << ",\"rounds\":" << kRounds
      << ",\"grid\":" << gridX << "}";
    res.params_json = p.str();
    res.metadata["access_pattern"] = "shared_crossbar";
    res.metadata["memory_type"] = "shared";
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

    // warmup shared mem crossbar
    for (int w = 0; w < 3; ++w) {
        sharedMemCrossbarKernel<<<gridX, 256, 256 * sizeof(float), str>>>(dSrc, dDst, numElems, 100);
    }
    chk(cudaStreamSynchronize(str), "wsm");

    struct TD { const char* label; double byteMult; };
    TD tests[] = {
        {"lpddr5x_read",  1.0},
        {"lpddr5x_write", 1.0},
        {"lpddr5x_copy",  2.0},
    };

    for (const auto& td : tests) {
        try {
            results.push_back(measure(td.label, td.byteMult, dSrc, dDst, numElems, allocBytes, gridX, iterations, evS, evE, str));
        } catch (const std::exception& ex) {
            BenchResult r{};
            r.suite_name = "memory";
            r.test_name  = td.label;
            r.unit       = "GB/s";
            r.peak_pct   = 0.0;
            if (td.label[8] == 'r') r.metadata["access_pattern"] = "read";
            else if (td.label[8] == 'w') r.metadata["access_pattern"] = "write";
            else r.metadata["access_pattern"] = "copy";
            r.metadata["memory_type"] = "device";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"bytes\":";
            err += std::to_string(allocBytes);
            err += "}";
            r.params_json = err;
            results.push_back(r);
        }
    }

    // shared mem crossbar
    try {
        results.push_back(measureSharedMem(numElems, gridX, iterations, evS, evE, str));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "memory";
        r.test_name  = "shared_mem_crossbar";
        r.unit       = "GB/s";
        r.peak_pct   = 0.0;
        r.metadata["access_pattern"] = "shared_crossbar";
        r.metadata["memory_type"] = "shared";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    chk(cudaFree(dSrc), "fs");
    chk(cudaFree(dDst), "fd");

    return results;
}

} // namespace deusridet::bench

static std::string getSweepTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf{};
    localtime_r(&time_t_now, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

BENCH_REGISTER_SWEEP_SUITE(memory, "LPDDR5X memory bandwidth (read/write/copy)",
    [](deusridet::bench::BenchRunner& runner, int device) -> std::vector<deusridet::bench::SweepReport> {
        deusridet::bench::SweepReport report;
        report.suite_name = "memory";
        report.description = "LPDDR5X memory bandwidth (read/write/copy)";
        report.param_names.push_back("transfer_bytes");

        deusridet::bench::PowerMonitor pm;
        pm.init();

        for (size_t transferSize : std::vector<size_t>{64*1024*1024, 128*1024*1024, 256*1024*1024, 512*1024*1024}) {
            deusridet::bench::SweepResult point;
            point.suite_name = "memory";
            point.test_name = "lpddr5x_read/write/copy/shared";
            {
                std::ostringstream p;
                p << "{\"transfer_bytes\":" << transferSize << "}";
                point.params_json = p.str();
            }
            pm.markStart();
            try {
                auto benchResults = deusridet::bench::runMemoryBench(device, transferSize, 10);
                if (!benchResults.empty()) {
                    point.result = benchResults[0];
                }
            } catch (const std::exception& e) {
                point.error_message = e.what();
            }
            point.power_watts = pm.markEnd();
            point.timestamp = getSweepTimestamp();
            report.points.push_back(point);
        }

        pm.shutdown();

        report.total_points   = static_cast<int>(report.points.size());
        report.success_points = 0;
        report.error_points   = 0;
        for (const auto& pt : report.points) {
            if (pt.error_message.has_value()) ++report.error_points;
            else ++report.success_points;
        }

        return {report};
    });
