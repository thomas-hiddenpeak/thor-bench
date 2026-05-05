#include "memory/l2_cache.h"
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

constexpr int kTpb = 256;
constexpr int kGridX = 65535;
constexpr size_t kL2BufferBytes = 0x08000000; // 128 MB
constexpr size_t kMaxBufferBytes = 256ULL * 1024 * 1024; // 256 MB cap

// ---- L2 read kernel: sequential access via __ldg() ----
__global__ void l2SequentialReadKernel(const float* data, float* out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x)
        s += __ldg(&data[i]);
    out[tid] = s;
}

// ---- L2 read kernel: strided access via __ldg() to bypass L1 ----
__global__ void l2StridedReadKernel(const float* data, float* out, size_t n, size_t stride) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    size_t strideBytes = stride * sizeof(float);
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        // Each thread reads with a large stride to bypass L1 cache lines
        size_t actualIdx = (i * strideBytes) % n;
        s += __ldg(&data[actualIdx]);
    }
    out[tid] = s;
}

// ---- L2 read kernel: pseudo-random access via __ldg() ----
// Simple LCG seeded by thread ID to generate pseudo-random offsets
__global__ void l2RandomReadKernel(const float* data, float* out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;

    // LCG constants (Numerical Recipes)
    unsigned long long seed = static_cast<unsigned long long>(tid) ^ 0x5DEECE66DULL;
    const unsigned long long a = 6364136223846793005ULL;
    const unsigned long long c = 1442695040888963407ULL;
    const unsigned long long m = 1ULL << 63; // 2^63

    size_t logN = 0;
    size_t tmp = n;
    while (tmp > 1) { logN++; tmp >>= 1; }
    size_t mask = (logN > 0) ? (n - 1) : 1;

    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        seed = seed * a + c;
        seed = seed & (m - 1);
        size_t randIdx = (seed >> 16) & mask;
        s += __ldg(&data[randIdx]);
    }
    out[tid] = s;
}

// ---- L2 write kernel: sequential write to large buffer ----
__global__ void l2WriteKernel(float* data, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x)
        data[i] = static_cast<float>(i) * 1.234f;
    __threadfence();
}

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

BenchResult measureRead(const char* label, float* dSrc, float* dDst,
                        size_t numElems, size_t allocBytes, int gridX,
                        int iters, cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        l2SequentialReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "l2_cache";
    res.test_name  = label;
    res.unit       = "GB/s";
    res.peak_pct   = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);

    std::ostringstream p;
    p << "{\"bytes\":" << allocBytes
      << ",\"mult\":1.0"
      << ",\"grid\":" << gridX
      << ",\"tpb\":" << kTpb << "}";
    res.params_json = p.str();
    return res;
}

BenchResult measureStridedRead(float* dSrc, float* dDst,
                               size_t numElems, size_t allocBytes, int gridX,
                               int iters, cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str) {
    // Stride = 128 KB in elements (bypasses L1 cache lines)
    size_t stride = 128 * 1024 / sizeof(float);
    if (stride == 0) stride = 1;

    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        l2StridedReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems, stride);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "l2_cache";
    res.test_name  = "l2_strided_read";
    res.unit       = "GB/s";
    res.peak_pct   = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);

    std::ostringstream p;
    p << "{\"bytes\":" << allocBytes
      << ",\"stride_elements\":" << stride
      << ",\"grid\":" << gridX
      << ",\"tpb\":" << kTpb << "}";
    res.params_json = p.str();
    return res;
}

BenchResult measureRandomRead(float* dSrc, float* dDst,
                              size_t numElems, size_t allocBytes, int gridX,
                              int iters, cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        l2RandomReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "l2_cache";
    res.test_name  = "l2_random_read";
    res.unit       = "GB/s";
    res.peak_pct   = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);

    std::ostringstream p;
    p << "{\"bytes\":" << allocBytes
      << ",\"grid\":" << gridX
      << ",\"tpb\":" << kTpb << "}";
    res.params_json = p.str();
    return res;
}

BenchResult measureWrite(float* dDst,
                         size_t numElems, size_t allocBytes, int gridX,
                         int iters, cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        l2WriteKernel<<<gridX, kTpb, 0, str>>>(dDst, numElems);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "l2_cache";
    res.test_name  = "l2_write";
    res.unit       = "GB/s";
    res.peak_pct   = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);

    std::ostringstream p;
    p << "{\"bytes\":" << allocBytes
      << ",\"mult\":1.0"
      << ",\"grid\":" << gridX
      << ",\"tpb\":" << kTpb << "}";
    res.params_json = p.str();
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runL2CacheBench(int device, int iterations) {
    std::vector<BenchResult> results;

    // Buffer size: ~128 MB, capped at 256 MB
    size_t bufferBytes = kL2BufferBytes;
    if (bufferBytes > kMaxBufferBytes) bufferBytes = kMaxBufferBytes;

    // Guard: check total allocation < 512 MB (2 buffers)
    size_t totalAlloc = bufferBytes * 2;
    constexpr size_t kMaxTotalAlloc = 512ULL * 1024 * 1024;
    if (totalAlloc > kMaxTotalAlloc) {
        bufferBytes = kMaxTotalAlloc / 2;
        totalAlloc = bufferBytes * 2;
    }

    size_t numElems = bufferBytes / sizeof(float);
    int gridX = std::min(kGridX, static_cast<int>((numElems + kTpb - 1) / kTpb));
    gridX = std::max(1, gridX);

    float* dSrc = nullptr;
    float* dDst = nullptr;
    cudaEvent_t evS, evE;
    cudaStream_t str;

    try {
        chk(cudaSetDevice(device), "dev");
        chk(cudaMalloc(&dSrc, bufferBytes), "ms");
        chk(cudaMalloc(&dDst, bufferBytes), "md");
        chk(cudaMemset(dSrc, 0xAA, bufferBytes), "ms");
        chk(cudaEventCreate(&evS), "es");
        chk(cudaEventCreate(&evE), "ee");
        chk(cudaStreamCreate(&str), "st");

        // warmup
        for (int w = 0; w < 3; ++w) {
            l2SequentialReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
            l2StridedReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems, 128 * 1024 / sizeof(float));
            l2RandomReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
            l2WriteKernel<<<gridX, kTpb, 0, str>>>(dDst, numElems);
        }
        chk(cudaStreamSynchronize(str), "ws");

        // l2_sequential_read
        try {
            results.push_back(measureRead("l2_sequential_read", dSrc, dDst,
                                          numElems, bufferBytes, gridX, iterations,
                                          evS, evE, str));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "l2_cache";
            r.test_name  = "l2_sequential_read";
            r.unit       = "GB/s";
            r.peak_pct   = 0.0;
            r.sample_count = 0;
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"bytes\":";
            err += std::to_string(bufferBytes);
            err += "}";
            r.params_json = err;
            results.push_back(r);
        }

        // l2_strided_read
        try {
            results.push_back(measureStridedRead(dSrc, dDst,
                                                 numElems, bufferBytes, gridX, iterations,
                                                 evS, evE, str));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "l2_cache";
            r.test_name  = "l2_strided_read";
            r.unit       = "GB/s";
            r.peak_pct   = 0.0;
            r.sample_count = 0;
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"bytes\":";
            err += std::to_string(bufferBytes);
            err += "}";
            r.params_json = err;
            results.push_back(r);
        }

        // l2_random_read
        try {
            results.push_back(measureRandomRead(dSrc, dDst,
                                                numElems, bufferBytes, gridX, iterations,
                                                evS, evE, str));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "l2_cache";
            r.test_name  = "l2_random_read";
            r.unit       = "GB/s";
            r.peak_pct   = 0.0;
            r.sample_count = 0;
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"bytes\":";
            err += std::to_string(bufferBytes);
            err += "}";
            r.params_json = err;
            results.push_back(r);
        }

        // l2_write
        try {
            results.push_back(measureWrite(dDst,
                                           numElems, bufferBytes, gridX, iterations,
                                           evS, evE, str));
        } catch (const std::exception& ex) {
            BenchResult r;
            r.suite_name = "l2_cache";
            r.test_name  = "l2_write";
            r.unit       = "GB/s";
            r.peak_pct   = 0.0;
            r.sample_count = 0;
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"bytes\":";
            err += std::to_string(bufferBytes);
            err += "}";
            r.params_json = err;
            results.push_back(r);
        }

        chk(cudaStreamDestroy(str), "ds");
        chk(cudaEventDestroy(evS), "de");
        chk(cudaEventDestroy(evE), "de");
        chk(cudaFree(dSrc), "fs");
        chk(cudaFree(dDst), "fd");
    } catch (const std::exception& ex) {
        // Cleanup on error
        if (dSrc) chk(cudaFree(dSrc), "fs_err");
        if (dDst) chk(cudaFree(dDst), "fd_err");
        if (evS) chk(cudaEventDestroy(evS), "de_err");
        if (evE) chk(cudaEventDestroy(evE), "de_err");
        if (str) chk(cudaStreamDestroy(str), "ds_err");

        // Push failure results for each test
        const char* tests[] = {"l2_sequential_read", "l2_strided_read", "l2_random_read", "l2_write"};
        for (const char* t : tests) {
            BenchResult r;
            r.suite_name = "l2_cache";
            r.test_name  = t;
            r.unit       = "GB/s";
            r.peak_pct   = 0.0;
            r.sample_count = 0;
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"bytes\":";
            err += std::to_string(bufferBytes);
            err += "}";
            r.params_json = err;
            results.push_back(r);
        }
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(l2_cache, "L2 cache bandwidth (sequential/strided/random read + write)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runL2CacheBench(0, 10);
    });
