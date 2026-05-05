#include "memory/tegra_memory.h"
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

// ---- kernels (same pattern as memory_bench.cu) ----

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

// ---- helpers ----

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

void runWarmup(float* dSrc, float* dDst, size_t numElems, int gridX, cudaStream_t str) {
    for (int w = 0; w < 3; ++w) {
        memReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
        memWriteKernel<<<gridX, kTpb, 0, str>>>(dDst, numElems);
    }
    chk(cudaStreamSynchronize(str), "ws");
}

// ---- device memory benchmark ----

void benchDevice(const float* dSrc, float* dDst, size_t numElems,
                 size_t allocBytes, int gridX, int iters,
                 cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                 std::vector<BenchResult>& results) {

    // Read bandwidth: GPU reads from device memory
    {
        std::vector<double> vals;
        for (int i = 0; i < iters; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            memReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        BenchResult res = computeStats(vals, 3);
        res.suite_name = "tegra_memory";
        res.test_name  = "device_read";
        res.unit       = "GB/s";
        res.metadata["memory_type"] = "device";
        res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
        {
            std::ostringstream p;
            p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << gridX << "}";
            res.params_json = p.str();
        }
        results.push_back(res);
    }

    // Write bandwidth: GPU writes to device memory
    {
        std::vector<double> vals;
        for (int i = 0; i < iters; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            memWriteKernel<<<gridX, kTpb, 0, str>>>(dDst, numElems);
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        BenchResult res = computeStats(vals, 3);
        res.suite_name = "tegra_memory";
        res.test_name  = "device_write";
        res.unit       = "GB/s";
        res.metadata["memory_type"] = "device";
        res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
        {
            std::ostringstream p;
            p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << gridX << "}";
            res.params_json = p.str();
        }
        results.push_back(res);
    }
}

// ---- pinned memory benchmark ----

void benchPinned(const float* dSrc, float* dDst, size_t numElems,
                 size_t allocBytes, int gridX, int iters,
                 cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                 std::vector<BenchResult>& results) {

    float* hBuf = nullptr;
    chk(cudaHostAlloc(&hBuf, allocBytes, cudaHostAllocDefault), "pha");

    // Copy host data to device so GPU reads from device copy of pinned data
    chk(cudaMemcpy(dDst, hBuf, allocBytes, cudaMemcpyHostToDevice), "h2d");

    // Read bandwidth: GPU reads from device copy (pinned-backed)
    {
        std::vector<double> vals;
        for (int i = 0; i < iters; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            memReadKernel<<<gridX, kTpb, 0, str>>>(dSrc, dDst, numElems);
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        BenchResult res = computeStats(vals, 3);
        res.suite_name = "tegra_memory";
        res.test_name  = "pinned_read";
        res.unit       = "GB/s";
        res.metadata["memory_type"] = "pinned";
        res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
        {
            std::ostringstream p;
            p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << gridX << "}";
            res.params_json = p.str();
        }
        results.push_back(res);
    }

    // Write bandwidth: GPU writes to device copy of pinned data
    {
        std::vector<double> vals;
        for (int i = 0; i < iters; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            memWriteKernel<<<gridX, kTpb, 0, str>>>(dDst, numElems);
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        BenchResult res = computeStats(vals, 3);
        res.suite_name = "tegra_memory";
        res.test_name  = "pinned_write";
        res.unit       = "GB/s";
        res.metadata["memory_type"] = "pinned";
        res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
        {
            std::ostringstream p;
            p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << gridX << "}";
            res.params_json = p.str();
        }
        results.push_back(res);
    }

    chk(cudaFreeHost(hBuf), "pfh");
}

// ---- registered host memory benchmark (Thor Sysmem Full Coherency) ----

void benchRegistered(float* dDst, size_t numElems,
                     size_t allocBytes, int gridX, int iters,
                     cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                     std::vector<BenchResult>& results) {

    float* hBuf = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&hBuf), 4096, allocBytes) != 0) {
        BenchResult r;
        r.suite_name = "tegra_memory";
        r.test_name  = "registered_read";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.metadata["memory_type"] = "registered";
        r.params_json = "{\"error\":\"posix_memalign failed\",\"bytes\":" + std::to_string(allocBytes) + "}";
        results.push_back(r);

        BenchResult r2 = r;
        r2.test_name = "registered_write";
        results.push_back(r2);
        return;
    }

    // Register with portable + mapped flags (Thor Sysmem Full Coherency)
    cudaError_t regErr = cudaHostRegister(hBuf, allocBytes,
                                           cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (regErr != cudaSuccess) {
        free(hBuf);
        BenchResult r;
        r.suite_name = "tegra_memory";
        r.test_name  = "registered_read";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.metadata["memory_type"] = "registered";
        std::ostringstream e;
        e << "{\"error\":\"cudaHostRegister failed: " << cudaGetErrorString(regErr)
          << "\",\"bytes\":" << allocBytes << "}";
        r.params_json = e.str();
        results.push_back(r);

        BenchResult r2 = r;
        r2.test_name = "registered_write";
        results.push_back(r2);
        return;
    }

    // Get device pointer to the registered host memory
    float* dReg = nullptr;
    chk(cudaHostGetDevicePointer(&dReg, hBuf, 0), "hgp");

    // Initialize the buffer via device pointer
    chk(cudaMemset(dReg, 0xAA, allocBytes), "rm");

    // Warmup on registered memory
    for (int w = 0; w < 3; ++w) {
        memReadKernel<<<gridX, kTpb, 0, str>>>(dReg, dDst, numElems);
        memWriteKernel<<<gridX, kTpb, 0, str>>>(dReg, numElems);
    }
    chk(cudaStreamSynchronize(str), "wr");

    // Read bandwidth: GPU reads from registered host memory (Sysmem Full Coherency)
    {
        std::vector<double> vals;
        for (int i = 0; i < iters; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            memReadKernel<<<gridX, kTpb, 0, str>>>(dReg, dDst, numElems);
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        BenchResult res = computeStats(vals, 3);
        res.suite_name = "tegra_memory";
        res.test_name  = "registered_read";
        res.unit       = "GB/s";
        res.metadata["memory_type"] = "registered";
        res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
        {
            std::ostringstream p;
            p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << gridX
              << ",\"coherency\":\"sysmem_full\"}";
            res.params_json = p.str();
        }
        results.push_back(res);
    }

    // Write bandwidth: GPU writes to registered host memory (Sysmem Full Coherency)
    {
        std::vector<double> vals;
        for (int i = 0; i < iters; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            memWriteKernel<<<gridX, kTpb, 0, str>>>(dReg, numElems);
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        BenchResult res = computeStats(vals, 3);
        res.suite_name = "tegra_memory";
        res.test_name  = "registered_write";
        res.unit       = "GB/s";
        res.metadata["memory_type"] = "registered";
        res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
        {
            std::ostringstream p;
            p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << gridX
              << ",\"coherency\":\"sysmem_full\"}";
            res.params_json = p.str();
        }
        results.push_back(res);
    }

    chk(cudaHostUnregister(hBuf), "hur");
    free(hBuf);
}

// ---- pageable memory benchmark (Thor pageableMemoryAccess=1) ----

void benchPageable(float* dDst, size_t numElems,
                   size_t allocBytes, int gridX, int iters,
                   cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                   std::vector<BenchResult>& results) {

    // Check if pageableMemoryAccess is supported
    int pageableAccess = 0;
    cudaError_t attrErr = cudaDeviceGetAttribute(&pageableAccess,
                                                  cudaDevAttrPageableMemoryAccess, 0);
    if (attrErr != cudaSuccess || pageableAccess != 1) {
        BenchResult r;
        r.suite_name = "tegra_memory";
        r.test_name  = "pageable_read";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.metadata["memory_type"] = "pageable";
        r.metadata["pageableMemoryAccess"] = "0";
        std::string err = "{\"error\":\"pageableMemoryAccess not supported\"}";
        r.params_json = err;
        results.push_back(r);

        BenchResult r2 = r;
        r2.test_name = "pageable_write";
        results.push_back(r2);
        return;
    }

    // Allocate pageable host memory with plain malloc
    float* hBuf = static_cast<float*>(std::malloc(allocBytes));
    if (!hBuf) {
        BenchResult r;
        r.suite_name = "tegra_memory";
        r.test_name  = "pageable_read";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.metadata["memory_type"] = "pageable";
        r.metadata["pageableMemoryAccess"] = "1";
        r.params_json = "{\"error\":\"malloc failed\",\"bytes\":" + std::to_string(allocBytes) + "}";
        results.push_back(r);

        BenchResult r2 = r;
        r2.test_name = "pageable_write";
        results.push_back(r2);
        return;
    }

    // Initialize buffer via host
    for (size_t i = 0; i < numElems; ++i) hBuf[i] = 0.5f;

    // Warmup: launch kernels directly on the host pointer (pageableMemoryAccess=1)
    for (int w = 0; w < 3; ++w) {
        memReadKernel<<<gridX, kTpb, 0, str>>>(hBuf, dDst, numElems);
        memWriteKernel<<<gridX, kTpb, 0, str>>>(hBuf, numElems);
    }
    chk(cudaStreamSynchronize(str), "wp");

    // Read bandwidth: GPU reads from pageable host memory
    {
        std::vector<double> vals;
        for (int i = 0; i < iters; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            memReadKernel<<<gridX, kTpb, 0, str>>>(hBuf, dDst, numElems);
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        BenchResult res = computeStats(vals, 3);
        res.suite_name = "tegra_memory";
        res.test_name  = "pageable_read";
        res.unit       = "GB/s";
        res.metadata["memory_type"] = "pageable";
        res.metadata["pageableMemoryAccess"] = "1";
        res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
        {
            std::ostringstream p;
            p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << gridX << "}";
            res.params_json = p.str();
        }
        results.push_back(res);
    }

    // Write bandwidth: GPU writes to pageable host memory
    {
        std::vector<double> vals;
        for (int i = 0; i < iters; ++i) {
            chk(cudaEventRecord(evS, str), "rs");
            memWriteKernel<<<gridX, kTpb, 0, str>>>(hBuf, numElems);
            chk(cudaEventRecord(evE, str), "re");
            chk(cudaStreamSynchronize(str), "sy");
            float ms = 0;
            chk(cudaEventElapsedTime(&ms, evS, evE), "et");
            double sec = ms / 1000.0;
            double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
            vals.push_back(gb);
        }
        BenchResult res = computeStats(vals, 3);
        res.suite_name = "tegra_memory";
        res.test_name  = "pageable_write";
        res.unit       = "GB/s";
        res.metadata["memory_type"] = "pageable";
        res.metadata["pageableMemoryAccess"] = "1";
        res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
        {
            std::ostringstream p;
            p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << gridX << "}";
            res.params_json = p.str();
        }
        results.push_back(res);
    }

    std::free(hBuf);
}

} // anonymous namespace

std::vector<BenchResult> runTegraMemoryBench(int device, size_t transferSize, int iterations) {
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
    runWarmup(dSrc, dDst, numElems, gridX, str);

    try {
        benchDevice(dSrc, dDst, numElems, allocBytes, gridX, iterations, evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "tegra_memory";
        r.test_name  = "device";
        r.unit       = "GB/s";
        r.metadata["memory_type"] = "device";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchPinned(dSrc, dDst, numElems, allocBytes, gridX, iterations, evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "tegra_memory";
        r.test_name  = "pinned";
        r.unit       = "GB/s";
        r.metadata["memory_type"] = "pinned";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchRegistered(dDst, numElems, allocBytes, gridX, iterations, evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "tegra_memory";
        r.test_name  = "registered";
        r.unit       = "GB/s";
        r.metadata["memory_type"] = "registered";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchPageable(dDst, numElems, allocBytes, gridX, iterations, evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "tegra_memory";
        r.test_name  = "pageable";
        r.unit       = "GB/s";
        r.metadata["memory_type"] = "pageable";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
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

#include "bench_suites.h"

BENCH_REGISTER_SUITE(tegra_memory, "Tegra SoC memory architecture benchmark (Device/Pinned/Registered)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runTegraMemoryBench(0, 256*1024*1024, 10);
    });
