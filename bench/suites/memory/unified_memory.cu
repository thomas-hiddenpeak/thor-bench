#include "memory/unified_memory.h"
#include "bench_peaks.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace deusridet::bench {

namespace {

constexpr int kTpb = 256;
constexpr int kGridX = 65535;
constexpr size_t kRequestedBytes = 128ULL * 1024ULL * 1024ULL; // 128 MB
constexpr size_t kMaxBytes = 256ULL * 1024ULL * 1024ULL;       // 256 MB cap

// ---- kernels ----

__global__ void umReadKernel(const float* data, float* out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x)
        s += data[i];
    out[tid] = s;
}

__global__ void umWriteKernel(float* data, size_t n) {
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

void runWarmup(float* umBuf, float* dOut, size_t numElems, cudaStream_t str) {
    for (int w = 0; w < 3; ++w) {
        umReadKernel<<<kGridX, kTpb, 0, str>>>(umBuf, dOut, numElems);
        umWriteKernel<<<kGridX, kTpb, 0, str>>>(umBuf, numElems);
    }
    chk(cudaStreamSynchronize(str), "wu");
}

// ---- read benchmark (no prefetch) ----

void benchReadNoPrefetch(float* umBuf, float* dOut, size_t numElems,
                         size_t allocBytes, int iters,
                         cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                         std::vector<BenchResult>& results) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        umReadKernel<<<kGridX, kTpb, 0, str>>>(umBuf, dOut, numElems);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }
    BenchResult res = computeStats(vals, 3);
    res.suite_name = "unified_memory";
    res.test_name  = "um_read_no_prefetch";
    res.unit       = "GB/s";
    res.metadata["prefetch"] = "false";
    res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
    {
        std::ostringstream p;
        p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << kGridX << "}";
        res.params_json = p.str();
    }
    results.push_back(res);
}

// ---- read benchmark (with prefetch) ----

void benchReadPrefetch(float* umBuf, float* dOut, size_t numElems,
                       size_t allocBytes, int iters, int device,
                       cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                       std::vector<BenchResult>& results) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaMemPrefetchAsync(umBuf, allocBytes, cudaMemLocation{cudaMemLocationTypeDevice, device}, 0, str), "pf");
        chk(cudaEventRecord(evS, str), "rs");
        umReadKernel<<<kGridX, kTpb, 0, str>>>(umBuf, dOut, numElems);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }
    BenchResult res = computeStats(vals, 3);
    res.suite_name = "unified_memory";
    res.test_name  = "um_read_prefetch";
    res.unit       = "GB/s";
    res.metadata["prefetch"] = "true";
    res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
    {
        std::ostringstream p;
        p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << kGridX << "}";
        res.params_json = p.str();
    }
    results.push_back(res);
}

// ---- write benchmark (no prefetch) ----

void benchWriteNoPrefetch(float* umBuf, size_t numElems,
                          size_t allocBytes, int iters,
                          cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                          std::vector<BenchResult>& results) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        umWriteKernel<<<kGridX, kTpb, 0, str>>>(umBuf, numElems);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }
    BenchResult res = computeStats(vals, 3);
    res.suite_name = "unified_memory";
    res.test_name  = "um_write_no_prefetch";
    res.unit       = "GB/s";
    res.metadata["prefetch"] = "false";
    res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
    {
        std::ostringstream p;
        p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << kGridX << "}";
        res.params_json = p.str();
    }
    results.push_back(res);
}

// ---- write benchmark (with prefetch) ----

void benchWritePrefetch(float* umBuf, size_t numElems,
                        size_t allocBytes, int iters, int device,
                        cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                        std::vector<BenchResult>& results) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaMemPrefetchAsync(umBuf, allocBytes, cudaMemLocation{cudaMemLocationTypeDevice, device}, 0, str), "pf");
        chk(cudaEventRecord(evS, str), "rs");
        umWriteKernel<<<kGridX, kTpb, 0, str>>>(umBuf, numElems);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }
    BenchResult res = computeStats(vals, 3);
    res.suite_name = "unified_memory";
    res.test_name  = "um_write_prefetch";
    res.unit       = "GB/s";
    res.metadata["prefetch"] = "true";
    res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
    {
        std::ostringstream p;
        p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << kGridX << "}";
        res.params_json = p.str();
    }
    results.push_back(res);
}

// ---- copy benchmark (no prefetch) ----
// H2D copy via unified memory pointer: host memcpy → GPU read, no prefetch

void benchCopyNoPrefetch(float* hBuf, float* umBuf, float* dOut,
                         size_t numElems, size_t allocBytes, int iters,
                         cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                         std::vector<BenchResult>& results) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        // Host-side copy into unified memory
        memcpy(umBuf, hBuf, allocBytes);

        chk(cudaEventRecord(evS, str), "rs");
        umReadKernel<<<kGridX, kTpb, 0, str>>>(umBuf, dOut, numElems);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }
    BenchResult res = computeStats(vals, 3);
    res.suite_name = "unified_memory";
    res.test_name  = "um_copy_no_prefetch";
    res.unit       = "GB/s";
    res.metadata["prefetch"] = "false";
    res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
    {
        std::ostringstream p;
        p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << kGridX << "}";
        res.params_json = p.str();
    }
    results.push_back(res);
}

// ---- copy benchmark (with prefetch) ----
// H2D copy via unified memory pointer: host memcpy → prefetch → GPU read

void benchCopyPrefetch(float* hBuf, float* umBuf, float* dOut,
                       size_t numElems, size_t allocBytes, int iters, int device,
                       cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                       std::vector<BenchResult>& results) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        // Host-side copy into unified memory
        memcpy(umBuf, hBuf, allocBytes);

        chk(cudaMemPrefetchAsync(umBuf, allocBytes, cudaMemLocation{cudaMemLocationTypeDevice, device}, 0, str), "pf");
        chk(cudaEventRecord(evS, str), "rs");
        umReadKernel<<<kGridX, kTpb, 0, str>>>(umBuf, dOut, numElems);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (allocBytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }
    BenchResult res = computeStats(vals, 3);
    res.suite_name = "unified_memory";
    res.test_name  = "um_copy_prefetch";
    res.unit       = "GB/s";
    res.metadata["prefetch"] = "true";
    res.peak_pct = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
    {
        std::ostringstream p;
        p << "{\"bytes\":" << allocBytes << ",\"tpb\":" << kTpb << ",\"grid\":" << kGridX << "}";
        res.params_json = p.str();
    }
    results.push_back(res);
}

} // anonymous namespace

std::vector<BenchResult> runUnifiedMemoryBench(int device, int iterations) {
    std::vector<BenchResult> results;

    // Cap buffer size
    size_t allocBytes = std::min(kRequestedBytes, kMaxBytes);
    size_t numElems   = allocBytes / sizeof(float);

    // Check concurrentManagedAccess support
    int concurrentAccess = 0;
    cudaError_t attrErr = cudaDeviceGetAttribute(&concurrentAccess,
                                                  cudaDevAttrConcurrentManagedAccess, device);
    if (attrErr != cudaSuccess || concurrentAccess != 1) {
        BenchResult r;
        r.suite_name = "unified_memory";
        r.test_name  = "um_read_no_prefetch";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.metadata["prefetch"] = "false";
        r.metadata["concurrentManagedAccess"] =
            (attrErr == cudaSuccess) ? std::to_string(concurrentAccess) : "error";
        std::string err = "{\"error\":\"concurrentManagedAccess not supported\"}";
        r.params_json = err;
        results.push_back(r);
        return results;
    }

    // Allocate unified memory
    float* umBuf = nullptr;
    cudaError_t allocErr = cudaMallocManaged(&umBuf, allocBytes);
    if (allocErr != cudaSuccess) {
        BenchResult r;
        r.suite_name = "unified_memory";
        r.test_name  = "um_read_no_prefetch";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.metadata["prefetch"] = "false";
        std::ostringstream e;
        e << "{\"error\":\"cudaMallocManaged failed: " << cudaGetErrorString(allocErr)
          << "\",\"bytes\":" << allocBytes << "}";
        r.params_json = e.str();
        results.push_back(r);
        return results;
    }

    // Allocate device output buffer for read kernels
    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, allocBytes), "md");

    // Allocate pinned host buffer for copy benchmarks
    float* hBuf = nullptr;
    chk(cudaHostAlloc(&hBuf, allocBytes, cudaHostAllocDefault), "pha");
    // Initialize host buffer
    for (size_t i = 0; i < numElems; ++i) hBuf[i] = 0.5f;

    cudaEvent_t evS, evE;
    cudaStream_t str;
    chk(cudaSetDevice(device), "dev");
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    chk(cudaStreamCreate(&str), "st");

    // Initialize unified memory via host
    memset(umBuf, 0xAA, allocBytes);

    // Warmup
    runWarmup(umBuf, dOut, numElems, str);

    // Force prefetch to device once to ensure pages are allocated before benchmarking
    chk(cudaMemPrefetchAsync(umBuf, allocBytes, cudaMemLocation{cudaMemLocationTypeDevice, device}, 0, str), "init_pf");
    chk(cudaStreamSynchronize(str), "init_sy");

    try {
        benchReadNoPrefetch(umBuf, dOut, numElems, allocBytes, iterations,
                            evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "unified_memory";
        r.test_name  = "um_read_no_prefetch";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.metadata["prefetch"] = "false";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchReadPrefetch(umBuf, dOut, numElems, allocBytes, iterations, device,
                          evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "unified_memory";
        r.test_name  = "um_read_prefetch";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.metadata["prefetch"] = "true";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchWriteNoPrefetch(umBuf, numElems, allocBytes, iterations,
                             evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "unified_memory";
        r.test_name  = "um_write_no_prefetch";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.metadata["prefetch"] = "false";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchWritePrefetch(umBuf, numElems, allocBytes, iterations, device,
                           evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "unified_memory";
        r.test_name  = "um_write_prefetch";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.metadata["prefetch"] = "true";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchCopyNoPrefetch(hBuf, umBuf, dOut, numElems, allocBytes, iterations,
                            evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "unified_memory";
        r.test_name  = "um_copy_no_prefetch";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.metadata["prefetch"] = "false";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchCopyPrefetch(hBuf, umBuf, dOut, numElems, allocBytes, iterations, device,
                          evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "unified_memory";
        r.test_name  = "um_copy_prefetch";
        r.unit       = "GB/s";
        r.sample_count = 0;
        r.metadata["prefetch"] = "true";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    // Cleanup
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    chk(cudaFree(dOut), "fd");
    chk(cudaFreeHost(hBuf), "fh");
    chk(cudaFree(umBuf), "fu");

    return results;
}

} // namespace deusridet::bench

#include "bench_suites.h"

BENCH_REGISTER_SUITE(unified_memory, "Unified Memory benchmark with/without cudaMemPrefetchAsync",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runUnifiedMemoryBench(0, 10);
    });
