#include "memory/tma_copy.h"
#include "bench_schema.h"
#include "bench_peaks.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
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

void benchDirection(void* src, void* dst, size_t bytes,
                    cudaMemcpyKind kind, int iters,
                    cudaEvent_t evS, cudaEvent_t evE, cudaStream_t str,
                    const char* testName, const char* allocType,
                    std::vector<BenchResult>& results) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        chk(cudaMemcpyAsync(dst, src, bytes, kind, str), "cpy");
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double gb = sec > 0.0 ? (bytes / 1073741824.0) / sec : 0.0;
        vals.push_back(gb);
    }
    BenchResult res = computeStats(vals, 3);
    res.suite_name = "tma_copy";
    res.test_name  = testName;
    res.unit       = "GB/s";
    res.peak_pct   = computePeakPctSame(res.median, T5000Peaks::memory_bandwidth_gbs);
    res.metadata["alloc_type"] = allocType;
    if (std::string(allocType) == "fallback")
        res.metadata["fallback_reason"] = "cudaMemPoolCreate not available";
    res.metadata["copy_kind"] = kind == cudaMemcpyHostToDevice ? "h2d" :
                             kind == cudaMemcpyDeviceToHost ? "d2h" : "d2d";
    {
        std::ostringstream p;
        p << "{\"bytes\":" << bytes << "}";
        res.params_json = p.str();
    }
    results.push_back(res);
}

} // anonymous namespace

std::vector<BenchResult> runTMACopyBench(int device, size_t transferSize, int iterations) {
    std::vector<BenchResult> results;

    size_t allocBytes = (transferSize + 3) / 4 * 4;

    cudaEvent_t evS, evE;
    cudaStream_t str;
    cudaMemPool_t pool = nullptr;

    chk(cudaSetDevice(device), "dev");
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    chk(cudaStreamCreate(&str), "st");

    // Check if memory pools are supported on this device
    int memPoolsSupported = 0;
    chk(cudaDeviceGetAttribute(&memPoolsSupported, cudaDevAttrMemoryPoolsSupported, device), "attr");
    // This is tracked in benchDirection metadata via allocType

    bool useMempool = false;
    const char* allocType = "fallback";

    void* dBuf1 = nullptr;
    void* dBuf2 = nullptr;
    void* hBuf = nullptr;

    if (memPoolsSupported) {
        // Attempt mempool creation
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        cudaError_t memPoolErr = cudaMemPoolCreate(&pool, &props);

        if (memPoolErr == cudaSuccess) {
            useMempool = true;
            allocType = "mempool";
            size_t poolLimit = allocBytes * 4;
            chk(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &poolLimit), "mpa");

            chk(cudaMallocFromPoolAsync(&dBuf1, allocBytes, pool, str), "ma1");
            chk(cudaMallocFromPoolAsync(&dBuf2, allocBytes, pool, str), "ma2");
            chk(cudaStreamSynchronize(str), "mas");
        } else {
            // MemPoolCreate failed despite attribute saying supported — fallback
            allocType = "fallback";
            chk(cudaMalloc(&dBuf1, allocBytes), "ma1");
            chk(cudaMalloc(&dBuf2, allocBytes), "ma2");
        }
    } else {
        // Memory pools not supported — direct fallback
        chk(cudaMalloc(&dBuf1, allocBytes), "ma1");
        chk(cudaMalloc(&dBuf2, allocBytes), "ma2");
    }

    // Host pinned buffer for H2D/D2H
    chk(cudaHostAlloc(&hBuf, allocBytes, cudaHostAllocDefault), "hst");

    // Initialize buffers
    chk(cudaMemset(dBuf1, 0xAA, allocBytes), "ms1");
    chk(cudaMemset(dBuf2, 0xBB, allocBytes), "ms2");
    std::fill(static_cast<char*>(hBuf), static_cast<char*>(hBuf) + allocBytes, 0xCC);

    // Warmup copies
    for (int w = 0; w < 3; ++w) {
        chk(cudaMemcpyAsync(dBuf1, hBuf, allocBytes, cudaMemcpyHostToDevice, str), "wh2d");
        chk(cudaMemcpyAsync(hBuf, dBuf1, allocBytes, cudaMemcpyDeviceToHost, str), "wd2h");
        chk(cudaMemcpyAsync(dBuf2, dBuf1, allocBytes, cudaMemcpyDeviceToDevice, str), "wd2d");
    }
    chk(cudaStreamSynchronize(str), "ws");

    try {
        benchDirection(hBuf, dBuf1, allocBytes,
                        cudaMemcpyHostToDevice, iterations,
                        evS, evE, str, "tma_copy_h2d", allocType, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "tma_copy";
        r.test_name  = "tma_copy_h2d";
        r.unit       = "GB/s";
        r.metadata["alloc_type"] = allocType;
        r.metadata["copy_kind"] = "h2d";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchDirection(dBuf1, hBuf, allocBytes,
                        cudaMemcpyDeviceToHost, iterations,
                        evS, evE, str, "tma_copy_d2h", allocType, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "tma_copy";
        r.test_name  = "tma_copy_d2h";
        r.unit       = "GB/s";
        r.metadata["alloc_type"] = allocType;
        r.metadata["copy_kind"] = "d2h";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchDirection(dBuf1, dBuf2, allocBytes,
                        cudaMemcpyDeviceToDevice, iterations,
                        evS, evE, str, "tma_copy_d2d", allocType, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "tma_copy";
        r.test_name  = "tma_copy_d2d";
        r.unit       = "GB/s";
        r.metadata["alloc_type"] = allocType;
        r.metadata["copy_kind"] = "d2d";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"bytes\":";
        err += std::to_string(allocBytes);
        err += "}";
        r.params_json = err;
        results.push_back(r);
    }

    // Cleanup
    chk(cudaFree(dBuf1), "f1");
    chk(cudaFree(dBuf2), "f2");
    if (useMempool) {
        chk(cudaMemPoolTrimTo(pool, 0), "mpt");
        chk(cudaMemPoolDestroy(pool), "mpd");
    }
    chk(cudaFreeHost(hBuf), "fh");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return results;
}

} // namespace deusridet::bench

#include "bench_suites.h"

BENCH_REGISTER_SUITE(tma_copy, "TMA async copy bandwidth (mempool)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runTMACopyBench(0, 256*1024*1024, 10);
    });
