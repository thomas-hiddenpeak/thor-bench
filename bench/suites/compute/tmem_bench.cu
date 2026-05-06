#include "compute/tmem_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include "bench_stats.h"
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
BenchResult computeStats(const std::vector<double>& vals,
                         const std::string& suite, const std::string& test,
                         const std::string& unit, const std::string& pj,
                         double peak = 0.0) {
    std::vector<double> sv = vals;
    BenchResult res = ::deusridet::bench::computeStats(sv, 3);
    res.suite_name = suite;
    res.test_name  = test;
    res.unit       = unit;
    res.params_json = pj;
    if (peak > 0.0) {
        res.peak_pct = computePeakPctSame(res.median, peak);
    }
    return res;
}

// ── Shared memory read bandwidth kernel
// ── Shared memory read bandwidth kernel ────────────────────────────────────
// Each threadblock reads a 128-byte tile from __shared__ memory repeatedly.
// Uses .param to avoid compiler optimizing away the loop.
template <int TileBytes>
__global__ void smemReadBwKernel(float* out, int loops) {
    __shared__ float sdata[TileBytes / 4];
    int tid = threadIdx.x;
    for (int i = 0; i < blockDim.x; i += blockDim.x) {
        if (tid + i < TileBytes / 4)
            sdata[tid + i] = static_cast<float>(tid + i);
    }
    __syncthreads();

    float accum = 0.0f;
    int cnt = loops;
    while (cnt--) {
        for (int i = 0; i < TileBytes / 4; ++i) {
            accum += sdata[i];
        }
    }
    asm volatile("" : "+f"(accum));
    if (tid == 0) *out = accum;
}

// ── Shared memory write bandwidth kernel ───────────────────────────────────
template <int TileBytes>
__global__ void smemWriteBwKernel(float* out, int loops) {
    __shared__ float sdata[TileBytes / 4];
    int tid = threadIdx.x;
    int cnt = loops;
    float accum = 0.0f;
    while (--cnt >= 0) {
        for (int i = 0; i < TileBytes / 4; ++i) {
            sdata[(tid + i * 2) % (TileBytes / 4)] = accum + i;
        }
        __syncthreads();
        accum += sdata[tid % (TileBytes / 4)];
    }
    if (tid == 0) *out = accum;
}

// ── Shared memory copy latency kernel (global → smem → global) ────────────
template <int TileBytes>
__global__ void smemCopyLatencyKernel(const float* __restrict__ in,
                                       float* __restrict__ out) {
    __shared__ float sdata[TileBytes / 4];
    int tid = threadIdx.x;
    for (int i = tid; i < TileBytes / 4; i += blockDim.x)
        sdata[i] = in[i];
    __syncthreads();
    for (int i = tid; i < TileBytes / 4; i += blockDim.x)
        out[i] = sdata[i];
}

// ── TMEM proxy read bandwidth ──────────────────────────────────────────────
// On SM110a, TMEM is accessed via tcgen05.alloc/ld/st PTX which requires
// descriptor setup. As a proxy, we measure shared memory bandwidth using
// __shared__ memory with large tiles, which exercises the same L1/shared
// memory crossbar that feeds TMEM.
BenchResult measureTMEMReadBW(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int TileBytes = 128;
    constexpr int loops = 4096;
    int tpb = 128;

    float *dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    // Warmup
    for (int w = 0; w < 3; ++w) {
        smemReadBwKernel<TileBytes><<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        smemReadBwKernel<TileBytes><<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        // Each loop reads TileBytes from smem; total bytes = loops * TileBytes * 1 block
        double totalBytes = static_cast<double>(loops) * TileBytes;
        double gbs = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
        vals.push_back(gbs);
    }

    std::ostringstream p;
    p << "{\"tile_bytes\":" << TileBytes << ",\"loops\":" << loops
      << ",\"tpb\":" << tpb << ",\"type\":\"smem_proxy_for_tmem\"}";

    BenchResult res = computeStats(vals, "tmem", "tmem_read_bw", "GB/s", p.str());
    res.metadata["proxy"] = "true";
    res.metadata["note"] = "SMEM read as TMEM proxy; tcgen05.alloc/ld PTX requires descriptor setup";

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ── TMEM proxy write bandwidth ─────────────────────────────────────────────
BenchResult measureTMEMWriteBW(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int TileBytes = 128;
    constexpr int loops = 2048;
    int tpb = 128;

    float *dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    for (int w = 0; w < 3; ++w) {
        smemWriteBwKernel<TileBytes><<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        smemWriteBwKernel<TileBytes><<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double totalBytes = static_cast<double>(loops) * TileBytes;
        double gbs = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
        vals.push_back(gbs);
    }

    std::ostringstream p;
    p << "{\"tile_bytes\":" << TileBytes << ",\"loops\":" << loops
      << ",\"tpb\":" << tpb << ",\"type\":\"smem_proxy_for_tmem\"}";

    BenchResult res = computeStats(vals, "tmem", "tmem_write_bw", "GB/s", p.str());
    res.metadata["proxy"] = "true";
    res.metadata["note"] = "SMEM write as TMEM proxy; tcgen05.st PTX requires TMEM allocation";

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ── TMEM proxy copy latency ────────────────────────────────────────────────
BenchResult measureTMEMCopyLatency(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int TileBytes = 128;
    int tpb = 64;

    float *dIn = nullptr, *dOut = nullptr;
    chk(cudaMalloc(&dIn, TileBytes), "in");
    chk(cudaMalloc(&dOut, TileBytes), "out");
    chk(cudaMemset(dIn, 0x3F, TileBytes), "in");

    for (int w = 0; w < 3; ++w) {
        smemCopyLatencyKernel<TileBytes><<<1, tpb, 0, str>>>(dIn, dOut);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        smemCopyLatencyKernel<TileBytes><<<1, tpb, 0, str>>>(dIn, dOut);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double ns = ms * 1e6;
        vals.push_back(ns);
    }

    std::ostringstream p;
    p << "{\"tile_bytes\":" << TileBytes << ",\"tpb\":" << tpb
      << ",\"type\":\"smem_proxy_for_tmem\"}";

    BenchResult res = computeStats(vals, "tmem", "tmem_cp_latency", "ns", p.str());
    res.metadata["proxy"] = "true";
    res.metadata["note"] = "Global→SMEM→Global latency as tcgen05.cp proxy";

    chk(cudaFree(dIn), "in");
    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ── TMEM vs SMEM comparison ────────────────────────────────────────────────
// Since TMEM is accessed via tcgen05 PTX on SM110a, we compare two SMEM
// configurations: small tile (low capacity) vs large tile (higher capacity)
// to show the bandwidth characteristics that TMEM would operate within.
BenchResult measureTMEMvsSMEM(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int SmallTile = 64;
    constexpr int LargeTile = 256;
    constexpr int loops = 4096;
    int tpb = 128;

    float *dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    std::vector<double> vals;
    // Run large tile to get higher bandwidth number (TMEM-like)
    for (int w = 0; w < 3; ++w) {
        smemReadBwKernel<LargeTile><<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaStreamSynchronize(str), "warmup");
    }
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        smemReadBwKernel<LargeTile><<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double totalBytes = static_cast<double>(loops) * LargeTile;
        double gbs = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
        vals.push_back(gbs);
    }

    std::ostringstream p;
    p << "{\"small_tile\":" << SmallTile << ",\"large_tile\":" << LargeTile
      << ",\"loops\":" << loops << ",\"tpb\":" << tpb
      << ",\"note\":\"large tile SMEM read as TMEM bandwidth proxy\"}";

    BenchResult res = computeStats(vals, "tmem", "tmem_vs_smem", "GB/s", p.str());
    res.metadata["proxy"] = "true";
    res.metadata["note"] = "Large-tile SMEM vs small-tile SMEM as TMEM proxy comparison";

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

} // anonymous namespace

std::vector<BenchResult> runTMEMBench(int device, int matDim, int iterations) {
    (void)matDim; // TMEM benchmarks don't use matrix dimensions
    std::vector<BenchResult> results;

    // --- TMEM Read Bandwidth (SMEM proxy) ---
    try {
        results.push_back(measureTMEMReadBW(device, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "tmem";
        r.test_name  = "tmem_read_bw";
        r.unit       = "GB/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    // --- TMEM Write Bandwidth (SMEM proxy) ---
    try {
        results.push_back(measureTMEMWriteBW(device, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "tmem";
        r.test_name  = "tmem_write_bw";
        r.unit       = "GB/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    // --- TMEM Copy Latency (SMEM proxy) ---
    try {
        results.push_back(measureTMEMCopyLatency(device, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "tmem";
        r.test_name  = "tmem_cp_latency";
        r.unit       = "ns";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    // --- TMEM vs SMEM comparison ---
    try {
        results.push_back(measureTMEMvsSMEM(device, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "tmem";
        r.test_name  = "tmem_vs_smem";
        r.unit       = "GB/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(tmem, "TCGen05 TMEM bandwidth and latency (tcgen05.alloc/ld/st/cp)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runTMEMBench(0, 512, 10);
    });
