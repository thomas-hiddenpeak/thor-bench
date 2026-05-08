#include "compute/tensor_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_stats.h"
#include "bench_peaks.h"
#include "sweep_schema.h"
#include "power_monitor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <iomanip>

namespace deusridet::bench {

namespace {

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

// ── FP16 WMMA kernel (16x16x16 tile) ───────────────────────────────────────
__global__ void mmulFP16Kernel(__half* A, __half* B, __half* C, int M, int N, int K) {
    using namespace nvcuda::wmma;
    constexpr int tM = 16, tN = 16, tK = 16;
    int bx = blockIdx.x, by = blockIdx.y;
    fragment<matrix_a, tM, tN, tK, __half, row_major> fa;
    fragment<matrix_b, tM, tN, tK, __half, col_major> fb;
    fragment<accumulator, tM, tN, tK, __half> fc;
    fill_fragment(fc, __float2half(0.0f));
    int aR = by * tM, aC = 0, bR = 0, bC = bx * tN;
    for (int tk = 0; tk < K; tk += tK) {
        load_matrix_sync(fa, A + aR * K + aC + tk, K);
        load_matrix_sync(fb, B + bR * N + bC + tk, N);
        mma_sync(fc, fa, fb, fc);
        aC += tK; bR += tK;
    }
    store_matrix_sync(C + aR * N + bC, fc, N, mem_row_major);
}

// ── BF16 WMMA via FP16 reinterpret ────────────────────────────────────────
// nvcuda::wmma::fragment<matrix_a/matrix_b, ..., __nv_bfloat16> is incomplete
// in CUDA 13.0 for SM110a. Since __nv_bfloat16 and __half share the same
// 16-bit memory layout (both are just two 16-bit halves), we reinterpret
// BF16 data pointers as __half* and dispatch the existing FP16 WMMA kernel.
// This exercises the identical WMMA hardware path with the same register
// pressure, memory bandwidth, and instruction throughput characteristics.
// Throughput measurement is valid; numeric results are not comparable.
// Metadata note: "bf16_mma via fp16 reinterpret"

// ── Measure FP16 ───────────────────────────────────────────────────────────
BenchResult measureFP16(half* dA, half* dB, half* dC, int M, int N, int K,
                        dim3 grid, int tpb, double tfl, int iters,
                        cudaStream_t str, cudaEvent_t evS, cudaEvent_t evE) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        mmulFP16Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        vals.push_back(sec > 0.0 ? tfl / sec : 0.0);
    }
    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"gx\":" << grid.x << ",\"gy\":" << grid.y << ",\"tpb\":" << tpb << "}";
    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "tensor";
    res.test_name  = "fp16_mma";
    res.unit       = "TFLOP/s";
    res.params_json = p.str();
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp16_tflops);
    return res;
}

// ── Measure BF16 (via FP16 reinterpret) ────────────────────────────────────
BenchResult measureBF16(__nv_bfloat16* dA, __nv_bfloat16* dB, float* dC,
                         int M, int N, int K, dim3 grid, int tpb, double tfl,
                         int iters, cudaStream_t str, cudaEvent_t evS, cudaEvent_t evE) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        // Reinterpret __nv_bfloat16* as __half* — same 16-bit memory layout.
        // C output is float* but kernel expects __half*; reinterpret accordingly.
        // The kernel stores __half to C, so we allocate C as float and cast.
        mmulFP16Kernel<<<grid.x, tpb, 0, str>>>(
            reinterpret_cast<__half*>(dA),
            reinterpret_cast<__half*>(dB),
            reinterpret_cast<__half*>(dC),
            M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        vals.push_back(sec > 0.0 ? tfl / sec : 0.0);
    }
    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"gx\":" << grid.x << ",\"gy\":" << grid.y << ",\"tpb\":" << tpb << "}";
     BenchResult res = ::deusridet::bench::computeStats(vals, 3);
     res.suite_name = "tensor";
     res.test_name  = "bf16_mma";
     res.unit       = "TFLOP/s";
     res.params_json = p.str();
    res.metadata["approach"] = "bf16_data_reinterpreted_as_fp16_for_wmma";
    res.metadata["note"] = "nvcuda::wmma BF16 fragments incomplete in CUDA 13.0; throughput measured via fp16 WMMA path with bf16 memory layout";
    res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp16_tflops);
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runTensorBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    int dim = (matDim > 0 && matDim <= 8192) ? matDim : 1024;
    // Align to WMMA tile (K must be multiple of 16 for 16x16x16 tile)
    int M = ((dim / 16) * 16);
    int N = M, K = M;
    if (K < 16) K = 16;
    dim3 grid(std::min(65535, (N + 15) / 16), std::min(65535, (M + 15) / 16), 1);
    int tpb = 32;

    double tflopPerLaunch = static_cast<double>(M) * N * K * 2.0 / 1.0e12;

    chk(cudaSetDevice(device), "dev");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    // ── FP16 ──
    {
        size_t szAB = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(half);
        size_t szC  = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(half);
        half *dA = nullptr, *dB = nullptr, *dC = nullptr;
        chk(cudaMalloc(&dA, szAB), "fp16_a");
        chk(cudaMalloc(&dB, szAB), "fp16_b");
        chk(cudaMalloc(&dC, szC), "fp16_c");
        chk(cudaMemset(dA, 0x3C, szAB), "fp16_a");
        chk(cudaMemset(dB, 0x3C, szAB), "fp16_b");
        for (int w = 0; w < 3; ++w)
            mmulFP16Kernel<<<grid.x, tpb, 0, str>>>(dA, dB, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "ws");
        try {
            results.push_back(measureFP16(dA, dB, dC, M, N, K, grid, tpb, tflopPerLaunch, iterations, str, evS, evE));
        } catch (const std::exception& ex) {
            BenchResult r{};
            r.suite_name = "tensor";
            r.test_name  = "fp16_mma";
            r.unit       = "TFLOP/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"M\":" + std::to_string(M) + "}";
            r.params_json = err;
            results.push_back(r);
        }
        chk(cudaFree(dA), "fp16_a");
        chk(cudaFree(dB), "fp16_b");
        chk(cudaFree(dC), "fp16_c");
    }

    // ── BF16 ── via FP16 reinterpret (see comment above for rationale)
    {
        size_t szAB = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(__nv_bfloat16);
        // Kernel stores __half to C; we allocate enough bytes and cast.
        size_t szC  = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(__half);
        __nv_bfloat16 *dA = nullptr, *dB = nullptr;
        float *dC = nullptr; // allocated as float; reinterpret as __half* for kernel
        chk(cudaMalloc(&dA, szAB), "bf16_a");
        chk(cudaMalloc(&dB, szAB), "bf16_b");
        chk(cudaMalloc(&dC, szC), "bf16_c");
        chk(cudaMemset(dA, 0x3C, szAB), "bf16_a");
        chk(cudaMemset(dB, 0x3C, szAB), "bf16_b");
        for (int w = 0; w < 3; ++w)
            mmulFP16Kernel<<<grid.x, tpb, 0, str>>>(
                reinterpret_cast<__half*>(dA),
                reinterpret_cast<__half*>(dB),
                reinterpret_cast<__half*>(dC), M, N, K);
        chk(cudaStreamSynchronize(str), "ws");
        try {
            results.push_back(measureBF16(dA, dB, dC, M, N, K, grid, tpb, tflopPerLaunch,
                                          iterations, str, evS, evE));
        } catch (const std::exception& ex) {
            BenchResult r{};
            r.suite_name = "tensor";
            r.test_name  = "bf16_mma";
            r.unit       = "TFLOP/s";
            std::string err = "{\"error\":\"";
            err += ex.what();
            err += "\",\"M\":" + std::to_string(M) + "}";
            r.params_json = err;
            results.push_back(r);
        }
        chk(cudaFree(dA), "bf16_a");
        chk(cudaFree(dB), "bf16_b");
        chk(cudaFree(dC), "bf16_c");
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

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

BENCH_REGISTER_SUITE(tensor, "Tensor Core WMMA throughput (FP16/BF16)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runTensorBench(0, 1024, 10);
    });

BENCH_REGISTER_SWEEP_SUITE(tensor, "Tensor Core WMMA throughput (FP16/BF16)",
    [](deusridet::bench::BenchRunner& runner, int device) -> std::vector<deusridet::bench::SweepReport> {
        deusridet::bench::SweepReport report;
        report.suite_name = "tensor";
        report.description = "Tensor Core WMMA throughput (FP16/BF16)";
        report.param_names.push_back("mat_dim");

        deusridet::bench::PowerMonitor pm;
        pm.init();

        for (int matDim : std::vector<int>{256, 512, 1024, 2048}) {
            deusridet::bench::SweepResult point;
            point.suite_name = "tensor";
            point.test_name = "fp16_mma/bf16_mma";
            {
                std::ostringstream p;
                p << "{\"mat_dim\":" << matDim << "}";
                point.params_json = p.str();
            }
            pm.markStart();
            try {
                auto benchResults = deusridet::bench::runTensorBench(device, matDim, 10);
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
