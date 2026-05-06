#include "compute/fp8_scalar_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace deusridet::bench {

namespace {

constexpr int kTpb = 256;
constexpr int kTile = 8;

// ── FP8 dense matmul kernel ────────────────────────────────────────────────
// C[M,N] = A[M,K] @ B[K,N], tiled scalar kernel.
// Inputs stored as FP8 (__nv_fp8_storage_t), accumulated in float.
__global__ void fp8MatmulKernel(
    const __nv_fp8_storage_t* __restrict__ A,
    const __nv_fp8_storage_t* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.x * kTile + threadIdx.x;
    int col = blockIdx.y * kTile + threadIdx.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = __half2float(static_cast<__half>(
            __nv_cvt_fp8_to_halfraw(A[row * K + k], __NV_E4M3)));
        float b = __half2float(static_cast<__half>(
            __nv_cvt_fp8_to_halfraw(B[k * N + col], __NV_E4M3)));
        sum = fmaf(a, b, sum);
    }
    C[row * N + col] = sum;
}

// ── Float → FP8 conversion kernel ──────────────────────────────────────────
__global__ void floatToFp8Kernel(
    __nv_fp8_storage_t* __restrict__ out,
    const float* __restrict__ inp,
    size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        out[idx] = __nv_cvt_float_to_fp8(inp[idx], __NV_SATFINITE, __NV_E4M3);
}

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}
BenchResult computeStats(const std::vector<double>& vals,
                         const std::string& suite, const std::string& test,
                         const std::string& unit, const std::string& pj,
                         double peakTflops = 0.0) {
    std::vector<double> sv = vals;
    BenchResult res = ::deusridet::bench::computeStats(sv, 3);
    res.suite_name = suite;
    res.test_name  = test;
    res.unit       = unit;
    res.params_json = pj;
    if (peakTflops > 0.0) {
        res.peak_pct = computePeakPctFromT(res.median, peakTflops);
    }
    return res;
}

// ── Measure FP8 Dense
// ── Measure FP8 Dense ──────────────────────────────────────────────────────
BenchResult measureFP8Dense(int device, int matDim, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    chk(cudaSetDevice(device), "dev");

    matDim = std::min(matDim, 8192);
    int M = matDim, N = matDim, K = matDim;

    size_t sizeF32 = static_cast<size_t>(M) * static_cast<size_t>(K);
    size_t sizeC   = static_cast<size_t>(M) * static_cast<size_t>(N);

    float *dA = nullptr, *dB = nullptr;
    __nv_fp8_storage_t *dA8 = nullptr, *dB8 = nullptr;
    float *dC = nullptr;

    chk(cudaMalloc(&dA, sizeF32 * sizeof(float)), "a");
    chk(cudaMalloc(&dB, sizeF32 * sizeof(float)), "b");
    chk(cudaMalloc(&dA8, sizeF32 * sizeof(__nv_fp8_storage_t)), "a8");
    chk(cudaMalloc(&dB8, sizeF32 * sizeof(__nv_fp8_storage_t)), "b8");
    chk(cudaMalloc(&dC, sizeC * sizeof(float)), "c");

    chk(cudaMemset(dA, 0x3E, sizeF32 * sizeof(float)), "a");
    chk(cudaMemset(dB, 0x3E, sizeF32 * sizeof(float)), "b");

    int gridConv = std::max(1, static_cast<int>((sizeF32 + kTpb - 1) / kTpb));
    floatToFp8Kernel<<<gridConv, kTpb, 0, str>>>(dA8, dA, M * K);
    floatToFp8Kernel<<<gridConv, kTpb, 0, str>>>(dB8, dB, K * N);
    chk(cudaStreamSynchronize(str), "sync");

    dim3 grid(std::min(65535, (M + kTile - 1) / kTile),
              std::min(65535, (N + kTile - 1) / kTile), 1);
    dim3 block(kTile, kTile, 1);

    for (int w = 0; w < 3; ++w) {
        fp8MatmulKernel<<<grid, block, 0, str>>>(dA8, dB8, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        fp8MatmulKernel<<<grid, block, 0, str>>>(dA8, dB8, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        size_t totalFlops = static_cast<size_t>(M) * N * K * 2;
        double tflops = sec > 0.0 ? (totalFlops / 1e12) / sec : 0.0;
        vals.push_back(tflops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"tile\":" << kTile << ",\"format\":\"e4m3\""
      << ",\"type\":\"scalar_fp8_kernel\"}";

    BenchResult res = computeStats(vals, "fp8_scalar", "fp8_scalar_dense",
                                     "TFLOP/s", p.str(), T5000Peaks::fp8_dense_tflops);
    res.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_dense_tflops));
    res.metadata["note"] = "scalar FP8 kernel; tcgen05.mma kind::f8f6f4 PTX requires SMEM descriptors + TMEM alloc";

    chk(cudaFree(dA), "fa");
    chk(cudaFree(dB), "fb");
    chk(cudaFree(dA8), "fa8");
    chk(cudaFree(dB8), "fb8");
    chk(cudaFree(dC), "fc");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ── FP8 Sparse stub ────────────────────────────────────────────────────────
// FP8 2:4 sparse requires tcgen05.mma.sp inline PTX with sparsity metadata.
// Not feasible as a simple kernel without cuSPARSELt tcgen05 extensions.
BenchResult measureFP8Sparse(int device, int matDim, int iterations) {
    (void)device; (void)matDim; (void)iterations;
    BenchResult res{};
    res.suite_name = "fp8_scalar";
    res.test_name  = "fp8_scalar_sparse";
    res.unit       = "TFLOP/s";
    res.sample_count = 0;
    res.warmup_count = 0;
    res.params_json = "{\"note\":\"FP8 2:4 sparse requires tcgen05.mma.sp with sparsity metadata descriptor\"}";
    res.metadata["stub"] = "true";
    res.metadata["stub_reason"] = "FP8 sparse tcgen05.mma.sp requires sparsity metadata + descriptor-based layout";
    res.metadata["peak_sparse_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_sparse_tflops));
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runFP8ScalarBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    // --- FP8 Dense ---
    try {
        results.push_back(measureFP8Dense(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "fp8_scalar";
        r.test_name  = "fp8_scalar_dense";
        r.unit       = "TFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        r.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_dense_tflops));
        results.push_back(r);
    }

    // --- FP8 Sparse ---
    try {
        results.push_back(measureFP8Sparse(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "fp8_scalar";
        r.test_name  = "fp8_scalar_sparse";
        r.unit       = "TFLOP/s";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        r.metadata["peak_sparse_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_sparse_tflops));
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(fp8_scalar, "Scalar FP8 GEMM (no Tensor Core — ~0.04% of peak)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runFP8ScalarBench(0, 512, 10);
    });
