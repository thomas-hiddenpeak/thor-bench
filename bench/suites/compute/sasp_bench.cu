#include "compute/sasp_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_peaks.h"
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
constexpr int kTile = 8;   // tile dimension per thread-block row/col

// --- FP8 dense matmul kernel ---
// C[M,N] = A[M,K] @ B[K,N], tiled.
// Inputs stored as FP8 (__nv_fp8_storage_t, 1 byte each), accumulated in float.
// Uses fmaf for FP8→float→float multiply-add per element.
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

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

__global__ void floatToFp8Kernel(
    __nv_fp8_storage_t* __restrict__ out,
    const float* __restrict__ inp,
    size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        out[idx] = __nv_cvt_float_to_fp8(inp[idx], __NV_SATFINITE, __NV_E4M3);
}

BenchResult measureFP8Dense(int device, int matDim, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    chk(cudaSetDevice(device), "dev");

    matDim = std::min(matDim, 8192); // upper bound guard
    int M = matDim, N = matDim, K = matDim;
    size_t sizeFp8 = static_cast<size_t>(M) * static_cast<size_t>(K) + static_cast<size_t>(K) * static_cast<size_t>(N); // A + B in fp8
    size_t sizeF32 = static_cast<size_t>(M) * static_cast<size_t>(K) + static_cast<size_t>(K) * static_cast<size_t>(N); // float inputs
    size_t sizeC   = static_cast<size_t>(M) * static_cast<size_t>(N);                                                   // float output

    float *dA = nullptr, *dB = nullptr;
    __nv_fp8_storage_t *dA8 = nullptr, *dB8 = nullptr;
    float *dC = nullptr;

    chk(cudaMalloc(&dA, sizeF32 * sizeof(float)), "a");
    chk(cudaMalloc(&dB, sizeF32 * sizeof(float)), "b");
    chk(cudaMalloc(&dA8, sizeFp8 * sizeof(__nv_fp8_storage_t)), "a8");
    chk(cudaMalloc(&dB8, sizeFp8 * sizeof(__nv_fp8_storage_t)), "b8");
    chk(cudaMalloc(&dC, sizeC * sizeof(float)), "c");

    // Initialize with small non-zero values to avoid NaN/Inf in FP8
    chk(cudaMemset(dA, 0x3E, sizeF32 * sizeof(float)), "a"); // ~0.23
    chk(cudaMemset(dB, 0x3E, sizeF32 * sizeof(float)), "b");

    // Convert to FP8 once (outside timing)
    int gridConv = std::max(1, static_cast<int>((static_cast<size_t>(M) * static_cast<size_t>(K) + kTpb - 1) / kTpb));
    floatToFp8Kernel<<<gridConv, kTpb, 0, str>>>(dA8, dA, M * K);
    floatToFp8Kernel<<<gridConv, kTpb, 0, str>>>(dB8, dB, K * N);
    chk(cudaStreamSynchronize(str), "sync");

    // Grid for matmul: one thread per output element in kTile x kTile blocks
    dim3 grid(std::min(65535, (M + kTile - 1) / kTile), std::min(65535, (N + kTile - 1) / kTile), 1);
    dim3 block(kTile, kTile, 1);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        fp8MatmulKernel<<<grid, block, 0, str>>>(dA8, dB8, dC, M, N, K);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    // Measure
    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        fp8MatmulKernel<<<grid, block, 0, str>>>(dA8, dB8, dC, M, N, K);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;

        // Each element of C requires K multiply + K add = 2K FLOPs
        // Total = M * N * 2 * K FLOPs
        size_t totalFlops = static_cast<size_t>(M) * N * K * 2;
        double tflops = sec > 0.0 ? (totalFlops / 1e12) / sec : 0.0;
        vals.push_back(tflops);
    }

    // Compute stats
    BenchResult res;
    res.suite_name = "sasp";
    res.test_name  = "sasp_fp8_dense";
    res.unit       = "TFLOP/s";
    res.warmup_count = 3;

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
        res.peak_pct = computePeakPctFromT(res.median, T5000Peaks::fp8_dense_tflops);
    }

    std::ostringstream p;
    p << "{\"M\":" << M << ",\"N\":" << N << ",\"K\":" << K
      << ",\"tile\":" << kTile
      << ",\"format\":\"e4m3\""
      << ",\"type\":\"scalar_fp8_kernel\"}";
    res.params_json = p.str();
    res.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_dense_tflops));
    res.metadata["note"] = "scalar kernel with FP8 storage + float accumulate; no WMMA/tcgen05";

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

BenchResult measureFP8Sparse(int device, int matDim, int iterations) {
    // cuSPARSELt does NOT support FP8 structured sparsity (only INT8 via INT8X4).
    // FP8 2:4 sparse requires tcgen05.mma.sp inline PTX which is a separate benchmark
    // (fp8_scalar suite). cuSPARSELt's structuredDescriptorInit + matmulDescriptorInit
    // fails with "operation not supported" for CUDA_R_8F_E4M3 sparse inputs.
    BenchResult res{};
    res.suite_name = "sasp";
    res.test_name  = "sasp_fp8_sparse";
    res.unit       = "TFLOP/s";
    res.sample_count = 0;
    res.warmup_count = 0;
    res.median = 0.0;
    res.params_json = "{\"note\":\"FP8 2:4 sparse requires tcgen05.mma.sp PTX; cuSPARSELt only supports INT8X4\"}";
    res.metadata["stub"] = "true";
    res.metadata["stub_reason"] = "cuSPARSELt structured sparsity only supports INT8X4, not FP8 E4M3";
    res.metadata["peak_sparse_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_sparse_tflops));
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runSASPBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    // --- FP8 Dense ---
    try {
        results.push_back(measureFP8Dense(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "sasp";
        r.test_name  = "sasp_fp8_dense";
        r.unit       = "TFLOP/s";
        r.sample_count = 0;
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        r.metadata["peak_dense_tflops"] = std::to_string(static_cast<int>(T5000Peaks::fp8_dense_tflops));
        results.push_back(r);
    }

    // --- FP8 Sparse (2:4) via cuSPARSELt ---
    try {
        results.push_back(measureFP8Sparse(device, matDim, iterations));
    } catch (const std::exception& ex) {
        BenchResult r{};
        r.suite_name = "sasp";
        r.test_name  = "sasp_fp8_sparse";
        r.unit       = "TFLOP/s";
        r.sample_count = 0;
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

BENCH_REGISTER_SUITE(sasp, "FP8 dense matmul + 2:4 structured sparse (scalar kernel, no Tensor Core)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runSASPBench(0, 512, 10);
    });
