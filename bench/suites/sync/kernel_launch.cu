#include "sync/kernel_launch.h"
#include "bench_schema.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace deusridet::bench {

namespace {

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void emptyKernel() {
    // intentionally empty — measures launch latency only
}

__global__ void smallKernel(float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = static_cast<float>(tid);
}

// ---------------------------------------------------------------------------
// Kernel launch latency benchmarks
// ---------------------------------------------------------------------------

void benchEmptyKernel(int iters, cudaEvent_t evS, cudaEvent_t evE,
                      cudaStream_t str, std::vector<BenchResult>& results) {
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "es");
        emptyKernel<<<1, 1, 0, str>>>();
        chk(cudaEventRecord(evE, str), "ee");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        vals.push_back(ms * 1000.0); // ms → µs
    }
    BenchResult res = computeStats(vals, 3);
    res.suite_name = "kernel_launch";
    res.test_name  = "kernel_launch_empty";
    res.unit       = "µs";
    res.params_json = "{\"blocks\":1,\"threads\":1}";
    results.push_back(res);
}

void benchSmallKernel(int iters, cudaEvent_t evS, cudaEvent_t evE,
                      cudaStream_t str, std::vector<BenchResult>& results) {
    constexpr int kTpb = 256;
    constexpr int kN = 256;
    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, kN * sizeof(float)), "m1");

    // warmup
    for (int w = 0; w < 3; ++w) {
        smallKernel<<<1, kTpb, 0, str>>>(dOut, kN);
    }
    chk(cudaStreamSynchronize(str), "ws");

    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "es");
        smallKernel<<<1, kTpb, 0, str>>>(dOut, kN);
        chk(cudaEventRecord(evE, str), "ee");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        vals.push_back(ms * 1000.0); // ms → µs
    }

    chk(cudaFree(dOut), "f1");

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "kernel_launch";
    res.test_name  = "kernel_launch_small";
    res.unit       = "µs";
    res.params_json = "{\"blocks\":1,\"threads\":256,\"elements\":256}";
    results.push_back(res);
}

// ---------------------------------------------------------------------------
// CUDA Graph benchmarks
// ---------------------------------------------------------------------------

BenchResult makeGraphStub(const char* testName, const char* extraJson, const char* reason) {
    BenchResult r;
    r.suite_name = "kernel_launch";
    r.test_name  = testName;
    r.unit       = "µs";
    r.sample_count = 0;
    r.warmup_count = 0;
    std::string p = "{\"error\":\"";
    p += reason;
    p += "\",";
    if (extraJson && extraJson[0]) p += extraJson;
    p += "}";
    r.params_json = p;
    r.metadata["stub"] = "true";
    r.metadata["reason"] = reason;
    return r;
}

void benchGraphCapture(int iters, cudaEvent_t evS, cudaEvent_t evE,
                       cudaStream_t str, std::vector<BenchResult>& results) {
    constexpr int kTpb = 256;
    constexpr int kN = 256;
    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, kN * sizeof(float)), "mc");

    // Warmup capture
    for (int w = 0; w < 3; ++w) {
        chk(cudaStreamBeginCapture(str, cudaStreamCaptureModeGlobal), "b1");
        smallKernel<<<1, kTpb, 0, str>>>(dOut, kN);
        chk(cudaStreamEndCapture(str, nullptr), "e1");
    }
    chk(cudaStreamSynchronize(str), "ws");

    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "es");
        chk(cudaStreamBeginCapture(str, cudaStreamCaptureModeGlobal), "b2");
        smallKernel<<<1, kTpb, 0, str>>>(dOut, kN);
        chk(cudaStreamEndCapture(str, nullptr), "e2");
        chk(cudaEventRecord(evE, str), "ee");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        vals.push_back(ms * 1000.0);
    }

    chk(cudaFree(dOut), "fc");

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "kernel_launch";
    res.test_name  = "cuda_graph_capture";
    res.unit       = "µs";
    res.params_json = "{\"mode\":\"global\",\"blocks\":1,\"threads\":256}";
    results.push_back(res);
}

void benchGraphReplay(int iters, cudaEvent_t evS, cudaEvent_t evE,
                      cudaStream_t str, std::vector<BenchResult>& results) {
    constexpr int kTpb = 256;
    constexpr int kN = 256;
    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, kN * sizeof(float)), "mr");

    // Capture graph once
    cudaGraph_t graph = nullptr;
    chk(cudaStreamBeginCapture(str, cudaStreamCaptureModeGlobal), "b3");
    smallKernel<<<1, kTpb, 0, str>>>(dOut, kN);
    chk(cudaStreamEndCapture(str, &graph), "e3");
    chk(cudaStreamSynchronize(str), "ws");

    // Instantiate
    cudaGraphExec_t exec = nullptr;
    chk(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "inst");

    // Measure replay
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "es");
        chk(cudaGraphLaunch(exec, str), "gl");
        chk(cudaEventRecord(evE, str), "ee");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        vals.push_back(ms * 1000.0);
    }

    chk(cudaGraphExecDestroy(exec), "dg");
    chk(cudaGraphDestroy(graph), "dgr");
    chk(cudaFree(dOut), "fr");

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "kernel_launch";
    res.test_name  = "cuda_graph_replay";
    res.unit       = "µs";
    res.params_json = "{\"mode\":\"global\",\"blocks\":1,\"threads\":256}";
    results.push_back(res);
}

void benchGraphReplayWarm(int iters, cudaEvent_t evS, cudaEvent_t evE,
                          cudaStream_t str, std::vector<BenchResult>& results) {
    constexpr int kTpb = 256;
    constexpr int kN = 256;
    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, kN * sizeof(float)), "mwr");

    // Capture graph once
    cudaGraph_t graph = nullptr;
    chk(cudaStreamBeginCapture(str, cudaStreamCaptureModeGlobal), "bw");
    smallKernel<<<1, kTpb, 0, str>>>(dOut, kN);
    chk(cudaStreamEndCapture(str, &graph), "ew");
    chk(cudaStreamSynchronize(str), "ws");

    // Instantiate
    cudaGraphExec_t exec = nullptr;
    chk(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "inst");

    // Warmup replays
    for (int w = 0; w < 10; ++w) {
        chk(cudaGraphLaunch(exec, str), "glw");
    }
    chk(cudaStreamSynchronize(str), "wss");

    // Measure replay
    std::vector<double> vals;
    for (int i = 0; i < iters; ++i) {
        chk(cudaEventRecord(evS, str), "es");
        chk(cudaGraphLaunch(exec, str), "gl");
        chk(cudaEventRecord(evE, str), "ee");
        chk(cudaStreamSynchronize(str), "sy");
        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        vals.push_back(ms * 1000.0);
    }

    chk(cudaGraphExecDestroy(exec), "dg");
    chk(cudaGraphDestroy(graph), "dgr");
    chk(cudaFree(dOut), "fr");

    BenchResult res = computeStats(vals, 3);
    res.suite_name = "kernel_launch";
    res.test_name  = "cuda_graph_replay_warm";
    res.unit       = "µs";
    res.params_json = "{\"mode\":\"global\",\"warmup\":10,\"blocks\":1,\"threads\":256}";
    results.push_back(res);
}

} // anonymous namespace

std::vector<BenchResult> runKernelLaunchBench(int device, int iterations) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    cudaEvent_t evS, evE;
    cudaStream_t str;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    chk(cudaStreamCreate(&str), "st");

    // warmup
    emptyKernel<<<1, 1, 0, str>>>();
    chk(cudaStreamSynchronize(str), "ws");

    try {
        benchEmptyKernel(iterations, evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "kernel_launch";
        r.test_name  = "kernel_launch_empty";
        r.unit       = "µs";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    try {
        benchSmallKernel(iterations, evS, evE, str, results);
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "kernel_launch";
        r.test_name  = "kernel_launch_small";
        r.unit       = "µs";
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\"}";
        r.params_json = err;
        results.push_back(r);
    }

    // CUDA Graph capture benchmark
    try {
        benchGraphCapture(iterations, evS, evE, str, results);
    } catch (const std::exception& ex) {
        results.push_back(makeGraphStub("cuda_graph_capture", "\"mode\":\"global\"", ex.what()));
    }

    // CUDA Graph replay benchmark
    try {
        benchGraphReplay(iterations, evS, evE, str, results);
    } catch (const std::exception& ex) {
        results.push_back(makeGraphStub("cuda_graph_replay", "\"mode\":\"global\"", ex.what()));
    }

    // CUDA Graph replay benchmark (warmed)
    try {
        benchGraphReplayWarm(iterations, evS, evE, str, results);
    } catch (const std::exception& ex) {
        results.push_back(makeGraphStub("cuda_graph_replay_warm", "\"mode\":\"global\",\"warmup\":10", ex.what()));
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return results;
}

} // namespace deusridet::bench

#include "bench_suites.h"

BENCH_REGISTER_SUITE(kernel_launch, "Kernel launch latency and CUDA Graph replay",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runKernelLaunchBench(0, 10);
    });
