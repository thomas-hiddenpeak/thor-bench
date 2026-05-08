#include "system/thermal_throttle.h"
#include "bench_peaks.h"
#include "bench_suites.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <thread>

namespace deusridet::bench {

namespace {

constexpr int kTpb = 256;

// ---- heavy FP32 FMA kernel ----

__global__ void sustainedFMAKernel(float* a, float* b, float* c, size_t n, int burns) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float va = 1.0001f, vb = 0.9999f, vc = 1.0f;
    for (size_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        va += a[i] * 0.001f;
        vb += b[i] * 0.001f;
    }
    for (int b = 0; b < burns; ++b) {
        vc = fmaf(vc, va, vb);
    }
    if (tid < n)
        c[tid] = vc;
}

// ---- helpers ----

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

struct ThermalSample {
    double gflops;
    double gpuTempC;
    double gpuClockMhz;
};

// Read GPU temp and clock from tegrastats --once; returns true on success.
// Output format: "gpu@40.406C" for temp. No GPU clock field in default output.
bool readTegrastats(double& tempC, double& clockMhz) {
    tempC = -1.0;
    clockMhz = -1.0;

    FILE* fp = popen("tegrastats --once", "r");
    if (!fp) return false;

    char line[1024];
    if (std::fgets(line, sizeof(line), fp)) {
        // Parse "gpu@40.406C" pattern for temperature
        char* gpuStr = strstr(line, "gpu@");
        if (gpuStr) {
            tempC = std::strtod(gpuStr + 4, nullptr);
        }
    }

    pclose(fp);
    return (tempC > 0.0);
}

BenchResult computeStats(const std::vector<ThermalSample>& samples, const std::string& testName, int durationSec) {
    std::vector<double> gflops;
    double firstGflops = 0.0, lastGflops = 0.0;
    double minTemp = 999.0, maxTemp = -999.0;

    for (const auto& s : samples) {
        gflops.push_back(s.gflops);
        if (s.gpuTempC > 0.0) {
            minTemp = std::min(minTemp, s.gpuTempC);
            maxTemp = std::max(maxTemp, s.gpuTempC);
        }
    }

    if (!gflops.empty()) {
        firstGflops = gflops.front();
        lastGflops = gflops.back();
    }

    BenchResult res;
    res.suite_name = "thermal_throttle";
    res.test_name  = testName;
    res.unit       = "GFLOP/s";
    res.warmup_count = 0; // sustained test, no traditional warmup

    int n = static_cast<int>(gflops.size());
    res.sample_count = n;

    if (!gflops.empty()) {
        std::sort(gflops.begin(), gflops.end());
        double sum = 0;
        for (double v : gflops) sum += v;
        double mean = sum / n;

        res.min_val  = gflops.front();
        res.max_val  = gflops.back();
        res.mean     = mean;
        res.median   = (n % 2 == 1) ? gflops[n / 2] : (gflops[n / 2 - 1] + gflops[n / 2]) / 2.0;

        double sq = 0;
        for (double v : gflops) { double d = v - mean; sq += d * d; }
        res.stddev = std::sqrt(sq / n);

        auto pct = [&](double p) -> double {
            if (n <= 1) return gflops[0];
            double r = p * (n - 1);
            int lo = static_cast<int>(std::floor(r));
            int hi = static_cast<int>(std::ceil(r));
            if (hi >= n) return gflops.back();
            return gflops[lo] * (1.0 - (r - lo)) + gflops[hi] * (r - lo);
        };
        res.p95 = pct(0.95);
        res.p99 = pct(0.99);
    }

    res.peak_pct = computePeakPctFromG(res.median, T5000Peaks::fp32_tflops);

    double sustainPct = (firstGflops > 0.0) ? (lastGflops / firstGflops) * 100.0 : 100.0;
    res.metadata["sustain_pct"] = std::to_string(sustainPct);
    res.metadata["duration_sec"] = std::to_string(durationSec);
    if (minTemp < 999.0) {
        res.metadata["gpu_temp_min_c"] = std::to_string(minTemp);
        res.metadata["gpu_temp_max_c"] = std::to_string(maxTemp);
    }

    std::ostringstream p;
    p << "{\"duration_sec\":" << durationSec
      << ",\"interval_sec\":10"
      << ",\"sample_count\":" << n
      << ",\"first_sample_gflops\":" << firstGflops
      << ",\"last_sample_gflops\":" << lastGflops
      << ",\"sustain_pct\":" << sustainPct << "}";
    res.params_json = p.str();

    return res;
}

std::vector<BenchResult> runSustainedTest(int device, int durationSec, const std::string& testName) {
    std::vector<BenchResult> results;

    chk(cudaSetDevice(device), "dev");

    // Allocate large buffer to maximize GPU utilization
    size_t numElems = 16ULL * 1024 * 1024; // 16M floats = 64MB
    int burns = 256; // burn cycles per thread per launch
    int gridX = std::max(1, std::min(65535, static_cast<int>((numElems + kTpb - 1) / kTpb)));

    float* dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t sz = numElems * sizeof(float);
    chk(cudaMalloc(&dA, sz), "a");
    chk(cudaMalloc(&dB, sz), "b");
    chk(cudaMalloc(&dC, sz), "c");
    chk(cudaMemset(dA, 0x3F, sz), "a");
    chk(cudaMemset(dB, 0x3F, sz), "b");

    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");

    // Warmup: run briefly to stabilize
    auto warmupStart = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - warmupStart).count() < 3) {
        sustainedFMAKernel<<<gridX, kTpb, 0, str>>>(dA, dB, dC, numElems, burns);
        chk(cudaStreamSynchronize(str), "ws");
    }

    // Sustained measurement: collect samples every ~10 seconds
    std::vector<ThermalSample> samples;
    auto testStart = std::chrono::steady_clock::now();

    while (true) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - testStart).count();
        if (elapsed >= durationSec) break;

        // Read thermal state before this sample
        double tempC = -1.0, clockMhz = -1.0;
        readTegrastats(tempC, clockMhz);

        // Run kernel and measure
        chk(cudaEventRecord(evS, str), "rs");
        sustainedFMAKernel<<<gridX, kTpb, 0, str>>>(dA, dB, dC, numElems, burns);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        size_t totalFlops = static_cast<size_t>(gridX) * kTpb * burns * 2;
        double gflops = sec > 0.0 ? (totalFlops / 1e9) / sec : 0.0;

        ThermalSample s;
        s.gflops = gflops;
        s.gpuTempC = tempC;
        s.gpuClockMhz = clockMhz;
        samples.push_back(s);

        // Wait to hit ~10 second intervals
        auto now = std::chrono::steady_clock::now();
        auto elapsedNow = std::chrono::duration_cast<std::chrono::seconds>(
            now - testStart).count();
        int nextSampleSec = static_cast<int>(elapsedNow) / 10 + 1;
        nextSampleSec *= 10;
        if (nextSampleSec < durationSec) {
            auto sleepUntil = testStart + std::chrono::seconds(nextSampleSec);
            std::this_thread::sleep_until(sleepUntil);
        }
    }

    if (!samples.empty()) {
        results.push_back(computeStats(samples, testName, durationSec));
    }

    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");
    chk(cudaFree(dA), "fa");
    chk(cudaFree(dB), "fb");
    chk(cudaFree(dC), "fc");

    return results;
}

} // anonymous namespace

std::vector<BenchResult> runThermalThrottleBench(int device, int durationSec, int iterations) {
    std::vector<BenchResult> results;

    // iterations parameter is unused; sustained tests use fixed intervals
    (void)iterations;

    try {
        auto r60 = runSustainedTest(device, durationSec, "sustained_fp32_" + std::to_string(durationSec) + "s");
        results.insert(results.end(), r60.begin(), r60.end());
    } catch (const std::exception& ex) {
        BenchResult r;
        r.suite_name = "thermal_throttle";
        r.test_name  = "sustained_fp32_" + std::to_string(durationSec) + "s";
        r.unit       = "GFLOP/s";
        r.sample_count = 0;
        r.warmup_count = 0;
        r.metadata["duration_sec"] = std::to_string(durationSec);
        std::string err = "{\"error\":\"";
        err += ex.what();
        err += "\",\"duration_sec\":" + std::to_string(durationSec) + "}";
        r.params_json = err;
        results.push_back(r);
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(thermal_throttle, "Sustained performance under thermal constraints",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runThermalThrottleBench(0, 10, 1);
    });
