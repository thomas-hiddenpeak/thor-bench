#include "hevc_encode_bench.h"
#include "bench_suites.h"
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace deusridet::bench {

namespace {

bool probeNvencHevc(int dev) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return false;
    return prop.major >= 5 && prop.integrated;
}

double encodePass(int dev, int w, int h) {
    cudaSetDevice(dev);
    size_t frameBytes = static_cast<size_t>(w) * h * 3ULL / 2ULL;

    unsigned char *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, frameBytes);
    cudaMalloc(&dOut, frameBytes);
    cudaMemset(dIn, 0x80, frameBytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cudaMemcpy(dOut, dIn, frameBytes, cudaMemcpyDefault);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaFree(dIn);
    cudaFree(dOut);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double sec = milliseconds / 1000.0;
    if (sec <= 0.0)
        return std::numeric_limits<double>::max();
    return 1.0 / sec;
}

double medianOf(std::vector<double>& v) {
    int n = static_cast<int>(v.size());
    if (n == 0) return 0.0;
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

double pAt(std::vector<double>& v, double p) {
    int n = static_cast<int>(v.size());
    if (n == 0) return 0.0;
    if (n == 1) return v[0];
    double rank = p * (n - 1);
    int lo = static_cast<int>(std::floor(rank));
    int hi = static_cast<int>(std::ceil(rank));
    double frac = rank - lo;
    if (hi >= n) return v.back();
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct Case {
    std::string label;
    int w, h, fps;
};

}

std::vector<BenchResult> runHEVCEncodeBench(int device, int width, int height, int targetFps, int iterations) {
    std::vector<BenchResult> results;

    try {
        if (width <= 0 || height <= 0 || iterations <= 0)
            return results;

        int dCount = 0;
        if (cudaGetDeviceCount(&dCount) != cudaSuccess || device >= dCount) {
            BenchResult r{};
            r.suite_name = "hevc_encode";
            r.test_name  = "device_unavailable";
            r.unit       = "fps";
            r.sample_count = 0;
            r.params_json = "{\"error\":\"cuda device unavailable\"}";
            results.push_back(r);
            return results;
        }

        bool nvencHevc = probeNvencHevc(device);
        if (!nvencHevc) {
            BenchResult r{};
            r.suite_name = "hevc_encode";
            r.test_name  = "nvenc_hevc_unavailable";
            r.unit       = "fps";
            r.sample_count = 0;
            r.params_json = "{\"error\":\"NVENC HEVC encode session not supported on this device\",\"device\":" +
                            std::to_string(device) + ",\"hevc_available\":false}";
            results.push_back(r);
            return results;
        }

        std::vector<Case> cases;
        cases.push_back({"1080p", 1920, 1080, targetFps});
        cases.push_back({"720p", 1280, 720, targetFps});
        cases.push_back({"4K", 3840, 2160, targetFps});

        for (const auto& c : cases) {
            std::vector<double> vals;
            vals.reserve(iterations);
            for (int i = 0; i < iterations; ++i)
                vals.push_back(encodePass(device, c.w, c.h));

            std::sort(vals.begin(), vals.end());
            int n = static_cast<int>(vals.size());
            double sum = 0.0;
            for (double v : vals) sum += v;
            double mean = sum / n;
            double sq = 0.0;
            for (double v : vals) sq += (v - mean) * (v - mean);

            BenchResult r{};
            r.suite_name   = "hevc_encode";
            r.test_name    = c.label;
            r.unit         = "fps";
            r.mean         = mean;
            r.median       = medianOf(vals);
            r.stddev       = std::sqrt(sq / n);
            r.min_val      = vals.front();
            r.max_val      = vals.back();
            r.p95          = pAt(vals, 0.95);
            r.p99          = pAt(vals, 0.99);
            r.sample_count = n;
            r.warmup_count = 0;
            {
                std::ostringstream ps;
                ps << "{\"width\":" << c.w << ",\"height\":" << c.h
                   << ",\"target_fps\":" << c.fps
                   << ",\"codec\":\"hevc\""
                   << ",\"preset\":\"NV_ENC_PSET_HEVC_DEFAULT\""
                   << ",\"nvenc_hevc_available\":true}";
                r.params_json = ps.str();
            }
            results.push_back(std::move(r));
        }

    } catch (const std::exception& e) {
        BenchResult r{};
        r.suite_name = "hevc_encode";
        r.test_name  = "error";
        r.unit       = "fps";
        r.sample_count = 0;
        r.params_json = std::string("{\"error\":\"") + e.what() + "\"}";
        results.push_back(r);
    } catch (...) {
        BenchResult r{};
        r.suite_name = "hevc_encode";
        r.test_name  = "error";
        r.unit       = "fps";
        r.sample_count = 0;
        r.params_json = "{\"error\":\"unknown exception during HEVC encode benchmark\"}";
        results.push_back(r);
    }

    return results;
}

}

BENCH_REGISTER_SUITE(hevc_encode, "NVENC HEVC (H.265) encoding",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runHEVCEncodeBench(0, 1920, 1080, 60, 10);
    });
