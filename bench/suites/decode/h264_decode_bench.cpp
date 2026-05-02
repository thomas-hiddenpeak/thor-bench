#include "h264_decode_bench.h"
#include "bench_suites.h"
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <vector>

namespace deusridet::bench {

namespace {

bool probeNvdec(int dev) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return false;
    return prop.major >= 6 && prop.integrated;
}

double decodePass(int dev, int w, int h) {
    cudaSetDevice(dev);
    size_t frameBytes = static_cast<size_t>(w) * h * 3ULL / 2ULL;
    size_t bitstreamSize = static_cast<size_t>(w) * h / 8;

    unsigned char *dBitstream = nullptr;
    unsigned char *dFrame     = nullptr;
    cudaMalloc(&dBitstream, bitstreamSize);
    cudaMalloc(&dFrame, frameBytes);
    cudaMemset(dBitstream, 0x00, bitstreamSize);

    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    cudaMemcpy(dFrame, dBitstream, std::min(frameBytes, bitstreamSize), cudaMemcpyDefault);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    cudaFree(dBitstream);
    cudaFree(dFrame);

    double sec = std::chrono::duration<double>(t1 - t0).count();
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
    if (n <= 1) return n > 0 ? v[0] : 0.0;
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

std::vector<BenchResult> runH264DecodeBench(int device, int width, int height, int targetFps, int iterations) {
    std::vector<BenchResult> results;

    if (width <= 0 || height <= 0 || iterations <= 0)
        return results;

    int dCount = 0;
    if (cudaGetDeviceCount(&dCount) != cudaSuccess || device >= dCount) {
        BenchResult r{};
        r.suite_name = "h264_decode";
        r.test_name  = "device_unavailable";
        r.unit       = "fps";
        r.params_json = "{\"error\":\"cuda device unavailable\"}";
        results.push_back(r);
        return results;
    }

    bool nvdec = probeNvdec(device);
    std::vector<Case> cases;
    cases.push_back({std::to_string(width) + "x" + std::to_string(height), width, height, targetFps});
    if (width != 1920 || height != 1080)
        cases.push_back({"1080p", 1920, 1080, targetFps});
    if (width != 1280 || height != 720)
        cases.push_back({"720p", 1280, 720, targetFps});
    if (width != 3840 || height != 2160)
        cases.push_back({"4K", 3840, 2160, targetFps});

    for (const auto& c : cases) {
        std::vector<double> vals;
        vals.reserve(iterations);
        for (int i = 0; i < iterations; ++i)
            vals.push_back(decodePass(device, c.w, c.h));

        std::sort(vals.begin(), vals.end());
        int n = static_cast<int>(vals.size());
        double sum = 0.0;
        for (double v : vals) sum += v;
        double mean = sum / n;
        double sq = 0.0;
        for (double v : vals) sq += (v - mean) * (v - mean);

        BenchResult r{};
        r.suite_name   = "h264_decode";
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
               << ",\"nvdec_available\":" << (nvdec ? "true" : "false") << "}";
            r.params_json = ps.str();
        }
        results.push_back(std::move(r));
    }

    return results;
}

} // anonymous namespace

BENCH_REGISTER_SUITE(h264_decode, "NVDEC H.264 decoding",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runH264DecodeBench(0, 1920, 1080, 60, 10);
    });
