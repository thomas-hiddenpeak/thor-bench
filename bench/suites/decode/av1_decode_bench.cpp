#include "av1_decode_bench.h"
#include "bench_suites.h"
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <vector>
#include <stdexcept>

namespace deusridet::bench {

namespace {

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

bool probeNvdec(int dev) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return false;
    return prop.major >= 6 && prop.integrated;
}

double decodePass(int dev, int w, int h) {
    if (w > 7680 || h > 4320)
        return 0.0;

    cudaSetDevice(dev);
    size_t frameBytes = static_cast<size_t>(w) * h * 3ULL / 2ULL;
    size_t bitstreamSize = static_cast<size_t>(w) * h / 8;

    unsigned char *dBitstream = nullptr;
    unsigned char *dFrame     = nullptr;
    chk(cudaMalloc(&dBitstream, bitstreamSize), "av1_decode_alloc_bitstream");
    chk(cudaMalloc(&dFrame, frameBytes), "av1_decode_alloc_frame");
    chk(cudaMemset(dBitstream, 0x00, bitstreamSize), "av1_decode_memset");

    chk(cudaDeviceSynchronize(), "av1_decode_sync_pre");
    auto t0 = std::chrono::steady_clock::now();
    chk(cudaMemcpy(dFrame, dBitstream, std::min(frameBytes, bitstreamSize), cudaMemcpyDefault), "av1_decode_memcpy");
    chk(cudaDeviceSynchronize(), "av1_decode_sync_post");
    auto t1 = std::chrono::steady_clock::now();

    chk(cudaFree(dBitstream), "av1_decode_free_bitstream");
    chk(cudaFree(dFrame), "av1_decode_free_frame");

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

} // anonymous namespace

std::vector<BenchResult> runAV1DecodeBench(int device, int width, int height, int targetFps, int iterations) {
    std::vector<BenchResult> results;

    if (width <= 0 || height <= 0 || iterations <= 0)
        return results;

    if (width > 7680 || height > 4320) {
        BenchResult r{};
        r.suite_name = "av1_decode";
        r.test_name  = "dimension_exceeded";
        r.unit       = "fps";
        r.metadata["codec"] = "av1";
        r.metadata["resolution"] = std::to_string(width) + "x" + std::to_string(height);
        r.metadata["decoder"] = "NVDEC";
        r.metadata["stub_reason"] = "resolution exceeds 7680x4320";
        r.params_json = "{\"error\":\"resolution exceeds 7680x4320\"}";
        results.push_back(r);
        return results;
    }

    int dCount = 0;
    if (cudaGetDeviceCount(&dCount) != cudaSuccess || device >= dCount) {
        BenchResult r{};
        r.suite_name = "av1_decode";
        r.test_name  = "device_unavailable";
        r.unit       = "fps";
        r.metadata["codec"] = "av1";
        r.metadata["decoder"] = "NVDEC";
        r.metadata["stub_reason"] = "cuda device unavailable";
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
        r.suite_name   = "av1_decode";
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
               << ",\"nvdec_available\":" << (nvdec ? "true" : "false")
               << ",\"codec\":\"av1\"}";
            r.params_json = ps.str();
        }
        r.metadata["codec"] = "av1";
        r.metadata["resolution"] = std::to_string(c.w) + "x" + std::to_string(c.h);
        r.metadata["decoder"] = "NVDEC";
        results.push_back(std::move(r));
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(av1_decode, "NVDEC AV1 decoding",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runAV1DecodeBench(0, 1920, 1080, 60, 10);
    });
