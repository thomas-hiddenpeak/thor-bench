#include "encode/nvjpeg_bench.h"
#include "bench_suites.h"
#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <vector>

namespace deusridet::bench {

namespace {

BenchResult makeStub(const std::string& testName, const std::string& note) {
    BenchResult r;
    r.suite_name = "nvjpeg";
    r.test_name  = testName;
    r.unit       = "images/s";
    r.sample_count = 0;
    r.warmup_count = 0;
    r.median = 0.0;
    std::ostringstream p;
    p << "{\"note\":\"" << note << "\"}";
    r.params_json = p.str();
    r.metadata["stub"] = "true";
    r.metadata["stub_reason"] = "nvjpeg not available on Tegra";
    return r;
}

} // anonymous namespace

std::vector<BenchResult> runNVJPEGBench(int device, int imageWidth, int imageHeight, int iterations) {
    std::vector<BenchResult> results;

    (void)device;
    (void)imageWidth;
    (void)imageHeight;
    (void)iterations;

    // NVJPEG is not available on Tegra/Thor per CUDA for Tegra AppNote constraints.
    results.push_back(makeStub("nvjpeg_encode", "nvjpeg not available on Tegra"));
    results.push_back(makeStub("nvjpeg_decode", "nvjpeg not available on Tegra"));

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(nvjpeg, "NVJPEG encode/decode (stub — not available on Tegra)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runNVJPEGBench(0, 1920, 1080, 10);
    });
