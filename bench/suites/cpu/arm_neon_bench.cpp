#include "arm_neon_bench.h"
#include "bench_suites.h"
#include <arm_neon.h>
#include <cmath>
#include <limits>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <sstream>

namespace deusridet::bench {

namespace {

constexpr int kWarmupRounds = 3;
constexpr int kIterations = 10;
constexpr int kDataLen = 1 << 20;

static std::vector<float> g_a, g_b, g_c;
static std::vector<int8_t> g_i8_a, g_i8_b;

static void initData() {
    g_a.resize(kDataLen);
    g_b.resize(kDataLen);
    g_c.resize(kDataLen);
    for (int i = 0; i < kDataLen; ++i) {
        g_a[i] = 0.5f + (i & 0xFF) / 255.0f;
        g_b[i] = 0.3f + ((i >> 3) & 0xFF) / 511.0f;
        g_c[i] = 0.7f + ((i >> 7) & 0xFF) / 255.0f;
    }
    g_i8_a.resize(kDataLen);
    g_i8_b.resize(kDataLen);
    for (int i = 0; i < kDataLen; ++i) {
        g_i8_a[i] = (int8_t)(i & 0x3F);
        g_i8_b[i] = (int8_t)((i >> 1) & 0x3F);
    }
}

static float run_fp32() {
    float32x4_t acc = vdupq_n_f32(0.0f);
    const float* __restrict__ a = g_a.data();
    const float* __restrict__ b = g_b.data();
    const float* __restrict__ c = g_c.data();
    for (int i = 0; i < kDataLen; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        acc = vmlaq_f32(acc, va, vc);
        acc = vmlaq_f32(acc, vb, vc);
    }
    return vaddvq_f32(acc);
}

static float run_fp16() {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    const float* __restrict__ a = g_a.data();
    const float* __restrict__ b = g_b.data();
    const float* __restrict__ c = g_c.data();
    for (int i = 0; i + 8 < kDataLen; i += 8) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        acc0 = vmlaq_f32(acc0, va, vc);
        acc0 = vmlaq_f32(acc0, vb, vc);

        float32x4_t va2 = vld1q_f32(a + i + 4);
        float32x4_t vb2 = vld1q_f32(b + i + 4);
        float32x4_t vc2 = vld1q_f32(c + i + 4);
        acc1 = vmlaq_f32(acc1, va2, vc2);
        acc1 = vmlaq_f32(acc1, vb2, vc2);
    }
    return vaddvq_f32(acc0) + vaddvq_f32(acc1);
}

static int run_int8() {
    int sum = 0;
    const int8_t* __restrict__ a = g_i8_a.data();
    const int8_t* __restrict__ b = g_i8_b.data();
    for (int i = 0; i + 16 <= kDataLen; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        int16x8_t lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t hi = vmull_high_s8(va, vb);
        int32x4_t lo32 = vpaddlq_s16(lo);
        int32x4_t hi32 = vpaddlq_s16(hi);
        sum += vaddlvq_s32(vaddq_s32(lo32, hi32));
    }
    return sum;
}

static BenchResult makeResult(
    const std::string& test_name,
    std::vector<double>& samples,
    const std::string& unit)
{
    int n = static_cast<int>(samples.size());
    std::sort(samples.begin(), samples.end());

    double sum = 0.0;
    for (double v : samples) sum += v;
    double mean = sum / n;

    double sq = 0.0;
    for (double v : samples) sq += (v - mean) * (v - mean);

    int mid = n / 2;
    double median = (n % 2 == 1) ? samples[mid] : (samples[mid - 1] + samples[mid]) / 2.0;

    auto percentile = [&](double p) {
        double rank = p * (n - 1);
        int lo = static_cast<int>(std::floor(rank));
        int hi = static_cast<int>(std::ceil(rank));
        double frac = rank - lo;
        if (hi >= n) return samples.back();
        return samples[lo] * (1.0 - frac) + samples[hi] * frac;
    };

    BenchResult r{};
    r.suite_name   = "arm_neon";
    r.test_name    = test_name;
    r.unit         = unit;
    r.median       = median;
    r.mean         = mean;
    r.stddev       = std::sqrt(sq / n);
    r.min_val      = samples.front();
    r.max_val      = samples.back();
    r.p95          = percentile(0.95);
    r.p99          = percentile(0.99);
    r.sample_count = n;
    r.warmup_count = kWarmupRounds;
    r.params_json  = R"({"note":"using NEON fallback, SVE2 intrinsics unavailable","neon_width":128,"data_len":1048576})";
    r.metadata["neon_fallback"] = "true";
    r.metadata["sve2_available"] = "false";
    return r;
}

static BenchResult runFTest(const std::string& test_name, double flopsPerCall) {
    int callsPerIter = 100;

    for (int w = 0; w < kWarmupRounds; ++w) {
        volatile float sink = 0;
        for (int j = 0; j < callsPerIter; ++j) sink = run_fp32();
        sink = sink;
    }

    std::vector<double> samples;
    samples.reserve(kIterations);
    for (int i = 0; i < kIterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        volatile float sink = 0;
        for (int j = 0; j < callsPerIter; ++j) sink = run_fp32();
        asm volatile("" : "+r"(sink) : : "memory");
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        samples.push_back(flopsPerCall * callsPerIter / (secs * 1e9));
    }
    return makeResult(test_name, samples, "GFLOP/s");
}

static BenchResult runF16Test(const std::string& test_name, double flopsPerCall) {
    int callsPerIter = 100;

    for (int w = 0; w < kWarmupRounds; ++w) {
        volatile float sink = 0;
        for (int j = 0; j < callsPerIter; ++j) sink = run_fp16();
        sink = sink;
    }

    std::vector<double> samples;
    samples.reserve(kIterations);
    for (int i = 0; i < kIterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        volatile float sink = 0;
        for (int j = 0; j < callsPerIter; ++j) sink = run_fp16();
        asm volatile("" : "+r"(sink) : : "memory");
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        samples.push_back(flopsPerCall * callsPerIter / (secs * 1e9));
    }
    return makeResult(test_name, samples, "GFLOP/s");
}

static BenchResult runI8Test(const std::string& test_name, double opsPerCall) {
    int callsPerIter = 100;

    for (int w = 0; w < kWarmupRounds; ++w) {
        volatile int sink = 0;
        for (int j = 0; j < callsPerIter; ++j) sink = run_int8();
        sink = sink;
    }

    std::vector<double> samples;
    samples.reserve(kIterations);
    for (int i = 0; i < kIterations; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        volatile int sink = 0;
        for (int j = 0; j < callsPerIter; ++j) sink = run_int8();
        asm volatile("" : "+r"(sink) : : "memory");
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        samples.push_back(opsPerCall * callsPerIter / (secs * 1e12));
    }
    return makeResult(test_name, samples, "TOP/s");
}

}

std::vector<BenchResult> runARMNEONBench(int, int) {
    std::vector<BenchResult> results;
    initData();

    // FP32: 2× vmlaq_f32 per 4 elements = 16 FLOPs per 4 elements = 4 FLOPs/element
    double fp32_flops = kDataLen * 4.0;
    // FP16: 2× vmlaq_f32 × 2 lanes per 8 elements = 32 FLOPs per 8 elements = 4 FLOPs/element
    double fp16_flops = (kDataLen / 8.0) * 32.0;
    // INT8: 16 mults per 16 elements = 1 op/element
    double int8_ops = kDataLen;

    results.push_back(runFTest("neon_fp32", fp32_flops));
    results.push_back(runF16Test("neon_fp16", fp16_flops));
    results.push_back(runI8Test("neon_int8", int8_ops));

    return results;
}

}

BENCH_REGISTER_SUITE(arm_neon, "ARM CPU NEON vector throughput (NEON fallback, SVE2 intrinsics unavailable)",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runARMNEONBench(4, 10);
    });
