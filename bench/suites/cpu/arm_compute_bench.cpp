#include "arm_compute_bench.h"
#include "bench_suites.h"
#include <cmath>
#include <limits>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <functional>
#include <sstream>

namespace deusridet::bench {

namespace {

double sineBench(int iters) {
    double sum = 0.0;
    for (int i = 0; i < iters; ++i)
        sum += std::sin(static_cast<double>(i)) * std::cos(static_cast<double>(i));
    return sum;
}

double sqrtBench(int iters) {
    double sum = 0.0;
    for (int i = 1; i <= iters; ++i)
        sum += std::sqrt(static_cast<double>(i));
    return sum;
}

double matMulBench(int size) {
    std::vector<double> a(size * size, 0.0);
    std::vector<double> b(size * size, 0.0);
    std::vector<double> c(size * size, 0.0);

    for (int i = 0; i < size * size; ++i) {
        a[i] = static_cast<double>(i % 100) / 100.0;
        b[i] = static_cast<double>((i + 1) % 100) / 100.0;
    }

    for (int i = 0; i < size; ++i)
        for (int k = 0; k < size; ++k)
            for (int j = 0; j < size; ++j)
                c[i * size + j] += a[i * size + k] * b[k * size + j];

    return c[0];
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
    std::function<double(int)> workload;
    double workPerCall;
    std::string unit;
};

}

std::vector<BenchResult> runArmComputeBench(int threadCount, int iterations) {
    std::vector<BenchResult> results;

    if (threadCount <= 0)
        threadCount = static_cast<int>(std::thread::hardware_concurrency());
    if (threadCount <= 0 || iterations <= 0)
        return results;

    std::vector<Case> cases = {
        {"sine_loop",     [](int n) { return sineBench(n); },   1e6, "gflops"},
        {"sqrt_loop",     [](int n) { return sqrtBench(n); },   1e6, "gflops"},
        {"matmul_64",     [](int)  { return matMulBench(64); },  64*64*64*2.0, "gflops"},
        {"matmul_128",    [](int)  { return matMulBench(128); }, 128*128*128*2.0, "gflops"},
        {"matmul_256",    [](int)  { return matMulBench(256); }, 256*256*256*2.0, "gflops"},
    };

    for (const auto& c : cases) {
        std::vector<double> vals;
        vals.reserve(iterations);

        for (int iter = 0; iter < iterations; ++iter) {
            double value;

            if (threadCount > 1 && c.label.substr(0, 7) == "matmul_") {
                std::atomic<bool> done{false};
                std::vector<std::thread> threads;
                threads.reserve(threadCount);

                auto t0 = std::chrono::steady_clock::now();
                for (int t = 0; t < threadCount; ++t)
                    threads.emplace_back([&c, &done]() {
                        while (!done.load()) c.workload(0);
                    });

                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                done.store(true);
                for (auto& th : threads) th.join();
                auto t1 = std::chrono::steady_clock::now();

                double sec = std::chrono::duration<double>(t1 - t0).count();
                if (sec > 0.0)
                    value = c.workPerCall * threadCount / sec / 1e9;
                else
                    value = std::numeric_limits<double>::max();
            } else {
                auto t0 = std::chrono::steady_clock::now();
                double dummy = c.workload(1000000);
                auto t1 = std::chrono::steady_clock::now();

                double sec = std::chrono::duration<double>(t1 - t0).count();
                if (sec > 0.0)
                    value = c.workPerCall / sec / 1e9;
                else
                    value = std::numeric_limits<double>::max();
                (void)dummy;
            }
            vals.push_back(value);
        }

        std::sort(vals.begin(), vals.end());
        int n = static_cast<int>(vals.size());
        double sum = 0.0;
        for (double v : vals) sum += v;
        double mean = sum / n;
        double sq = 0.0;
        for (double v : vals) sq += (v - mean) * (v - mean);

        BenchResult r{};
        r.suite_name   = "arm_compute";
        r.test_name    = c.label;
        r.unit         = c.unit;
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
            ps << "{\"threads\":" << threadCount << ",\"iterations\":" << iterations << "}";
            r.params_json = ps.str();
        }
        results.push_back(std::move(r));
    }

    return results;
}

} // anonymous namespace

BENCH_REGISTER_SUITE(arm_compute, "ARM CPU FP32 NEON/SVE throughput",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runArmComputeBench(4, 10);
    });
