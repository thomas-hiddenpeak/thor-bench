#include "bench_runner.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace deusridet::bench {

BenchRunner& BenchRunner::warmup(int count) {
    if (count < 0) count = 0;
    warmupCount_ = count;
    return *this;
}

BenchRunner& BenchRunner::iterations(int count) {
    if (count < 1) count = 1;
    iterationCount_ = count;
    return *this;
}

BenchRunner& BenchRunner::timeout(std::chrono::milliseconds ms) {
    if (ms.count() > 0)
        timeout_ = ms;
    return *this;
}

BenchRunner& BenchRunner::unit(const std::string& u) {
    unit_ = u;
    return *this;
}

BenchResult BenchRunner::run(
    const std::string& suite_name,
    const std::string& test_name,
    std::function<void()> fn,
    std::function<double(std::chrono::nanoseconds)> valueExtractor
) {
    BenchResult result;
    result.suite_name = suite_name;
    result.test_name  = test_name;
    result.unit       = unit_;
    result.warmup_count = warmupCount_;

    for (int i = 0; i < warmupCount_; ++i) {
        try {
            fn();
        } catch (const std::exception&) {
        } catch (...) {
        }
    }

    std::vector<double> values;
    auto deadline = std::chrono::steady_clock::now() + timeout_;

    auto fill = [this](std::vector<double>& vals, BenchResult& res) {
        int n = static_cast<int>(vals.size());
        res.sample_count = n;
        if (n == 0) return;
        std::sort(vals.begin(), vals.end());
        double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        double mean = sum / n;
        res.min_val  = vals.front();
        res.max_val  = vals.back();
        res.mean     = mean;
        res.median   = median(vals);
        res.stddev   = stddev(vals, mean);
        res.p95      = percentile(vals, 0.95);
        res.p99      = percentile(vals, 0.99);
    };

    for (int i = 0; i < iterationCount_; ++i) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }

        auto t0 = std::chrono::steady_clock::now();
        try {
            fn();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA kernel error: ") +
                                         cudaGetErrorString(err));
            }
        } catch (const std::exception& e) {
            result.params_json = std::string{"{\"error\":\""} + e.what() + "\"}";
            fill(values, result);
            return result;
        } catch (...) {
            result.params_json = "{\"error\":\"unknown_exception\"}";
            fill(values, result);
            return result;
        }
        auto t1 = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
        double value = valueExtractor(duration);
        values.push_back(value);
    }

    fill(values, result);
    return result;
}

BenchResult BenchRunner::runThroughput(
    const std::string& suite_name,
    const std::string& test_name,
    std::function<void()> fn,
    double workPerCall
) {
    auto valueExtractor = [workPerCall](std::chrono::nanoseconds duration) -> double {
        double seconds = duration.count() / 1e9;
        if (seconds <= 0.0)
            return -1.0; // zero-duration sentinel (was DBL_MAX)
        return workPerCall / seconds;
    };

    return run(suite_name, test_name, fn, valueExtractor);
}

double BenchRunner::percentile(std::vector<double>& values, double p) {
    int n = static_cast<int>(values.size());
    if (n == 0)
        return 0.0;
    if (n == 1)
        return values[0];

    double rank = p * (n - 1);
    int    lo   = static_cast<int>(std::floor(rank));
    int    hi   = static_cast<int>(std::ceil(rank));
    double frac = rank - lo;

    if (hi >= n)
        return values.back();

    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

double BenchRunner::median(std::vector<double>& values) {
    int n = static_cast<int>(values.size());
    if (n == 0)
        return 0.0;
    if (n % 2 == 1)
        return values[n / 2];
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

double BenchRunner::stddev(const std::vector<double>& values, double mean) {
    int n = static_cast<int>(values.size());
    if (n <= 1)
        return 0.0;

    double sq_sum = 0.0;
    for (const double v : values) {
        double diff = v - mean;
        sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum / n);
}

} // namespace deusridet::bench
