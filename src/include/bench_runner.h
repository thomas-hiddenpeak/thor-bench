#pragma once

#include "bench_schema.h"
#include <functional>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace deusridet::bench {

class BenchRunner {
public:
    BenchRunner() = default;

    BenchRunner& warmup(int count);
    BenchRunner& iterations(int count);
    BenchRunner& timeout(std::chrono::milliseconds ms);
    BenchRunner& unit(const std::string& u);

    BenchResult run(
        const std::string& suite_name,
        const std::string& test_name,
        std::function<void()> fn,
        std::function<double(std::chrono::nanoseconds)> valueExtractor
    );

    BenchResult runThroughput(
        const std::string& suite_name,
        const std::string& test_name,
        std::function<void()> fn,
        double workPerCall
    );

private:
    int warmupCount_ = 3;
    int iterationCount_ = 10;
    std::chrono::milliseconds timeout_ = std::chrono::minutes(5);
    std::string unit_ = "ops/s";

    std::vector<double> computeStats(std::vector<double> values);
    double percentile(std::vector<double>& values, double p);
    double median(std::vector<double>& values);
    double stddev(const std::vector<double>& values, double mean);
};

}
