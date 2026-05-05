#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <optional>
#include <limits>

namespace deusridet::bench {

struct Measurement {
    std::chrono::nanoseconds duration;
    double throughput;
    double latency_us;
    std::string unit;
};

struct BenchResult {
    std::string suite_name;
    std::string test_name;
    std::string unit;

    double median = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    std::optional<double> score;
    std::optional<double> peak_pct;

    int sample_count = 0;
    int warmup_count = 0;
    std::string params_json;
    std::string probe_snapshot;
    std::map<std::string, std::string> metadata;
};

struct BenchReport {
    std::string version = "0.1.0";
    std::string timestamp;
    std::string hostname;
    std::vector<BenchResult> results;

    std::optional<double> overall_score;
};

}
