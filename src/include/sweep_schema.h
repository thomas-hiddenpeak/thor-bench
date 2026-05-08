#pragma once

#include <string>
#include <vector>
#include <map>
#include <variant>
#include <optional>

#include "bench_schema.h"

namespace deusridet::bench {

struct SweepParams {
    std::string name;
    std::vector<std::variant<int, double, std::string>> values;
};

struct SweepPoint {
    std::map<std::string, std::string> param_values;
};

struct SweepResult {
    std::string suite_name;
    std::string test_name;
    std::string params_json;
    BenchResult result;
    std::optional<double> power_watts;
    std::optional<std::string> error_message;
    std::string timestamp;
};

struct SweepReport {
    std::string suite_name;
    std::string description;
    std::vector<std::string> param_names;
    std::vector<SweepResult> points;
    std::string sweep_timestamp;
    int total_points = 0;
    int success_points = 0;
    int error_points = 0;
};

}
