#pragma once

#include "bench_schema.h"
#include "sweep_schema.h"
#include <string>
#include <vector>

namespace deusridet::bench {

std::string serializeJson(const BenchReport& report);
std::string serializeJson(const SweepResult& result);
std::string serializeJson(const SweepReport& report);
std::string serializeSweepJson(const std::vector<SweepReport>& reports, const std::string& hostname);

}
