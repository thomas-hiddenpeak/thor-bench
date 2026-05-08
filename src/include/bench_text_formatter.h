#pragma once

#include "bench_schema.h"
#include "sweep_schema.h"
#include <string>
#include <vector>

namespace deusridet::bench {

std::string formatText(const BenchReport& report);
std::string formatText(const SweepReport& report);
std::string formatText(const std::vector<SweepReport>& reports);

}
