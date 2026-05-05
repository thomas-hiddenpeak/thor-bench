#pragma once

#include <vector>

namespace deusridet::bench {

struct BenchResult;

std::vector<BenchResult> runThermalThrottleBench(int device, int durationSec, int iterations);

}
