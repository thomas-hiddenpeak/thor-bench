#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runArmComputeBench(int threadCount, int iterations);

}
