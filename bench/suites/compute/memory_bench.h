#pragma once

#include <chrono>
#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runMemoryBench(int device, size_t transferSize, int iterations);

}
