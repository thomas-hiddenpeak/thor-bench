#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runTegraMemoryBench(int device, size_t transferSize, int iterations);

}
