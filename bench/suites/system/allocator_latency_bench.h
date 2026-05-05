#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runAllocatorLatencyBench(int device, int iterations);

}
