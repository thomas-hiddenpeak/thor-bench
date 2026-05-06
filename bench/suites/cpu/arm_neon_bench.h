#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runARMNEONBench(int threadCount, int iterations);

}
