#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runTMEMBench(int device, int matDim, int iterations);

}
