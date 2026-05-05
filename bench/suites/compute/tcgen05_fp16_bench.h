#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runTCGen05FP16Bench(int device, int matDim, int iterations);

}
