#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runINT8TensorBench(int device, int matDim, int iterations);

}
