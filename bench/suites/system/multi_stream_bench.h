#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runMultiStreamBench(int device, int bufferSize, int iterations);

}
