#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runNVJPEGBench(int device, int imageWidth, int imageHeight, int iterations);

}
