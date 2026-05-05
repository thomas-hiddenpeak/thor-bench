#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

std::vector<BenchResult> runAV1DecodeBench(int device, int width, int height, int targetFps, int iterations);

}
