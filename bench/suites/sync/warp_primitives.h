#pragma once

#include <vector>
#include <cstddef>

namespace deusridet::bench {

struct BenchResult;

std::vector<BenchResult> runWarpPrimitivesBench(int device, int iterations);

}
