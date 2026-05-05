#pragma once

#include <vector>
#include <cstddef>

namespace deusridet::bench {

struct BenchResult;

std::vector<BenchResult> runMBarrierBench(int device, int iterations);

}
