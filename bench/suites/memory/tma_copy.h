#pragma once

#include <vector>
#include <cstddef>

namespace deusridet::bench {

struct BenchResult;

std::vector<BenchResult> runTMACopyBench(int device, size_t transferSize, int iterations);

}
