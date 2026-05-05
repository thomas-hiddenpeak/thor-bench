#pragma once

#include <vector>
#include <cstddef>

namespace deusridet::bench {

struct BenchResult;

std::vector<BenchResult> runClusterSyncBench(int device, int iterations);

}
