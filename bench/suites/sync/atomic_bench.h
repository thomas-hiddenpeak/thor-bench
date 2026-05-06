#pragma once

#include <vector>
#include <cstddef>

namespace deusridet::bench {

struct BenchResult;

std::vector<BenchResult> runAtomicBench(int device, int iterations);

}
