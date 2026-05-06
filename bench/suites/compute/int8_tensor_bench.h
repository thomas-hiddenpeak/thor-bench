#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

// INT8 Tensor Core benchmark via tcgen05.mma.kind::i8 inline PTX.
// Real implementation: 16x16x16 tiles, SMEM descriptors, TMEM accumulation.
std::vector<BenchResult> runINT8TensorBench(int device, int matDim, int iterations);

} // namespace deusridet::bench
