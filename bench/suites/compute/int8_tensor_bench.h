#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

// INT8 Tensor Core benchmark (tcgen05.mma via nvcuda::wmma).
// Stub: nvcuda::wmma INT8 fragments are incomplete in CUDA 13.0
// for SM110a; tcgen05.mma kind::i8 requires descriptor-based PTX + SMEM.
std::vector<BenchResult> runINT8TensorBench(int device, int matDim, int iterations);

} // namespace deusridet::bench
