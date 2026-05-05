#pragma once

#include <vector>
#include "bench_schema.h"

namespace deusridet::bench {

// FP4 e2m1 Tensor Core dense benchmark via tcgen05.mma inline PTX.
// Uses kind::mxf4nvf4 (NVFP4 block-scaled) with raw PTX descriptors.
// Returns results with peak_pct vs T5000Peaks::fp4_dense_tflops (1035 TFLOPS).
std::vector<BenchResult> runFP4Bench(int device, int matDim, int iterations);

}
