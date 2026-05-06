#include "compute/int8_tensor_bench.h"
#include "bench_schema.h"
#include "bench_stats.h"
#include "bench_suites.h"
#include "bench_peaks.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace deusridet::bench {

namespace {

// ── Stub result builder ────────────────────────────────────────────────────
// nvcuda::wmma INT8 fragments (kind::i8) are incomplete in CUDA 13.0 for SM110a.
// tcgen05.mma kind::i8 requires descriptor-based PTX + SMEM layouts not yet
// exposed in CUDA 13.0 WMMA API. Return stub results for both dense and sparse.
BenchResult makeStub(const std::string& testName) {
    BenchResult res{};
    res.suite_name = "int8_tensor";
    res.test_name  = testName;
    res.unit       = "TOP/s";
    res.sample_count = 0;
    res.warmup_count = 0;
    std::ostringstream p;
    p << "{\"M\":256,\"N\":256,\"K\":256,\"tile\":\"16x16x16\"}";
    res.params_json = p.str();
    res.metadata["tcgen05"] = "true";
    res.metadata["precision"] = "int8";
    res.metadata["stub"] = "true";
    res.metadata["stub_reason"] = "nvcuda::wmma INT8 fragments incomplete in CUDA 13.0; tcgen05.mma kind::i8 requires descriptor-based PTX + SMEM";
    res.peak_pct = 0.0; // stub — no measurable result
    return res;
}

} // anonymous namespace

std::vector<BenchResult> runINT8TensorBench(int device, int matDim, int iterations) {
    std::vector<BenchResult> results;

    // nvcuda::wmma INT8 fragments are not available in CUDA 13.0 for SM110a.
    // The header includes mma.h which defines the wmma namespace, but kind::i8
    // fragment specializations are incomplete — they require tcgen05.mma inline
    // PTX with SMEM descriptor layouts that are not yet exposed in the CUDA 13.0
    // WMMA API. Return stub results for both dense and sparse tests.
    (void)device; (void)matDim; (void)iterations;

    results.push_back(makeStub("int8_tensor_dense"));
    results.push_back(makeStub("int8_tensor_sparse"));

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(int8_tensor, "INT8 Tensor Core throughput (tcgen05)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        try {
            return deusridet::bench::runINT8TensorBench(0, 256, 10);
        } catch (const std::exception& ex) {
            std::vector<deusridet::bench::BenchResult> fallback;
            deusridet::bench::BenchResult r{};
            r.suite_name = "int8_tensor";
            r.test_name  = "int8_tensor_dense";
            r.unit       = "TOP/s";
            r.params_json = std::string("{\"error\":\"") + ex.what() + "\"}";
            r.metadata["stub"] = "true";
            r.metadata["stub_reason"] = std::string("runtime error: ") + ex.what();
            r.peak_pct = 0.0;
            fallback.push_back(r);
            return fallback;
        }
    });
