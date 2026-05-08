#pragma once

#include "bench_suites.h"
#include "sweep_schema.h"

#include <chrono>
#include <vector>

namespace deusridet::bench {

class SweepRunner {
public:
    SweepRunner() = default;

    SweepRunner& warmup(int n);
    SweepRunner& iterations(int n);
    SweepRunner& timeout(std::chrono::milliseconds ms);

    SweepReport run(const BenchSweepSuite& suite, int device);

    std::vector<SweepReport> runMultiple(
        const std::vector<BenchSweepSuite>& suites, int device);

private:
    int warmup_ = 3;
    int iterations_ = 10;
    std::chrono::milliseconds timeout_ = std::chrono::minutes(5);
};

} // namespace deusridet::bench
