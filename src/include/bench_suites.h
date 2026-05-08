#pragma once

#include "bench_schema.h"
#include "bench_runner.h"
#include "sweep_schema.h"
#include <vector>
#include <string>
#include <functional>

namespace deusridet::bench {

struct BenchSuite {
    std::string name;
    std::string description;
    std::function<std::vector<BenchResult>(BenchRunner&)> runFn;
};

struct BenchSweepSuite {
    std::string name;
    std::string description;
    std::vector<SweepParams> params;
    std::function<std::vector<SweepReport>(BenchRunner&, int device)> runFn;
};

class BenchSuiteRegistry {
public:
    static BenchSuiteRegistry& instance();

    void registerSuite(BenchSuite suite);
    std::vector<BenchSuite>& allSuites();
    std::vector<BenchSuite>& filteredSuites(const std::vector<std::string>& names);

    void registerSweepSuite(BenchSweepSuite suite);
    std::vector<BenchSweepSuite>& allSweepSuites();
    std::vector<BenchSweepSuite>& filteredSweepSuites(const std::vector<std::string>& names);

private:
    std::vector<BenchSuite> suites_;
    std::vector<BenchSweepSuite> sweepSuites_;
    BenchSuiteRegistry() = default;
};

class BenchSuiteRegistrar {
public:
    BenchSuiteRegistrar(BenchSuite suite);
};

#define BENCH_REGISTER_SUITE(name_, desc_, fn_) \
    static deusridet::bench::BenchSuiteRegistrar _bench_reg_##name_( \
        deusridet::bench::BenchSuite{#name_, desc_, fn_})

class BenchSweepSuiteRegistrar {
public:
    BenchSweepSuiteRegistrar(BenchSweepSuite suite);
};

#define BENCH_REGISTER_SWEEP_SUITE(name_, desc_, fn_) \
    static deusridet::bench::BenchSweepSuiteRegistrar _sweep_bench_reg_##name_( \
        deusridet::bench::BenchSweepSuite{#name_, desc_, {}, fn_})

}
