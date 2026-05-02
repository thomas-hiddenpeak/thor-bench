#pragma once

#include "bench_schema.h"
#include "bench_runner.h"
#include <vector>
#include <string>
#include <functional>

namespace deusridet::bench {

struct BenchSuite {
    std::string name;
    std::string description;
    std::function<void(BenchRunner&, BenchResult&)> runFn;
};

class BenchSuiteRegistry {
public:
    static BenchSuiteRegistry& instance();

    void registerSuite(BenchSuite suite);
    std::vector<BenchSuite>& allSuites();
    std::vector<BenchSuite>& filteredSuites(const std::vector<std::string>& names);

private:
    std::vector<BenchSuite> suites_;
    BenchSuiteRegistry() = default;
};

class BenchSuiteRegistrar {
public:
    BenchSuiteRegistrar(BenchSuite suite);
};

#define BENCH_REGISTER_SUITE(name_, desc_, fn_) \
    static deusridet::bench::BenchSuiteRegistrar _bench_reg_##name_( \
        deusridet::bench::BenchSuite{#name_, desc_, fn_})

}
