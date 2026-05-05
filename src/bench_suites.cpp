#include "bench_suites.h"
#include <algorithm>

namespace deusridet::bench {

BenchSuiteRegistry& BenchSuiteRegistry::instance() {
    static BenchSuiteRegistry inst;
    return inst;
}

void BenchSuiteRegistry::registerSuite(BenchSuite suite) {
    suites_.push_back(std::move(suite));
}

std::vector<BenchSuite>& BenchSuiteRegistry::allSuites() {
    std::sort(suites_.begin(), suites_.end(),
        [](const BenchSuite& a, const BenchSuite& b) { return a.name < b.name; });
    return suites_;
}

std::vector<BenchSuite>& BenchSuiteRegistry::filteredSuites(const std::vector<std::string>& names) {
    if (names.empty()) return allSuites();
    suites_.erase(
        std::remove_if(suites_.begin(), suites_.end(),
            [&](const BenchSuite& s) {
                return std::find(names.begin(), names.end(), s.name) == names.end();
            }),
        suites_.end());
    return suites_;
}

BenchSuiteRegistrar::BenchSuiteRegistrar(BenchSuite suite) {
    BenchSuiteRegistry::instance().registerSuite(std::move(suite));
}

} // namespace deusridet::bench
