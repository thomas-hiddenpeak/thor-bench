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
    // Copy all suites first, then filter in-place to avoid clearing
    // the reference that allSuites() returned.
    std::vector<BenchSuite> all = allSuites();
    suites_.clear();
    for (const auto& s : all) {
        if (std::find(names.begin(), names.end(), s.name) != names.end()) {
            suites_.push_back(s);
        }
    }
    return suites_;
}

BenchSuiteRegistrar::BenchSuiteRegistrar(BenchSuite suite) {
    BenchSuiteRegistry::instance().registerSuite(std::move(suite));
}

} // namespace deusridet::bench
