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
    std::vector<BenchSuite> filtered;
    std::vector<BenchSuite> all = allSuites();
    for (const auto& s : all) {
        if (std::find(names.begin(), names.end(), s.name) != names.end()) {
            filtered.push_back(s);
        }
    }
    suites_ = std::move(filtered);
    return suites_;
}

BenchSuiteRegistrar::BenchSuiteRegistrar(BenchSuite suite) {
    BenchSuiteRegistry::instance().registerSuite(std::move(suite));
}

void BenchSuiteRegistry::registerSweepSuite(BenchSweepSuite suite) {
    sweepSuites_.push_back(std::move(suite));
}

std::vector<BenchSweepSuite>& BenchSuiteRegistry::allSweepSuites() {
    std::sort(sweepSuites_.begin(), sweepSuites_.end(),
        [](const BenchSweepSuite& a, const BenchSweepSuite& b) { return a.name < b.name; });
    return sweepSuites_;
}

std::vector<BenchSweepSuite>& BenchSuiteRegistry::filteredSweepSuites(const std::vector<std::string>& names) {
    if (names.empty()) return allSweepSuites();
    std::vector<BenchSweepSuite> filtered;
    std::vector<BenchSweepSuite> all = allSweepSuites();
    for (const auto& s : all) {
        if (std::find(names.begin(), names.end(), s.name) != names.end()) {
            filtered.push_back(s);
        }
    }
    sweepSuites_ = std::move(filtered);
    return sweepSuites_;
}

BenchSweepSuiteRegistrar::BenchSweepSuiteRegistrar(BenchSweepSuite suite) {
    BenchSuiteRegistry::instance().registerSweepSuite(std::move(suite));
}

} // namespace deusridet::bench
