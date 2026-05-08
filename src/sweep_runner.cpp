#include "sweep_runner.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace deusridet::bench {

namespace {

std::string getSweepTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf{};
    localtime_r(&time_t_now, &tm_buf);

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

} // namespace

SweepRunner& SweepRunner::warmup(int n) {
    if (n < 0) n = 0;
    warmup_ = n;
    return *this;
}

SweepRunner& SweepRunner::iterations(int n) {
    if (n < 1) n = 1;
    iterations_ = n;
    return *this;
}

SweepRunner& SweepRunner::timeout(std::chrono::milliseconds ms) {
    if (ms.count() > 0)
        timeout_ = ms;
    return *this;
}

SweepReport SweepRunner::run(const BenchSweepSuite& suite, int device) {
    SweepReport report;
    report.suite_name    = suite.name;
    report.description   = suite.description;
    report.sweep_timestamp = getSweepTimestamp();

    for (const auto& p : suite.params) {
        report.param_names.push_back(p.name);
    }

    BenchRunner runner;
    runner.warmup(warmup_);
    runner.iterations(iterations_);
    runner.timeout(timeout_);

    std::cerr << "[sweep] Running: " << suite.name
              << " (" << suite.description << ")\n";

    try {
        auto results = suite.runFn(runner, device);

        for (auto& sr : results) {
            for (const auto& pn : sr.param_names) {
                if (std::find(report.param_names.begin(), report.param_names.end(), pn)
                    == report.param_names.end())
                {
                    report.param_names.push_back(pn);
                }
            }
            report.points.insert(report.points.end(),
                                 sr.points.begin(), sr.points.end());
        }
    } catch (const std::exception& e) {
        std::cerr << "[sweep] " << suite.name << " FAILED: " << e.what() << "\n";

        SweepResult errPoint;
        errPoint.suite_name    = suite.name;
        errPoint.test_name     = "error";
        errPoint.error_message = e.what();
        errPoint.timestamp     = getSweepTimestamp();
        report.points.push_back(errPoint);
    } catch (...) {
        std::cerr << "[sweep] " << suite.name
                  << " FAILED with unknown exception\n";

        SweepResult errPoint;
        errPoint.suite_name    = suite.name;
        errPoint.test_name     = "unknown_error";
        errPoint.error_message = "unknown_exception";
        errPoint.timestamp     = getSweepTimestamp();
        report.points.push_back(errPoint);
    }

    report.total_points   = static_cast<int>(report.points.size());
    report.success_points = 0;
    report.error_points   = 0;
    for (const auto& pt : report.points) {
        if (pt.error_message.has_value()) {
            ++report.error_points;
        } else {
            ++report.success_points;
        }
    }

    std::cerr << "[sweep] " << suite.name
              << " complete: " << report.success_points << "/"
              << report.total_points << " points OK\n";

    return report;
}

std::vector<SweepReport> SweepRunner::runMultiple(
    const std::vector<BenchSweepSuite>& suites, int device)
{
    std::vector<SweepReport> reports;
    reports.reserve(suites.size());

    for (const auto& suite : suites) {
        reports.push_back(run(suite, device));
    }

    return reports;
}

} // namespace deusridet::bench
