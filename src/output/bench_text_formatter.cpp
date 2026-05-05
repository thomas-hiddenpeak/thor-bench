#include "bench_text_formatter.h"
#include "bench_schema.h"

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cmath>

namespace deusridet::bench {

namespace {

constexpr char RESET[]    = "\033[0m";
constexpr char BOLD[]     = "\033[1m";
constexpr char GREEN[]    = "\033[32m";
constexpr char YELLOW[]   = "\033[33m";
constexpr char CYAN[]     = "\033[36m";
constexpr char DIM[]      = "\033[2m";
constexpr char RED[]      = "\033[31m";

std::string separator(char c = '=') {
    std::string s(60, c);
    return s;
}

std::string score_tag(std::optional<double> score) {
    if (!score) return {};
    if (*score >= 0.8) return std::string(GREEN) + " PASS " + RESET;
    if (*score >= 0.5) return std::string(YELLOW) + " WARN " + RESET;
    return std::string(RED) + " FAIL " + RESET;
}

} // anonymous namespace

std::string formatText(const BenchReport& report) {
    std::ostringstream os;

    // ── header ──
    os << BOLD << "Thor Bench v" << report.version << RESET << "\n";
    os << separator('=') << "\n";
    if (!report.hostname.empty())
        os << DIM << "  Host    " << RESET << " " << report.hostname << "\n";
    if (!report.timestamp.empty())
        os << DIM << "  Time    " << RESET << " " << report.timestamp << "\n";
    os << "\n";

    // ── group results by suite name ──
    std::vector<const BenchResult*> grouped;
    std::string current_suite;

    for (const auto& r : report.results) {
        // Insert separator when suite changes
        if (!current_suite.empty() && r.suite_name != current_suite)
            grouped.push_back(nullptr); // sentinel for section break
        current_suite = r.suite_name;
        grouped.push_back(&r);
    }

    for (const auto* ptr : grouped) {
        if (!ptr) {
            os << "\n";
            continue;
        }

        // ── suite header ──
        static std::string last_suite;
        if (ptr->suite_name != last_suite) {
            last_suite = ptr->suite_name;
            os << BOLD << CYAN << "  " << ptr->suite_name << RESET << "\n";
            os << "  " << separator('-') << "\n";
        }

        // ── test row ──
        os << "    " << BOLD << ptr->test_name << RESET;

        // score badge
        if (ptr->score) {
            os << "  " << score_tag(ptr->score);
        }
        if (ptr->peak_pct) {
            os << "  " << DIM << "peak=" << RESET << std::fixed << std::setprecision(1) << *ptr->peak_pct << DIM << "%" << RESET;
        }
        os << "\n";

        // stats line
        os << "        "
           << "mean=" << std::fixed << std::setprecision(2) << ptr->mean
           << "  median=" << ptr->median
           << "  stddev=" << ptr->stddev
           << "  p95=" << ptr->p95
           << "  p99=" << ptr->p99
           << " " << ptr->unit << "\n";

        // range line
        os << "        "
           << "min=" << ptr->min_val
           << "  max=" << ptr->max_val
           << "  samples=" << ptr->sample_count
           << "  warmup=" << ptr->warmup_count
           << "\n";

        // params (if present)
        if (!ptr->params_json.empty()) {
            os << DIM << "        params: " << RESET << ptr->params_json << "\n";
        }

        // metadata (if present)
        if (!ptr->metadata.empty()) {
            os << DIM << "        metadata:" << RESET;
            for (const auto& [k, v] : ptr->metadata) {
                os << " " << k << "=" << v;
            }
            os << "\n";
        }
    }

    // ── footer ──
    os << "\n" << separator('=') << "\n";
    if (report.overall_score) {
        os << "  Overall Score: " << BOLD
           << std::fixed << std::setprecision(1) << *report.overall_score << RESET
           << "\n";
    }
    os << "  " << GREEN << report.results.size() << RESET << " tests completed.\n";

    return os.str();
}

} // namespace deusridet::bench
