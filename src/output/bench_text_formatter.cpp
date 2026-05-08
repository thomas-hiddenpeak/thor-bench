#include "bench_text_formatter.h"

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

std::map<std::string, std::string> parseParamsJson(const std::string& json) {
    std::map<std::string, std::string> result;
    if (json.empty()) return result;

    size_t pos = 0;
    while (pos < json.size()) {
        size_t eq = json.find('"', pos);
        if (eq == std::string::npos) break;

        size_t keyStart = eq + 1;
        size_t keyEnd = json.find('"', keyStart);
        if (keyEnd == std::string::npos) break;
        std::string key = json.substr(keyStart, keyEnd - keyStart);

        size_t afterColon = keyEnd + 1;
        while (afterColon < json.size() && (json[afterColon] == ' ' || json[afterColon] == ':'))
            afterColon++;

        if (afterColon >= json.size()) break;

        if (json[afterColon] == '"') {
            size_t valStart = afterColon + 1;
            size_t valEnd = valStart;
            while (valEnd < json.size()) {
                if (json[valEnd] == '"') break;
                if (json[valEnd] == '\\' && valEnd + 1 < json.size()) valEnd += 2;
                else valEnd++;
            }
            result[key] = json.substr(valStart, valEnd - valStart);
            pos = valEnd + 1;
        } else {
            size_t valEnd = afterColon;
            while (valEnd < json.size() && json[valEnd] != ',' && json[valEnd] != '}' && json[valEnd] != ' ' && json[valEnd] != '\n' && json[valEnd] != '\r')
                valEnd++;
            result[key] = json.substr(afterColon, valEnd - afterColon);
            pos = valEnd;
        }
    }
    return result;
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
        std::string last_suite;
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

std::string formatText(const SweepReport& report) {
    std::ostringstream os;

    // ── suite header ──
    os << BOLD << CYAN << report.suite_name << RESET << "\n";
    if (!report.description.empty()) {
        os << DIM << "  " << report.description << RESET << "\n";
    }
    os << separator('-') << "\n";

    // ── build header row ──
    std::vector<std::string> cols;
    cols.push_back("test");
    for (const auto& pn : report.param_names) cols.push_back(pn);
    cols.push_back("median");
    cols.push_back("stddev");
    cols.push_back("peak%");
    cols.push_back("power(W)");

    std::vector<std::size_t> widths(cols.size());
    for (std::size_t c = 0; c < cols.size(); c++) {
        widths[c] = cols[c].size();
    }

    for (const auto& pt : report.points) {
        std::size_t c = 0;
        widths[c++] = std::max(widths[c - 1], pt.test_name.size());
        auto params = parseParamsJson(pt.params_json);
        for (const auto& pn : report.param_names) {
            auto it = params.find(pn);
            std::size_t len = (it != params.end()) ? it->second.size() : 0;
            widths[c] = std::max(widths[c], len);
            c++;
        }
        { std::string s = std::to_string(pt.result.median); widths[c] = std::max(widths[c], s.size()); c++; }
        { std::string s = std::to_string(pt.result.stddev); widths[c] = std::max(widths[c], s.size()); c++; }
        { std::string s = pt.result.peak_pct ? std::to_string(*pt.result.peak_pct) : "-"; widths[c] = std::max(widths[c], s.size()); c++; }
        { std::string s = pt.power_watts ? std::to_string(*pt.power_watts) : "-"; widths[c] = std::max(widths[c], s.size()); c++; }
    }
    for (std::size_t c = 0; c < cols.size(); c++) {
        os << " " << BOLD << std::left << std::setw(static_cast<int>(widths[c])) << cols[c] << RESET;
    }
    os << "\n";

    // ── data rows ──
    for (const auto& pt : report.points) {
        bool isError = pt.error_message.has_value() && !pt.error_message->empty();

        if (isError) {
            os << " " << RED << "[ERROR]" << RESET;
        } else {
            os << " ";
        }

        os << std::left << std::setw(static_cast<int>(widths[0])) << pt.test_name;

        std::size_t c = 1;
        auto params = parseParamsJson(pt.params_json);
        for (const auto& pn : report.param_names) {
            auto it = params.find(pn);
            std::string val = (it != params.end()) ? it->second : "-";
            os << " " << std::left << std::setw(static_cast<int>(widths[c])) << val;
            c++;
        }

        os << " " << std::left << std::setw(static_cast<int>(widths[c]))
           << std::fixed << std::setprecision(2) << pt.result.median;
        c++;

        os << " " << std::left << std::setw(static_cast<int>(widths[c]))
           << std::fixed << std::setprecision(4) << pt.result.stddev;
        c++;

        if (pt.result.peak_pct) {
            os << " " << std::left << std::setw(static_cast<int>(widths[c]))
               << std::fixed << std::setprecision(1) << *pt.result.peak_pct;
        } else {
            os << " " << std::left << std::setw(static_cast<int>(widths[c])) << "-";
        }
        c++;

        if (pt.power_watts) {
            os << " " << std::left << std::setw(static_cast<int>(widths[c]))
               << std::fixed << std::setprecision(1) << *pt.power_watts;
        } else {
            os << " " << std::left << std::setw(static_cast<int>(widths[c])) << "-";
        }
        c++;
        if (isError) {
            std::string msg = *pt.error_message;
            if (msg.size() > 40) msg = msg.substr(0, 40) + "..";
            os << " " << RED << msg << RESET;
        }

        os << "\n";
    }

    // ── summary ──
    os << separator('-') << "\n";
    os << "  " << GREEN << report.success_points << RESET << "/"
       << report.total_points << " points succeeded";
    if (report.error_points > 0) {
        os << " (" << DIM << report.error_points << " errors" << RESET << ")";
    }
    os << "\n";

    return os.str();
}

std::string formatText(const std::vector<SweepReport>& reports) {
    std::ostringstream os;
    for (std::size_t i = 0; i < reports.size(); i++) {
        if (i > 0) os << "\n";
        os << formatText(reports[i]);
    }
    return os.str();
}

} // namespace deusridet::bench
