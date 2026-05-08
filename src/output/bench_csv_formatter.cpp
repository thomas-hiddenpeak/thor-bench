#include "bench_csv_formatter.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <filesystem>

namespace deusridet::bench {

namespace {

std::string csvEscape(const std::string& field) {
    bool needsQuote = false;
    for (char c : field) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needsQuote = true;
            break;
        }
    }
    if (!needsQuote) return field;

    std::string result = "\"";
    for (char c : field) {
        if (c == '"') result += "\"\"";
        else result += c;
    }
    result += '"';
    return result;
}

std::string fmtDouble(double val) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << val;
    return os.str();
}

// Minimal JSON object parser: extracts {"key": "value"} pairs.
// Handles string values with escaped quotes and numeric values.
// Returns a map from key to raw string value.
std::map<std::string, std::string> parseParamsJson(const std::string& json) {
    std::map<std::string, std::string> result;
    if (json.empty()) return result;

    size_t pos = 0;
    while (pos < json.size()) {
        // Find next '"'
        size_t eq = json.find('\"', pos);
        if (eq == std::string::npos) break;

        // key is the content between the opening ']' and closing '"'
        size_t keyStart = eq + 1;
        size_t keyEnd = json.find('\"', keyStart);
        if (keyEnd == std::string::npos) break;
        std::string key = json.substr(keyStart, keyEnd - keyStart);

        // Skip ': '
        size_t afterColon = keyEnd + 1;
        while (afterColon < json.size() && (json[afterColon] == ' ' || json[afterColon] == ':'))
            afterColon++;

        if (afterColon >= json.size()) break;

        // Value starts here
        if (json[afterColon] == '"') {
            // String value
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
            // Numeric or boolean/null value
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

std::string formatCsvHeader(const SweepReport& report) {
    std::ostringstream os;
    os << "suite_name,test_name";
    for (const auto& p : report.param_names) {
        os << "," << csvEscape(p);
    }
    os << ",median,mean,stddev,p95,p99,peak_pct,power_watts,score,sample_count,error_message";
    os << "\n";
    return os.str();
}

std::string formatCsvRows(const SweepReport& report) {
    std::ostringstream os;

    for (const auto& pt : report.points) {
        auto params = parseParamsJson(pt.params_json);

        os << csvEscape(pt.suite_name) << ",";
        os << csvEscape(pt.test_name) << ",";

        for (const auto& p : report.param_names) {
            auto it = params.find(p);
            if (it != params.end())
                os << csvEscape(it->second);
            os << ",";
        }

        // Statistics
        os << fmtDouble(pt.result.median) << ",";
        os << fmtDouble(pt.result.mean) << ",";
        os << fmtDouble(pt.result.stddev) << ",";
        os << fmtDouble(pt.result.p95) << ",";
        os << fmtDouble(pt.result.p99) << ",";

        // peak_pct (optional)
        if (pt.result.peak_pct)
            os << fmtDouble(*pt.result.peak_pct);
        os << ",";

        // power_watts (optional)
        if (pt.power_watts)
            os << fmtDouble(*pt.power_watts);
        os << ",";

        // score (optional)
        if (pt.result.score)
            os << fmtDouble(*pt.result.score);
        os << ",";

        // sample_count
        os << pt.result.sample_count << ",";

        // error_message (optional)
        if (pt.error_message)
            os << csvEscape(*pt.error_message);

        os << "\n";
    }

    return os.str();
}

bool writeCsvFile(const std::string& outputPath, const SweepReport& report) {
    std::filesystem::path p(outputPath);
    std::filesystem::create_directories(p.parent_path());

    std::ofstream ofs(outputPath, std::ios::binary);
    if (!ofs) return false;

    ofs << formatCsvHeader(report);
    ofs << formatCsvRows(report);

    ofs.close();
    return ofs.good() || ofs.eof();
}

std::string defaultCsvPath(const std::string& suiteName) {
    return "sweep_results/" + suiteName + ".csv";
}

} // namespace deusridet::bench
