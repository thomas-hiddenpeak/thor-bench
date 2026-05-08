#include "bench_json_serializer.h"
#include <cmath>
#include "communis/json_writer.h"

#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>

namespace deusridet::bench {

std::string serializeJson(const BenchReport& report) {
    deusridet::json::Writer w;

    w.begin_object();

    // -- metadata --
    w.field_string("version", report.version);
    w.field_string("timestamp", report.timestamp);
    w.field_string("hostname", report.hostname);

    // -- overall score --
    w.field_optional_double("overall_score", report.overall_score);

    // -- results array --
    w.begin_array("results");
    for (const auto& r : report.results) {
        w.begin_object();
        w.field_string("suite_name", r.suite_name);
        w.field_string("test_name", r.test_name);
        w.field_string("unit", r.unit);
        w.field_double("median", r.median);
        w.field_double("mean", r.mean);
        w.field_double("stddev", r.stddev);
        w.field_double("p95", r.p95);
        w.field_double("p99", r.p99);
        w.field_double("min", r.min_val);
        w.field_double("max", r.max_val);
        w.field_optional_double("score", r.score);
        w.field_optional_double("peak_pct", r.peak_pct);
        w.field_int("sample_count", r.sample_count);
        w.field_int("warmup_count", r.warmup_count);
        if (!r.params_json.empty())
            w.field_string("params", r.params_json);
        if (!r.metadata.empty()) {
            w.begin_object("metadata");
            for (const auto& [k, v] : r.metadata) {
                w.field_string(k, v);
            }
            w.end_object();
        }
        if (!r.probe_snapshot.empty())
            w.field_string("probe_snapshot", r.probe_snapshot);
        w.end_object();
    }
    w.end_array();

    w.end_object();

    return std::move(w).finalize();
}

static void serializeBenchResult(const BenchResult& r, deusridet::json::Writer& w) {
    w.field_string("suite_name", r.suite_name);
    w.field_string("test_name", r.test_name);
    w.field_string("unit", r.unit);
    w.field_double("median", r.median);
    w.field_double("mean", r.mean);
    w.field_double("stddev", r.stddev);
    w.field_double("p95", r.p95);
    w.field_double("p99", r.p99);
    w.field_double("min", r.min_val);
    w.field_double("max", r.max_val);
    w.field_optional_double("score", r.score);
    w.field_optional_double("peak_pct", r.peak_pct);
    w.field_int("sample_count", r.sample_count);
    w.field_int("warmup_count", r.warmup_count);
    if (!r.params_json.empty())
        w.field_string("params", r.params_json);
    if (!r.metadata.empty()) {
        w.begin_object("metadata");
        for (const auto& [k, v] : r.metadata) {
            w.field_string(k, v);
        }
        w.end_object();
    }
    if (!r.probe_snapshot.empty())
        w.field_string("probe_snapshot", r.probe_snapshot);
}

static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm_now;
    gmtime_r(&time_t_now, &tm_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

std::string serializeJson(const SweepResult& result) {
    deusridet::json::Writer w;

    w.begin_object();

    w.field_string("suite_name", result.suite_name);
    w.field_string("test_name", result.test_name);
    if (!result.params_json.empty())
        w.field_string("params_json", result.params_json);

    w.begin_object("result");
    serializeBenchResult(result.result, w);
    w.end_object();

    w.field_optional_double("power_watts", result.power_watts);

    if (result.error_message) {
        w.field_string("error_message", *result.error_message);
    }

    w.field_string("timestamp", result.timestamp);

    w.end_object();

    return std::move(w).finalize();
}

static void serializeSweepPoint(const SweepResult& pt, deusridet::json::Writer& w) {
    w.begin_object();
    w.field_string("suite_name", pt.suite_name);
    w.field_string("test_name", pt.test_name);
    if (!pt.params_json.empty())
        w.field_string("params_json", pt.params_json);

    w.begin_object("result");
    serializeBenchResult(pt.result, w);
    w.end_object();

    w.field_optional_double("power_watts", pt.power_watts);

    if (pt.error_message) {
        w.field_string("error_message", *pt.error_message);
    }

    w.field_string("timestamp", pt.timestamp);
    w.end_object();
}

static void serializeSweepReport(const SweepReport& rpt, deusridet::json::Writer& w) {
    w.begin_object();

    w.field_string("suite_name", rpt.suite_name);
    w.field_string("description", rpt.description);

    w.begin_array("param_names");
    for (const auto& name : rpt.param_names) {
        w.begin_object();
        w.field_string("name", name);
        w.end_object();
    }
    w.end_array();

    w.begin_array("points");
    for (const auto& pt : rpt.points) {
        serializeSweepPoint(pt, w);
    }
    w.end_array();

    w.field_string("sweep_timestamp", rpt.sweep_timestamp);
    w.field_int("total_points", rpt.total_points);
    w.field_int("success_points", rpt.success_points);
    w.field_int("error_points", rpt.error_points);

    w.end_object();
}

std::string serializeJson(const SweepReport& report) {
    deusridet::json::Writer w;

    serializeSweepReport(report, w);

    return std::move(w).finalize();
}

std::string serializeSweepJson(const std::vector<SweepReport>& reports, const std::string& hostname) {
    deusridet::json::Writer w;

    w.begin_object();

    w.field_string("version", "0.1.0");
    w.field_string("timestamp", get_timestamp());
    w.field_string("hostname", hostname);
    w.field_string("mode", "sweep");

    w.begin_array("sweepReports");
    for (const auto& rpt : reports) {
        serializeSweepReport(rpt, w);
    }
    w.end_array();

    w.end_object();

    return std::move(w).finalize();
}

} // namespace deusridet::bench
