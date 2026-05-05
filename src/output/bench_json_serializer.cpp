#include "bench_json_serializer.h"
#include <cmath>
#include "communis/json_writer.h"

#include <sstream>
#include <iomanip>

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

} // namespace deusridet::bench
