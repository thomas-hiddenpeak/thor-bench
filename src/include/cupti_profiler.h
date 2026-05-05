#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <mutex>

// Header-safe: no <cupti.h> included — void* handles keep CUPTI types
// private to the .cu. Builds without CUPTI still compile cleanly.
namespace deusridet::bench {

struct CuptiActivityRecord {
    std::string type;
    uint64_t start_ns     = 0;
    uint64_t end_ns       = 0;
    uint64_t correlationId = 0;
    uint64_t correlator   = 0;
    std::string name;
};

struct CuptiSuiteData {
    std::string suite_name;
    std::chrono::nanoseconds wall_ns;
    std::vector<CuptiActivityRecord> activities;
    std::map<std::string, double> metrics;
};

struct CuptiOverhead {
    double baseline_ns     = 0.0;
    double instrumented_ns = 0.0;
    double overhead_pct    = 0.0;
};

// Singleton CUPTI profiler — one subscriber per process (CUPTI limitation).
// init() returns false if CUPTI unavailable; all methods are no-ops when !isActive().
class CuptiProfiler {
public:
    static CuptiProfiler& instance();

    bool init(int device);
    void startRange(const char* name);
    void stopRange();
    std::vector<CuptiSuiteData> getResults() const;
    CuptiOverhead measureOverhead();
    void shutdown();

    // Public method for CUPTI v1 callback to push records (no userData param in v1)
    void pushActivityRecord(CuptiActivityRecord&& record);

    bool isActive() const { return active_; }

    CuptiProfiler(const CuptiProfiler&)            = delete;
    CuptiProfiler& operator=(const CuptiProfiler&) = delete;
    CuptiProfiler(CuptiProfiler&&)                 = delete;
    CuptiProfiler& operator=(CuptiProfiler&&)      = delete;

 private:
    CuptiProfiler();
    ~CuptiProfiler();

    bool active_ = false;
    int  device_ = 0;

    mutable std::mutex mutex_;

    std::string currentRangeName_;
    std::chrono::steady_clock::time_point rangeStart_;

    std::vector<CuptiSuiteData> results_;
    CuptiSuiteData currentResult_;
};

} // namespace deusridet::bench
