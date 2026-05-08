#pragma once

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace deusridet::bench {

class PowerMonitor {
public:
    PowerMonitor();
    ~PowerMonitor();

    // No copy/move
    PowerMonitor(const PowerMonitor&) = delete;
    PowerMonitor& operator=(const PowerMonitor&) = delete;
    PowerMonitor(PowerMonitor&&) = delete;
    PowerMonitor& operator=(PowerMonitor&&) = delete;

    /// Probe sysfs INA3221/INA238 paths and start 100 Hz polling thread.
    /// Returns true if a valid power sensor was found and the thread started.
    bool init();

    /// Stop the polling thread and clean up.
    void shutdown();

    /// Mark measurement start — records current steady_clock time.
    void markStart();

    /// Compute average power (W) since last markStart().
    /// Returns nullopt if unavailable, no samples, or markStart never called.
    std::optional<double> markEnd();

    /// Return latest instantaneous power reading in Watts, or nullopt.
    std::optional<double> readInstant();

    /// Whether a power sensor was successfully initialised and polling is active.
    bool isAvailable() const { return running_.load(std::memory_order_relaxed); }

private:
    /// Poll one sample: read sysfs file, push to ring buffer.
    void poll();

    /// Resolve the best sysfs path (INA3221 first, then INA238 fallback).
    std::string resolvePath();

    // ── state ────────────────────────────────────────────────────
    std::string sysfsPath_;
    std::atomic<bool> running_{false};
    std::thread pollThread_;

    mutable std::mutex bufMutex_;
    std::vector<std::pair<std::chrono::steady_clock::time_point, double>> ringBuf_;

    std::chrono::steady_clock::time_point startTime_;
    bool hasStart_{false};

    static constexpr int MAX_SAMPLES = 1000;
    static constexpr int POLL_MS = 10; // 100 Hz
};

} // namespace deusridet::bench
