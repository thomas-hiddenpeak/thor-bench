#include "power_monitor.h"

#include <dirent.h>
#include <fstream>

namespace deusridet::bench {

namespace {

bool probeHwmon(const std::string& basePath, std::string& outPath) {
    DIR* dir = opendir(basePath.c_str());
    if (!dir) return false;

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] != 'h' || entry->d_name[1] != 'w' ||
            entry->d_name[2] != 'm' || entry->d_name[3] != 'o' ||
            entry->d_name[4] != 'n') continue;

        std::string powerFile = basePath + entry->d_name + "/in_power1_input";
        std::ifstream ifs(powerFile);
        if (ifs) {
            std::string val;
            if (ifs >> val) {
                outPath = powerFile;
                closedir(dir);
                return true;
            }
        }
    }
    closedir(dir);
    return false;
}

} // namespace

PowerMonitor::PowerMonitor() = default;

PowerMonitor::~PowerMonitor() {
    shutdown();
}

bool PowerMonitor::init() {
    if (running_.load(std::memory_order_relaxed)) return true;

    sysfsPath_ = resolvePath();
    if (sysfsPath_.empty()) return false;

    running_.store(true, std::memory_order_relaxed);
    pollThread_ = std::thread(&PowerMonitor::poll, this);
    return true;
}

void PowerMonitor::shutdown() {
    if (!running_.load(std::memory_order_relaxed)) return;

    running_.store(false, std::memory_order_relaxed);
    if (pollThread_.joinable()) {
        pollThread_.join();
    }
    ringBuf_.clear();
    hasStart_ = false;
}

void PowerMonitor::markStart() {
    std::lock_guard<std::mutex> lock(bufMutex_);
    startTime_ = std::chrono::steady_clock::now();
    hasStart_ = true;
    ringBuf_.clear();
}

std::optional<double> PowerMonitor::markEnd() {
    std::lock_guard<std::mutex> lock(bufMutex_);

    if (!hasStart_ || ringBuf_.empty()) return std::nullopt;

    auto now = std::chrono::steady_clock::now();

    double sum = 0.0;
    int count = 0;
    for (const auto& [ts, val] : ringBuf_) {
        if (ts >= startTime_ && ts <= now) {
            sum += val;
            ++count;
        }
    }

    if (count == 0) return std::nullopt;
    return sum / count;
}

std::optional<double> PowerMonitor::readInstant() {
    std::lock_guard<std::mutex> lock(bufMutex_);
    if (ringBuf_.empty()) return std::nullopt;
    return ringBuf_.back().second;
}

void PowerMonitor::poll() {
    while (running_.load(std::memory_order_relaxed)) {
        std::ifstream ifs(sysfsPath_);
        if (!ifs) {
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_MS));
            continue;
        }

        std::string val;
        if (!(ifs >> val)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(POLL_MS));
            continue;
        }

        double microwatts = std::stod(val);
        double watts = microwatts / 1e6;

        {
            std::lock_guard<std::mutex> lock(bufMutex_);
            ringBuf_.push_back({std::chrono::steady_clock::now(), watts});
            if (static_cast<int>(ringBuf_.size()) > MAX_SAMPLES) {
                ringBuf_.erase(ringBuf_.begin());
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_MS));
    }
}

std::string PowerMonitor::resolvePath() {
    // Try INA3221 first
    std::string path;
    if (probeHwmon("/sys/bus/i2c/devices/2-0040/hwmon/", path)) return path;

    // Fallback: INA238 carrier board
    if (probeHwmon("/sys/bus/i2c/devices/2-0044/hwmon/", path)) return path;

    return {};
}

} // namespace deusridet::bench
