#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/utsname.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>

#include <cuda_runtime.h>

#include "bench_csv_formatter.h"
#include "bench_json_serializer.h"
#include "bench_suites.h"
#include "bench_text_formatter.h"
#include "cupti_profiler.h"
#include "power_monitor.h"
#include "sweep_runner.h"
#include "communis/cuda_check.h"

using namespace deusridet::bench;

// ── Exit codes ────────────────────────────────────────────────────────────
constexpr int EXIT_SUCCESS_CODE  = 0;
constexpr int EXIT_PARAM_ERROR   = 1;
constexpr int EXIT_CUDA_ERROR    = 2;
constexpr int EXIT_EXEC_FAIL     = 3;

// ── CLI parser ────────────────────────────────────────────────────────────
static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [OPTIONS]\n"
              << "\n"
              << "  --json            Output results as JSON\n"
              << "  --suites NAME1,NAME2,...  Run only named suites\n"
              << "  --iterations N    Number of benchmark iterations (default: 10)\n"
              << "  --warmup N        Number of warmup runs per test (default: 3)\n"
              << "  --timeout SEC     Per-suite timeout in seconds (default: 300)\n"
              << "  --device N        CUDA device index (default: 0)\n"
               << "  --cupti           Enable CUPTI activity profiling\n"
               << "  --sweep           Run in sweep mode (parameter matrix testing)\n"
               << "  --csv             Output sweep results as CSV (requires --sweep)\n"
               << "  --help            Show this message\n";
}

struct CliArgs {
    bool     json         = false;
    bool     cupti        = false;
    bool     sweep        = false;
    bool     csv          = false;
    std::vector<std::string> suites;
    int      iterations   = 10;
    int      warmup       = 3;
    int      timeout_sec  = 300;
    int      device       = 0;
};

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    if (s.empty()) return out;

    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        auto start = token.find_first_not_of(" \t");
        auto end   = token.find_last_not_of(" \t");
        if (start != std::string::npos) {
            out.push_back(token.substr(start, end - start + 1));
        }
    }
    return out;
}

static CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS_CODE);
        } else if (arg == "--json") {
            args.json = true;
        } else if (arg == "--cupti") {
            args.cupti = true;
        } else if (arg == "--suites") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --suites requires an argument" << std::endl;
                print_usage(argv[0]);
                exit(EXIT_PARAM_ERROR);
            }
            ++i;
            args.suites = split_csv(argv[i]);
            if (args.suites.empty()) {
                std::cerr << "Error: --suites list is empty" << std::endl;
                exit(EXIT_PARAM_ERROR);
            }
        } else if (arg == "--iterations") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --iterations requires an argument" << std::endl;
                print_usage(argv[0]);
                exit(EXIT_PARAM_ERROR);
            }
            ++i;
            try {
                args.iterations = std::stoi(argv[i]);
                if (args.iterations < 1) {
                    std::cerr << "Error: --iterations must be >= 1" << std::endl;
                    exit(EXIT_PARAM_ERROR);
                }
            } catch (const std::exception&) {
                std::cerr << "Error: invalid --iterations value: " << argv[i] << std::endl;
                exit(EXIT_PARAM_ERROR);
            }
        } else if (arg == "--warmup") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --warmup requires an argument" << std::endl;
                print_usage(argv[0]);
                exit(EXIT_PARAM_ERROR);
            }
            ++i;
            try {
                args.warmup = std::stoi(argv[i]);
                if (args.warmup < 0) {
                    std::cerr << "Error: --warmup must be >= 0" << std::endl;
                    exit(EXIT_PARAM_ERROR);
                }
            } catch (const std::exception&) {
                std::cerr << "Error: invalid --warmup value: " << argv[i] << std::endl;
                exit(EXIT_PARAM_ERROR);
            }
        } else if (arg == "--timeout") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --timeout requires an argument" << std::endl;
                print_usage(argv[0]);
                exit(EXIT_PARAM_ERROR);
            }
            ++i;
            try {
                args.timeout_sec = std::stoi(argv[i]);
                if (args.timeout_sec < 0) {
                    std::cerr << "Error: --timeout must be >= 0" << std::endl;
                    exit(EXIT_PARAM_ERROR);
                }
            } catch (const std::exception&) {
                std::cerr << "Error: invalid --timeout value: " << argv[i] << std::endl;
                exit(EXIT_PARAM_ERROR);
            }
        } else if (arg == "--device") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --device requires an argument" << std::endl;
                print_usage(argv[0]);
                exit(EXIT_PARAM_ERROR);
            }
            ++i;
            try {
                args.device = std::stoi(argv[i]);
                if (args.device < 0) {
                    std::cerr << "Error: --device must be >= 0" << std::endl;
                    exit(EXIT_PARAM_ERROR);
                }
            } catch (const std::exception&) {
                std::cerr << "Error: invalid --device value: " << argv[i] << std::endl;
                exit(EXIT_PARAM_ERROR);
            }
        } else if (arg == "--sweep") {
            args.sweep = true;
        } else if (arg == "--csv") {
            args.csv = true;
        } else {
            std::cerr << "Error: unrecognized option: " << arg << std::endl;
            print_usage(argv[0]);
            exit(EXIT_PARAM_ERROR);
        }
    }

    if (args.sweep && args.cupti) {
        std::cerr << "Error: --sweep and --cupti are mutually exclusive" << std::endl;
        print_usage(argv[0]);
        exit(EXIT_PARAM_ERROR);
    }

    return args;
}

// ── Hostname helper ───────────────────────────────────────────────────────
static std::string get_hostname() {
    struct utsname buf{};
    if (uname(&buf) == 0 && std::string(buf.nodename).size() > 1) {
        return buf.nodename;
    }
    return "unknown";
}

// ── Timestamp helper ──────────────────────────────────────────────────────
static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf{};
    localtime_r(&time_t_now, &tm_buf);

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

// ── Main ──────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    CliArgs args = parse_args(argc, argv);

    // ── Sweep mode (before CUDA device validation) ──
    if (args.sweep) {
        auto& sweepRegistry = BenchSuiteRegistry::instance();
        const auto& sweepSuites = args.suites.empty()
            ? sweepRegistry.allSweepSuites()
            : sweepRegistry.filteredSweepSuites(args.suites);

        if (sweepSuites.empty()) {
            std::cerr << "Error: no sweep suites registered (or none matched filter)" << std::endl;
            return EXIT_PARAM_ERROR;
        }

        std::cerr << "[sweep] Mode: sweep | Suites: " << sweepSuites.size()
                  << " | Device: " << args.device << "\n";

        PowerMonitor powerMonitor;
        powerMonitor.init();

        if (args.csv) {
            std::filesystem::create_directories("sweep_results");
        }

        SweepRunner sweepRunner;
        sweepRunner.warmup(args.warmup);
        sweepRunner.iterations(args.iterations);
        sweepRunner.timeout(std::chrono::milliseconds(args.timeout_sec * 1000));

        auto reports = sweepRunner.runMultiple(sweepSuites, args.device);

        if (args.csv) {
            for (const auto& report : reports) {
                std::string path = "sweep_results/" + report.suite_name + ".csv";
                writeCsvFile(path, report);
                std::cerr << "[csv] Written: " << path << "\n";
            }
        }

        if (args.json) {
            std::cout << serializeSweepJson(reports, get_hostname()) << std::endl;
        } else {
            std::cout << formatText(reports) << std::endl;
        }

        powerMonitor.shutdown();

        return EXIT_SUCCESS_CODE;
    }

    // ── CUDA device validation ──
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaGetDeviceCount failed: "
                  << cudaGetErrorString(err) << std::endl;
        return EXIT_CUDA_ERROR;
    }
    if (deviceCount == 0) {
        std::cerr << "Error: no CUDA devices available" << std::endl;
        return EXIT_CUDA_ERROR;
    }
    if (args.device < 0 || args.device >= deviceCount) {
        std::cerr << "Error: invalid device " << args.device
                  << " (available: 0.." << deviceCount - 1 << ")" << std::endl;
        return EXIT_CUDA_ERROR;
    }

    try {
        cudaCheck(cudaSetDevice(args.device));
    } catch (const std::exception& e) {
        std::cerr << "Error: cudaSetDevice(" << args.device << ") failed: "
                  << e.what() << std::endl;
        return EXIT_CUDA_ERROR;
    }

    // ── Get suites (filtered or all) ──
    auto& registry = BenchSuiteRegistry::instance();
    const auto& suites = args.suites.empty()
        ? registry.allSuites()
        : registry.filteredSuites(args.suites);

    if (suites.empty()) {
        if (!args.suites.empty()) {
            std::cerr << "Error: no suites matched filter: ";
            for (size_t i = 0; i < args.suites.size(); ++i) {
                if (i) std::cerr << ",";
                std::cerr << args.suites[i];
            }
            std::cerr << std::endl;
            return EXIT_PARAM_ERROR;
        }
        std::cerr << "Error: no suites registered" << std::endl;
        return EXIT_PARAM_ERROR;
    }

    // ── CUPTI profiler ──
    CuptiOverhead cuptiOverhead{};
    if (args.cupti) {
        auto& cupti = CuptiProfiler::instance();
        if (!cupti.init(args.device)) {
            std::cerr << "Warning: CUPTI init failed, disabling profiling" << std::endl;
            args.cupti = false;
        } else {
            std::cerr << "[CUPTI] Profiler initialized" << std::endl;
        }
    }

    // ── Build runner ──
    BenchRunner runner;
    runner.warmup(args.warmup);
    runner.iterations(args.iterations);
    runner.timeout(std::chrono::milliseconds(args.timeout_sec * 1000));

    // ── Banner ──
    const char* banner = "DeusRidet-Thor Benchmark Suite v0.1.0\n"
                          "======================================\n";
    if (args.json) {
        std::cerr << banner;
    } else {
        std::cout << banner;
        std::cout << "Device: " << args.device << " | ";
        std::cout << "Suites: " << suites.size() << " | ";
        std::cout << "Iterations: " << args.iterations << " | ";
        std::cout << "Warmup: " << args.warmup;
        if (args.cupti) std::cout << " | CUPTI: ON";
        std::cout << "\n";
        std::cout << std::string(72, '=') << "\n";
        std::cout << std::endl;
    }

    // ── Execute suites ──
    BenchReport report;
    report.version    = "0.1.0";
    report.timestamp  = get_timestamp();
    report.hostname   = get_hostname();

    bool anyFailed = false;

    for (const auto& suite : suites) {
        if (!args.json) {
            std::cout << "[" << suite.name << "] " << suite.description << std::endl;
        }

        // Drain async errors from prior suites (non-blocking check only).
        {
            cudaError_t preSync = cudaGetLastError();
            if (preSync != cudaSuccess) {
                std::cerr << "CUDA(" << cudaGetErrorString(preSync) << ") before suite " << suite.name
                          << " — resetting device context\n";
                cudaDeviceReset();
                {
                    int retry = 0;
                    cudaError_t e;
                    do {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        e = cudaSetDevice(args.device);
                    } while (e != cudaSuccess && ++retry < 10);
                    if (e != cudaSuccess) {
                        std::cerr << "Cannot recover before suite " << suite.name << " — stopping benchmark\n";
                        break;
                    }
                }
            }
        }

        if (args.cupti) {
            CuptiProfiler::instance().startRange(suite.name.c_str());
        }

        try {
            auto results = suite.runFn(runner);
            report.results.insert(report.results.end(), results.begin(), results.end());
        } catch (const std::exception& e) {
            std::cerr << "[" << suite.name << "] FAILED: " << e.what() << std::endl;
            BenchResult errResult;
            errResult.suite_name = suite.name;
            errResult.test_name  = "error";
            errResult.sample_count = 0;
            report.results.push_back(errResult);
            anyFailed = true;
        } catch (...) {
            std::cerr << "[" << suite.name << "] FAILED with unknown exception" << std::endl;
            BenchResult errResult;
            errResult.suite_name = suite.name;
            errResult.test_name  = "unknown_error";
            errResult.sample_count = 0;
            report.results.push_back(errResult);
            anyFailed = true;
        }

        if (args.cupti) {
            CuptiProfiler::instance().stopRange();
        }

        // Use cudaEventRecord + polling loop to avoid blocking forever on async workstream errors.
        cudaError_t syncErr = cudaSuccess;
        {
            cudaEvent_t syncEvent;
            syncErr = cudaEventCreate(&syncEvent);
            if (syncErr == cudaSuccess) {
                cudaEventRecord(syncEvent);
                auto start = std::chrono::steady_clock::now();
                bool completed = false;
                while (std::chrono::steady_clock::now() - start < std::chrono::seconds(30)) {
                    cudaError_t queryErr = cudaEventQuery(syncEvent);
                    if (queryErr == cudaSuccess) {
                        completed = true;
                        break;
                    } else if (queryErr != cudaErrorNotReady) {
                        syncErr = queryErr;
                        completed = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (!completed) {
                    std::cerr << "[suite " << suite.name << "] cudaDeviceSynchronize() timed out (30s) — resetting context\n";
                    cudaEventDestroy(syncEvent);
                    cudaDeviceReset();
                    {
                        int retry = 0;
                        cudaError_t e;
                        do {
                            std::this_thread::sleep_for(std::chrono::milliseconds(200));
                            e = cudaSetDevice(args.device);
                        } while (e != cudaSuccess && ++retry < 10);
                    }
                    syncErr = cudaErrorUnknown;
                } else {
                    cudaEventDestroy(syncEvent);
                }
            }
        }
        if (syncErr != cudaSuccess) {
            std::cerr << "CUDA(" << cudaGetErrorString(syncErr) << ") after suite " << suite.name << "\n";
            anyFailed = true;
            // Only IllegalInstruction poisons context. Other async errors (ws) do not —
            // cudaDeviceReset() would block forever on those.
            if (syncErr == cudaErrorIllegalInstruction) {
                cudaDeviceReset();
                {
                    int retry = 0;
                    cudaError_t e;
                    do {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        e = cudaSetDevice(args.device);
                    } while (e != cudaSuccess && ++retry < 10);
                    if (e != cudaSuccess) {
                        std::cerr << "Cannot recover after IllegalInstruction — stopping benchmark\n";
                        break;
                    }
                }
            }
        }
        cudaGetLastError();

        if (!args.json) std::cout << std::endl;
    }

    // ── CUPTI overhead measurement ──
    if (args.cupti) {
        cuptiOverhead = CuptiProfiler::instance().measureOverhead();
        std::cerr << "[CUPTI] Overhead: baseline="
                  << cuptiOverhead.baseline_ns << "ns "
                  << "instrumented=" << cuptiOverhead.instrumented_ns << "ns "
                  << "overhead=" << std::fixed << std::setprecision(1)
                  << cuptiOverhead.overhead_pct << "%" << std::endl;
    }

    // ── Output ──
    if (args.json) {
        std::cout << serializeJson(report) << std::endl;
    } else {
        std::cout << formatText(report) << std::endl;
    }

    // ── CUPTI results ──
    if (args.cupti) {
        auto cuptiResults = CuptiProfiler::instance().getResults();
        std::cout << "\n" << std::string(72, '=') << "\n";
        std::cout << "[CUPTI Activity Summary]\n";
        std::cout << std::string(72, '-') << "\n";
        for (const auto& data : cuptiResults) {
            std::cout << "  " << data.suite_name
                      << " | wall=" << data.wall_ns.count() / 1e6 << "ms"
                      << " | activities=" << data.activities.size();
            if (!data.metrics.empty()) {
                std::cout << " | metrics:";
                for (const auto& [k, v] : data.metrics) {
                    std::cout << " " << k << "=" << v;
                }
            }
            std::cout << "\n";
        }
        std::cout << std::string(72, '=') << "\n";

        CuptiProfiler::instance().shutdown();
    }

    return anyFailed ? EXIT_EXEC_FAIL : EXIT_SUCCESS_CODE;
}
