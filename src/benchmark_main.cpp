#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/utsname.h>

#include <cuda_runtime.h>

#include "bench_schema.h"
#include "bench_runner.h"
#include "bench_suites.h"
#include "output/bench_json_serializer.h"
#include "output/bench_text_formatter.h"
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
              << "  --help            Show this message\n";
}

struct CliArgs {
    bool     json         = false;
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
        } else {
            std::cerr << "Error: unrecognized option: " << arg << std::endl;
            print_usage(argv[0]);
            exit(EXIT_PARAM_ERROR);
        }
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
        std::cout << "Warmup: " << args.warmup << "\n";
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
        std::cout << "[" << suite.name << "] " << suite.description << std::endl;

        try {
            auto results = suite.runFn(runner);
            report.results.insert(report.results.end(), results.begin(), results.end());
        } catch (const std::exception& e) {
            std::cerr << "[" << suite.name << "] FAILED: " << e.what() << std::endl;
            BenchResult errResult{};
            errResult.suite_name = suite.name;
            errResult.test_name  = "error";
            errResult.sample_count = 0;
            report.results.push_back(errResult);
            anyFailed = true;
        } catch (...) {
            std::cerr << "[" << suite.name << "] FAILED with unknown exception" << std::endl;
            BenchResult errResult{};
            errResult.suite_name = suite.name;
            errResult.test_name  = "unknown_error";
            errResult.sample_count = 0;
            report.results.push_back(errResult);
            anyFailed = true;
        }

        std::cout << std::endl;
    }

    // ── Output ──
    if (args.json) {
        std::cout << serializeJson(report) << std::endl;
    } else {
        std::cout << formatText(report) << std::endl;
    }

    return anyFailed ? EXIT_EXEC_FAIL : EXIT_SUCCESS_CODE;
}
