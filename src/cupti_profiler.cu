#include "cupti_profiler.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <utility>

#include <cupti.h>
#include <cupti_activity.h>

#include <cuda_runtime.h>

namespace deusridet::bench {

namespace {

static CUptiResult checkCupti(CUptiResult result, const char* msg) {
    if (result != CUPTI_SUCCESS) {
        const char* errStr = nullptr;
        cuptiGetResultString(result, &errStr);
        std::cerr << "[CUPTI v1] " << msg << " failed: "
                  << result << " (" << (errStr ? errStr : "unknown") << ")" << std::endl;
        return result;
    }
    return CUPTI_SUCCESS;
}

// ── Overhead measurement kernel ──
__global__ void cuptiOverheadKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 1.0001f;
    }
}

// ── CUPTI v1 buffer callback infrastructure ──
// CUPTI v1 (CUDA 13.0) uses a buffer-pair callback model:
//   1. Request callback: we provide ONE buffer pointer and size
//   2. Complete callback: CUPTI notifies when the buffer is full, we parse records
// No userData parameter — we use a global pointer.

constexpr size_t kCuptiBufferSize = 1ULL << 24; // 16 MB per buffer

static uint8_t* gCuptiBuffer = nullptr;
static CuptiProfiler* gCuptiProfiler = nullptr;

// Request callback: called once at startup, allocate buffer
static void CUPTIAPI cuptiBufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    gCuptiBuffer = static_cast<uint8_t*>(std::malloc(kCuptiBufferSize));
    if (!gCuptiBuffer) {
        std::cerr << "[CUPTI v1] Failed to allocate activity buffer" << std::endl;
        *buffer = nullptr;
        *size = 0;
        *maxNumRecords = 0;
        return;
    }
    std::memset(gCuptiBuffer, 0, kCuptiBufferSize);
    *buffer = gCuptiBuffer;
    *size = kCuptiBufferSize;
    *maxNumRecords = 0; // let CUPTI decide
}

// Complete callback: called when buffer is full, parse records
static void CUPTIAPI cuptiBufferComplete(CUcontext context,
                                          uint32_t streamId,
                                          uint8_t* buffer, size_t size,
                                          size_t validSize) {
    (void)context;
    (void)streamId;
    (void)size;

    auto* profiler = gCuptiProfiler;
    if (!profiler || !buffer || validSize == 0) return;

    uint8_t* recordPtr = buffer;
    size_t remaining = validSize;

    while (remaining > 0) {
        CUpti_Activity* record = nullptr;
        CUptiResult res = cuptiActivityGetNextRecord(recordPtr, remaining, &record);
        if (res != CUPTI_SUCCESS || !record) break;

        CUpti_ActivityKind kind = record->kind;

        CuptiActivityRecord act;
        switch (kind) {
            case CUPTI_ACTIVITY_KIND_RUNTIME:
                act.type = "CUDA_RUNTIME";
                break;
            case CUPTI_ACTIVITY_KIND_DRIVER:
                act.type = "CUDA_DRIVER";
                break;
            case CUPTI_ACTIVITY_KIND_KERNEL:
                act.type = "KERNEL";
                break;
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                act.type = "CONCURRENT_KERNEL";
                break;
            case CUPTI_ACTIVITY_KIND_MEMCPY:
                act.type = "MEMCPY";
                break;
            case CUPTI_ACTIVITY_KIND_MEMSET:
                act.type = "MEMSET";
                break;
            case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
                act.type = "EXTERNAL_CORRELATION";
                break;
            default:
                act.type = "OTHER";
                break;
        }

        // Cast to specific record types for field access
        if (kind == CUPTI_ACTIVITY_KIND_DRIVER || kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            auto* api = reinterpret_cast<CUpti_ActivityAPI*>(record);
            act.start_ns = api->start;
            act.end_ns   = api->end;
            act.correlationId = api->correlationId;
            act.correlator    = api->correlationId;
        } else if (kind == CUPTI_ACTIVITY_KIND_KERNEL || kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto* kern = reinterpret_cast<CUpti_ActivityKernel10*>(record);
            act.start_ns = kern->start;
            act.end_ns   = kern->end;
            act.correlationId = kern->correlationId;
            act.correlator    = kern->gridId;
            act.name = kern->name ? kern->name : "";
        } else if (kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto* mc = reinterpret_cast<CUpti_ActivityMemcpy6*>(record);
            act.start_ns = mc->start;
            act.end_ns   = mc->end;
            act.correlationId = mc->correlationId;
            act.correlator    = mc->correlationId;
        } else if (kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            auto* ms = reinterpret_cast<CUpti_ActivityMemset4*>(record);
            act.start_ns = ms->start;
            act.end_ns   = ms->end;
            act.correlationId = ms->correlationId;
            act.correlator    = ms->correlationId;
        }

        profiler->pushActivityRecord(std::move(act));

        // cuptiActivityGetNextRecord advances its internal pointer on success.
        // Reset remaining to allow the next call; API stops on CUPTI_INVALID_ARGUMENT.
        remaining = validSize;
    }
}

} // anonymous namespace

// ── Constructor / Destructor / Singleton ──

CuptiProfiler::CuptiProfiler() = default;

CuptiProfiler::~CuptiProfiler() {
    if (active_) {
        shutdown();
    }
}

CuptiProfiler& CuptiProfiler::instance() {
    static CuptiProfiler inst;
    return inst;
}

// ── Init ──

bool CuptiProfiler::init(int device) {
    if (active_) return true;

    device_ = device;

    // Set activity buffer size attribute before enabling
    size_t bufSize = kCuptiBufferSize;
    size_t attrSize = sizeof(bufSize);
    checkCupti(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                                          &attrSize, &bufSize),
                "set_buffer_size");

    // Register buffer callbacks (v1 API: no userData, uses globals)
    gCuptiBuffer = nullptr;
    gCuptiProfiler = this;

    checkCupti(cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferComplete),
                "cuptiActivityRegisterCallbacks");

    // Enable activity kinds (v1: single param, no subscriber/device)
    auto enable = [&](CUpti_ActivityKind kind) {
        checkCupti(cuptiActivityEnable(kind), "cuptiActivityEnable");
    };

    enable(CUPTI_ACTIVITY_KIND_RUNTIME);
    enable(CUPTI_ACTIVITY_KIND_DRIVER);
    enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    enable(CUPTI_ACTIVITY_KIND_MEMCPY);
    enable(CUPTI_ACTIVITY_KIND_MEMSET);
    enable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);

    active_ = true;
    return true;
}

// ── Range control ──

void CuptiProfiler::startRange(const char* name) {
    if (!active_) return;

    // Flush BEFORE acquiring lock — flush triggers buffer complete callback
    // which also acquires the lock. Calling flush inside the lock = deadlock.
    cuptiActivityFlushAll(0);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        currentResult_ = CuptiSuiteData{};
        currentResult_.suite_name = name;
        currentRangeName_ = name;
    }
    rangeStart_ = std::chrono::steady_clock::now();
}

void CuptiProfiler::stopRange() {
    if (!active_) return;

    cuptiActivityFlushAll(0);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto end = std::chrono::steady_clock::now();
        currentResult_.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - rangeStart_);
        results_.push_back(std::move(currentResult_));
        currentRangeName_.clear();
    }
}

std::vector<CuptiSuiteData> CuptiProfiler::getResults() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return results_;
}

void CuptiProfiler::pushActivityRecord(CuptiActivityRecord&& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    currentResult_.activities.push_back(std::move(record));
}

// ── Overhead measurement ──

CuptiOverhead CuptiProfiler::measureOverhead() {
    CuptiOverhead overhead{};
    if (!active_) return overhead;

    const int n = 1 << 20;
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    float* dData = nullptr;
    cudaError_t err = cudaMalloc(&dData, n * sizeof(float));
    if (err != cudaSuccess) return overhead;

    // Warmup launch
    cuptiOverheadKernel<<<blocks, threads>>>(dData, n);
    chk(cudaGetLastError(), "warmup_kernel");
    chk(cudaDeviceSynchronize(), "warmup_sync");

    // Activity kinds to toggle
    constexpr CUpti_ActivityKind kinds[] = {
        CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_DRIVER,
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_MEMSET,
        CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION
    };

    auto disableAll = [&]() {
        for (auto kind : kinds)
            cuptiActivityDisable(kind);
    };
    auto enableAll = [&]() {
        for (auto kind : kinds)
            cuptiActivityEnable(kind);
    };

    // Measure baseline (activities disabled)
    disableAll();
    chk(cuptiActivityFlushAll(0), "flush_baseline");

    constexpr int samples = 5;
    std::vector<double> baseline;
    baseline.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        cuptiOverheadKernel<<<blocks, threads>>>(dData, n);
        chk(cudaGetLastError(), "baseline_kernel");
        chk(cudaDeviceSynchronize(), "baseline_sync");
        auto t1 = std::chrono::steady_clock::now();
        baseline.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    // Measure instrumented (activities enabled)
    enableAll();
    std::vector<double> instrumented;
    instrumented.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        cuptiOverheadKernel<<<blocks, threads>>>(dData, n);
        chk(cudaGetLastError(), "instrumented_kernel");
        chk(cudaDeviceSynchronize(), "instrumented_sync");
        auto t1 = std::chrono::steady_clock::now();
        instrumented.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    chk(cuptiActivityFlushAll(0), "flush_instrumented");

    std::sort(baseline.begin(), baseline.end());
    std::sort(instrumented.begin(), instrumented.end());
    overhead.baseline_ns     = baseline[samples / 2];
    overhead.instrumented_ns = instrumented[samples / 2];
    overhead.overhead_pct    = (overhead.baseline_ns > 0)
        ? (overhead.instrumented_ns - overhead.baseline_ns) / overhead.baseline_ns * 100.0
        : 0.0;

    cudaFree(dData);
    return overhead;
}

// ── Shutdown ──

void CuptiProfiler::shutdown() {
    if (!active_) return;

    cuptiActivityFlushAll(0);

    constexpr CUpti_ActivityKind kinds[] = {
        CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_DRIVER,
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_MEMSET,
        CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION
    };
    for (auto kind : kinds)
        cuptiActivityDisable(kind);

    // Free callback-allocated buffer
    if (gCuptiBuffer) { std::free(gCuptiBuffer); gCuptiBuffer = nullptr; }
    gCuptiProfiler = nullptr;

    active_  = false;
}

} // namespace deusridet::bench
