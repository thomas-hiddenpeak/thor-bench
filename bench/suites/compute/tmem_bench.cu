#include "compute/tmem_bench.h"
#include "bench_schema.h"
#include "bench_suites.h"
#include "bench_stats.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace deusridet::bench {

namespace {

inline void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + m + "): " + cudaGetErrorString(e));
}

// Valid tcgen05 PTX instructions on SM110a:
//   alloc, dealloc, relinquish_alloc_permit, mma, ld, fence
// NOTE: st and cp DO NOT exist in PTX ISA. TMEM is written ONLY via mma.

constexpr int WARP_M = 128;
constexpr int WARP_N =   8;

__device__ static uint32_t buildIdesc(int M, int N) {
    constexpr uint8_t E2M1 = 1;
    uint32_t idesc = 0;
    idesc  = E2M1;
    idesc |= (E2M1 << 10);
    idesc |= (0 << 15);
    idesc |= (1 << 16);
    idesc |= ((N >> 3) & 0x3F) << 17;
    idesc |= (0 << 23);
    idesc |= ((M >> 4) & 0x1F) << 24;
    idesc |= (0 << 29);
    idesc |= (0 << 31);
    return idesc;
}

// ── TMEM probe kernel (MUST match fp4_bench.cu probe exactly) ───────────────
// ONLY alloc -> mma -> dealloc. No ld on uninitialized TMEM.
__global__ void tmemProbeKernel() {
    if (threadIdx.x != 0) return;

    __shared__ uint32_t tmemHandleSmem;

    uint32_t nCols = WARP_N;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmemHandleSmem));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    uint64_t aDesc = 0, bDesc = 0;
    uint32_t idesc = buildIdesc(WARP_M, WARP_N);
    uint32_t tmemHandle = tmemHandleSmem;
    uint32_t saPtr = 0, sbPtr = 0;

    // mma is the ONLY supported TCGen05 instruction that writes to TMEM.
    // If this fails, the entire tcgen05 suite is unavailable.
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.ne.b32 p, 0, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "  [%0], %1, %2, %3, [%4], [%5], p;}\n"
        :
        : "r"(tmemHandle), "l"(aDesc), "l"(bDesc), "r"(idesc),
          "r"(saPtr), "r"(sbPtr)
        : "memory"
    );

    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : : "r"(tmemHandle), "r"(nCols) : "memory");
}

static bool tmemSupported(int device) {
    chk(cudaSetDevice(device), "probe_dev");

    int major = 0, minor = 0;
    chk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "major");
    chk(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "minor");
    if (major < 11 || (major == 11 && minor < 0)) return false;

    cudaStream_t str;
    chk(cudaStreamCreate(&str), "probe_stream");

    tmemProbeKernel<<<1, 32, 0, str>>>();
    cudaError_t e = cudaStreamSynchronize(str);

    if (e != cudaSuccess) {
        cudaStreamDestroy(str);
        return false;
    }

    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaStreamDestroy(str);
        return false;
    }

    cudaStreamDestroy(str);

    return true;
}

// ── TMEM Read Bandwidth kernel ──────────────────────────────────────────────
// mma primes TMEM, then repeatedly reads via tcgen05.ld
__global__ void tmemReadBwKernel(float* out, int loops) {
    __shared__ uint32_t tmemHandleSmem;

    int tid = threadIdx.x;

    uint32_t nCols = WARP_N;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmemHandleSmem));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    uint32_t tmemHandle = tmemHandleSmem;

    // Prime TMEM with mma
    uint64_t aDesc = 0, bDesc = 0;
    uint32_t idesc = buildIdesc(WARP_M, WARP_N);
    uint32_t saPtr = 0, sbPtr = 0;
    asm volatile(
        "{.reg .pred p;\n\t"
        "setp.ne.b32 p, 0, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "  [%0], %1, %2, %3, [%4], [%5], p;}\n"
        :
        : "r"(tmemHandle), "l"(aDesc), "l"(bDesc), "r"(idesc),
          "r"(saPtr), "r"(sbPtr)
        : "memory"
    );

    // Repeatedly read from TMEM via bulk ld
    uint32_t accum = 0;
    int cnt = loops;
    while (cnt--) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile(
            "tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
              "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
            : "r"(tmemHandle)
            : "memory");
        accum += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    }
    asm volatile("" : "+r"(accum));

    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : : "r"(tmemHandle), "r"(nCols) : "memory");

    if (tid == 0) *out = __int_as_float(accum);
}

// ── TMEM Write Bandwidth kernel ──────────────────────────────────────────────
// Repeatedly mma to write to TMEM
__global__ void tmemWriteBwKernel(float* out, int loops) {
    __shared__ uint32_t tmemHandleSmem;

    int tid = threadIdx.x;

    uint32_t nCols = WARP_N;
    uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmemHandleSmem));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : : "r"(smemTmemPtr), "r"(nCols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");

    uint32_t tmemHandle = tmemHandleSmem;

    uint64_t aDesc = 0, bDesc = 0;
    uint32_t idesc = buildIdesc(WARP_M, WARP_N);
    uint32_t saPtr = 0, sbPtr = 0;

    uint32_t accum = 0;
    int cnt = loops;
    while (cnt--) {
        accum++;
        uint32_t enableD = (accum == 1) ? 0 : 1;
        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
            "  [%0], %1, %2, %3, [%4], [%5], p;}\n"
            :
            : "r"(tmemHandle), "l"(aDesc), "l"(bDesc), "r"(idesc),
              "r"(saPtr), "r"(sbPtr), "r"(enableD)
            : "memory"
        );
    }
    asm volatile("" : "+r"(accum));

    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : : "r"(tmemHandle), "r"(nCols) : "memory");

    if (tid == 0) *out = __int_as_float(accum);
}

// ── TMEM Latency kernel ─────────────────────────────────────────────────────
// Single-thread: alloc -> mma -> ld -> dealloc -> relinquish per iteration
__global__ void tmemLatencyKernel(float* out, int loops) {
    int tid = threadIdx.x;
    if (tid != 0) return;

    uint32_t accum = 0;
    int cnt = loops;

    while (cnt--) {
        __shared__ uint32_t tmemHandleSmem;
        uint32_t nCols = WARP_N;
        uint32_t smemTmemPtr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmemHandleSmem));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            : : "r"(smemTmemPtr), "r"(nCols) : "memory");
        uint32_t tmemHandle = tmemHandleSmem;

        uint64_t aDesc = 0, bDesc = 0;
        uint32_t idesc = buildIdesc(WARP_M, WARP_N);
        uint32_t saPtr = 0, sbPtr = 0, enableD = 0;
        asm volatile(
            "{.reg .pred p;\n\t"
            "setp.ne.b32 p, %6, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
            "  [%0], %1, %2, %3, [%4], [%5], p;}\n"
            :
            : "r"(tmemHandle), "l"(aDesc), "l"(bDesc), "r"(idesc),
              "r"(saPtr), "r"(sbPtr), "r"(enableD)
            : "memory"
        );

        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile(
            "tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
              "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
            : "r"(tmemHandle)
            : "memory");
        (void)r1; (void)r2; (void)r3; (void)r4; (void)r5; (void)r6; (void)r7;
        accum += r0;

        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : : "r"(tmemHandle), "r"(nCols) : "memory");

        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }

    asm volatile("" : "+r"(accum));
    *out = __int_as_float(accum);
}

// ── SMEM Read Bandwidth kernel (for comparison) ─────────────────────────────
template <int TileBytes>
__global__ void smemReadBwKernel(float* out, int loops) {
    __shared__ float sdata[TileBytes / 4];
    int tid = threadIdx.x;
    for (int i = 0; i < blockDim.x; i += blockDim.x) {
        if (tid + i < TileBytes / 4)
            sdata[tid + i] = static_cast<float>(tid + i);
    }
    __syncthreads();

    float accum = 0.0f;
    int cnt = loops;
    while (cnt--) {
        for (int i = 0; i < TileBytes / 4; ++i) {
            accum += sdata[i];
        }
    }
    asm volatile("" : "+f"(accum));
    if (tid == 0) *out = accum;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Test 1: TMEM Read Bandwidth ──────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult measureTMEMReadBW(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int loops = 4096;
    int tpb = 32;

    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    for (int w = 0; w < 3; ++w) {
        tmemReadBwKernel<<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        tmemReadBwKernel<<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        // Each ld reads 8*4=32 bytes per thread, 32 threads
        double totalBytes = static_cast<double>(loops) * 32 * sizeof(uint32_t);
        double gbs = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
        vals.push_back(gbs);
    }

    std::ostringstream p;
    p << "{\"loops\":" << loops << ",\"tpb\":" << tpb
      << ",\"bytes_per_ld\":" << (32 * sizeof(uint32_t))
      << ",\"type\":\"tcgen05_tmem\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "tmem";
    res.test_name  = "tmem_read_bandwidth";
    res.unit       = "GB/s";
    res.params_json = p.str();
    res.metadata["proxy"] = "false";
    res.metadata["tcgen05"] = "true";
    res.metadata["note"] = "Real TMEM read via tcgen05.ld (primed by tcgen05.mma)";
    res.peak_pct = 0.0;

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Test 2: TMEM Write Bandwidth (via mma) ───────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult measureTMEMWriteBW(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int loops = 2048;
    int tpb = 32;

    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    for (int w = 0; w < 3; ++w) {
        tmemWriteBwKernel<<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        tmemWriteBwKernel<<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        // Each mma writes 8 FP32 = 32 bytes to TMEM per thread
        double totalBytes = static_cast<double>(loops) * 32 * sizeof(uint32_t);
        double gbs = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
        vals.push_back(gbs);
    }

    std::ostringstream p;
    p << "{\"loops\":" << loops << ",\"tpb\":" << tpb
      << ",\"bytes_per_mma\":" << (32 * sizeof(uint32_t))
      << ",\"type\":\"tcgen05_tmem\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "tmem";
    res.test_name  = "tmem_write_bandwidth";
    res.unit       = "GB/s";
    res.params_json = p.str();
    res.metadata["proxy"] = "false";
    res.metadata["tcgen05"] = "true";
    res.metadata["note"] = "Real TMEM write via tcgen05.mma";
    res.peak_pct = 0.0;

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Test 3: TMEM Latency ─────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult measureTMEMLatency(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    constexpr int loops = 256;
    int tpb = 1;

    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    for (int w = 0; w < 3; ++w) {
        tmemLatencyKernel<<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaStreamSynchronize(str), "warmup");
    }

    std::vector<double> vals;
    for (int i = 0; i < iterations; ++i) {
        chk(cudaEventRecord(evS, str), "rs");
        tmemLatencyKernel<<<1, tpb, 0, str>>>(dOut, loops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double totalNs = ms * 1e6;
        double nsPerRoundTrip = totalNs / loops;
        vals.push_back(nsPerRoundTrip);
    }

    std::ostringstream p;
    p << "{\"loops\":" << loops << ",\"tpb\":" << tpb
      << ",\"type\":\"tcgen05_tmem\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 3);
    res.suite_name = "tmem";
    res.test_name  = "tmem_latency";
    res.unit       = "ns";
    res.params_json = p.str();
    res.metadata["proxy"] = "false";
    res.metadata["tcgen05"] = "true";
    res.metadata["note"] = "TMEM round-trip: alloc->mma->ld->dealloc->relinquish";
    res.peak_pct = 0.0;

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Test 4: TMEM vs SMEM Bandwidth Comparison ────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult measureTMEMvsSMEM(int device, int iterations) {
    cudaEvent_t evS, evE;
    chk(cudaEventCreate(&evS), "es");
    chk(cudaEventCreate(&evE), "ee");
    cudaStream_t str;
    chk(cudaStreamCreate(&str), "st");
    chk(cudaSetDevice(device), "dev");

    // TMEM bandwidth
    constexpr int TmemLoops = 4096;
    int tpb = 32;

    float* dOut = nullptr;
    chk(cudaMalloc(&dOut, sizeof(float)), "out");

    double tmemBw = 0.0;
    {
        for (int w = 0; w < 3; ++w) {
            tmemReadBwKernel<<<1, tpb, 0, str>>>(dOut, TmemLoops);
            chk(cudaStreamSynchronize(str), "warmup");
        }

        chk(cudaEventRecord(evS, str), "rs");
        tmemReadBwKernel<<<1, tpb, 0, str>>>(dOut, TmemLoops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double totalBytes = static_cast<double>(TmemLoops) * 32 * sizeof(uint32_t);
        tmemBw = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
    }

    // SMEM bandwidth
    constexpr int TileBytes = 128;
    constexpr int SmemLoops = 4096;
    double smemBw = 0.0;
    {
        for (int w = 0; w < 3; ++w) {
            smemReadBwKernel<TileBytes><<<1, tpb, 0, str>>>(dOut, SmemLoops);
            chk(cudaStreamSynchronize(str), "warmup");
        }

        chk(cudaEventRecord(evS, str), "rs");
        smemReadBwKernel<TileBytes><<<1, tpb, 0, str>>>(dOut, SmemLoops);
        chk(cudaEventRecord(evE, str), "re");
        chk(cudaStreamSynchronize(str), "sy");

        float ms = 0;
        chk(cudaEventElapsedTime(&ms, evS, evE), "et");
        double sec = ms / 1000.0;
        double totalBytes = static_cast<double>(SmemLoops) * TileBytes;
        smemBw = sec > 0.0 ? (totalBytes / 1e9) / sec : 0.0;
    }

    std::vector<double> vals;
    vals.push_back(tmemBw);

    std::ostringstream p;
    p << "{\"tmem_loops\":" << TmemLoops
      << ",\"smem_tile_bytes\":" << TileBytes << ",\"smem_loops\":" << SmemLoops
      << ",\"tpb\":" << tpb
      << ",\"tmem_bw_gb_s\":" << tmemBw
      << ",\"smem_bw_gb_s\":" << smemBw
      << ",\"tmem_smem_ratio\":" << (smemBw > 0 ? tmemBw / smemBw : 0.0)
      << ",\"type\":\"tcgen05_tmem\"}";

    BenchResult res = ::deusridet::bench::computeStats(vals, 0);
    res.suite_name = "tmem";
    res.test_name  = "tmem_vs_smem";
    res.unit       = "GB/s";
    res.params_json = p.str();
    res.metadata["proxy"] = "false";
    res.metadata["tcgen05"] = "true";
    res.metadata["note"] = "TMEM vs SMEM read bandwidth (tcgen05 vs __shared__)";
    res.peak_pct = 0.0;

    chk(cudaFree(dOut), "out");
    chk(cudaStreamDestroy(str), "ds");
    chk(cudaEventDestroy(evS), "de");
    chk(cudaEventDestroy(evE), "de");

    return res;
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Fallback stub results ────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
BenchResult makeStubResult(const char* test_name, const char* unit, const char* reason) {
    BenchResult r{};
    r.suite_name = "tmem";
    r.test_name  = test_name;
    r.unit       = unit;
    r.peak_pct   = 0.0;
    std::string err = "{\"error\":\"";
    err += reason;
    err += "\",\"fallback\":\"smem_proxy\"}";
    r.params_json = err;
    r.metadata["proxy"] = "true";
    r.metadata["tcgen05"] = "false";
    r.metadata["stub"] = "true";
    r.metadata["stub_reason"] = reason;
    return r;
}

} // anonymous namespace

std::vector<BenchResult> runTMEMBench(int device, int matDim, int iterations) {
    (void)matDim;
    std::vector<BenchResult> results;

    // Probe: if tcgen05.mma (mxf4nvf4) is not supported, return stubs
    bool supported = tmemSupported(device);

    if (!supported) {
        results.push_back(makeStubResult("tmem_read_bandwidth", "GB/s",
            "tcgen05 alloc/mma/ld not supported by current driver/firmware"));
        results.push_back(makeStubResult("tmem_write_bandwidth", "GB/s",
            "tcgen05 alloc/mma not supported by current driver/firmware"));
        results.push_back(makeStubResult("tmem_latency", "ns",
            "tcgen05 alloc/mma/ld not supported by current driver/firmware"));
        results.push_back(makeStubResult("tmem_vs_smem", "GB/s",
            "tcgen05 alloc/mma/ld not supported by current driver/firmware"));
        return results;
    }

    try {
        results.push_back(measureTMEMReadBW(device, iterations));
    } catch (const std::exception& ex) {
        results.push_back(makeStubResult("tmem_read_bandwidth", "GB/s",
            (std::string("exception: ") + ex.what()).c_str()));
    }

    try {
        results.push_back(measureTMEMWriteBW(device, iterations));
    } catch (const std::exception& ex) {
        results.push_back(makeStubResult("tmem_write_bandwidth", "GB/s",
            (std::string("exception: ") + ex.what()).c_str()));
    }

    try {
        results.push_back(measureTMEMLatency(device, iterations));
    } catch (const std::exception& ex) {
        results.push_back(makeStubResult("tmem_latency", "ns",
            (std::string("exception: ") + ex.what()).c_str()));
    }

    try {
        results.push_back(measureTMEMvsSMEM(device, iterations));
    } catch (const std::exception& ex) {
        results.push_back(makeStubResult("tmem_vs_smem", "GB/s",
            (std::string("exception: ") + ex.what()).c_str()));
    }

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(tmem, "TCGen05 TMEM bandwidth and latency (tcgen05.alloc/ld/mma)",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runTMEMBench(0, 512, 10);
    });
