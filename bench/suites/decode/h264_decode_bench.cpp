#include "h264_decode_bench.h"
#include "bench_suites.h"
#include <cuda.h>
#include <cuviddec.h>
#include <string>
#include <sstream>
#include <vector>

namespace deusridet::bench {

namespace {

bool probeNvdecH264(int dev) {
    CUcontext ctx = nullptr;
    CUresult res = cuCtxGetCurrent(&ctx);
    if (res != CUDA_SUCCESS || ctx == nullptr) {
        res = cuInit(0);
        if (res != CUDA_SUCCESS) return false;
        CUdevice cuDev;
        res = cuDeviceGet(&cuDev, dev);
        if (res != CUDA_SUCCESS) return false;
        CUctxCreateParams params{};
        params.execAffinityParams = nullptr;
        params.numExecAffinityParams = 0;
        params.cigParams = nullptr;
        res = cuCtxCreate(&ctx, &params, 0, cuDev);
        if (res != CUDA_SUCCESS) return false;
    }

    CUVIDDECODECAPS caps{};
    caps.eCodecType = cudaVideoCodec_H264;
    caps.eChromaFormat = cudaVideoChromaFormat_420;
    caps.nBitDepthMinus8 = 0;
    res = cuvidGetDecoderCaps(&caps);
    cuCtxDestroy(ctx);
    return (res == CUDA_SUCCESS && caps.bIsSupported);
}

}

std::vector<BenchResult> runH264DecodeBench(int device, int, int, int, int) {
    std::vector<BenchResult> results;

    bool nvdecSupported = probeNvdecH264(device);

    BenchResult r{};
    r.suite_name = "h264_decode";
    r.test_name = "nvdec_stub";
    r.unit = "fps";
    r.sample_count = 0;
    r.warmup_count = 0;
    r.metadata["codec"] = "h264";
    r.metadata["decoder"] = "NVDEC";
    r.metadata["stub"] = "true";
    r.metadata["stub_reason"] = "NVDEC decode requires real H.264 bitstream files; hardware capability probed only";
    std::ostringstream ps;
    ps << "{\"nvdec_supported\":" << (nvdecSupported ? "true" : "false")
       << ",\"codec\":\"h264\""
       << ",\"stub_reason\":\"requires_real_bitstream\"}";
    r.params_json = ps.str();
    results.push_back(r);

    return results;
}

}

BENCH_REGISTER_SUITE(h264_decode, "NVDEC H.264 decoding",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runH264DecodeBench(0, 1920, 1080, 60, 10);
    });
