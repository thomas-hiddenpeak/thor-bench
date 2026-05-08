#include "hevc_decode_bench.h"
#include "bench_suites.h"
#include <cuda.h>
#include <cuviddec.h>
#include <string>
#include <sstream>
#include <vector>

namespace deusridet::bench {

namespace {

bool probeNvdecHevc(int dev) {
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
    caps.eCodecType = cudaVideoCodec_HEVC;
    caps.eChromaFormat = cudaVideoChromaFormat_420;
    caps.nBitDepthMinus8 = 0;
    res = cuvidGetDecoderCaps(&caps);
    cuCtxDestroy(ctx);
    return (res == CUDA_SUCCESS && caps.bIsSupported);
}

}

std::vector<BenchResult> runHEVCDecodeBench(int device, int, int, int, int) {
    std::vector<BenchResult> results;

    bool nvdecSupported = probeNvdecHevc(device);

    BenchResult r{};
    r.suite_name = "hevc_decode";
    r.test_name = "nvdec_stub";
    r.unit = "fps";
    r.sample_count = 0;
    r.warmup_count = 0;
    r.metadata["codec"] = "hevc";
    r.metadata["decoder"] = "NVDEC";
    r.metadata["stub"] = "true";
    r.metadata["stub_reason"] = "NVDEC decode requires real HEVC bitstream files; hardware capability probed only";
    std::ostringstream ps;
    ps << "{\"nvdec_supported\":" << (nvdecSupported ? "true" : "false")
       << ",\"codec\":\"hevc\""
       << ",\"stub_reason\":\"requires_real_bitstream\"}";
    r.params_json = ps.str();
    results.push_back(r);

    return results;
}

}

BENCH_REGISTER_SUITE(hevc_decode, "NVDEC HEVC (H.265) decoding",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runHEVCDecodeBench(0, 1920, 1080, 60, 10);
    });
