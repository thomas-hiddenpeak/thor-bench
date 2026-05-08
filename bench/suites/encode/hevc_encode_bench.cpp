#include "hevc_encode_bench.h"
#include "bench_suites.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvEncodeAPI.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace deusridet::bench {

namespace {

struct NvencSession {
    NV_ENCODE_API_FUNCTION_LIST api{};
    void* session = nullptr;
    NV_ENC_INPUT_PTR inputBuf = nullptr;
    NV_ENC_OUTPUT_PTR bitstreamBuf = nullptr;
    bool ownsCtx = false;
    CUcontext cuCtx = nullptr;

    ~NvencSession() { cleanup(); }

    void cleanup() {
        if (session && api.nvEncDestroyInputBuffer && api.nvEncDestroyBitstreamBuffer) {
            if (inputBuf) api.nvEncDestroyInputBuffer(session, inputBuf);
            if (bitstreamBuf) api.nvEncDestroyBitstreamBuffer(session, bitstreamBuf);
        }
        if (session && api.nvEncDestroyEncoder) {
            api.nvEncDestroyEncoder(session);
        }
        session = nullptr;
        inputBuf = nullptr;
        bitstreamBuf = nullptr;
        if (ownsCtx && cuCtx) {
            cuCtxDestroy(cuCtx);
            cuCtx = nullptr;
        }
    }

    bool init(int dev, int w, int h, const GUID& codecGuid) {
        memset(&api, 0, sizeof(api));
        api.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        if (NvEncodeAPICreateInstance(&api) != NV_ENC_SUCCESS) {
            return false;
        }

        cudaSetDevice(dev);
        cudaFree(0);
        CUcontext ctx = nullptr;
        CUresult res = cuCtxGetCurrent(&ctx);
        if (res != CUDA_SUCCESS || ctx == nullptr) {
            cuInit(0);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, dev);
            CUctxCreateParams params{};
            cuCtxCreate(&ctx, &params, 0, cuDev);
            cuCtxSetCurrent(ctx);
        }
        cuCtx = ctx;

        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS openParams{};
        openParams.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        openParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        openParams.device = cuCtx;
        openParams.apiVersion = NVENCAPI_VERSION;

        NVENCSTATUS status = api.nvEncOpenEncodeSessionEx(&openParams, &session);
        if (status != NV_ENC_SUCCESS) {
            cleanup();
            return false;
        }

        NV_ENC_PRESET_CONFIG presetConfig{};
        presetConfig.version = NV_ENC_PRESET_CONFIG_VER;
        presetConfig.presetCfg.version = NV_ENC_CONFIG_VER;
        status = api.nvEncGetEncodePresetConfigEx(session, codecGuid,
            NV_ENC_PRESET_P2_GUID, NV_ENC_TUNING_INFO_LOW_LATENCY, &presetConfig);
        if (status != NV_ENC_SUCCESS) {
            cleanup();
            return false;
        }

        presetConfig.presetCfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
        presetConfig.presetCfg.rcParams.constQP.qpIntra = 24;
        presetConfig.presetCfg.rcParams.constQP.qpInterP = 24;
        presetConfig.presetCfg.rcParams.constQP.qpInterB = 24;

        NV_ENC_INITIALIZE_PARAMS initParams{};
        initParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
        initParams.encodeGUID = codecGuid;
        initParams.presetGUID = NV_ENC_PRESET_P2_GUID;
        initParams.tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY;
        initParams.encodeWidth = w;
        initParams.encodeHeight = h;
        initParams.darWidth = w;
        initParams.darHeight = h;
        initParams.frameRateNum = 60;
        initParams.frameRateDen = 1;
        initParams.enablePTD = 1;
        initParams.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
        initParams.encodeConfig = &presetConfig.presetCfg;

        status = api.nvEncInitializeEncoder(session, &initParams);
        if (status != NV_ENC_SUCCESS) {
            cleanup();
            return false;
        }

         NV_ENC_CREATE_INPUT_BUFFER createIn{};
        createIn.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
        createIn.width = w;
        createIn.height = h;
        createIn.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
        status = api.nvEncCreateInputBuffer(session, &createIn);
        if (status != NV_ENC_SUCCESS) {
            cleanup();
            return false;
        }
        inputBuf = createIn.inputBuffer;

        // Fill input buffer with NV12 data (Y=128, UV=128 — flat gray)
        NV_ENC_LOCK_INPUT_BUFFER lockIn{};
        lockIn.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
        lockIn.inputBuffer = inputBuf;
        status = api.nvEncLockInputBuffer(session, &lockIn);
        if (status == NV_ENC_SUCCESS) {
            uint8_t* ptr = static_cast<uint8_t*>(lockIn.bufferDataPtr);
            int pitch = lockIn.pitch;
            for (int y = 0; y < h; ++y) {
                memset(ptr + y * pitch, 0x80, w);
            }
            // UV plane (half height, interleaved, same pitch)
            int uvBase = pitch * h;
            for (int y = 0; y < h / 2; ++y) {
                memset(ptr + uvBase + y * pitch, 0x80, w);
            }
            api.nvEncUnlockInputBuffer(session, inputBuf);
        }

        NV_ENC_CREATE_BITSTREAM_BUFFER createBs{};
        createBs.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        status = api.nvEncCreateBitstreamBuffer(session, &createBs);
        if (status != NV_ENC_SUCCESS) {
            cleanup();
            return false;
        }
        bitstreamBuf = createBs.bitstreamBuffer;

        return true;
    }

    bool encodeFrame() {
        NV_ENC_PIC_PARAMS picParams{};
        picParams.version = NV_ENC_PIC_PARAMS_VER;
        picParams.inputBuffer = inputBuf;
        picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
        picParams.outputBitstream = bitstreamBuf;
        picParams.frameIdx = 0;
        picParams.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;

        return api.nvEncEncodePicture(session, &picParams) == NV_ENC_SUCCESS;
    }
};

std::vector<BenchResult> runEncodeBench(int device, int width, int height, int targetFps, int iterations,
                                         const GUID& codecGuid, const char* codecStr) {
    std::vector<BenchResult> results;

    if (width <= 0 || height <= 0 || iterations <= 0) return results;

    struct Case {
        std::string label;
        int w, h;
    };

    std::vector<Case> cases;
    cases.push_back({std::to_string(width) + "x" + std::to_string(height), width, height});
    if (width != 1920 || height != 1080)
        cases.push_back({"1080p", 1920, 1080});
    if (width != 1280 || height != 720)
        cases.push_back({"720p", 1280, 720});
    if (width != 3840 || height != 2160)
        cases.push_back({"4K", 3840, 2160});

    for (const auto& c : cases) {
        if (c.w > 7680 || c.h > 4320) {
            BenchResult r{};
            r.suite_name = codecStr;
            r.test_name = c.label;
            r.unit = "fps";
            r.sample_count = 0;
            r.metadata["codec"] = codecStr;
            r.metadata["encoder"] = "NVENC";
            r.metadata["stub"] = "true";
            r.metadata["stub_reason"] = "resolution exceeds limit";
            r.params_json = "{\"error\":\"resolution exceeds limit\"}";
            results.push_back(r);
            continue;
        }

        NvencSession enc;
        try {
            if (!enc.init(device, c.w, c.h, codecGuid)) {
                BenchResult r{};
                r.suite_name = codecStr;
                r.test_name = c.label;
                r.unit = "fps";
                r.sample_count = 0;
                r.metadata["codec"] = codecStr;
                r.metadata["encoder"] = "NVENC";
                r.metadata["stub"] = "true";
                r.metadata["stub_reason"] = "NVENC session creation failed";
                r.params_json = "{\"error\":\"NVENC session creation failed\"}";
                results.push_back(r);
                continue;
            }
        } catch (const std::exception& e) {
            BenchResult r{};
            r.suite_name = codecStr;
            r.test_name = c.label;
            r.unit = "fps";
            r.sample_count = 0;
            r.metadata["codec"] = codecStr;
            r.metadata["encoder"] = "NVENC";
            r.metadata["stub"] = "true";
            r.metadata["stub_reason"] = std::string("init error: ") + e.what();
            r.params_json = std::string("{\"error\":\"") + e.what() + "\"}";
            results.push_back(r);
            continue;
        }

        for (int w = 0; w < 3; ++w) {
            enc.encodeFrame();
        }

        constexpr int framesPerSample = 10;
        std::vector<double> fpsVals;
        fpsVals.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            cudaEvent_t evS, evE;
            if (cudaEventCreate(&evS) != cudaSuccess || cudaEventCreate(&evE) != cudaSuccess) {
                break;
            }

            cudaEventRecord(evS);
            cudaDeviceSynchronize();
            for (int f = 0; f < framesPerSample; ++f) {
                if (!enc.encodeFrame()) {
                    cudaEventDestroy(evS);
                    cudaEventDestroy(evE);
                    break;
                }
            }
            cudaDeviceSynchronize();
            cudaEventRecord(evE);
            cudaEventSynchronize(evE);

            float ms = 0;
            cudaEventElapsedTime(&ms, evS, evE);
            cudaEventDestroy(evS);
            cudaEventDestroy(evE);

            double fps = (framesPerSample * 1000.0) / ms;
            fpsVals.push_back(fps);
        }

        if (fpsVals.empty()) {
            BenchResult r{};
            r.suite_name = codecStr;
            r.test_name = c.label;
            r.unit = "fps";
            r.sample_count = 0;
            r.metadata["codec"] = codecStr;
            r.metadata["encoder"] = "NVENC";
            r.metadata["stub"] = "true";
            r.metadata["stub_reason"] = "encode failed";
            r.params_json = "{\"error\":\"encode failed\"}";
            results.push_back(r);
            continue;
        }

        std::sort(fpsVals.begin(), fpsVals.end());
        int n = static_cast<int>(fpsVals.size());
        double sum = 0.0;
        for (double v : fpsVals) sum += v;
        double mean = sum / n;
        double sq = 0.0;
        for (double v : fpsVals) sq += (v - mean) * (v - mean);

        BenchResult r{};
        r.suite_name = codecStr;
        r.test_name = c.label;
        r.unit = "fps";
        r.mean = mean;
        r.median = (n % 2 == 1) ? fpsVals[n / 2] : (fpsVals[n / 2 - 1] + fpsVals[n / 2]) / 2.0;
        r.stddev = std::sqrt(sq / n);
        r.min_val = fpsVals.front();
        r.max_val = fpsVals.back();
        r.sample_count = n;
        r.warmup_count = 3;

        std::ostringstream ps;
        ps << "{\"width\":" << c.w << ",\"height\":" << c.h
           << ",\"target_fps\":" << targetFps
           << ",\"frames_per_sample\":" << framesPerSample
           << ",\"codec\":\"" << codecStr << "\"}";
        r.params_json = ps.str();
        r.metadata["codec"] = codecStr;
        r.metadata["encoder"] = "NVENC";
        r.metadata["resolution"] = std::to_string(c.w) + "x" + std::to_string(c.h);
        results.push_back(r);
    }

    return results;
}

}

std::vector<BenchResult> runHEVCEncodeBench(int device, int width, int height, int targetFps, int iterations) {
    return runEncodeBench(device, width, height, targetFps, iterations, NV_ENC_CODEC_HEVC_GUID, "hevc_encode");
}

}

BENCH_REGISTER_SUITE(hevc_encode, "NVENC HEVC (H.265) encoding",
    [](deusridet::bench::BenchRunner&) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runHEVCEncodeBench(0, 1920, 1080, 60, 10);
    });
