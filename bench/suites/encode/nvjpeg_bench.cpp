#include "encode/nvjpeg_bench.h"
#include "bench_schema.h"
#include "bench_stats.h"
#include "bench_peaks.h"
#include "bench_suites.h"
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace deusridet::bench {

namespace {

static void chkCUDA(cudaError_t e, const char* msg) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA(") + msg + "): " + cudaGetErrorString(e));
}

static void chkNVJPEG(nvjpegStatus_t s, const char* msg) {
    if (s != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error(std::string("NVJPEG(") + msg + "): status=" + std::to_string(static_cast<int>(s)));
}

static std::vector<unsigned char> makeRGBPattern(int w, int h) {
    std::vector<unsigned char> buf(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * 3;
            buf[idx]     = static_cast<unsigned char>((x * 17 + y * 31) % 256);
            buf[idx + 1] = static_cast<unsigned char>((x * 53 + y * 71) % 256);
            buf[idx + 2] = static_cast<unsigned char>((x * 89 + y * 113) % 256);
        }
    }
    return buf;
}

static BenchResult makeStub(const std::string& testName, const std::string& reason) {
    BenchResult r;
    r.suite_name   = "nvjpeg";
    r.test_name    = testName;
    r.unit         = "fps";
    r.sample_count = 0;
    r.warmup_count = 0;
    r.median       = 0.0;
    r.metadata["stub"]        = "true";
    r.metadata["stub_reason"] = reason;
    return r;
}

static void runEncodeOp(nvjpegHandle_t handle,
                        nvjpegEncoderState_t encState,
                        nvjpegEncoderParams_t encParams,
                        unsigned char* dRGB,
                        unsigned char* dJpegBuf,
                        size_t jpegBufSize,
                        int w, int h,
                        cudaStream_t stream) {
    nvjpegImage_t src{};
    src.channel[0] = dRGB;
    src.pitch[0]   = static_cast<size_t>(w) * 3;
    chkNVJPEG(nvjpegEncode(handle, encState, encParams, &src,
                            NVJPEG_CSS_420, NVJPEG_INPUT_RGBI,
                            w, h, stream), "nvjpegEncode");
    chkNVJPEG(nvjpegEncodeRetrieveBitstream(handle, encState, dJpegBuf, nullptr, stream),
              "nvjpegEncodeRetrieveBitstream");
}

static void runDecodeOp(nvjpegHandle_t handle,
                        nvjpegJpegState_t decState,
                        const unsigned char* dJpegData,
                        size_t jpegLen,
                        unsigned char* dRGB,
                        int w, int h,
                        cudaStream_t stream) {
    nvjpegImage_t dst{};
    dst.channel[0] = dRGB;
    dst.pitch[0]   = static_cast<size_t>(w) * 3;
    chkNVJPEG(nvjpegDecode(handle, decState, dJpegData, jpegLen,
                            NVJPEG_OUTPUT_RGBI, &dst, stream),
              "nvjpegDecode");
}

static const std::vector<std::pair<std::string, std::pair<int, int>>> RESOLUTIONS = {
    {"1080p", {1920, 1080}},
    {"4K",    {3840, 2160}},
};

} // anonymous namespace

std::vector<BenchResult> runNVJPEGBench(int device, int imageWidth, int imageHeight, int iterations) {
    std::vector<BenchResult> results;
    (void)imageWidth;
    (void)imageHeight;

    int dCount = 0;
    if (cudaGetDeviceCount(&dCount) != cudaSuccess || dCount == 0) {
        std::string reason = "CUDA device unavailable";
        for (const auto& [label, wh] : RESOLUTIONS) {
            results.push_back(makeStub("encode_" + label, reason));
            results.push_back(makeStub("decode_" + label, reason));
        }
        return results;
    }

    chkCUDA(cudaSetDevice(device), "cudaSetDevice");
    cudaStream_t stream;
    chkCUDA(cudaStreamCreate(&stream), "cudaStreamCreate");

    nvjpegHandle_t nvjpegHandle = nullptr;
    nvjpegBackend_t backend = NVJPEG_BACKEND_HARDWARE;

    if (nvjpegCreate(NVJPEG_BACKEND_HARDWARE, nullptr, &nvjpegHandle) != NVJPEG_STATUS_SUCCESS) {
        if (nvjpegCreate(NVJPEG_BACKEND_GPU_HYBRID, nullptr, &nvjpegHandle) != NVJPEG_STATUS_SUCCESS) {
            std::string reason = "nvJPEG init failed (hardware & GPU_HYBRID backend)";
            for (const auto& [label, wh] : RESOLUTIONS) {
                results.push_back(makeStub("encode_" + label, reason));
                results.push_back(makeStub("decode_" + label, reason));
            }
            chkCUDA(cudaStreamDestroy(stream), "cudaStreamDestroy");
            return results;
        }
        backend = NVJPEG_BACKEND_GPU_HYBRID;
    }

    nvjpegEncBackend_t encBackend = NVJPEG_ENC_BACKEND_HARDWARE;
    {
        nvjpegEncoderState_t testEnc = nullptr;
        nvjpegStatus_t s = nvjpegEncoderStateCreateWithBackend(nvjpegHandle, &testEnc, NVJPEG_ENC_BACKEND_HARDWARE, stream);
        if (s != NVJPEG_STATUS_SUCCESS) {
            s = nvjpegEncoderStateCreateWithBackend(nvjpegHandle, &testEnc, NVJPEG_ENC_BACKEND_GPU, stream);
            if (s != NVJPEG_STATUS_SUCCESS) {
                encBackend = NVJPEG_ENC_BACKEND_DEFAULT;
            } else {
                encBackend = NVJPEG_ENC_BACKEND_GPU;
            }
        }
        if (testEnc) nvjpegEncoderStateDestroy(testEnc);
    }

    std::string encBackendName = (encBackend == NVJPEG_ENC_BACKEND_HARDWARE ? "HARDWARE" :
                                  encBackend == NVJPEG_ENC_BACKEND_GPU      ? "GPU" : "DEFAULT");
    std::string decBackendName = (backend == NVJPEG_BACKEND_HARDWARE   ? "HARDWARE" :
                                  backend == NVJPEG_BACKEND_GPU_HYBRID ? "GPU_HYBRID" : "DEFAULT");

    for (const auto& [label, wh] : RESOLUTIONS) {
        int w = wh.first, h = wh.second;
        size_t rgbBytes = static_cast<size_t>(w) * h * 3ULL;

        unsigned char* dRGB = nullptr;
        chkCUDA(cudaMalloc(&dRGB, rgbBytes), "malloc RGB");
        {
            auto hPattern = makeRGBPattern(w, h);
            chkCUDA(cudaMemcpy(dRGB, hPattern.data(), rgbBytes, cudaMemcpyHostToDevice), "memcpy pattern");
        }

        nvjpegEncoderState_t encState = nullptr;
        nvjpegEncoderParams_t encParams = nullptr;
        size_t jpegBufSize = 0;
        bool encodeAvailable = true;

        if (nvjpegEncoderStateCreateWithBackend(nvjpegHandle, &encState, encBackend, stream) != NVJPEG_STATUS_SUCCESS) {
            encodeAvailable = false;
        }
        if (encodeAvailable && nvjpegEncoderParamsCreate(nvjpegHandle, &encParams, stream) != NVJPEG_STATUS_SUCCESS) {
            encodeAvailable = false;
        }

        unsigned char* dJpegBuf = nullptr;
        if (encodeAvailable) {
            chkNVJPEG(nvjpegEncoderParamsSetQuality(encParams, 80, stream), "setQuality");
            size_t maxLen = 0;
            if (nvjpegEncodeGetBufferSize(nvjpegHandle, encParams, w, h, &maxLen) != NVJPEG_STATUS_SUCCESS) {
                maxLen = rgbBytes;
            }
            jpegBufSize = maxLen;
            chkCUDA(cudaMalloc(&dJpegBuf, jpegBufSize), "malloc JPEG buf");
        }

        std::vector<double> encodeFPS;
        encodeFPS.reserve(iterations);

        if (encodeAvailable) {
            bool encodeWorks = false;
            try {
                for (int i = 0; i < 3; ++i) {
                    runEncodeOp(nvjpegHandle, encState, encParams, dRGB, dJpegBuf, jpegBufSize, w, h, stream);
                    chkCUDA(cudaStreamSynchronize(stream), "encode warmup");
                }
                encodeWorks = true;
            } catch (...) {
                encodeWorks = false;
                encodeAvailable = false;
            }

            if (encodeWorks) {
                auto hPattern = makeRGBPattern(w, h);
                for (int i = 0; i < iterations; ++i) {
                    chkCUDA(cudaMemcpy(dRGB, hPattern.data(), rgbBytes, cudaMemcpyHostToDevice), "memcpy pattern");

                    auto t0 = std::chrono::steady_clock::now();
                    runEncodeOp(nvjpegHandle, encState, encParams, dRGB, dJpegBuf, jpegBufSize, w, h, stream);
                    chkCUDA(cudaStreamSynchronize(stream), "encode sync");
                    auto t1 = std::chrono::steady_clock::now();

                    double sec = std::chrono::duration<double>(t1 - t0).count();
                    encodeFPS.push_back(sec > 0.0 ? 1.0 / sec : 0.0);
                }
            }
        }

        if (encParams)  nvjpegEncoderParamsDestroy(encParams);
        if (encState)   nvjpegEncoderStateDestroy(encState);

        {
            BenchResult r;
            if (encodeFPS.empty()) {
                r = makeStub("encode_" + label, "NVJPEG hardware encoder not available on this device");
            } else {
                r = computeStats(encodeFPS);
                r.suite_name  = "nvjpeg";
                r.test_name   = "encode_" + label;
                r.unit        = "fps";
            }
            if (encodeFPS.empty()) {
                std::ostringstream ps;
                ps << "{\"width\":" << w << ",\"height\":" << h
                   << ",\"quality\":80"
                   << ",\"input_format\":\"RGBI\""
                   << ",\"subsampling\":\"420\""
                   << ",\"encoder_backend\":\"" << encBackendName << "\"}";
                r.params_json = ps.str();
            }
            r.metadata["codec"]      = "jpeg";
            r.metadata["resolution"] = std::to_string(w) + "x" + std::to_string(h);
            r.metadata["engine"]     = "NVJPEG";
            r.metadata["mode"]       = "encode";
            results.push_back(std::move(r));
        }

        nvjpegJpegState_t decState = nullptr;
        unsigned char* dRGBOut = nullptr;
        unsigned char* dJpegData = nullptr;
        size_t jpegLen = 0;
        bool decodeAvailable = encodeAvailable;

        if (decodeAvailable) {
            nvjpegEncoderState_t tmpEnc = nullptr;
            nvjpegEncoderParams_t tmpParams = nullptr;
            unsigned char* tmpJpegBuf = nullptr;

            if (nvjpegEncoderStateCreateWithBackend(nvjpegHandle, &tmpEnc, encBackend, stream) != NVJPEG_STATUS_SUCCESS) {
                decodeAvailable = false;
            }
            if (decodeAvailable && nvjpegEncoderParamsCreate(nvjpegHandle, &tmpParams, stream) != NVJPEG_STATUS_SUCCESS) {
                decodeAvailable = false;
            }
            if (decodeAvailable) {
                try {
                    chkNVJPEG(nvjpegEncoderParamsSetQuality(tmpParams, 80, stream), "setQuality");
                    size_t maxLen = 0;
                    if (nvjpegEncodeGetBufferSize(nvjpegHandle, tmpParams, w, h, &maxLen) != NVJPEG_STATUS_SUCCESS) {
                        maxLen = rgbBytes;
                    }
                    chkCUDA(cudaMalloc(&tmpJpegBuf, maxLen), "malloc decode JPEG buf");

                    {
                        auto hPat = makeRGBPattern(w, h);
                        chkCUDA(cudaMemcpy(dRGB, hPat.data(), rgbBytes, cudaMemcpyHostToDevice), "memcpy pattern");
                        chkCUDA(cudaStreamSynchronize(stream), "fill sync");
                    }

                    runEncodeOp(nvjpegHandle, tmpEnc, tmpParams, dRGB, tmpJpegBuf, maxLen, w, h, stream);
                    chkCUDA(cudaStreamSynchronize(stream), "pre-encode sync");

                    size_t* hLen = nullptr;
                    chkCUDA(cudaMallocHost(&hLen, sizeof(size_t)), "mallocHost decode len");
                    nvjpegEncodeRetrieveBitstream(nvjpegHandle, tmpEnc, tmpJpegBuf, hLen, stream);
                    chkCUDA(cudaStreamSynchronize(stream), "decode retrieve sync");
                    jpegLen = *hLen;
                    chkCUDA(cudaFreeHost(hLen), "freeHost decode len");

                    chkCUDA(cudaMalloc(&dJpegData, jpegLen), "malloc decode jpeg data");
                    chkCUDA(cudaMemcpy(dJpegData, tmpJpegBuf, jpegLen, cudaMemcpyDefault), "memcpy decode jpeg");

                    chkCUDA(cudaFree(tmpJpegBuf), "free tmp jpeg buf");
                } catch (...) {
                    jpegLen = 0;
                    if (tmpJpegBuf) chkCUDA(cudaFree(tmpJpegBuf), "free tmp jpeg buf");
                }
                nvjpegEncoderParamsDestroy(tmpParams);
                nvjpegEncoderStateDestroy(tmpEnc);
            } else {
                if (tmpParams) nvjpegEncoderParamsDestroy(tmpParams);
                if (tmpEnc)    nvjpegEncoderStateDestroy(tmpEnc);
            }
        }

        if (decodeAvailable && jpegLen > 0) {
            if (nvjpegJpegStateCreate(nvjpegHandle, &decState) != NVJPEG_STATUS_SUCCESS) {
                decodeAvailable = false;
            } else {
                chkCUDA(cudaMalloc(&dRGBOut, rgbBytes), "malloc RGB out");
            }
        }

        std::vector<double> decodeFPS;
        decodeFPS.reserve(iterations);

        if (decodeAvailable && decState != nullptr) {
            for (int i = 0; i < 3; ++i) {
                runDecodeOp(nvjpegHandle, decState, dJpegData, jpegLen, dRGBOut, w, h, stream);
                chkCUDA(cudaStreamSynchronize(stream), "decode warmup");
            }

            for (int i = 0; i < iterations; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                runDecodeOp(nvjpegHandle, decState, dJpegData, jpegLen, dRGBOut, w, h, stream);
                chkCUDA(cudaStreamSynchronize(stream), "decode sync");
                auto t1 = std::chrono::steady_clock::now();

                double sec = std::chrono::duration<double>(t1 - t0).count();
                decodeFPS.push_back(sec > 0.0 ? 1.0 / sec : 0.0);
            }
        }

        if (dRGBOut)  chkCUDA(cudaFree(dRGBOut), "free RGB out");
        if (decState) nvjpegJpegStateDestroy(decState);
        if (dJpegData) chkCUDA(cudaFree(dJpegData), "free JPEG data");

        {
            BenchResult r = computeStats(decodeFPS);
            r.suite_name  = "nvjpeg";
            r.test_name   = "decode_" + label;
            r.unit        = "fps";
            {
                std::ostringstream ps;
                ps << "{\"width\":" << w << ",\"height\":" << h
                   << ",\"output_format\":\"RGBI\""
                   << ",\"decoder_backend\":\"" << decBackendName << "\"}";
                r.params_json = ps.str();
            }
            r.metadata["codec"]      = "jpeg";
            r.metadata["resolution"] = std::to_string(w) + "x" + std::to_string(h);
            r.metadata["engine"]     = "NVJPEG";
            r.metadata["mode"]       = "decode";
            results.push_back(std::move(r));
        }

        chkCUDA(cudaFree(dRGB), "free RGB in");
    }

    nvjpegDestroy(nvjpegHandle);
    chkCUDA(cudaStreamDestroy(stream), "cudaStreamDestroy");

    return results;
}

} // namespace deusridet::bench

BENCH_REGISTER_SUITE(nvjpeg, "NVJPEG hardware-accelerated encode/decode benchmark",
    [](deusridet::bench::BenchRunner& runner) -> std::vector<deusridet::bench::BenchResult> {
        return deusridet::bench::runNVJPEGBench(0, 1920, 1080, 10);
    });
