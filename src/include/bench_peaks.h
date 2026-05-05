#pragma once

namespace deusridet::bench {

// T5000 theoretical peak values (from NVIDIA datasheet + Blackwell tuning guide).
// Values expressed in natural units per second.
struct T5000Peaks {
    // FP32: 2560 cores × 2 FLOP/cycle × 1.575 GHz
    static constexpr double fp32_tflops  = 8.064;
    // FP4: Dense / Sparse (2:4) from datasheet
    static constexpr double fp4_dense_tflops  = 1035.0;
    static constexpr double fp4_sparse_tflops = 2070.0;
    // FP8: Dense / Sparse (2:4) from datasheet
    static constexpr double fp8_dense_tflops  = 517.0;
    static constexpr double fp8_sparse_tflops = 1035.0;
    // LPDDR5X: 256-bit @ 4266 MHz
    static constexpr double memory_bandwidth_gbs = 273.0;
};

// Compute % of theoretical peak from a measured G-unit value (GFLOP/s) and T-unit peak (TFLOPS).
inline double computePeakPctFromG(double measuredGval, double peakTflopsOrGbs) {
    double peakGval = peakTflopsOrGbs * 1000.0;
    return (measuredGval / peakGval) * 100.0;
}

// Compute % of theoretical peak when both measured and peak are in the same unit (e.g. GB/s vs GB/s).
inline double computePeakPctSame(double measured, double peak) {
    return (measured / peak) * 100.0;
}

inline double computePeakPctFromT(double measuredTflops, double peakTflops) {
    return (measuredTflops / peakTflops) * 100.0;
}

}
