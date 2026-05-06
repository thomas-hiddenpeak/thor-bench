#pragma once

namespace deusridet::bench {

// T5000 theoretical peak values (from NVIDIA datasheet + Blackwell tuning guide).
// Values expressed in natural units per second.
//
// NOTE: These are THEORETICAL PEAKS derived from the datasheet spec formulas:
//   SM_count × TCs_per_SM × clock × ops_per_cycle × sparsity_multiplier
// The community has NOT measured anywhere close to these peaks on a Thor DevKit.
// CUTLASS v4.4.1 profiler on Thor achieves ~0.4 TFLOPS dense FP4 vs 1035 TFLOPS peak.
// Sparse FP4/FP8 CUTLASS kernels do not compile for SM110a (tcgen05.alloc unsupported).
// See: https://forums.developer.nvidia.com/t/verifying-claimed-tops-performance-on-jetson-thor/352063
//
struct T5000Peaks {
    // FP32: 2560 cores × 2 FLOP/cycle × 1.575 GHz
    static constexpr double fp32_tflops  = 8.064;
    // FP64: Blackwell SM 1:64 FP64:FP32 ratio (8.064 / 64)
    static constexpr double fp64_tflops  = 0.126;
    // FP16/BF16 TC peak not in datasheet; use FP32 scalar as fallback reference
    static constexpr double fp16_tflops  = 8.064;
    // FP4: Dense / Sparse (2:4) from datasheet — THEORETICAL, not measured.
    // Community best: ~0.88 TFLOPS dense FP4 via CUTLASS (john_c, Nov 2025).
    static constexpr double fp4_dense_tflops  = 1035.0;
    static constexpr double fp4_sparse_tflops = 2070.0;
    // FP8: Dense / Sparse (2:4) from datasheet — THEORETICAL, not measured.
    // Community best: ~0.29 TFLOPS dense FP8 via CUTLASS (AastaLLL/NVIDIA, Mar 2026).
    static constexpr double fp8_dense_tflops  = 517.0;
    static constexpr double fp8_sparse_tflops = 1035.0;
    // INT8 Tensor Core peak not in T5000 datasheet; scalar fallback uses FP32 reference.
    // TODO: Replace with actual INT8 TC peak when available (likely ~517 TFLOPS, same as FP8 dense).
    static constexpr double int8_top     = 8.064;
    // LPDDR5X: 256-bit @ 4266 MHz
    static constexpr double memory_bandwidth_gbs = 273.0;
    // L2 cache: ~50 MB estimated, bandwidth benchmark empirical;
    // 640-bit internal bus @ 1.575 GHz ≈ 246 GB/s theoretical internal bandwidth
    static constexpr double l2_bandwidth_gbs = 246.0;
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
