#pragma once

#include "bench_schema.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace deusridet::bench {

inline BenchResult computeStats(std::vector<double>& vals, int warmup = 0) {
    BenchResult res;
    res.warmup_count = warmup;
    int n = static_cast<int>(vals.size());
    res.sample_count = n;
    if (n == 0) return res;

    std::sort(vals.begin(), vals.end());
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    double mean = sum / n;

    res.min_val  = vals.front();
    res.max_val  = vals.back();
    res.mean     = mean;
    res.median   = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;

    double sq = 0.0;
    for (double v : vals) { double d = v - mean; sq += d * d; }
    res.stddev = std::sqrt(sq / n);

    auto pct = [&](double p) -> double {
        if (n <= 1) return vals[0];
        double r = p * (n - 1);
        int lo = static_cast<int>(std::floor(r));
        int hi = static_cast<int>(std::ceil(r));
        if (hi >= n) return vals.back();
        return vals[lo] * (1.0 - (r - lo)) + vals[hi] * (r - lo);
    };
    res.p95 = pct(0.95);
    res.p99 = pct(0.99);
    return res;
}

} // namespace deusridet::bench
