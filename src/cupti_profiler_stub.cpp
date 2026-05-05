#include "cupti_profiler.h"

namespace deusridet::bench {

CuptiProfiler::CuptiProfiler() = default;
CuptiProfiler::~CuptiProfiler() = default;

CuptiProfiler& CuptiProfiler::instance() {
    static CuptiProfiler inst;
    return inst;
}

bool CuptiProfiler::init(int) { return false; }
void CuptiProfiler::startRange(const char*) {}
void CuptiProfiler::stopRange() {}
std::vector<CuptiSuiteData> CuptiProfiler::getResults() const { return {}; }
CuptiOverhead CuptiProfiler::measureOverhead() { return {}; }
void CuptiProfiler::shutdown() {}

} // namespace deusridet::bench
