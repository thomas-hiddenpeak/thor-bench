#pragma once

#include "bench_schema.h"
#include <string>

namespace deusridet::bench {

std::string serializeJson(const BenchReport& report);

}
