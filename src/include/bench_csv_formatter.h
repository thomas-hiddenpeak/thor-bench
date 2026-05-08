#pragma once

#include "sweep_schema.h"
#include <string>

namespace deusridet::bench {

std::string formatCsvHeader(const SweepReport& report);
std::string formatCsvRows(const SweepReport& report);
bool writeCsvFile(const std::string& outputPath, const SweepReport& report);
std::string defaultCsvPath(const std::string& suiteName);

} // namespace deusridet::bench
