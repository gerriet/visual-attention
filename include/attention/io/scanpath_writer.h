#pragma once

#include "attention/system/attention_system.h"
#include <string>

namespace attention
{
namespace io
{

/**
 * Writes the scanpath of an AttentionSystem run (the ordered foci over the
 * stream, plus the final object files) as JSON in the interchange family
 * ("attention-scanpath/v1"). Sibling to ResultWriter, which handles the
 * single-frame first-stage result.
 */
class ScanpathWriter
{
 public:
  static void write(const system::AttentionSystem& system, const std::string& json_path);
};

} // namespace io
} // namespace attention
