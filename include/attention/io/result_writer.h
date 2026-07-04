#pragma once

#include "attention/pipeline/attention_pipeline.h"
#include <string>

namespace attention
{
namespace io
{

/**
 * ResultWriter emits pipeline results in the interchange format
 * (attention-result/v1): a JSON file with the ordered fixation sequence and
 * run parameters, plus a sibling 16-bit grayscale PNG of the saliency map.
 *
 * Every model in the framework — C++ pipelines and Python-side models alike —
 * produces this format, and the Python evaluation layer only ever consumes it.
 * See docs/INTERCHANGE_FORMAT.md for the schema.
 */
class ResultWriter
{
 public:
  /**
   * Write results of a processed pipeline.
   * @param pipeline Pipeline after a successful process() call
   * @param json_path Destination JSON path; the saliency map is written next
   *                  to it as "<stem>_saliency.png". Parent directories are
   *                  created as needed.
   * @throws std::runtime_error if the pipeline is unprocessed or writing fails
   */
  static void write(const pipeline::AttentionPipeline& pipeline, const std::string& json_path);
};

} // namespace io
} // namespace attention
