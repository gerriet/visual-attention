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

  /**
   * Write every feature map of a processed pipeline as "feature_<name>.png"
   * into `directory` — 16-bit grayscale on a *fixed* scale ([0, 1] -> [0,
   * 65535], values outside clipped), never stretched. For measurements that
   * need absolute feature responses (the replication dossier's variation
   * curves); the visualizations elsewhere are normalized for viewing.
   * @throws std::runtime_error if the pipeline is unprocessed or writing fails
   */
  static void write_features(const pipeline::AttentionPipeline& pipeline, const std::string& directory);
};

} // namespace io
} // namespace attention
