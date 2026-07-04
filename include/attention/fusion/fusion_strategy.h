#pragma once

#include "attention/core/feature_map.h"
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace attention
{
namespace fusion
{

/**
 * FusionStrategy combines feature maps into a single saliency map.
 *
 * Implementations are selected by name via config ("pipeline: fusion: ...").
 * The classic thesis-era alternative to the weighted sum — neural-field
 * dynamics — lands here in roadmap M3.
 */
class FusionStrategy
{
 public:
  virtual ~FusionStrategy() = default;

  virtual std::string name() const = 0;

  /**
   * Fuse feature maps into a saliency map normalized to [0, 1].
   * @param features Extracted feature maps (each CV_32F, frame-sized)
   * @param weights Per-feature weights keyed by feature name (missing = 1.0)
   * @param frame_size Expected map size; mismatching features are an error
   */
  virtual cv::Mat fuse(const std::vector<core::FeatureMap>& features,
                       const std::map<std::string, float>& weights, const cv::Size& frame_size) const = 0;
};

/**
 * Create a fusion strategy by name.
 * @param name Currently: "weighted-sum"
 * @throws std::runtime_error for unknown names
 */
std::unique_ptr<FusionStrategy> create_fusion_strategy(const std::string& name);

} // namespace fusion
} // namespace attention
