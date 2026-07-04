#include "attention/fusion/fusion_strategy.h"
#include <stdexcept>

namespace attention
{
namespace fusion
{

namespace
{

/**
 * Weighted linear sum of feature maps (weight * confidence * data), then
 * min-max normalization to [0, 1]. This is the classic v1 pipeline fusion.
 */
class WeightedSumFusion : public FusionStrategy
{
 public:
  std::string name() const override { return "weighted-sum"; }

  cv::Mat fuse(const std::vector<core::FeatureMap>& features, const std::map<std::string, float>& weights,
               const cv::Size& frame_size) const override
  {
    if (features.empty())
    {
      throw std::runtime_error("WeightedSumFusion: no features to fuse");
    }

    cv::Mat integrated = cv::Mat::zeros(frame_size, CV_32F);

    for (const auto& feature : features)
    {
      if (feature.data.empty())
      {
        throw std::runtime_error("Feature '" + feature.name + "' has empty data");
      }
      if (feature.data.size() != frame_size)
      {
        throw std::runtime_error("Feature '" + feature.name + "' size mismatch: expected " +
                                 std::to_string(frame_size.width) + "x" + std::to_string(frame_size.height) +
                                 " but got " + std::to_string(feature.data.cols) + "x" +
                                 std::to_string(feature.data.rows));
      }

      float weight = 1.0f;
      auto it = weights.find(feature.name);
      if (it != weights.end())
      {
        weight = it->second;
      }

      integrated += weight * feature.confidence * feature.data;
    }

    cv::normalize(integrated, integrated, 0.0f, 1.0f, cv::NORM_MINMAX);
    return integrated;
  }
};

} // namespace

std::unique_ptr<FusionStrategy> create_fusion_strategy(const std::string& name)
{
  if (name == "weighted-sum")
  {
    return std::make_unique<WeightedSumFusion>();
  }
  throw std::runtime_error("Unknown fusion strategy '" + name + "'. Available: weighted-sum");
}

} // namespace fusion
} // namespace attention
