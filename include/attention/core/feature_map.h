#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace attention
{
namespace core
{

/**
 * FeatureMap represents a computed feature with its saliency data.
 * Each feature (color, intensity, etc.) produces a FeatureMap.
 */
struct FeatureMap
{
  // Feature identification
  std::string name;

  // Feature data - normalized saliency values [0, 1]
  // Type: CV_32F (float), single channel
  cv::Mat data;

  // Confidence/weight for this feature [0, 1]
  // Can be used to dynamically weight features based on quality
  float confidence = 1.0f;

  // Default constructor
  FeatureMap() = default;

  // Constructor with name and data
  FeatureMap(const std::string& feature_name, const cv::Mat& feature_data) : name(feature_name), data(feature_data) {}

  // Constructor with name, data, and confidence
  FeatureMap(const std::string& feature_name, const cv::Mat& feature_data, float conf)
    : name(feature_name), data(feature_data), confidence(conf)
  {
  }

  // Move semantics
  FeatureMap(FeatureMap&& other) noexcept = default;
  FeatureMap& operator=(FeatureMap&& other) noexcept = default;

  // Copy semantics
  FeatureMap(const FeatureMap& other) = default;
  FeatureMap& operator=(const FeatureMap& other) = default;

  // Query methods
  bool empty() const { return data.empty(); }
  cv::Size size() const { return data.size(); }
  bool is_valid() const { return !data.empty() && data.type() == CV_32F && data.channels() == 1; }

  // Normalize the feature map to [0, 1]
  void normalize()
  {
    if (!data.empty())
    {
      cv::normalize(data, data, 0.0, 1.0, cv::NORM_MINMAX);
    }
  }
};

} // namespace core
} // namespace attention
