#pragma once

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>

namespace attention
{
namespace core
{

/**
 * Peak represents a salient location in the saliency map.
 */
struct Peak
{
  cv::Point location;
  float value;

  Peak() : location(0, 0), value(0.0f) {}
  Peak(const cv::Point& loc, float val) : location(loc), value(val) {}

  // For sorting peaks by value (descending)
  bool operator<(const Peak& other) const
  {
    return value > other.value; // Note: reversed for descending order
  }
};

/**
 * SaliencyMap holds the integrated attention map with its selected peaks.
 * Pure data: peak selection lives in selection::SelectionStrategy, fusion in
 * fusion::FusionStrategy.
 */
struct SaliencyMap
{
  // Saliency data - normalized values [0, 1]
  // Type: CV_32F (float), single channel
  cv::Mat map;

  // Selected salient locations, in scanpath order
  std::vector<Peak> peaks;

  // Default constructor
  SaliencyMap() = default;

  // Constructor from cv::Mat
  explicit SaliencyMap(const cv::Mat& saliency_data) : map(saliency_data) {}

  // Move semantics
  SaliencyMap(SaliencyMap&& other) noexcept = default;
  SaliencyMap& operator=(SaliencyMap&& other) noexcept = default;

  // Copy semantics
  SaliencyMap(const SaliencyMap& other) = default;
  SaliencyMap& operator=(const SaliencyMap& other) = default;

  // Query methods
  bool empty() const { return map.empty(); }
  cv::Size size() const { return map.size(); }
  bool is_valid() const { return !map.empty() && map.type() == CV_32F && map.channels() == 1; }

  // Get the global maximum
  Peak get_global_max() const
  {
    if (map.empty())
    {
      return Peak();
    }

    cv::Point max_loc;
    double max_val;
    cv::minMaxLoc(map, nullptr, &max_val, nullptr, &max_loc);
    return Peak(max_loc, static_cast<float>(max_val));
  }

  // Get top N peaks (sorted by value, descending)
  std::vector<Peak> get_top_peaks(int n) const
  {
    if (peaks.empty())
    {
      return std::vector<Peak>();
    }

    std::vector<Peak> sorted_peaks = peaks;
    std::sort(sorted_peaks.begin(), sorted_peaks.end());

    int count = std::min(n, static_cast<int>(sorted_peaks.size()));
    return std::vector<Peak>(sorted_peaks.begin(), sorted_peaks.begin() + count);
  }

  // Normalize the saliency map to [0, 1]
  void normalize()
  {
    if (!map.empty())
    {
      cv::normalize(map, map, 0.0f, 1.0f, cv::NORM_MINMAX);
    }
  }
};

} // namespace core
} // namespace attention
