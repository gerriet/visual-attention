#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace attention
{
namespace features
{

/**
 * ColorFeature extracts color saliency using opponent color channels.
 *
 * Based on the Itti-Koch model, this feature:
 * 1. Converts RGB to opponent color space (RG = R-G, BY = B-Y)
 * 2. Creates Gaussian pyramids for multi-scale processing
 * 3. Computes center-surround differences across scales
 * 4. Combines into a single color saliency map
 *
 * References:
 * - Itti, Koch, Niebur (1998): A Model of Saliency-Based Visual Attention
 * - Original dissertation implementation (2003-2005)
 */
class ColorFeature
{
 public:
  /**
   * Configuration for color feature extraction.
   */
  struct Config
  {
    int pyramid_levels;      // Number of pyramid levels (default: 5)
    int center_levels[3];    // Center scales (c)
    int surround_deltas[2];  // Surround offsets (delta)
    bool normalize_channels; // Normalize color channels before processing

    Config() : pyramid_levels(6), center_levels{2, 3, 4}, surround_deltas{1, 2}, normalize_channels(true) {}
    // Original Itti-Koch uses c ∈ {2,3,4}, δ ∈ {3,4} with 9-level pyramid
    // We use smaller deltas to fit in 6-level pyramid: surround = c + δ ∈ {3,4,5,6}
  };

  explicit ColorFeature(const Config& config = Config());

  /**
   * Extract color feature from frame.
   * @param frame Input frame (must be color image)
   * @return Color feature map with saliency values [0, 1]
   * @throws std::runtime_error if frame is not color
   */
  core::FeatureMap extract(const core::Frame& frame) const;

 private:
  Config config_;

  // Convert RGB to opponent color channels
  void compute_opponent_colors(const cv::Mat& rgb, cv::Mat& rg, cv::Mat& by) const;

  // Create Gaussian pyramid
  std::vector<cv::Mat> create_pyramid(const cv::Mat& input, int levels) const;

  // Compute center-surround differences for one channel
  cv::Mat compute_center_surround(const std::vector<cv::Mat>& pyramid) const;

  // Normalize and resize to original size
  cv::Mat normalize_and_resize(const cv::Mat& feature, const cv::Size& target_size) const;
};

} // namespace features
} // namespace attention
