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
 * IntensityFeature extracts intensity/brightness saliency.
 *
 * Based on the Itti-Koch model, this feature:
 * 1. Converts image to grayscale (intensity)
 * 2. Creates Gaussian pyramid for multi-scale processing
 * 3. Computes center-surround differences across scales
 * 4. Detects regions with intensity contrast
 *
 * This feature responds to bright/dark contrasts independent of color.
 *
 * References:
 * - Itti, Koch, Niebur (1998): A Model of Saliency-Based Visual Attention
 */
class IntensityFeature
{
 public:
  /**
   * Configuration for intensity feature extraction.
   */
  struct Config
  {
    int pyramid_levels;     // Number of pyramid levels (0 = auto-detect from image size)
    int center_levels[3];   // Center scales (c)
    int surround_deltas[2]; // Surround offsets (delta)

    Config() : pyramid_levels(0), center_levels{2, 3, 4}, surround_deltas{3, 4} {}
    // Setting pyramid_levels=0 enables adaptive sizing based on input dimensions
  };

  explicit IntensityFeature(const Config& config = Config());

  /**
   * Extract intensity feature from frame.
   * @param frame Input frame (color or grayscale)
   * @return Intensity feature map with saliency values [0, 1]
   */
  core::FeatureMap extract(const core::Frame& frame) const;

 private:
  Config config_;

  // Create Gaussian pyramid
  std::vector<cv::Mat> create_pyramid(const cv::Mat& input, int levels) const;

  // Compute center-surround differences
  cv::Mat compute_center_surround(const std::vector<cv::Mat>& pyramid) const;

  // Normalize and resize to original size
  cv::Mat normalize_and_resize(const cv::Mat& feature, const cv::Size& target_size) const;
};

} // namespace features
} // namespace attention
