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
 * SymmetryFeature detects bilateral (mirror) symmetry in images.
 *
 * This feature computes symmetry at multiple scales and orientations:
 * - Vertical symmetry (left-right mirror)
 * - Horizontal symmetry (top-bottom mirror)
 * - Diagonal symmetries (optional)
 *
 * Symmetric regions (faces, objects, patterns) have high saliency.
 *
 * References:
 * - Reisfeld et al. (1995): Context-free attentional operators
 * - Original dissertation symmetry feature
 */
class SymmetryFeature
{
 public:
  /**
   * Configuration for symmetry feature extraction.
   */
  struct Config
  {
    int pyramid_levels;             // Number of pyramid levels (0 = auto-detect)
    bool compute_vertical = true;   // Compute vertical (left-right) symmetry
    bool compute_horizontal = true; // Compute horizontal (top-bottom) symmetry
    int kernel_size = 5;            // Smoothing kernel size

    Config() : pyramid_levels(0) {}
  };

  explicit SymmetryFeature(const Config& config = Config());

  /**
   * Extract symmetry feature from frame.
   * @param frame Input frame (color or grayscale)
   * @return Symmetry feature map with saliency values [0, 1]
   */
  core::FeatureMap extract(const core::Frame& frame) const;

 private:
  Config config_;

  // Compute symmetry for a single orientation
  cv::Mat compute_symmetry(const cv::Mat& image, bool vertical) const;

  // Create Gaussian pyramid
  std::vector<cv::Mat> create_pyramid(const cv::Mat& input, int levels) const;

  // Combine multi-scale symmetry
  cv::Mat combine_scales(const std::vector<cv::Mat>& symmetry_maps) const;

  // Normalize and resize to target size
  cv::Mat normalize_and_resize(const cv::Mat& feature, const cv::Size& target_size) const;
};

} // namespace features
} // namespace attention
