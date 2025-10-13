#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/features/feature_extractor.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace attention
{
namespace features
{

/**
 * SymmetryFeature detects radial symmetry in images using gradient voting.
 *
 * This feature implements the approach from Reisfeld, Wolfson & Yeshurun (1995).
 * For each edge pixel, the gradient direction votes for potential symmetry centers.
 * Regions with converging gradients (faces, circles, symmetric objects) receive
 * high votes and thus high saliency.
 *
 * The algorithm:
 * 1. Compute gradients at multiple scales
 * 2. For each edge pixel with gradient (gx, gy):
 *    - Project along gradient direction to find candidate centers
 *    - Vote with weight = magnitude * distance_falloff
 * 3. Accumulate votes across scales
 *
 * References:
 * - Reisfeld et al. (1995): Context-free attentional operators
 * - Original dissertation symmetry feature
 */
class SymmetryFeature : public FeatureExtractor
{
 public:
  /**
   * Configuration for symmetry feature extraction.
   */
  struct Config
  {
    int pyramid_levels;       // Number of pyramid levels (0 = auto-detect)
    float gradient_threshold; // Minimum gradient magnitude (fraction of max, 0-1)
    float distance_alpha;     // Distance falloff exponent (1.0 = 1/d, 2.0 = 1/d²)
    int max_radius_factor;    // Max search radius = min(w,h) / factor
    int compute_at_scale;     // Pyramid scale to compute at (0=full res, 1=half, 2=quarter, etc.)

    Config()
      : pyramid_levels(0), gradient_threshold(0.1f), distance_alpha(1.0f), max_radius_factor(4), compute_at_scale(0)
    {
    }
  };

  explicit SymmetryFeature(const Config& config = Config());

  /**
   * Extract symmetry feature from frame.
   * @param frame Input frame (color or grayscale)
   * @return Symmetry feature map with saliency values [0, 1]
   */
  core::FeatureMap extract(const core::Frame& frame) const override;

  /**
   * Get feature name.
   * @return "symmetry"
   */
  std::string name() const override { return "symmetry"; }

 private:
  Config config_;

  // Compute radial symmetry contribution map for a single scale
  cv::Mat compute_radial_symmetry(const cv::Mat& image) const;

  // Combine multi-scale symmetry
  cv::Mat combine_scales(const std::vector<cv::Mat>& symmetry_maps) const;

  // Normalize and resize to target size
  cv::Mat normalize_and_resize(const cv::Mat& feature, const cv::Size& target_size) const;
};

} // namespace features
} // namespace attention
