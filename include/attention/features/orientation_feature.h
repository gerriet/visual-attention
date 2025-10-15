#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/features/feature_extractor.h"

namespace attention
{
namespace features
{

/**
 * Orientation Feature Extractor (Itti-Koch style).
 * Extracts orientation-selective features using Gabor filters at 4 orientations.
 * Uses center-surround differences across pyramid scales to detect orientation contrasts.
 */
class OrientationFeature : public FeatureExtractor
{
public:
  /**
   * Configuration for orientation feature extraction.
   */
  struct Config
  {
    int num_orientations = 4;     // Number of orientations (default 4: 0°, 45°, 90°, 135°)
    double wavelength = 4.0;      // Wavelength for Gabor filters
    double bandwidth = 1.0;       // Bandwidth parameter
    int compute_at_scale = 0;     // Pyramid level for computation (0 = full resolution)
  };

  OrientationFeature();
  explicit OrientationFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;
  std::string name() const override { return "orientation"; }

private:
  Config config_;

  /**
   * Compute center-surround differences for a specific orientation.
   * @param gabor_pyramid Gabor responses for one orientation at all scales
   * @param center_scale Center scale index
   * @param surround_scale Surround scale index
   * @return Center-surround difference map
   */
  cv::Mat compute_center_surround(const std::vector<cv::Mat>& gabor_pyramid, int center_scale, int surround_scale) const;
};

} // namespace features
} // namespace attention
