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
    int num_orientations = 4; // Number of orientations (default 4: 0°, 45°, 90°, 135°)
    double wavelength = 4.0;  // Wavelength for Gabor filters
    double bandwidth = 1.0;   // Bandwidth parameter
    int compute_at_scale = 0; // Pyramid level for computation (0 = full resolution)
    // -1 (default) combines every orientation into one map, as the thesis and
    // Itti-Koch do. >= 0 emits *only* that orientation, so a config can run
    // four instances of this feature as four separate channels — which is what
    // a learned weight vector needs in order to say "this target is
    // horizontal" (M20/H8). The angular spacing still comes from
    // num_orientations, so index i is i * 180 / num_orientations degrees.
    int only_orientation = -1;
  };

  OrientationFeature();
  explicit OrientationFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;
  std::string name() const override { return "orientation"; }

  GaborRequirement gabor_requirement() const override
  {
    return {config_.num_orientations, config_.wavelength, config_.bandwidth};
  }

 private:
  Config config_;

  /**
   * Compute center-surround differences for a specific orientation.
   * @param gabor_pyramid Gabor responses for one orientation at all scales
   * @param center_scale Center scale index
   * @param surround_scale Surround scale index
   * @return Center-surround difference map
   */
  cv::Mat compute_center_surround(const std::vector<cv::Mat>& gabor_pyramid, int center_scale,
                                  int surround_scale) const;

  // Debug helper: capture intermediate results (keeps algorithm code clean)
  void capture_debug_data(DebugContext& debug, const core::Frame& frame,
                          const std::vector<std::vector<cv::Mat>>& orientation_pyramids, const cv::Mat& combined,
                          const cv::Mat& result, double total_ms, double gabor_computation_ms,
                          double center_surround_ms, double normalize_ms, int num_maps) const;
};

} // namespace features
} // namespace attention
