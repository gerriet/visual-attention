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
 * SymmetryFeature detects radial symmetry using Gabor filter responses.
 *
 * Based on the thesis approach: At each pyramid level, for each pixel, we test
 * multiple radii. For each radius and orientation, we sum Gabor filter values
 * within a specified width orthogonal to the radius direction. From all radii,
 * we select the maximum value and assign it to the pixel at this resolution.
 *
 * The algorithm:
 * 1. Use precomputed Gabor pyramids from Frame (multiple orientations and scales)
 * 2. For each pyramid level (e.g., 64x64, 128x128, 256x256):
 *    - Test multiple radii (e.g., 6, 9, 12, 15 pixels)
 *    - For each orientation and radius:
 *      * Sum Gabor responses within 'width' pixels orthogonal to the radius
 *      * Add responses from opposite sides (bilateral symmetry)
 *    - Select maximum across all radii for each pixel
 * 3. Combine results from multiple scales
 *
 * This matches the original dissertation's radius-sampling approach.
 */
class SymmetryFeature : public FeatureExtractor
{
 public:
  /**
   * Configuration for a single scale/pyramid level.
   */
  struct ScaleConfig
  {
    int pyramid_level;            // Which pyramid level to use (-1 = auto-select)
    int min_radius;               // Minimum radius to test (e.g., 3)
    int max_radius;               // Maximum radius to test (e.g., 20)
    int radius_step;              // Step between radii (1 = all radii, 2 = every other, etc.)
    int width;                    // Width of orthogonal summation box (e.g., 3)
    float symmetry_threshold;     // Minimum symmetry value to consider (suppress weak symmetry)

    ScaleConfig(int level = -1, int min_r = 3, int max_r = 20, int r_step = 1, int w = 3, float thresh = 0.3f)
      : pyramid_level(level), min_radius(min_r), max_radius(max_r),
        radius_step(r_step), width(w), symmetry_threshold(thresh)
    {
    }
  };

  /**
   * Configuration for symmetry feature extraction.
   */
  struct Config
  {
    int num_orientations;              // Number of Gabor orientations to use (e.g., 12)
    double wavelength;                 // Wavelength for Gabor filters
    double bandwidth;                  // Bandwidth parameter
    std::vector<ScaleConfig> scales;   // Configuration for each pyramid level
    bool use_multi_scale;              // Whether to combine multiple scales

    Config()
      : num_orientations(12), wavelength(4.0), bandwidth(1.0), use_multi_scale(true)
    {
      // Default: use pyramid levels with appropriate radius ranges
      // Level 0 (full res): small radii for local symmetry
      scales.push_back(ScaleConfig(0, 3, 15, 1, 3));
      // Level 2 (1/4 res): medium radii
      scales.push_back(ScaleConfig(2, 6, 25, 1, 3));
      // Level 4 (1/16 res): large radii for global symmetry
      scales.push_back(ScaleConfig(4, 10, 35, 1, 3));
    }
  };

  explicit SymmetryFeature(const Config& config = Config());

  /**
   * Extract symmetry feature from frame.
   * @param frame Input frame (color or grayscale)
   * @return Symmetry feature map with saliency values [0, 1]
   */
  core::FeatureMap extract(const core::Frame& frame) const override;
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;

  /**
   * Get feature name.
   * @return "symmetry"
   */
  std::string name() const override { return "symmetry"; }

 private:
  Config config_;

  /**
   * Compute radial symmetry at a single scale using the thesis approach.
   * For each pixel, tests multiple radii and selects the maximum response.
   * @param gabor_responses Vector of Gabor filter responses (one per orientation)
   * @param scale_config Configuration for this scale (radii, width, etc.)
   * @return Symmetry map for this scale
   */
  cv::Mat compute_radial_symmetry_at_scale(const std::vector<cv::Mat>& gabor_responses,
                                            const ScaleConfig& scale_config) const;

  /**
   * Compute symmetry for a single orientation and radius.
   * Sums Gabor responses within 'width' pixels orthogonal to the radius direction.
   * This corresponds to the symmetry_intern() function in the old code.
   * @param gabor_orientation Single orientation's Gabor response
   * @param orientation_angle Angle in degrees
   * @param radius Radius to test
   * @param width Width of orthogonal summation box
   * @param num_orientations Total number of orientations
   * @return Contribution to symmetry map from this orientation and radius
   */
  cv::Mat compute_orientation_radius_contribution(const cv::Mat& gabor_orientation,
                                                   float orientation_angle,
                                                   int radius,
                                                   int width,
                                                   int num_orientations) const;

  // Debug helper: capture intermediate results (keeps algorithm code clean)
  void capture_debug_data(DebugContext& debug,
                          const core::Frame& frame,
                          const std::vector<std::vector<cv::Mat>>& scale_gabor_responses,
                          const std::vector<cv::Mat>& scale_results,
                          const cv::Mat& result,
                          double total_ms,
                          double gabor_computation_ms,
                          const std::vector<double>& scale_computation_times,
                          double combine_ms,
                          double resize_ms) const;
};

} // namespace features
} // namespace attention
