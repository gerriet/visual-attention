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
 * SymmetryFeature detects radial and bilateral symmetry using Gabor filter responses.
 *
 * This feature uses precomputed Gabor pyramids to detect symmetry patterns by
 * analyzing orientation relationships. Symmetric regions show characteristic
 * patterns in Gabor responses across different orientations.
 *
 * The algorithm:
 * 1. Use precomputed Gabor pyramids from Frame (multiple orientations and scales)
 * 2. For each location, analyze Gabor response patterns across orientations
 * 3. Detect symmetry by finding opposing/complementary orientation pairs
 * 4. Accumulate symmetry evidence across scales
 *
 * This approach leverages the existing Gabor computations from the orientation
 * feature, making it more efficient and consistent with the dissertation's
 * original Gabor-based symmetry detection.
 */
class SymmetryFeature : public FeatureExtractor
{
 public:
  /**
   * Configuration for symmetry feature extraction.
   */
  struct Config
  {
    int num_orientations;         // Number of Gabor orientations to use
    double wavelength;            // Wavelength for Gabor filters
    double bandwidth;             // Bandwidth parameter
    float response_threshold;     // Minimum Gabor response threshold
    int compute_at_scale;         // Pyramid scale to compute at (0=full res, 1=half, 2=quarter, etc.)

    Config()
      : num_orientations(12), wavelength(4.0), bandwidth(1.0), response_threshold(0.1f), compute_at_scale(0)
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

  /**
   * Compute symmetry from Gabor responses at a single scale.
   * Analyzes orientation patterns to detect symmetric regions.
   * @param gabor_responses Vector of Gabor filter responses (one per orientation)
   * @return Symmetry map
   */
  cv::Mat compute_gabor_symmetry(const std::vector<cv::Mat>& gabor_responses) const;

  /**
   * Compute bilateral symmetry by comparing opposite orientations.
   * @param gabor_responses Vector of Gabor filter responses
   * @return Bilateral symmetry map
   */
  cv::Mat compute_bilateral_symmetry(const std::vector<cv::Mat>& gabor_responses) const;

  /**
   * Compute radial symmetry by analyzing all orientations.
   * @param gabor_responses Vector of Gabor filter responses
   * @return Radial symmetry map
   */
  cv::Mat compute_radial_symmetry(const std::vector<cv::Mat>& gabor_responses) const;
};

} // namespace features
} // namespace attention
