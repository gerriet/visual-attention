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
class ColorFeature : public FeatureExtractor
{
 public:
  /**
   * Configuration for color feature extraction.
   */
  struct Config
  {
    int pyramid_levels;      // Number of pyramid levels (0 = auto-detect from image size)
    int center_levels[3];    // Center scales (c)
    int surround_deltas[2];  // Surround offsets (delta)
    bool normalize_channels; // Normalize color channels before processing

    Config() : pyramid_levels(0), center_levels{2, 3, 4}, surround_deltas{3, 4}, normalize_channels(true) {}
    // Original Itti-Koch uses c ∈ {2,3,4}, δ ∈ {3,4} with 9-level pyramid
    // Creates 6 center-surround pairs: (2,5), (2,6), (3,6), (3,7), (4,7), (4,8)
    // Setting pyramid_levels=0 enables adaptive sizing based on input dimensions
  };

  explicit ColorFeature(const Config& config = Config());

  /**
   * Extract color feature from frame.
   * @param frame Input frame (must be color image)
   * @return Color feature map with saliency values [0, 1]
   * @throws std::runtime_error if frame is not color
   */
  core::FeatureMap extract(const core::Frame& frame) const override;

  /**
   * Extract color feature from frame with debug context.
   * @param frame Input frame (must be color image)
   * @param debug Debug context for capturing intermediate results
   * @return Color feature map with saliency values [0, 1]
   */
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;

  /**
   * Color features require a color image.
   */
  bool applicable(const core::Frame& frame) const override { return frame.channels() == 3; }

  /**
   * Get feature name.
   * @return "color"
   */
  std::string name() const override { return "color"; }

 private:
  Config config_;

  // Convert RGB to opponent color channels
  void compute_opponent_colors(const cv::Mat& rgb, cv::Mat& rg, cv::Mat& by) const;

  // Compute center-surround differences for one channel
  cv::Mat compute_center_surround(const std::vector<cv::Mat>& pyramid) const;

  // Normalize and resize to original size
  cv::Mat normalize_and_resize(const cv::Mat& feature, const cv::Size& target_size) const;

  // Debug helper: capture intermediate results (keeps algorithm code clean)
  void capture_debug_data(DebugContext& debug,
                          const core::Frame& frame,
                          const std::vector<cv::Mat>& rg_pyramid,
                          const std::vector<cv::Mat>& by_pyramid,
                          const cv::Mat& rg_saliency,
                          const cv::Mat& by_saliency,
                          const cv::Mat& combined,
                          const cv::Mat& result,
                          double total_ms,
                          double opponent_ms,
                          double center_surround_ms,
                          double normalize_ms) const;
};

} // namespace features
} // namespace attention
