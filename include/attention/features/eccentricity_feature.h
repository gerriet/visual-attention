#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/features/feature_extractor.h"
#include <opencv2/opencv.hpp>

namespace attention
{
namespace features
{

/**
 * Eccentricity Feature Extractor.
 *
 * Segments the image using watershed/edge-based region growing and computes
 * shape eccentricity for each region using image moments. Regions with high
 * eccentricity (elongated shapes) are highlighted as salient.
 *
 * This is based on the dissertation's approach of using moment-based shape
 * descriptors to detect elongated objects and structures.
 */
class EccentricityFeature : public FeatureExtractor
{
 public:
  /**
   * Configuration for eccentricity feature extraction.
   */
  struct Config
  {
    float edge_threshold = 0.75f;    // Threshold for edge detection (0-1)
    float min_area = 0.05f;          // Minimum segment area as fraction of image (0.0005 = 0.05%)
    float max_area = 30.0f;          // Maximum segment area as percentage of image
    float variance_threshold = 1.5f; // Variance ratio threshold for merging
    int compute_at_scale = 0;        // Pyramid level for computation (0 = full
                                     // resolution, -1 = auto: quarter resolution
                                     // for images larger than 640px)
  };

  EccentricityFeature();
  explicit EccentricityFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;
  std::string name() const override { return "eccentricity"; }

 private:
  Config config_;

  /**
   * Segment the image using edge-based watershed or similar method.
   * @param gray Input grayscale image
   * @param edges Edge magnitude image
   * @return Label image where each pixel has a segment ID
   */
  cv::Mat segment_image(const cv::Mat& gray, const cv::Mat& edges) const;

  /**
   * Compute eccentricity for a contour using image moments.
   * Eccentricity measures how elongated a shape is (0 = circle, 1 = line).
   * @param moments Image moments for the region
   * @return Eccentricity value [0, 1]
   */
  float compute_eccentricity(const cv::Moments& moments) const;

  /**
   * Filter segments by area constraints.
   * @param labels Label image
   * @param image_area Total image area
   * @return Map of valid segment IDs to their properties
   */
  std::map<int, cv::Moments> filter_segments(const cv::Mat& labels, int image_area) const;

  // Debug helper: capture intermediate results (keeps algorithm code clean)
  void capture_debug_data(DebugContext& debug, const core::Frame& frame, const cv::Mat& gray, const cv::Mat& edges,
                          const cv::Mat& labels, const cv::Mat& eccentricity_map, const cv::Mat& result,
                          double total_ms, double edge_computation_ms, double segmentation_ms,
                          double eccentricity_computation_ms, double resize_ms, int num_segments) const;
};

} // namespace features
} // namespace attention
