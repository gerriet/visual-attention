#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/core/saliency_map.h"
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace attention
{
namespace pipeline
{

/**
 * Configuration for the attention pipeline.
 */
struct PipelineConfig
{
  // Feature weights for integration
  // Keys are feature names (e.g., "color", "intensity", "symmetry")
  std::map<std::string, float> feature_weights;

  // Peak detection parameters
  int peak_min_distance = 30;  // Minimum distance between peaks (pixels)
  float peak_threshold = 0.3f; // Minimum saliency value for peaks
  int peak_max_count = 10;     // Maximum number of peaks to detect

  PipelineConfig()
  {
    // Default weights
    feature_weights["color"] = 1.0f;
    feature_weights["intensity"] = 1.0f;
    feature_weights["symmetry"] = 1.0f;
  }
};

/**
 * AttentionPipeline orchestrates the complete attention computation process.
 *
 * The pipeline follows these steps:
 * 1. Load image(s) from file or camera
 * 2. Extract features (color, intensity, etc.)
 * 3. Integrate features into saliency map
 * 4. Detect peaks (focus points)
 * 5. Visualize results
 *
 * Usage:
 *   AttentionPipeline pipeline;
 *   pipeline.load_image("input.jpg");
 *   pipeline.process();
 *   cv::Mat result = pipeline.visualize();
 */
class AttentionPipeline
{
 public:
  AttentionPipeline();
  explicit AttentionPipeline(const PipelineConfig& config);
  ~AttentionPipeline() = default;

  // Disable copy (use move semantics for efficiency)
  AttentionPipeline(const AttentionPipeline&) = delete;
  AttentionPipeline& operator=(const AttentionPipeline&) = delete;

  // Enable move semantics
  AttentionPipeline(AttentionPipeline&&) noexcept = default;
  AttentionPipeline& operator=(AttentionPipeline&&) noexcept = default;

  /**
   * Load image from file.
   * @param image_path Path to image file
   * @throws std::runtime_error if image cannot be loaded
   */
  void load_image(const std::string& image_path);

  /**
   * Load image directly from cv::Mat.
   * @param image Input image
   * @param source_name Optional name/description for the image
   */
  void load_image(const cv::Mat& image, const std::string& source_name = "");

  /**
   * Process the loaded image through the attention pipeline.
   * Extracts features, integrates them, and detects peaks.
   * @throws std::runtime_error if no image loaded
   */
  void process();

  /**
   * Visualize the results.
   * Creates a visualization showing original image, features, and saliency.
   * @param save_individual If true, save individual feature maps to results/
   * @return Visualization as BGR image
   * @throws std::runtime_error if process() not called yet
   */
  cv::Mat visualize(bool save_individual = false);

  /**
   * Get the computed saliency map.
   * @return Saliency map (empty if not yet processed)
   */
  const core::SaliencyMap& get_saliency_map() const { return saliency_; }

  /**
   * Get the extracted features.
   * @return Vector of feature maps (empty if not yet processed)
   */
  const std::vector<core::FeatureMap>& get_features() const { return features_; }

  /**
   * Get the loaded frame.
   * @return Current frame (empty if not yet loaded)
   */
  const core::Frame& get_frame() const { return frame_; }

  /**
   * Check if image is loaded.
   * @return true if image loaded
   */
  bool has_image() const { return !frame_.empty(); }

  /**
   * Check if processing is complete.
   * @return true if process() has been called
   */
  bool is_processed() const { return processed_; }

  /**
   * Set configuration.
   * @param config New configuration
   */
  void set_config(const PipelineConfig& config) { config_ = config; }

  /**
   * Get current configuration.
   * @return Current configuration
   */
  const PipelineConfig& get_config() const { return config_; }

 private:
  // Configuration
  PipelineConfig config_;

  // Current frame being processed
  core::Frame frame_;

  // Extracted features
  std::vector<core::FeatureMap> features_;

  // Integrated saliency map
  core::SaliencyMap saliency_;

  // Processing state
  bool processed_;

  // Internal processing methods
  int compute_pyramid_levels() const;
  void extract_features();
  void integrate_features();
  void detect_peaks();
};

} // namespace pipeline
} // namespace attention
