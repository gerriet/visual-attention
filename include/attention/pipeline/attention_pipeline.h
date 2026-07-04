#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/core/run_state.h"
#include "attention/core/saliency_map.h"
#include "attention/features/debug_context.h"
#include "attention/features/feature_extractor.h"
#include "attention/fusion/fusion_strategy.h"
#include "attention/pipeline/frame_source.h"
#include "attention/selection/selection_strategy.h"
#include <functional>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace attention
{
namespace pipeline
{

/**
 * Performance timing information for pipeline stages.
 */
struct PipelineTiming
{
  long pyramid_ms = 0;                    // Pyramid computation time
  long integration_ms = 0;                // Feature integration time
  long peak_detection_ms = 0;             // Peak detection time
  std::map<std::string, long> feature_ms; // Per-feature extraction time

  long total_ms() const
  {
    long feature_total = 0;
    for (const auto& pair : feature_ms)
    {
      feature_total += pair.second;
    }
    return pyramid_ms + feature_total + integration_ms + peak_detection_ms;
  }
};

/**
 * One feature in the pipeline's feature set.
 */
struct FeatureSpec
{
  std::string type;        // Registry key (e.g. "color", "symmetry")
  bool enabled = true;     // Disabled features are configured but not run
  float weight = 1.0f;     // Fusion weight
  std::string params_yaml; // Feature-specific params as a YAML snippet
                           // (empty = factory defaults). Stored as text so
                           // this header stays yaml-cpp-free.

  FeatureSpec() = default;
  explicit FeatureSpec(std::string t, float w = 1.0f) : type(std::move(t)), weight(w) {}
};

/**
 * Configuration for the attention pipeline. Everything algorithmic is
 * swappable: the feature set (via FeatureRegistry), the fusion strategy, and
 * the selection strategy are all chosen here — typically loaded from YAML
 * (config::ConfigLoader).
 */
struct PipelineConfig
{
  // Feature set (order = extraction and fusion order)
  std::vector<FeatureSpec> features;

  // Strategy names
  std::string fusion = "weighted-sum";
  std::string selection = ""; // "nms" or "ior"; empty = derive from enable_ior

  // Shared Gabor bank, precomputed once per frame for all features
  int gabor_orientations = 12;
  double gabor_wavelength = 4.0;
  double gabor_bandwidth = 1.0;

  // Peak selection parameters
  int peak_min_distance = 30;  // Minimum distance between peaks (pixels)
  float peak_threshold = 0.3f; // Minimum saliency value for peaks
  int peak_max_count = 10;     // Maximum number of peaks to detect

  // Inhibition of Return (IOR) parameters
  bool enable_ior = false;     // Legacy alias for selection = "ior"
  int ior_radius = 50;         // Radius of inhibition disk (pixels)
  float ior_strength = 0.8f;   // Inhibition strength (0-1, 1 = complete suppression)

  // Debug parameters
  features::DebugContext::Level debug_level = features::DebugContext::Level::None;
  std::string debug_output_dir = "debug_output";  // Directory for debug output
  bool debug_save_images = true;                   // Save debug images to disk
  bool debug_print_info = false;                   // Print debug info to console

  PipelineConfig()
  {
    // Default feature set (all weights 1.0)
    features.emplace_back("color");
    features.emplace_back("intensity");
    features.emplace_back("orientation");
    features.emplace_back("eccentricity");
    features.emplace_back("symmetry");
  }

  /**
   * The selection strategy actually in effect ("nms" or "ior").
   */
  std::string effective_selection() const
  {
    if (!selection.empty())
    {
      return selection;
    }
    return enable_ior ? "ior" : "nms";
  }
};

/**
 * AttentionPipeline orchestrates the attention computation:
 *
 * 1. Frame acquisition (single image or a FrameSource stream)
 * 2. Shared pyramid computation (Gaussian + Gabor)
 * 3. Feature extraction (config-driven set, via FeatureRegistry)
 * 4. Fusion into a saliency map (FusionStrategy)
 * 5. Peak selection into a scanpath (SelectionStrategy, with RunState)
 *
 * Usage (single image):
 *   AttentionPipeline pipeline(config);
 *   pipeline.load_image("input.jpg");
 *   pipeline.process();
 *
 * Usage (stream):
 *   ImageListSource source(paths);
 *   pipeline.process_stream(source, [](AttentionPipeline& p) { ... });
 */
class AttentionPipeline
{
 public:
  using FrameCallback = std::function<void(AttentionPipeline&)>;

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
   * Extracts features, fuses them, and selects peaks.
   * @throws std::runtime_error if no image loaded
   */
  void process();

  /**
   * Process all frames of a stream. Resets RunState first, then processes
   * frame by frame, invoking the callback after each.
   * @param source Frame source (single image = stream of length 1)
   * @param on_frame Called after each processed frame (may be empty)
   */
  void process_stream(FrameSource& source, const FrameCallback& on_frame = {});

  /**
   * Reset per-run state (frame index, future cross-frame state).
   */
  void reset_state() { run_state_.reset(); }

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
   * Get the per-run state.
   */
  const core::RunState& get_run_state() const { return run_state_; }

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
   * Get performance timing information from last process() call.
   * @return Timing breakdown
   */
  const PipelineTiming& get_timing() const { return timing_; }

  /**
   * Set configuration. Rebuilds extractors and strategies.
   * @param config New configuration
   * @throws std::runtime_error for unknown feature/strategy names
   */
  void set_config(const PipelineConfig& config);

  /**
   * Get current configuration.
   * @return Current configuration
   */
  const PipelineConfig& get_config() const { return config_; }

  /**
   * Get debug contexts for all features (if debugging was enabled).
   * @return Map of feature name to debug context
   */
  const std::map<std::string, features::DebugContext>& get_debug_contexts() const { return debug_contexts_; }

  /**
   * Check if debugging is enabled.
   * @return true if debug level is not None
   */
  bool is_debugging_enabled() const { return config_.debug_level != features::DebugContext::Level::None; }

 private:
  // Configuration
  PipelineConfig config_;

  // Components built from the configuration
  std::vector<std::unique_ptr<features::FeatureExtractor>> extractors_;
  std::map<std::string, float> feature_weights_; // instance name -> weight
  std::unique_ptr<fusion::FusionStrategy> fusion_;
  std::unique_ptr<selection::SelectionStrategy> selection_;

  // Current frame being processed
  core::Frame frame_;

  // Extracted features
  std::vector<core::FeatureMap> features_;

  // Integrated saliency map
  core::SaliencyMap saliency_;

  // Per-run state (persists across the frames of a stream)
  core::RunState run_state_;

  // Processing state
  bool processed_ = false;

  // Performance timing
  PipelineTiming timing_;

  // Debug contexts for each feature (if debugging enabled)
  std::map<std::string, features::DebugContext> debug_contexts_;

  // Internal processing methods
  void build_components();
  int compute_pyramid_levels() const;
  void extract_features();
  void integrate_features();
  void detect_peaks();
};

} // namespace pipeline
} // namespace attention
