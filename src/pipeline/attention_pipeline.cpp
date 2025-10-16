#include "attention/pipeline/attention_pipeline.h"
#include "attention/features/color_feature.h"
#include "attention/features/eccentricity_feature.h"
#include "attention/features/feature_extractor.h"
#include "attention/features/intensity_feature.h"
#include "attention/features/orientation_feature.h"
#include "attention/features/symmetry_feature.h"
#include "attention/visualization/visualizer.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace attention
{
namespace pipeline
{

AttentionPipeline::AttentionPipeline() : config_(), processed_(false) {}

AttentionPipeline::AttentionPipeline(const PipelineConfig& config) : config_(config), processed_(false) {}

void AttentionPipeline::load_image(const std::string& image_path)
{
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

  if (image.empty())
  {
    throw std::runtime_error("Failed to load image: " + image_path);
  }

  frame_ = core::Frame(image, image_path);
  processed_ = false; // Reset processing state
}

void AttentionPipeline::load_image(const cv::Mat& image, const std::string& source_name)
{
  if (image.empty())
  {
    throw std::runtime_error("Cannot load empty image");
  }

  frame_ = core::Frame(image, source_name);
  processed_ = false; // Reset processing state
}

void AttentionPipeline::process()
{
  if (!has_image())
  {
    throw std::runtime_error("No image loaded. Call load_image() first.");
  }

  // Clear previous results
  features_.clear();
  saliency_ = core::SaliencyMap();
  timing_ = PipelineTiming(); // Reset timing

  auto t_start = std::chrono::high_resolution_clock::now();

  // Step 0: Compute pyramids once (shared across features)
  auto t_pyramid_start = std::chrono::high_resolution_clock::now();
  int pyramid_levels = compute_pyramid_levels();
  frame_.compute_pyramids(pyramid_levels);
  auto t_pyramid_end = std::chrono::high_resolution_clock::now();
  timing_.pyramid_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_pyramid_end - t_pyramid_start).count();

  // Step 1: Extract features (with per-feature timing)
  extract_features();

  // Step 2: Integrate features into saliency map
  auto t_integrate_start = std::chrono::high_resolution_clock::now();
  integrate_features();
  auto t_integrate_end = std::chrono::high_resolution_clock::now();
  timing_.integration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_integrate_end - t_integrate_start).count();

  // Step 3: Detect peaks
  auto t_peaks_start = std::chrono::high_resolution_clock::now();
  detect_peaks();
  auto t_peaks_end = std::chrono::high_resolution_clock::now();
  timing_.peak_detection_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_peaks_end - t_peaks_start).count();

  // Print timing breakdown with aligned columns
  std::cout << "    Pyramid computation:     " << timing_.pyramid_ms << " ms" << std::endl;

  // Find longest feature name for alignment
  size_t max_name_length = 0;
  for (const auto& pair : timing_.feature_ms)
  {
    max_name_length = std::max(max_name_length, pair.first.length());
  }

  for (const auto& pair : timing_.feature_ms)
  {
    // Calculate padding needed after feature name
    size_t padding = max_name_length - pair.first.length();
    std::string spaces(padding, ' ');
    std::cout << "    Feature '" << pair.first << "'" << spaces << ": "
              << pair.second << " ms" << std::endl;
  }
  std::cout << "    Integration:             " << timing_.integration_ms << " ms" << std::endl;
  std::cout << "    Peak detection:          " << timing_.peak_detection_ms << " ms" << std::endl;

  // Handle debug output if debugging is enabled
  if (is_debugging_enabled())
  {
    std::cout << "\n  Debug Output:" << std::endl;

    // Print debug info if requested
    if (config_.debug_print_info)
    {
      for (const auto& [feature_name, debug_ctx] : debug_contexts_)
      {
        features::DebugVisualizer::print_debug_info(debug_ctx, feature_name);
      }
    }

    // Save debug images if requested
    if (config_.debug_save_images)
    {
      // Create debug output directory
      std::filesystem::create_directories(config_.debug_output_dir);

      for (const auto& [feature_name, debug_ctx] : debug_contexts_)
      {
        if (!debug_ctx.empty())
        {
          std::cout << "    Saving debug output for '" << feature_name << "'..." << std::endl;
          features::DebugVisualizer::save_debug_images(debug_ctx, config_.debug_output_dir, feature_name);

          // Create combined visualization
          cv::Mat combined_viz = features::DebugVisualizer::create_debug_visualization(debug_ctx);
          if (!combined_viz.empty())
          {
            std::string combined_path = config_.debug_output_dir + "/" + feature_name + "_combined.png";
            cv::imwrite(combined_path, combined_viz);
            std::cout << "      Saved combined: " << combined_path << std::endl;
          }
        }
      }
    }

    std::cout << "    ✓ Debug output complete" << std::endl;
  }

  processed_ = true;
}

int AttentionPipeline::compute_pyramid_levels() const
{
  // Adaptive pyramid levels based on image size
  int min_dim = std::min(frame_.width(), frame_.height());
  int levels = 0;
  while (min_dim > 16 && levels < 12)
  {
    min_dim /= 2;
    levels++;
  }
  return std::max(9, levels); // At least 9 levels for Itti-Koch
}

void AttentionPipeline::extract_features()
{
  // Pre-compute Gabor pyramids BEFORE parallel extraction to avoid race conditions
  // Maximum orientations needed: 12 (for symmetry feature)
  // This ensures thread-safe access during parallel feature extraction
  int pyramid_levels = compute_pyramid_levels();
  frame_.compute_gabor_pyramids(pyramid_levels, 12, 4.0, 1.0);

  // Create list of feature extractors to run in parallel
  std::vector<std::unique_ptr<features::FeatureExtractor>> extractors;

  // Add color feature (if image is color)
  if (frame_.channels() == 3)
  {
    extractors.push_back(std::make_unique<features::ColorFeature>());
  }

  // Add intensity feature (always)
  extractors.push_back(std::make_unique<features::IntensityFeature>());

  // Add orientation feature (always)
  extractors.push_back(std::make_unique<features::OrientationFeature>());

  // Add eccentricity feature (always)
  // Use quarter resolution for large images for performance
  features::EccentricityFeature::Config ecc_config;
  if (frame_.width() > 640 || frame_.height() > 640)
  {
    ecc_config.compute_at_scale = 2; // Quarter resolution for large images
  }
  else
  {
    ecc_config.compute_at_scale = 0; // Full resolution for small images
  }
  extractors.push_back(std::make_unique<features::EccentricityFeature>(ecc_config));

  // Add symmetry feature (always)
  // Use very coarse scale for global symmetry detection to improve performance
  // Use 12 orientations for better symmetry detection (vs 4 for basic orientation feature)
  features::SymmetryFeature::Config sym_config;
  sym_config.num_orientations = 12;
  sym_config.wavelength = 8.0;  // Larger wavelength for coarser features
  sym_config.bandwidth = 1.0;
  if (frame_.width() > 640 || frame_.height() > 640)
  {
    sym_config.compute_at_scale = 4; // 1/16 resolution for large images (very coarse for performance)
  }
  else
  {
    sym_config.compute_at_scale = 2; // Quarter resolution for small images
  }
  extractors.push_back(std::make_unique<features::SymmetryFeature>(sym_config));

  // Extract features (with optional debugging)
  features_.resize(extractors.size());
  std::vector<long> durations(extractors.size());

  // Clear previous debug contexts
  debug_contexts_.clear();

  // Check if debugging is enabled
  bool debugging = is_debugging_enabled();

  if (debugging)
  {
    // Extract serially to preserve debug context
    std::cout << "  Debug mode: extracting features serially" << std::endl;
    for (size_t i = 0; i < extractors.size(); ++i)
    {
      auto t_start = std::chrono::high_resolution_clock::now();

      // Create debug context for this feature
      features::DebugContext debug(config_.debug_level);
      features_[i] = extractors[i]->extract(frame_, debug);

      auto t_end = std::chrono::high_resolution_clock::now();
      durations[i] = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

      // Store debug context
      debug_contexts_[features_[i].name] = std::move(debug);
    }
  }
  else
  {
    // Extract features in parallel using threads (no debugging)
    std::vector<std::thread> threads;

    for (size_t i = 0; i < extractors.size(); ++i)
    {
      threads.emplace_back(
          [this, i, &extractors, &durations]()
          {
            auto t_start = std::chrono::high_resolution_clock::now();
            features_[i] = extractors[i]->extract(frame_);
            auto t_end = std::chrono::high_resolution_clock::now();
            durations[i] = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
          });
    }

    // Wait for all threads to complete
    for (auto& thread : threads)
    {
      thread.join();
    }
  }

  // Store per-feature timing
  for (size_t i = 0; i < features_.size(); ++i)
  {
    timing_.feature_ms[features_[i].name] = durations[i];
  }

  std::cout << "  Features extracted: " << features_.size();
  if (debugging)
  {
    std::cout << " (debug mode)";
  }
  else
  {
    std::cout << " (parallel)";
  }
  std::cout << std::endl;
}

void AttentionPipeline::integrate_features()
{
  if (features_.empty())
  {
    throw std::runtime_error("No features to integrate");
  }

  // Weighted integration using configured feature weights
  cv::Mat integrated = cv::Mat::zeros(frame_.size(), CV_32F);

  for (const auto& feature : features_)
  {
    // Validate feature size matches frame size
    if (feature.data.empty())
    {
      throw std::runtime_error("Feature '" + feature.name + "' has empty data");
    }

    if (feature.data.size() != frame_.size())
    {
      throw std::runtime_error("Feature '" + feature.name + "' size mismatch: expected " +
                               std::to_string(frame_.width()) + "x" + std::to_string(frame_.height()) +
                               " but got " + std::to_string(feature.data.cols) + "x" + std::to_string(feature.data.rows));
    }

    // Get weight from config (default to 1.0 if not configured)
    float weight = 1.0f;
    auto it = config_.feature_weights.find(feature.name);
    if (it != config_.feature_weights.end())
    {
      weight = it->second;
    }

    // Apply weight and feature confidence
    integrated += weight * feature.confidence * feature.data;
  }

  // Normalize to [0, 1]
  cv::normalize(integrated, integrated, 0.0f, 1.0f, cv::NORM_MINMAX);

  saliency_ = core::SaliencyMap(integrated);
}

void AttentionPipeline::detect_peaks()
{
  // Use the built-in peak detection
  // Parameters from configuration
  saliency_.detect_peaks(config_.peak_min_distance, config_.peak_threshold, config_.peak_max_count, config_.enable_ior,
                         config_.ior_radius, config_.ior_strength);

  std::cout << "  Peaks detected: " << saliency_.peaks.size();
  if (config_.enable_ior)
  {
    std::cout << " (IOR)";
  }
  std::cout << std::endl;
}

cv::Mat AttentionPipeline::visualize(bool save_individual)
{
  if (!is_processed())
  {
    throw std::runtime_error("Pipeline not processed. Call process() first.");
  }

  // Create side-by-side visualization
  std::vector<cv::Mat> vis_images;
  std::vector<std::string> labels;

  // Original image
  cv::Mat original_bgr;
  if (frame_.image.channels() == 3)
  {
    original_bgr = frame_.image;
  }
  else
  {
    cv::cvtColor(frame_.image, original_bgr, cv::COLOR_GRAY2BGR);
  }
  vis_images.push_back(original_bgr);
  labels.push_back("Original");

  if (save_individual)
  {
    cv::imwrite("../results/01_original.png", original_bgr);
  }

  // Feature visualizations
  int feature_idx = 2;
  for (const auto& feature : features_)
  {
    cv::Mat feature_vis = visualization::visualize_feature_map(feature);
    cv::Mat feature_bgr;
    cv::cvtColor(feature_vis, feature_bgr, cv::COLOR_GRAY2BGR);
    vis_images.push_back(feature_bgr);
    labels.push_back(feature.name);

    if (save_individual)
    {
      std::string filename = "../results/0" + std::to_string(feature_idx) + "_" + feature.name + ".png";
      cv::imwrite(filename, feature_vis);
      feature_idx++;
    }
  }

  // Saliency overlay
  cv::Mat saliency_vis = visualization::visualize_saliency_map(saliency_, frame_.image, "", true, false);
  vis_images.push_back(saliency_vis);
  labels.push_back("Saliency");

  if (save_individual)
  {
    cv::imwrite("../results/99_saliency.png", saliency_vis);
  }

  // Combine all visualizations
  return visualization::visualize_side_by_side(vis_images, labels, "", false);
}

} // namespace pipeline
} // namespace attention
