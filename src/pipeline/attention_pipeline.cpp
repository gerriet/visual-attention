#include "attention/pipeline/attention_pipeline.h"
#include "attention/features/feature_registry.h"
#include "attention/visualization/visualizer.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace attention
{
namespace pipeline
{

AttentionPipeline::AttentionPipeline() : config_()
{
  build_components();
}

AttentionPipeline::AttentionPipeline(const PipelineConfig& config) : config_(config)
{
  build_components();
}

void AttentionPipeline::set_config(const PipelineConfig& config)
{
  config_ = config;
  build_components();
}

void AttentionPipeline::build_components()
{
  features::register_builtin_features();
  auto& registry = features::FeatureRegistry::instance();

  extractors_.clear();
  feature_weights_.clear();

  for (const auto& spec : config_.features)
  {
    if (!spec.enabled)
    {
      continue;
    }

    YAML::Node params = spec.params_yaml.empty() ? YAML::Node() : YAML::Load(spec.params_yaml);
    auto extractor = registry.create(spec.type, params);
    feature_weights_[extractor->name()] = spec.weight;
    extractors_.push_back(std::move(extractor));
  }

  if (extractors_.empty())
  {
    throw std::runtime_error("PipelineConfig: no enabled features — enable at least one");
  }

  fusion_ = fusion::create_fusion_strategy(config_.fusion);
  top_down_ = std::make_unique<fusion::TopDownChannel>(config_.priority);

  selection::SelectionParams selection_params;
  selection_params.min_distance = config_.peak_min_distance;
  selection_params.threshold = config_.peak_threshold;
  selection_params.max_count = config_.peak_max_count;
  selection_params.ior_radius = config_.ior_radius;
  selection_params.ior_strength = config_.ior_strength;
  YAML::Node strategy_params =
      config_.selection_params_yaml.empty() ? YAML::Node() : YAML::Load(config_.selection_params_yaml);
  selection_ = selection::create_selection_strategy(config_.effective_selection(), selection_params, strategy_params);
}

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

void AttentionPipeline::load_stereo(const std::string& left_path, const std::string& right_path)
{
  cv::Mat left = cv::imread(left_path, cv::IMREAD_COLOR);
  if (left.empty())
  {
    throw std::runtime_error("Failed to load left image: " + left_path);
  }
  cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);
  if (right.empty())
  {
    throw std::runtime_error("Failed to load right image: " + right_path);
  }

  frame_ = core::Frame(left, left_path);
  frame_.stereo_right = right;
  processed_ = false;
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

  // Inject the previous frame's grayscale (carried in RunState) so temporal
  // features (onset) can compute a difference. Empty on the first frame and
  // for independent stills — those features declare themselves inapplicable.
  frame_.previous_gray = run_state_.previous_gray;

  // Step 0: Compute pyramids once (shared across features)
  auto t_pyramid_start = std::chrono::high_resolution_clock::now();
  int pyramid_levels = compute_pyramid_levels();
  frame_.compute_pyramids(pyramid_levels);
  auto t_pyramid_end = std::chrono::high_resolution_clock::now();
  timing_.pyramid_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_pyramid_end - t_pyramid_start).count();

  // Step 1: Extract features (with per-feature timing)
  extract_features(pyramid_levels);

  // Step 2: Fuse features into a saliency map
  auto t_integrate_start = std::chrono::high_resolution_clock::now();
  integrate_features();
  auto t_integrate_end = std::chrono::high_resolution_clock::now();
  timing_.integration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_integrate_end - t_integrate_start).count();

  // Step 3: Select peaks
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
    std::cout << "    Feature '" << pair.first << "'" << spaces << ": " << pair.second << " ms" << std::endl;
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

  // Remember this frame's grayscale for the next frame's temporal features.
  // Independent stills reset RunState between frames (main.cpp batch mode), so
  // this only chains within a temporal stream.
  if (frame_.image.channels() == 3)
  {
    cv::cvtColor(frame_.image, run_state_.previous_gray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    run_state_.previous_gray = frame_.image.clone();
  }

  processed_ = true;
  ++run_state_.frame_index;
}

void AttentionPipeline::process_stream(FrameSource& source, const FrameCallback& on_frame,
                                       const ErrorCallback& on_error)
{
  reset_state();

  while (true)
  {
    core::Frame frame;
    try
    {
      if (!source.next(frame))
      {
        break;
      }
      frame_ = std::move(frame);
      processed_ = false;
      process();

      if (on_frame)
      {
        on_frame(*this);
      }
    }
    catch (const std::exception& e)
    {
      // The error callback decides whether the stream continues (e.g. batch
      // mode logs and skips a corrupt image) or the error propagates
      if (on_error && on_error(e))
      {
        continue;
      }
      throw;
    }
  }
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

void AttentionPipeline::extract_features(int pyramid_levels)
{
  // Applicable extractors for this frame (e.g. color is skipped on grayscale)
  std::vector<features::FeatureExtractor*> active;
  for (const auto& extractor : extractors_)
  {
    if (extractor->applicable(frame_))
    {
      active.push_back(extractor.get());
    }
  }

  // Pre-compute every Gabor bank the active features require BEFORE parallel
  // extraction (banks are keyed by parameters and deduplicated); extraction
  // threads then only read shared Frame state
  for (const auto* extractor : active)
  {
    auto requirement = extractor->gabor_requirement();
    if (requirement.orientations > 0)
    {
      frame_.compute_gabor_bank(pyramid_levels, requirement.orientations, requirement.wavelength,
                                requirement.bandwidth);
    }
  }

  // Extract features (with optional debugging)
  features_.resize(active.size());
  std::vector<long> durations(active.size());

  // Clear previous debug contexts
  debug_contexts_.clear();

  // Check if debugging is enabled
  bool debugging = is_debugging_enabled();

  if (debugging)
  {
    // Extract serially to preserve debug context
    std::cout << "  Debug mode: extracting features serially" << std::endl;
    for (size_t i = 0; i < active.size(); ++i)
    {
      auto t_start = std::chrono::high_resolution_clock::now();

      // Create debug context for this feature
      features::DebugContext debug(config_.debug_level);
      features_[i] = active[i]->extract(frame_, debug);

      auto t_end = std::chrono::high_resolution_clock::now();
      durations[i] = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

      // Store debug context
      debug_contexts_[features_[i].name] = std::move(debug);
    }
  }
  else
  {
    // Extract features concurrently, one task thread per feature. Parallelism
    // model of the pipeline: coarse-grained tasks via std::thread here,
    // data-parallel loops via OpenMP inside the features (e.g. symmetry).
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(active.size());

    for (size_t i = 0; i < active.size(); ++i)
    {
      threads.emplace_back(
          [this, i, &active, &durations, &errors]()
          {
            // Capture any feature exception instead of letting it escape the
            // thread (which would call std::terminate). It is rethrown after
            // join, so process_stream's on_error handler can skip the frame —
            // matching the serial/debug path's behavior.
            try
            {
              auto t_start = std::chrono::high_resolution_clock::now();
              features_[i] = active[i]->extract(frame_);
              auto t_end = std::chrono::high_resolution_clock::now();
              durations[i] = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
            }
            catch (...)
            {
              errors[i] = std::current_exception();
            }
          });
    }

    // Wait for all threads to complete
    for (auto& thread : threads)
    {
      thread.join();
    }

    // Propagate the first feature exception (if any) now that every thread has
    // joined — no thread outlives this scope.
    for (const auto& error : errors)
    {
      if (error)
      {
        std::rethrow_exception(error);
      }
    }
  }

  // Store per-feature timing
  for (size_t i = 0; i < features_.size(); ++i)
  {
    timing_.feature_ms[features_[i].name] = durations[i];
  }

  // Publish the depth cue (if a depth-producing feature ran) so the 3D
  // neural-field selection can lift the fused saliency into a depth volume.
  // Reset first so a stale map never survives a frame without depth.
  run_state_.depth_map.release();
  for (size_t i = 0; i < active.size(); ++i)
  {
    if (active[i]->produces_depth() && !features_[i].data.empty())
    {
      run_state_.depth_map = features_[i].data.clone();
    }
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
  cv::Mat fused = fusion_->fuse(features_, feature_weights_, frame_.size());
  // M17: fold the top-down task-relevance channel into the master map,
  // turning it into a priority map. Inactive config returns `fused` untouched.
  fused = top_down_->apply(fused, frame_.image);
  saliency_ = core::SaliencyMap(fused);
}

void AttentionPipeline::detect_peaks()
{
  saliency_.peaks = selection_->select(saliency_.map, run_state_);

  std::cout << "  Peaks detected: " << saliency_.peaks.size();
  if (selection_->name() == "ior")
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
