#include "attention/pipeline/attention_pipeline.h"
#include "attention/features/color_feature.h"
#include "attention/features/intensity_feature.h"
#include "attention/features/symmetry_feature.h"
#include "attention/visualization/visualizer.h"
#include <stdexcept>

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

  // Step 1: Extract features
  extract_features();

  // Step 2: Integrate features into saliency map
  integrate_features();

  // Step 3: Detect peaks
  detect_peaks();

  processed_ = true;
}

void AttentionPipeline::extract_features()
{
  // Extract color feature (if image is color)
  if (frame_.channels() == 3)
  {
    features::ColorFeature color_extractor;
    core::FeatureMap color_feature = color_extractor.extract(frame_);
    features_.push_back(std::move(color_feature));
  }

  // Extract intensity feature (always)
  features::IntensityFeature intensity_extractor;
  core::FeatureMap intensity_feature = intensity_extractor.extract(frame_);
  features_.push_back(std::move(intensity_feature));

  // Extract symmetry feature (always)
  features::SymmetryFeature symmetry_extractor;
  core::FeatureMap symmetry_feature = symmetry_extractor.extract(frame_);
  features_.push_back(std::move(symmetry_feature));

  std::cout << "  Features extracted: " << features_.size() << std::endl;
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
  cv::normalize(integrated, integrated, 0.0, 1.0, cv::NORM_MINMAX);

  saliency_ = core::SaliencyMap(integrated);
}

void AttentionPipeline::detect_peaks()
{
  // Use the built-in peak detection with non-maximum suppression
  // Parameters from configuration
  saliency_.detect_peaks(config_.peak_min_distance, config_.peak_threshold, config_.peak_max_count);

  std::cout << "  Peaks detected: " << saliency_.peaks.size() << std::endl;
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
