#include "attention/pipeline/attention_pipeline.h"
#include "attention/visualization/visualizer.h"
#include <stdexcept>

namespace attention
{
namespace pipeline
{

AttentionPipeline::AttentionPipeline() : processed_(false) {}

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
  // TODO (Week 2): Implement real feature extraction
  // For now, create dummy features for testing

  // Dummy feature 1: Simple gradient
  cv::Mat dummy1(frame_.height(), frame_.width(), CV_32F);
  for (int y = 0; y < dummy1.rows; ++y)
  {
    for (int x = 0; x < dummy1.cols; ++x)
    {
      // Radial gradient
      float dx = x - dummy1.cols / 2.0f;
      float dy = y - dummy1.rows / 2.0f;
      float dist = std::sqrt(dx * dx + dy * dy);
      float max_dist = std::sqrt(dummy1.cols * dummy1.cols / 4.0f + dummy1.rows * dummy1.rows / 4.0f);
      dummy1.at<float>(y, x) = 1.0f - (dist / max_dist);
    }
  }

  features_.push_back(core::FeatureMap("dummy_radial", dummy1, 1.0f));

  // Dummy feature 2: Horizontal gradient
  cv::Mat dummy2(frame_.height(), frame_.width(), CV_32F);
  for (int y = 0; y < dummy2.rows; ++y)
  {
    for (int x = 0; x < dummy2.cols; ++x)
    {
      dummy2.at<float>(y, x) = static_cast<float>(x) / dummy2.cols;
    }
  }

  features_.push_back(core::FeatureMap("dummy_horizontal", dummy2, 0.5f));
}

void AttentionPipeline::integrate_features()
{
  if (features_.empty())
  {
    throw std::runtime_error("No features to integrate");
  }

  // TODO (Week 3): Implement proper weighted integration
  // For now, simple weighted sum

  cv::Mat integrated = cv::Mat::zeros(frame_.size(), CV_32F);

  for (const auto& feature : features_)
  {
    // Weight by confidence
    integrated += feature.confidence * feature.data;
  }

  // Normalize to [0, 1]
  cv::normalize(integrated, integrated, 0.0, 1.0, cv::NORM_MINMAX);

  saliency_ = core::SaliencyMap(integrated);
}

void AttentionPipeline::detect_peaks()
{
  // TODO (Week 3): Implement proper peak detection with non-maximum suppression
  // For now, just find global maximum and a few random peaks

  // Global maximum
  auto global_max = saliency_.get_global_max();
  saliency_.peaks.push_back(global_max);

  // Add a couple more peaks for visualization
  int w = saliency_.size().width;
  int h = saliency_.size().height;

  saliency_.peaks.push_back(core::Peak(cv::Point(w / 4, h / 4), 0.7f));
  saliency_.peaks.push_back(core::Peak(cv::Point(3 * w / 4, h / 4), 0.6f));
}

cv::Mat AttentionPipeline::visualize()
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

  // Feature visualizations
  for (const auto& feature : features_)
  {
    cv::Mat feature_vis = visualization::visualize_feature_map(feature);
    cv::Mat feature_bgr;
    cv::cvtColor(feature_vis, feature_bgr, cv::COLOR_GRAY2BGR);
    vis_images.push_back(feature_bgr);
    labels.push_back(feature.name);
  }

  // Saliency overlay
  cv::Mat saliency_vis = visualization::visualize_saliency_map(saliency_, frame_.image, "", true, false);
  vis_images.push_back(saliency_vis);
  labels.push_back("Saliency");

  // Combine all visualizations
  return visualization::visualize_side_by_side(vis_images, labels, "", false);
}

} // namespace pipeline
} // namespace attention
