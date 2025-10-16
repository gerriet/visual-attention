#include "attention/features/symmetry_feature.h"
#include "attention/core/constants.h"
#include <cmath>
#include <stdexcept>
#include <chrono>

namespace attention
{
namespace features
{

SymmetryFeature::SymmetryFeature(const Config& config) : config_(config) {}

core::FeatureMap SymmetryFeature::extract(const core::Frame& frame) const
{
  DebugContext dummy_debug;
  return extract(frame, dummy_debug);
}

core::FeatureMap SymmetryFeature::extract(const core::Frame& frame, DebugContext& debug) const
{
  // Timing (only if debugging)
  auto t_start = std::chrono::high_resolution_clock::now();

  // Validation
  if (frame.empty())
  {
    throw std::runtime_error("SymmetryFeature: Cannot extract from empty frame");
  }

  if (!frame.pyramids_computed || frame.gray_pyramid.empty())
  {
    throw std::runtime_error("SymmetryFeature: Grayscale pyramid not computed");
  }

  // Step 1: Ensure Gabor pyramids are computed with sufficient orientations
  auto t_gabor_start = std::chrono::high_resolution_clock::now();
  const_cast<core::Frame&>(frame).compute_gabor_pyramids(frame.gray_pyramid.size(), config_.num_orientations,
                                                          config_.wavelength, config_.bandwidth);

  if (!frame.gabor_pyramids_computed || frame.gabor_pyramids.empty())
  {
    throw std::runtime_error("SymmetryFeature: Failed to compute Gabor pyramids");
  }

  int scale_index = std::min(config_.compute_at_scale, static_cast<int>(frame.gabor_pyramids.size()) - 1);
  if (scale_index < 0)
  {
    throw std::runtime_error("SymmetryFeature: Invalid pyramid configuration");
  }
  const auto& gabor_level = frame.gabor_pyramids[scale_index];

  if (gabor_level.empty())
  {
    throw std::runtime_error("SymmetryFeature: Gabor level is empty");
  }
  auto t_gabor_end = std::chrono::high_resolution_clock::now();

  // Step 2: Compute bilateral symmetry (opposite orientations)
  auto t_bilateral_start = std::chrono::high_resolution_clock::now();
  cv::Mat bilateral = compute_bilateral_symmetry(gabor_level);
  auto t_bilateral_end = std::chrono::high_resolution_clock::now();

  // Step 3: Compute radial symmetry (all orientations)
  auto t_radial_start = std::chrono::high_resolution_clock::now();
  cv::Mat radial = compute_radial_symmetry(gabor_level);
  auto t_radial_end = std::chrono::high_resolution_clock::now();

  // Step 4: Combine and smooth
  auto t_smooth_start = std::chrono::high_resolution_clock::now();
  cv::Mat symmetry_map = constants::BILATERAL_SYMMETRY_WEIGHT * bilateral +
                         constants::RADIAL_SYMMETRY_WEIGHT * radial;
  cv::GaussianBlur(symmetry_map, symmetry_map, cv::Size(5, 5), 1.0);
  auto t_smooth_end = std::chrono::high_resolution_clock::now();

  // Step 5: Resize to original frame size and normalize
  auto t_resize_start = std::chrono::high_resolution_clock::now();
  cv::Mat result;
  if (scale_index > 0)
  {
    cv::resize(symmetry_map, result, frame.size(), 0, 0, cv::INTER_LINEAR);
  }
  else
  {
    result = symmetry_map.clone();
  }

  cv::normalize(result, result, 0.0f, 1.0f, cv::NORM_MINMAX);
  auto t_resize_end = std::chrono::high_resolution_clock::now();

  // Capture debug data if requested (keeps algorithm code clean above)
  if (debug.enabled)
  {
    double total_ms = std::chrono::duration<double, std::milli>(t_resize_end - t_start).count();
    double gabor_ms = std::chrono::duration<double, std::milli>(t_gabor_end - t_gabor_start).count();
    double bilateral_ms = std::chrono::duration<double, std::milli>(t_bilateral_end - t_bilateral_start).count();
    double radial_ms = std::chrono::duration<double, std::milli>(t_radial_end - t_radial_start).count();
    double smooth_ms = std::chrono::duration<double, std::milli>(t_smooth_end - t_smooth_start).count();
    double resize_ms = std::chrono::duration<double, std::milli>(t_resize_end - t_resize_start).count();

    capture_debug_data(debug, frame, gabor_level, bilateral, radial, symmetry_map, result,
                      total_ms, gabor_ms, bilateral_ms, radial_ms, smooth_ms, resize_ms);
  }

  return core::FeatureMap("symmetry", result, 1.0f);
}

cv::Mat SymmetryFeature::compute_bilateral_symmetry(const std::vector<cv::Mat>& gabor_responses) const
{
  if (gabor_responses.empty())
  {
    return cv::Mat();
  }

  cv::Mat symmetry = cv::Mat::zeros(gabor_responses[0].size(), CV_32F);
  int num_orientations = static_cast<int>(gabor_responses.size());

  // For bilateral symmetry, compare opposite orientations
  // Orientations separated by 180 degrees should have similar responses
  for (int i = 0; i < num_orientations / 2; ++i)
  {
    int opposite = i + num_orientations / 2;
    if (opposite < num_orientations)
    {
      // Compute correlation between opposite orientations
      cv::Mat response1 = gabor_responses[i];
      cv::Mat response2 = gabor_responses[opposite];

      // Element-wise minimum indicates agreement
      cv::Mat agreement;
      cv::min(response1, response2, agreement);

      symmetry += agreement;
    }
  }

  // Normalize by number of orientation pairs
  if (num_orientations > 0)
  {
    symmetry /= (num_orientations / 2.0f);
  }

  return symmetry;
}

cv::Mat SymmetryFeature::compute_radial_symmetry(const std::vector<cv::Mat>& gabor_responses) const
{
  if (gabor_responses.empty())
  {
    return cv::Mat();
  }

  cv::Mat symmetry = cv::Mat::zeros(gabor_responses[0].size(), CV_32F);
  int num_orientations = static_cast<int>(gabor_responses.size());

  // For radial symmetry, we want all orientations to be equally represented
  // Compute the variance of responses across orientations
  // Low variance = all orientations present = radial symmetry

  // First, compute mean response across orientations at each pixel
  cv::Mat mean_response = cv::Mat::zeros(gabor_responses[0].size(), CV_32F);
  for (const auto& response : gabor_responses)
  {
    mean_response += response;
  }
  mean_response /= static_cast<float>(num_orientations);

  // Compute variance
  cv::Mat variance = cv::Mat::zeros(gabor_responses[0].size(), CV_32F);
  for (const auto& response : gabor_responses)
  {
    cv::Mat diff = response - mean_response;
    cv::Mat diff_sq;
    cv::multiply(diff, diff, diff_sq);
    variance += diff_sq;
  }
  variance /= static_cast<float>(num_orientations);

  // Convert variance to symmetry: high variance = low symmetry
  // Symmetry = mean_response * (1 - normalized_variance)
  // This captures both the presence of edges and their isotropy

  // Normalize variance
  cv::Mat variance_norm;
  cv::normalize(variance, variance_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

  // Compute symmetry score
  cv::Mat isotropy = 1.0f - variance_norm;

  // Weight by mean response (only consider areas with strong Gabor responses)
  cv::Mat mean_norm;
  cv::normalize(mean_response, mean_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

  cv::multiply(isotropy, mean_norm, symmetry);

  return symmetry;
}

void SymmetryFeature::capture_debug_data(DebugContext& debug,
                                         const core::Frame& frame,
                                         const std::vector<cv::Mat>& gabor_responses,
                                         const cv::Mat& bilateral,
                                         const cv::Mat& radial,
                                         const cv::Mat& symmetry_map,
                                         const cv::Mat& result,
                                         double total_ms,
                                         double gabor_computation_ms,
                                         double bilateral_ms,
                                         double radial_ms,
                                         double smoothing_ms,
                                         double resize_ms) const
{
  // Annotations
  debug.add_annotation("num_orientations", std::to_string(config_.num_orientations));
  debug.add_annotation("compute_scale", std::to_string(config_.compute_at_scale));
  debug.add_annotation("bilateral_weight", std::to_string(constants::BILATERAL_SYMMETRY_WEIGHT));
  debug.add_annotation("radial_weight", std::to_string(constants::RADIAL_SYMMETRY_WEIGHT));
  debug.add_annotation("output_size", std::to_string(result.cols) + "x" + std::to_string(result.rows));

  // Timings
  debug.add_timing("gabor_computation", gabor_computation_ms);
  debug.add_timing("bilateral_symmetry", bilateral_ms);
  debug.add_timing("radial_symmetry", radial_ms);
  debug.add_timing("smoothing", smoothing_ms);
  debug.add_timing("resize_and_normalize", resize_ms);
  debug.add_timing("total_time", total_ms);

  // Basic level: Key symmetry components
  if (debug.is_level(DebugContext::Level::Basic))
  {
    debug.add_image("bilateral_symmetry", bilateral);
    debug.add_image("radial_symmetry", radial);
    debug.add_image("combined_before_resize", symmetry_map);
  }

  // Detailed level: Add individual Gabor orientations
  if (debug.is_level(DebugContext::Level::Detailed))
  {
    // Save first 4 orientations (0°, 30°, 60°, 90° for 12 orientations)
    int max_orientations_to_save = std::min(4, static_cast<int>(gabor_responses.size()));
    for (int i = 0; i < max_orientations_to_save; ++i)
    {
      if (!gabor_responses[i].empty())
      {
        int angle = (i * 180) / config_.num_orientations;
        std::string img_name = "gabor_response_" + std::to_string(angle) + "deg";
        debug.add_image(img_name, gabor_responses[i]);
      }
    }
  }
}

} // namespace features
} // namespace attention
