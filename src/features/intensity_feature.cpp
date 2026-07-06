#include "attention/features/intensity_feature.h"
#include <chrono>
#include <stdexcept>

namespace attention
{
namespace features
{

IntensityFeature::IntensityFeature(const Config& config) : config_(config) {}

core::FeatureMap IntensityFeature::extract(const core::Frame& frame) const
{
  DebugContext dummy_debug;
  return extract(frame, dummy_debug);
}

core::FeatureMap IntensityFeature::extract(const core::Frame& frame, DebugContext& debug) const
{
  // Timing (only if debugging)
  auto t_start = std::chrono::high_resolution_clock::now();

  // Validation
  if (frame.empty())
  {
    throw std::runtime_error("IntensityFeature: Cannot extract from empty frame");
  }
  if (!frame.pyramids_computed || frame.gray_pyramid.empty())
  {
    throw std::runtime_error("IntensityFeature: Grayscale pyramid not computed. Call frame.compute_pyramids() first.");
  }

  // Step 1: Compute center-surround differences using grayscale pyramid
  auto t_cs_start = std::chrono::high_resolution_clock::now();
  cv::Mat saliency = compute_center_surround(frame.gray_pyramid);
  auto t_cs_end = std::chrono::high_resolution_clock::now();

  // Step 2: Normalize and resize to original size
  auto t_norm_start = std::chrono::high_resolution_clock::now();
  cv::Mat result = normalize_and_resize(saliency, frame.size());
  auto t_norm_end = std::chrono::high_resolution_clock::now();

  // Capture debug data if requested (keeps algorithm code clean above)
  if (debug.enabled)
  {
    double total_ms = std::chrono::duration<double, std::milli>(t_norm_end - t_start).count();
    double cs_ms = std::chrono::duration<double, std::milli>(t_cs_end - t_cs_start).count();
    double norm_ms = std::chrono::duration<double, std::milli>(t_norm_end - t_norm_start).count();

    capture_debug_data(debug, frame, saliency, result, total_ms, cs_ms, norm_ms);
  }

  return core::FeatureMap("intensity", result, 1.0f);
}

cv::Mat IntensityFeature::compute_center_surround(const std::vector<cv::Mat>& pyramid) const
{
  if (pyramid.empty())
  {
    return cv::Mat();
  }

  // Use scale 4 as intermediate size (Itti-Koch approach)
  // This avoids excessive upsampling of coarse features
  int intermediate_scale = 4;
  if (intermediate_scale >= static_cast<int>(pyramid.size()))
  {
    intermediate_scale = std::max(0, static_cast<int>(pyramid.size()) - 1);
  }
  cv::Size target_size = pyramid[intermediate_scale].size();
  cv::Mat accumulated = cv::Mat::zeros(target_size, CV_32F);

  int num_differences = 0;

  for (int center_idx : config_.center_levels)
  {
    for (int delta : config_.surround_deltas)
    {
      int surround_idx = center_idx + delta;

      // Check bounds
      if (center_idx >= static_cast<int>(pyramid.size()) || surround_idx >= static_cast<int>(pyramid.size()))
      {
        continue;
      }

      // Get center and surround levels
      cv::Mat center = pyramid[center_idx];
      cv::Mat surround = pyramid[surround_idx];

      if (center.empty() || surround.empty())
      {
        continue;
      }

      // Resize surround to center size for comparison (interpolate coarser to finer)
      cv::Mat surround_at_center;
      cv::resize(surround, surround_at_center, center.size(), 0, 0, cv::INTER_LINEAR);

      // Compute absolute difference at center scale
      cv::Mat diff;
      cv::absdiff(center, surround_at_center, diff);

      // Normalize this difference (Itti-Koch normalization operator)
      double minVal, maxVal;
      cv::minMaxLoc(diff, &minVal, &maxVal);
      if (maxVal > minVal)
      {
        diff = (diff - minVal) / (maxVal - minVal);
      }

      // Resize to intermediate scale and accumulate
      cv::Mat diff_resized;
      cv::resize(diff, diff_resized, target_size, 0, 0, cv::INTER_LINEAR);
      accumulated += diff_resized;
      num_differences++;
    }
  }

  // Average across all center-surround combinations
  if (num_differences > 0)
  {
    accumulated /= static_cast<float>(num_differences);
  }

  // Final normalization
  if (!accumulated.empty())
  {
    cv::normalize(accumulated, accumulated, 0.0f, 1.0f, cv::NORM_MINMAX);
  }

  return accumulated;
}

cv::Mat IntensityFeature::normalize_and_resize(const cv::Mat& feature, const cv::Size& target_size) const
{
  cv::Mat result;

  // Resize to target size
  if (feature.size() != target_size)
  {
    cv::resize(feature, result, target_size, 0, 0, cv::INTER_LINEAR);
  }
  else
  {
    result = feature.clone();
  }

  // Final normalization
  cv::normalize(result, result, 0.0f, 1.0f, cv::NORM_MINMAX);

  return result;
}

void IntensityFeature::capture_debug_data(DebugContext& debug, const core::Frame& frame, const cv::Mat& saliency,
                                          const cv::Mat& result, double total_ms, double center_surround_ms,
                                          double normalize_ms) const
{
  // Annotations
  debug.add_annotation("pyramid_levels", std::to_string(frame.gray_pyramid.size()));
  debug.add_annotation("output_size", std::to_string(result.cols) + "x" + std::to_string(result.rows));

  // Timings
  debug.add_timing("center_surround_computation", center_surround_ms);
  debug.add_timing("normalize_and_resize", normalize_ms);
  debug.add_timing("total_time", total_ms);

  // Basic level: grayscale pyramid and center-surround result
  if (debug.is_level(DebugContext::Level::Basic))
  {
    debug.add_pyramid("gray_pyramid", frame.gray_pyramid);
    debug.add_image("center_surround_saliency", saliency);
  }

  // Detailed level: same as Basic for IntensityFeature (simpler than ColorFeature)
  // Could add individual scale differences here if needed in future
}

} // namespace features
} // namespace attention
