#include "attention/features/color_feature.h"
#include <stdexcept>
#include <chrono>

namespace attention
{
namespace features
{

ColorFeature::ColorFeature(const Config& config) : config_(config) {}

core::FeatureMap ColorFeature::extract(const core::Frame& frame) const
{
  DebugContext dummy_debug;
  return extract(frame, dummy_debug);
}

core::FeatureMap ColorFeature::extract(const core::Frame& frame, DebugContext& debug) const
{
  // Timing (only if debugging)
  auto t_start = std::chrono::high_resolution_clock::now();

  // Validation
  if (frame.empty())
  {
    throw std::runtime_error("ColorFeature: Cannot extract from empty frame");
  }
  if (frame.channels() != 3)
  {
    throw std::runtime_error("ColorFeature: Frame must be color (3 channels)");
  }
  if (!frame.pyramids_computed || frame.rgb_pyramid.empty())
  {
    throw std::runtime_error("ColorFeature: RGB pyramid not computed. Call frame.compute_pyramids() first.");
  }

  // Step 1: Compute opponent color pyramids from RGB pyramid
  auto t_opponent_start = std::chrono::high_resolution_clock::now();
  std::vector<cv::Mat> rg_pyramid, by_pyramid;
  for (const auto& rgb_level : frame.rgb_pyramid)
  {
    cv::Mat rg, by;
    compute_opponent_colors(rgb_level, rg, by);
    rg_pyramid.push_back(rg);
    by_pyramid.push_back(by);
  }
  auto t_opponent_end = std::chrono::high_resolution_clock::now();

  // Step 2: Compute center-surround differences for each channel
  auto t_cs_start = std::chrono::high_resolution_clock::now();
  cv::Mat rg_saliency = compute_center_surround(rg_pyramid);
  cv::Mat by_saliency = compute_center_surround(by_pyramid);
  auto t_cs_end = std::chrono::high_resolution_clock::now();

  // Step 3: Combine RG and BY saliencies
  cv::Mat combined = rg_saliency + by_saliency;

  // Step 4: Normalize and resize to original size
  auto t_norm_start = std::chrono::high_resolution_clock::now();
  cv::Mat result = normalize_and_resize(combined, frame.size());
  auto t_norm_end = std::chrono::high_resolution_clock::now();

  // Capture debug data if requested (keeps algorithm code clean above)
  if (debug.enabled)
  {
    double total_ms = std::chrono::duration<double, std::milli>(t_norm_end - t_start).count();
    double opponent_ms = std::chrono::duration<double, std::milli>(t_opponent_end - t_opponent_start).count();
    double cs_ms = std::chrono::duration<double, std::milli>(t_cs_end - t_cs_start).count();
    double norm_ms = std::chrono::duration<double, std::milli>(t_norm_end - t_norm_start).count();

    capture_debug_data(debug, frame, rg_pyramid, by_pyramid, rg_saliency, by_saliency,
                      combined, result, total_ms, opponent_ms, cs_ms, norm_ms);
  }

  return core::FeatureMap("color", result, 1.0f);
}

void ColorFeature::compute_opponent_colors(const cv::Mat& rgb, cv::Mat& rg, cv::Mat& by) const
{
  // Split RGB channels
  std::vector<cv::Mat> channels;
  cv::split(rgb, channels);
  cv::Mat b = channels[0]; // Blue
  cv::Mat g = channels[1]; // Green
  cv::Mat r = channels[2]; // Red

  // Compute opponent color channels
  // RG = R - G (red-green opponency)
  // BY = B - (R+G)/2 (blue-yellow opponency)
  rg = r - g;
  by = b - (r + g) * 0.5f;

  // Take absolute values for saliency (we care about differences, not direction)
  rg = cv::abs(rg);
  by = cv::abs(by);

  // Normalize each channel independently to [0, 1]
  double min_rg, max_rg, min_by, max_by;
  cv::minMaxLoc(rg, &min_rg, &max_rg);
  cv::minMaxLoc(by, &min_by, &max_by);

  // Only normalize if there's actually variation (use float literals)
  if (max_rg > min_rg)
  {
    cv::normalize(rg, rg, 0.0f, 1.0f, cv::NORM_MINMAX);
  }
  else
  {
    rg = cv::Mat::zeros(rg.size(), CV_32F);
  }

  if (max_by > min_by)
  {
    cv::normalize(by, by, 0.0f, 1.0f, cv::NORM_MINMAX);
  }
  else
  {
    by = cv::Mat::zeros(by.size(), CV_32F);
  }
}

cv::Mat ColorFeature::compute_center_surround(const std::vector<cv::Mat>& pyramid) const
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

cv::Mat ColorFeature::normalize_and_resize(const cv::Mat& feature, const cv::Size& target_size) const
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

void ColorFeature::capture_debug_data(DebugContext& debug,
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
                                      double normalize_ms) const
{
  // Annotations
  debug.add_annotation("pyramid_levels", std::to_string(frame.rgb_pyramid.size()));
  debug.add_annotation("output_size", std::to_string(result.cols) + "x" + std::to_string(result.rows));

  // Timings
  debug.add_timing("opponent_color_computation", opponent_ms);
  debug.add_timing("center_surround_computation", center_surround_ms);
  debug.add_timing("normalize_and_resize", normalize_ms);
  debug.add_timing("total_time", total_ms);

  // Basic level: pyramids and final combined result
  if (debug.is_level(DebugContext::Level::Basic))
  {
    debug.add_pyramid("rgb_pyramid", frame.rgb_pyramid);
    debug.add_pyramid("rg_pyramid", rg_pyramid);
    debug.add_pyramid("by_pyramid", by_pyramid);
    debug.add_image("combined_before_resize", combined);
  }

  // Detailed level: add intermediate processing results
  if (debug.is_level(DebugContext::Level::Detailed))
  {
    // Full resolution opponent colors (level 0)
    if (!rg_pyramid.empty() && !by_pyramid.empty())
    {
      debug.add_image("rg_channel_level0", rg_pyramid[0]);
      debug.add_image("by_channel_level0", by_pyramid[0]);
    }

    // Individual channel center-surround results
    debug.add_image("rg_center_surround", rg_saliency);
    debug.add_image("by_center_surround", by_saliency);
  }

  // Verbose level: reserved for future fine-grained debugging
  // (currently same as Detailed)
}

} // namespace features
} // namespace attention
