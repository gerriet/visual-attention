#include "attention/features/symmetry_feature.h"
#include "attention/core/constants.h"
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <algorithm>

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
  // Timing
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

  // Resolve the scale schedule (size-adaptive when auto_scale_schedule set)
  const std::vector<ScaleConfig> scales = resolve_scales(frame);

  // Step 1: Fetch this feature's Gabor bank (precomputed by the pipeline
  // with the wavelength/bandwidth this config specifies)
  auto t_gabor_start = std::chrono::high_resolution_clock::now();
  const auto& gabor_bank = frame.gabor_bank(config_.num_orientations, config_.wavelength, config_.bandwidth);
  auto t_gabor_end = std::chrono::high_resolution_clock::now();

  // Step 2: Compute symmetry at each scale
  std::vector<cv::Mat> scale_results;
  std::vector<std::vector<cv::Mat>> scale_gabor_responses;
  std::vector<double> scale_computation_times;

  for (size_t scale_idx = 0; scale_idx < scales.size(); ++scale_idx)
  {
    const auto& scale_config = scales[scale_idx];
    auto t_scale_start = std::chrono::high_resolution_clock::now();

    // Use specified pyramid level directly
    int pyramid_level = scale_config.pyramid_level;

    if (pyramid_level < 0 || pyramid_level >= static_cast<int>(gabor_bank.size()))
    {
      // Skip this scale if pyramid level is out of range
      std::cerr << "Warning: Skipping scale " << scale_idx << " - pyramid level " << pyramid_level
                << " out of range (available: 0-" << (gabor_bank.size() - 1) << ")" << std::endl;
      continue;
    }

    const auto& gabor_level = gabor_bank[pyramid_level];
    if (gabor_level.empty())
    {
      std::cerr << "Warning: Skipping scale " << scale_idx << " - gabor level " << pyramid_level << " is empty" << std::endl;
      continue;
    }

    std::cerr << "Processing scale " << scale_idx << " at pyramid level " << pyramid_level
              << " (size: " << gabor_level[0].cols << "x" << gabor_level[0].rows
              << ", radii: " << scale_config.min_radius << "-" << scale_config.max_radius << ")" << std::endl;

    // Use gabor responses directly from the pyramid (no resizing needed)
    // Compute symmetry at this scale using thesis approach
    cv::Mat scale_symmetry = compute_radial_symmetry_at_scale(gabor_level, scale_config);
    scale_results.push_back(scale_symmetry);
    scale_gabor_responses.push_back(gabor_level);

    auto t_scale_end = std::chrono::high_resolution_clock::now();
    double scale_ms = std::chrono::duration<double, std::milli>(t_scale_end - t_scale_start).count();
    scale_computation_times.push_back(scale_ms);
  }

  if (scale_results.empty())
  {
    throw std::runtime_error("SymmetryFeature: No valid scales could be computed");
  }

  // Step 3: Combine scales
  auto t_combine_start = std::chrono::high_resolution_clock::now();

  cv::Mat combined;
  if (config_.use_multi_scale && scale_results.size() > 1)
  {
    // Resize all scales to the largest size and take maximum
    int max_size = 0;
    for (const auto& scale_result : scale_results)
    {
      max_size = std::max(max_size, scale_result.cols);
    }

    combined = cv::Mat::zeros(max_size, max_size, CV_32F);
    for (size_t i = 0; i < scale_results.size(); ++i)
    {
      cv::Mat resized;
      if (scale_results[i].size() != combined.size())
      {
        cv::resize(scale_results[i], resized, combined.size(), 0, 0, cv::INTER_LINEAR);
      }
      else
      {
        resized = scale_results[i];
      }

      // Weight by scale factor (as in old code: factor = 0.5 + i*0.5)
      float scale_weight = 0.5f + i * 0.5f;
      combined = cv::max(combined, scale_weight * resized);
    }
  }
  else
  {
    // Single scale
    combined = scale_results[0].clone();
  }

  auto t_combine_end = std::chrono::high_resolution_clock::now();

  // Step 4: Resize to original frame size and normalize
  auto t_resize_start = std::chrono::high_resolution_clock::now();

  cv::Mat result;
  if (combined.size() != frame.size())
  {
    cv::resize(combined, result, frame.size(), 0, 0, cv::INTER_LINEAR);
  }
  else
  {
    result = combined.clone();
  }

  cv::normalize(result, result, 0.0f, 1.0f, cv::NORM_MINMAX);

  auto t_resize_end = std::chrono::high_resolution_clock::now();

  // Capture debug data if requested
  if (debug.enabled)
  {
    double total_ms = std::chrono::duration<double, std::milli>(t_resize_end - t_start).count();
    double gabor_ms = std::chrono::duration<double, std::milli>(t_gabor_end - t_gabor_start).count();
    double combine_ms = std::chrono::duration<double, std::milli>(t_combine_end - t_combine_start).count();
    double resize_ms = std::chrono::duration<double, std::milli>(t_resize_end - t_resize_start).count();

    capture_debug_data(debug, frame, scales, scale_gabor_responses, scale_results, result,
                      total_ms, gabor_ms, scale_computation_times, combine_ms, resize_ms);
  }

  return core::FeatureMap("symmetry", result, 1.0f);
}

std::vector<SymmetryFeature::ScaleConfig> SymmetryFeature::resolve_scales(const core::Frame& frame) const
{
  if (!config_.auto_scale_schedule)
  {
    return config_.scales;
  }

  // Size-adaptive schedule (moved verbatim from the v1 pipeline): start at the
  // first pyramid level where one side is below 256px, plus the two coarser
  // levels, with higher thresholds at coarser scales to suppress false
  // positives and radius_step=2 for speed
  int start_level = 0;
  int min_dim = std::min(frame.width(), frame.height());
  while (min_dim >= 256 && start_level < 10)
  {
    start_level++;
    min_dim /= 2;
  }

  std::vector<ScaleConfig> scales;
  scales.push_back(ScaleConfig(start_level, 5, 20, 2, 3, 0.3f));
  scales.push_back(ScaleConfig(start_level + 1, 8, 25, 2, 3, 0.5f));
  scales.push_back(ScaleConfig(start_level + 2, 10, 30, 2, 3, 0.65f));
  return scales;
}

cv::Mat SymmetryFeature::compute_radial_symmetry_at_scale(const std::vector<cv::Mat>& gabor_responses,
                                                           const ScaleConfig& scale_config) const
{
  if (gabor_responses.empty())
  {
    return cv::Mat();
  }

  const int width_px = gabor_responses[0].cols;
  const int height_px = gabor_responses[0].rows;
  const int num_orientations = static_cast<int>(gabor_responses.size());

  // Generate list of radii to test based on min, max, and step
  std::vector<int> radii;
  for (int r = scale_config.min_radius; r <= scale_config.max_radius; r += scale_config.radius_step)
  {
    radii.push_back(r);
  }

  if (radii.empty())
  {
    return cv::Mat();
  }

  const int num_radii = static_cast<int>(radii.size());

  // Create storage for each radius band (similar to symmetrybands in old code)
  std::vector<cv::Mat> radius_bands(num_radii);
  for (int i = 0; i < num_radii; ++i)
  {
    radius_bands[i] = cv::Mat::zeros(height_px, width_px, CV_32F);
  }

  // OPTIMIZATION #2: Parallelize across radius bands; each band accumulates
  // its orientations sequentially in fixed order, so no locking is needed and
  // float summation order is identical on every run (bit-reproducible results)
  float delta_alpha = 180.0f / num_orientations;

  #pragma omp parallel for schedule(dynamic) if(num_radii >= 2)
  for (int radius_idx = 0; radius_idx < num_radii; ++radius_idx)
  {
    int radius = radii[radius_idx];

    for (int orientation = 0; orientation < num_orientations; ++orientation)
    {
      float angle = orientation * delta_alpha;

      // Compute contribution from this orientation and radius
      cv::Mat contribution = compute_orientation_radius_contribution(
          gabor_responses[orientation], angle, radius, scale_config.width, num_orientations);

      radius_bands[radius_idx] += contribution;
    }
  }

  // Select maximum across radii, but with threshold and consistency
  // This makes symmetry centers pop out while suppressing weak/broad responses
  cv::Mat result = cv::Mat::zeros(height_px, width_px, CV_32F);

  // Normalize radius bands first to get consistent threshold values
  float max_response = 0.0f;
  for (int i = 0; i < num_radii; ++i)
  {
    double min_val, max_val;
    cv::minMaxLoc(radius_bands[i], &min_val, &max_val);
    max_response = std::max(max_response, static_cast<float>(max_val));
  }

  if (max_response > 0.0f)
  {
    for (int i = 0; i < num_radii; ++i)
    {
      radius_bands[i] /= max_response;
    }
  }

  // Apply threshold: only consider radii with strong symmetry
  // Count consecutive radii above threshold (radius consistency)
  for (int y = 0; y < height_px; ++y)
  {
    for (int x = 0; x < width_px; ++x)
    {
      float max_val = 0.0f;
      int num_above_threshold = 0;

      for (int i = 0; i < num_radii; ++i)
      {
        float val = radius_bands[i].at<float>(y, x);

        // Count radii above threshold
        if (val >= scale_config.symmetry_threshold)
        {
          num_above_threshold++;
          max_val = std::max(max_val, val);
        }
      }

      // Require at least 2 radii above threshold (consistency)
      // Weight by number of consistent radii (more = stronger center)
      if (num_above_threshold >= 2)
      {
        float consistency_weight = std::min(1.0f, num_above_threshold / 4.0f);
        result.at<float>(y, x) = max_val * consistency_weight;
      }
      // else: result stays 0 (weak symmetry suppressed)
    }
  }

  return result;
}

cv::Mat SymmetryFeature::compute_orientation_radius_contribution(
    const cv::Mat& gabor_orientation,
    float orientation_angle,
    int radius,
    int width,
    int num_orientations) const
{
  // This implements the core symmetry_intern logic from the old code (lines 109-171)

  const int width_px = gabor_orientation.cols;
  const int height_px = gabor_orientation.rows;
  cv::Mat contribution = cv::Mat::zeros(height_px, width_px, CV_32F);

  // Convert angle to radians (note: old code uses 180-alpha)
  float alpha_rad = (180.0f - orientation_angle) * M_PI / 180.0f;
  float cos_a = std::cos(alpha_rad);
  float sin_a = std::sin(alpha_rad);

  // Calculate summation box dimensions
  int box_length = static_cast<int>(M_PI * (radius + width) / num_orientations);
  int box_width = width;

  // Center point of summation area (ax, ay)
  int ax = -static_cast<int>(cos_a * (radius + width / 2.0f));
  int ay = static_cast<int>(sin_a * (radius + width / 2.0f));

  // Calculate search area size
  int area = static_cast<int>(std::sqrt((box_length + 1) * (box_length + 1) +
                                        (box_width + 1) * (box_width + 1)));

  // Build index array of points in summation box (old code lines 131-148)
  // OPTIMIZATION #1: This is computed once per orientation/radius, not per pixel
  std::vector<cv::Point> box_points;
  int max_x = 0, max_y = 0;

  for (int x = -area; x < area; ++x)
  {
    for (int y = -area; y < area; ++y)
    {
      float w = sin_a * y + cos_a * x;
      float l = cos_a * y - sin_a * x;

      if (std::abs(2 * w) <= box_width && std::abs(2 * l) <= box_length)
      {
        box_points.push_back(cv::Point(x, y));
        max_x = std::max(max_x, std::abs(x));
        max_y = std::max(max_y, std::abs(y));
      }
    }
  }

  // Normalization factor
  float normalization = static_cast<float>(box_points.size() * num_orientations) / 2.0f;
  if (normalization < 1.0f) normalization = 1.0f;

  // Core loop: for each pixel, sum responses in symmetry box (old code lines 154-169)
  int margin_y = std::abs(ay) + max_y;
  int margin_x = std::abs(ax) + max_x;

  // Pre-compute for faster access
  const int num_box_points = static_cast<int>(box_points.size());

  for (int y = margin_y; y < height_px - margin_y; ++y)
  {
    for (int x = margin_x; x < width_px - margin_x; ++x)
    {
      float sum = 0.0f;

      // Sample points in both opposite directions
      int x1_base = x + ax;
      int y1_base = y + ay;
      int x2_base = x - ax;
      int y2_base = y - ay;

      for (int i = 0; i < num_box_points; ++i)
      {
        const cv::Point& offset = box_points[i];
        int x1 = x1_base + offset.x;
        int y1 = y1_base + offset.y;
        int x2 = x2_base - offset.x;
        int y2 = y2_base - offset.y;

        // Bounds checking
        if (x1 >= 0 && x1 < width_px && y1 >= 0 && y1 < height_px &&
            x2 >= 0 && x2 < width_px && y2 >= 0 && y2 < height_px)
        {
          sum += gabor_orientation.at<float>(y1, x1) + gabor_orientation.at<float>(y2, x2);
        }
      }

      contribution.at<float>(y, x) = sum / normalization;
    }
  }

  return contribution;
}

void SymmetryFeature::capture_debug_data(DebugContext& debug,
                                         const core::Frame& frame,
                                         const std::vector<ScaleConfig>& scales,
                                         const std::vector<std::vector<cv::Mat>>& scale_gabor_responses,
                                         const std::vector<cv::Mat>& scale_results,
                                         const cv::Mat& result,
                                         double total_ms,
                                         double gabor_computation_ms,
                                         const std::vector<double>& scale_computation_times,
                                         double combine_ms,
                                         double resize_ms) const
{
  // Annotations
  debug.add_annotation("num_orientations", std::to_string(config_.num_orientations));
  debug.add_annotation("num_scales", std::to_string(scales.size()));
  debug.add_annotation("use_multi_scale", config_.use_multi_scale ? "true" : "false");
  debug.add_annotation("output_size", std::to_string(result.cols) + "x" + std::to_string(result.rows));

  // Scale configurations
  for (size_t i = 0; i < scales.size(); ++i)
  {
    const auto& sc = scales[i];
    debug.add_annotation("scale_" + std::to_string(i) + "_pyramid_level", std::to_string(sc.pyramid_level));
    debug.add_annotation("scale_" + std::to_string(i) + "_radius_range",
                        std::to_string(sc.min_radius) + "-" + std::to_string(sc.max_radius) +
                        " (step=" + std::to_string(sc.radius_step) + ")");
    debug.add_annotation("scale_" + std::to_string(i) + "_width", std::to_string(sc.width));
    debug.add_annotation("scale_" + std::to_string(i) + "_threshold", std::to_string(sc.symmetry_threshold));
  }

  // Timings
  debug.add_timing("gabor_computation", gabor_computation_ms);
  for (size_t i = 0; i < scale_computation_times.size(); ++i)
  {
    debug.add_timing("scale_" + std::to_string(i) + "_computation", scale_computation_times[i]);
  }
  debug.add_timing("combine_scales", combine_ms);
  debug.add_timing("resize_and_normalize", resize_ms);
  debug.add_timing("total_time", total_ms);

  // Basic level: Per-scale results
  if (debug.is_level(DebugContext::Level::Basic))
  {
    for (size_t i = 0; i < scale_results.size(); ++i)
    {
      int level = i < scales.size() ? scales[i].pyramid_level : -1;
      int size = scale_results[i].cols;
      std::string name = "scale_" + std::to_string(i) + "_level" + std::to_string(level) +
                         "_symmetry_" + std::to_string(size) + "x" + std::to_string(size);
      debug.add_image(name, scale_results[i]);
    }
  }

  // Detailed level: Add individual Gabor orientations for first scale
  if (debug.is_level(DebugContext::Level::Detailed) && !scale_gabor_responses.empty())
  {
    const auto& first_scale_gabor = scale_gabor_responses[0];
    int max_orientations_to_save = std::min(4, static_cast<int>(first_scale_gabor.size()));

    for (int i = 0; i < max_orientations_to_save; ++i)
    {
      if (!first_scale_gabor[i].empty())
      {
        int angle = (i * 180) / config_.num_orientations;
        std::string img_name = "scale0_gabor_" + std::to_string(angle) + "deg";
        debug.add_image(img_name, first_scale_gabor[i]);
      }
    }
  }
}

} // namespace features
} // namespace attention
