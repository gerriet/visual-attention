#include "attention/features/orientation_feature.h"
#include <chrono>
#include <iostream>

namespace attention
{
namespace features
{

OrientationFeature::OrientationFeature() : config_() {}

OrientationFeature::OrientationFeature(const Config& config) : config_(config) {}

core::FeatureMap OrientationFeature::extract(const core::Frame& frame) const
{
  DebugContext dummy_debug;
  return extract(frame, dummy_debug);
}

core::FeatureMap OrientationFeature::extract(const core::Frame& frame, DebugContext& debug) const
{
  // Timing (only if debugging)
  auto t_start = std::chrono::high_resolution_clock::now();

  // Validation
  if (frame.empty())
  {
    throw std::runtime_error("OrientationFeature: Cannot extract from empty frame");
  }

  // Step 1: Fetch this feature's Gabor bank (precomputed by the pipeline;
  // angular spacing matches config_.num_orientations, so orientation index
  // and assumed angle always agree)
  auto t_gabor_start = std::chrono::high_resolution_clock::now();
  const auto& bank = frame.gabor_bank(config_.num_orientations, config_.wavelength, config_.bandwidth);
  auto t_gabor_end = std::chrono::high_resolution_clock::now();

  // Step 2: Extract orientation pyramids (transpose from [level][orientation] to [orientation][level])
  std::vector<std::vector<cv::Mat>> orientation_pyramids(config_.num_orientations);
  for (int orient = 0; orient < config_.num_orientations; ++orient)
  {
    for (size_t level = 0; level < bank.size(); ++level)
    {
      if (orient < static_cast<int>(bank[level].size()))
      {
        orientation_pyramids[orient].push_back(bank[level][orient]);
      }
    }
  }

  // Step 3: Compute center-surround differences across orientations and scales
  auto t_cs_start = std::chrono::high_resolution_clock::now();
  std::vector<int> center_scales = {2, 3, 4};
  std::vector<int> delta_scales = {3, 4};

  cv::Mat combined;
  int num_maps = 0;

  for (int orient = 0; orient < config_.num_orientations; ++orient)
  {
    for (int c : center_scales)
    {
      for (int delta : delta_scales)
      {
        int s = c + delta;
        if (s < static_cast<int>(orientation_pyramids[orient].size()))
        {
          cv::Mat cs_map = compute_center_surround(orientation_pyramids[orient], c, s);
          if (!cs_map.empty())
          {
            cv::Mat cs_resized;
            cv::resize(cs_map, cs_resized, frame.size(), 0, 0, cv::INTER_LINEAR);

            if (combined.empty())
            {
              combined = cv::Mat::zeros(frame.size(), CV_32F);
            }

            combined += cs_resized;
            num_maps++;
          }
        }
      }
    }
  }

  if (num_maps > 0)
  {
    combined /= num_maps;
  }
  auto t_cs_end = std::chrono::high_resolution_clock::now();

  // Step 4: Normalize to [0, 1]
  auto t_norm_start = std::chrono::high_resolution_clock::now();
  cv::Mat result;
  cv::normalize(combined, result, 0.0f, 1.0f, cv::NORM_MINMAX);
  auto t_norm_end = std::chrono::high_resolution_clock::now();

  // Capture debug data if requested (keeps algorithm code clean above)
  if (debug.enabled)
  {
    double total_ms = std::chrono::duration<double, std::milli>(t_norm_end - t_start).count();
    double gabor_ms = std::chrono::duration<double, std::milli>(t_gabor_end - t_gabor_start).count();
    double cs_ms = std::chrono::duration<double, std::milli>(t_cs_end - t_cs_start).count();
    double norm_ms = std::chrono::duration<double, std::milli>(t_norm_end - t_norm_start).count();

    capture_debug_data(debug, frame, orientation_pyramids, combined, result, total_ms, gabor_ms, cs_ms, norm_ms,
                       num_maps);
  }

  return core::FeatureMap("orientation", result, 1.0f);
}

cv::Mat OrientationFeature::compute_center_surround(const std::vector<cv::Mat>& gabor_pyramid, int center_scale,
                                                    int surround_scale) const
{
  if (center_scale >= static_cast<int>(gabor_pyramid.size()) ||
      surround_scale >= static_cast<int>(gabor_pyramid.size()))
  {
    return cv::Mat();
  }

  const cv::Mat& center = gabor_pyramid[center_scale];
  const cv::Mat& surround = gabor_pyramid[surround_scale];

  if (center.empty() || surround.empty())
  {
    return cv::Mat();
  }

  // Resize surround to center size for subtraction
  cv::Mat surround_resized;
  cv::resize(surround, surround_resized, center.size(), 0, 0, cv::INTER_LINEAR);

  // Center-surround difference (absolute difference)
  cv::Mat diff;
  cv::absdiff(center, surround_resized, diff);

  return diff;
}

void OrientationFeature::capture_debug_data(DebugContext& debug, const core::Frame& frame,
                                            const std::vector<std::vector<cv::Mat>>& orientation_pyramids,
                                            const cv::Mat& combined, const cv::Mat& result, double total_ms,
                                            double gabor_computation_ms, double center_surround_ms, double normalize_ms,
                                            int num_maps) const
{
  // Annotations
  debug.add_annotation("num_orientations", std::to_string(config_.num_orientations));
  debug.add_annotation("pyramid_levels", std::to_string(frame.gray_pyramid.size()));
  debug.add_annotation("num_center_surround_maps", std::to_string(num_maps));
  debug.add_annotation("output_size", std::to_string(result.cols) + "x" + std::to_string(result.rows));

  // Timings
  debug.add_timing("gabor_computation", gabor_computation_ms);
  debug.add_timing("center_surround_computation", center_surround_ms);
  debug.add_timing("normalization", normalize_ms);
  debug.add_timing("total_time", total_ms);

  // Basic level: Gabor pyramids for each orientation and combined result
  if (debug.is_level(DebugContext::Level::Basic))
  {
    // Save each orientation pyramid
    for (int orient = 0; orient < config_.num_orientations; ++orient)
    {
      if (orient < static_cast<int>(orientation_pyramids.size()))
      {
        std::string pyramid_name =
            "gabor_orientation_" + std::to_string(orient * 180 / config_.num_orientations) + "deg";
        debug.add_pyramid(pyramid_name, orientation_pyramids[orient]);
      }
    }
    debug.add_image("combined_before_normalize", combined);
  }

  // Detailed level: Add individual orientation responses at level 0
  if (debug.is_level(DebugContext::Level::Detailed))
  {
    for (int orient = 0; orient < config_.num_orientations; ++orient)
    {
      if (orient < static_cast<int>(orientation_pyramids.size()) && !orientation_pyramids[orient].empty())
      {
        std::string img_name = "orientation_" + std::to_string(orient * 180 / config_.num_orientations) + "deg_level0";
        debug.add_image(img_name, orientation_pyramids[orient][0]);
      }
    }
  }
}

} // namespace features
} // namespace attention
