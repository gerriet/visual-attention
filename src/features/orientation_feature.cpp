#include "attention/features/orientation_feature.h"
#include <iostream>

namespace attention
{
namespace features
{

OrientationFeature::OrientationFeature() : config_() {}

OrientationFeature::OrientationFeature(const Config& config) : config_(config) {}

core::FeatureMap OrientationFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("OrientationFeature: Cannot extract from empty frame");
  }

  // Ensure Gabor pyramids are computed with required number of orientations
  const_cast<core::Frame&>(frame).compute_gabor_pyramids(frame.gray_pyramid.size(), config_.num_orientations,
                                                          config_.wavelength, config_.bandwidth);

  if (!frame.gabor_pyramids_computed || frame.gabor_pyramids.empty())
  {
    throw std::runtime_error("OrientationFeature: Failed to compute Gabor pyramids");
  }

  // Itti-Koch style center-surround differences
  // Use scales: c ∈ {2,3,4} and s = c + δ where δ ∈ {3,4}
  std::vector<int> center_scales = {2, 3, 4};
  std::vector<int> delta_scales = {3, 4};

  // Accumulate orientation responses across all orientations and scales
  cv::Mat combined;
  int num_maps = 0;

  // For each orientation
  for (int orient = 0; orient < config_.num_orientations; ++orient)
  {
    // Extract this orientation across all pyramid levels
    std::vector<cv::Mat> orientation_pyramid;
    for (size_t level = 0; level < frame.gabor_pyramids.size(); ++level)
    {
      if (orient < static_cast<int>(frame.gabor_pyramids[level].size()))
      {
        orientation_pyramid.push_back(frame.gabor_pyramids[level][orient]);
      }
    }

    // Compute center-surround differences for this orientation
    for (int c : center_scales)
    {
      for (int delta : delta_scales)
      {
        int s = c + delta;
        if (s < static_cast<int>(orientation_pyramid.size()))
        {
          cv::Mat cs_map = compute_center_surround(orientation_pyramid, c, s);
          if (!cs_map.empty())
          {
            // Resize to target frame size
            cv::Mat cs_resized;
            cv::resize(cs_map, cs_resized, frame.size(), 0, 0, cv::INTER_LINEAR);

            // Initialize combined on first map
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

  // Normalize by number of maps
  if (num_maps > 0)
  {
    combined /= num_maps;
  }

  // Normalize to [0, 1]
  cv::normalize(combined, combined, 0.0, 1.0, cv::NORM_MINMAX);

  // Create feature map
  core::FeatureMap feature;
  feature.name = "orientation";
  feature.data = combined;
  feature.confidence = 1.0f;

  return feature;
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

} // namespace features
} // namespace attention
