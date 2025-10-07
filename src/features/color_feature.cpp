#include "attention/features/color_feature.h"
#include <stdexcept>

namespace attention
{
namespace features
{

ColorFeature::ColorFeature(const Config& config) : config_(config) {}

core::FeatureMap ColorFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("ColorFeature: Cannot extract from empty frame");
  }

  if (frame.channels() != 3)
  {
    throw std::runtime_error("ColorFeature: Frame must be color (3 channels)");
  }

  // Convert to float for processing
  cv::Mat rgb_float;
  frame.image.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

  // Compute opponent color channels
  cv::Mat rg, by;
  compute_opponent_colors(rgb_float, rg, by);

  // Determine adaptive pyramid levels if set to 0
  int pyramid_levels = config_.pyramid_levels;
  if (pyramid_levels == 0)
  {
    // Compute adaptive pyramid levels: ensure we can reach at least 16x16 at finest scale
    // For Itti-Koch c∈{2,3,4}, δ∈{3,4}, we need up to level 8, so min_size >> 8 should be ≥16
    int min_dim = std::min(frame.width(), frame.height());
    pyramid_levels = 0;
    while (min_dim > 16 && pyramid_levels < 12)
    {
      min_dim /= 2;
      pyramid_levels++;
    }
    // Ensure we have enough levels for center-surround (need at least level 8)
    pyramid_levels = std::max(9, pyramid_levels);

    std::cout << "  Adaptive pyramid: " << pyramid_levels << " levels for " << frame.width() << "x" << frame.height()
              << " image" << std::endl;
  }

  // Create pyramids for each channel
  std::vector<cv::Mat> rg_pyramid = create_pyramid(rg, pyramid_levels);
  std::vector<cv::Mat> by_pyramid = create_pyramid(by, pyramid_levels);

  // Compute center-surround for each channel
  cv::Mat rg_saliency = compute_center_surround(rg_pyramid);
  cv::Mat by_saliency = compute_center_surround(by_pyramid);

  // Combine RG and BY saliencies
  cv::Mat combined = rg_saliency + by_saliency;

  // Normalize and resize to original size
  cv::Mat result = normalize_and_resize(combined, frame.size());

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

  // Only normalize if there's actually variation
  if (max_rg > min_rg)
  {
    cv::normalize(rg, rg, 0.0, 1.0, cv::NORM_MINMAX);
  }
  else
  {
    rg = cv::Mat::zeros(rg.size(), CV_32F);
  }

  if (max_by > min_by)
  {
    cv::normalize(by, by, 0.0, 1.0, cv::NORM_MINMAX);
  }
  else
  {
    by = cv::Mat::zeros(by.size(), CV_32F);
  }
}

std::vector<cv::Mat> ColorFeature::create_pyramid(const cv::Mat& input, int levels) const
{
  std::vector<cv::Mat> pyramid;
  pyramid.push_back(input.clone());

  cv::Mat current = input;
  for (int i = 1; i < levels; ++i)
  {
    cv::Mat downsampled;
    cv::pyrDown(current, downsampled);
    pyramid.push_back(downsampled);
    current = downsampled;
  }

  return pyramid;
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
    cv::normalize(accumulated, accumulated, 0.0, 1.0, cv::NORM_MINMAX);
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
  cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);

  return result;
}

} // namespace features
} // namespace attention
