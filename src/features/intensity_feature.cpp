#include "attention/features/intensity_feature.h"
#include <stdexcept>

namespace attention
{
namespace features
{

IntensityFeature::IntensityFeature(const Config& config) : config_(config) {}

core::FeatureMap IntensityFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("IntensityFeature: Cannot extract from empty frame");
  }

  // Convert to grayscale intensity
  cv::Mat intensity;
  if (frame.channels() == 1)
  {
    // Already grayscale
    frame.image.convertTo(intensity, CV_32F, 1.0 / 255.0);
  }
  else
  {
    // Convert BGR to grayscale
    cv::Mat gray;
    cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(intensity, CV_32F, 1.0 / 255.0);
  }

  // Determine adaptive pyramid levels if set to 0
  int pyramid_levels = config_.pyramid_levels;
  if (pyramid_levels == 0)
  {
    // Compute adaptive pyramid levels: ensure we can reach at least 16x16 at finest scale
    int min_dim = std::min(frame.width(), frame.height());
    pyramid_levels = 0;
    while (min_dim > 16 && pyramid_levels < 12)
    {
      min_dim /= 2;
      pyramid_levels++;
    }
    // Ensure we have enough levels for center-surround (need at least level 8)
    pyramid_levels = std::max(9, pyramid_levels);
  }

  // Create pyramid
  std::vector<cv::Mat> pyramid = create_pyramid(intensity, pyramid_levels);

  // Compute center-surround
  cv::Mat saliency = compute_center_surround(pyramid);

  // Normalize and resize to original size
  cv::Mat result = normalize_and_resize(saliency, frame.size());

  return core::FeatureMap("intensity", result, 1.0f);
}

std::vector<cv::Mat> IntensityFeature::create_pyramid(const cv::Mat& input, int levels) const
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
    cv::normalize(accumulated, accumulated, 0.0, 1.0, cv::NORM_MINMAX);
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
  cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);

  return result;
}

} // namespace features
} // namespace attention
