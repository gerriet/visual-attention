#include "attention/features/symmetry_feature.h"
#include <stdexcept>

namespace attention
{
namespace features
{

SymmetryFeature::SymmetryFeature(const Config& config) : config_(config) {}

core::FeatureMap SymmetryFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("SymmetryFeature: Cannot extract from empty frame");
  }

  // Convert to grayscale for symmetry computation
  cv::Mat gray;
  if (frame.channels() == 1)
  {
    gray = frame.image.clone();
  }
  else
  {
    cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);
  }

  // Convert to float
  cv::Mat gray_float;
  gray.convertTo(gray_float, CV_32F, 1.0 / 255.0);

  // Smooth to reduce noise
  cv::GaussianBlur(gray_float, gray_float, cv::Size(config_.kernel_size, config_.kernel_size), 0);

  // Determine adaptive pyramid levels if set to 0
  int pyramid_levels = config_.pyramid_levels;
  if (pyramid_levels == 0)
  {
    int min_dim = std::min(frame.width(), frame.height());
    pyramid_levels = 0;
    while (min_dim > 32 && pyramid_levels < 6)
    {
      min_dim /= 2;
      pyramid_levels++;
    }
    pyramid_levels = std::max(3, pyramid_levels); // At least 3 levels
  }

  // Create pyramid
  std::vector<cv::Mat> pyramid = create_pyramid(gray_float, pyramid_levels);

  // Compute symmetry at each scale
  std::vector<cv::Mat> symmetry_maps;
  for (const auto& level : pyramid)
  {
    cv::Mat combined = cv::Mat::zeros(level.size(), CV_32F);
    int num_orientations = 0;

    // Vertical symmetry (left-right)
    if (config_.compute_vertical)
    {
      cv::Mat vert_sym = compute_symmetry(level, true);
      combined += vert_sym;
      num_orientations++;
    }

    // Horizontal symmetry (top-bottom)
    if (config_.compute_horizontal)
    {
      cv::Mat horiz_sym = compute_symmetry(level, false);
      combined += horiz_sym;
      num_orientations++;
    }

    if (num_orientations > 0)
    {
      combined /= static_cast<float>(num_orientations);
    }

    symmetry_maps.push_back(combined);
  }

  // Combine scales
  cv::Mat combined_symmetry = combine_scales(symmetry_maps);

  // Resize to original size and normalize
  cv::Mat result = normalize_and_resize(combined_symmetry, frame.size());

  return core::FeatureMap("symmetry", result, 1.0f);
}

cv::Mat SymmetryFeature::compute_symmetry(const cv::Mat& image, bool vertical) const
{
  if (image.empty())
  {
    return cv::Mat();
  }

  // Compute gradients using Sobel
  cv::Mat grad_x, grad_y;
  cv::Sobel(image, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(image, grad_y, CV_32F, 0, 1, 3);

  // Compute gradient magnitude and orientation
  cv::Mat magnitude, orientation;
  cv::cartToPolar(grad_x, grad_y, magnitude, orientation);

  // Threshold: only consider significant gradients
  double max_mag;
  cv::minMaxLoc(magnitude, nullptr, &max_mag);
  float mag_threshold = static_cast<float>(max_mag * 0.1);

  // Initialize symmetry contribution map
  cv::Mat symmetry = cv::Mat::zeros(image.size(), CV_32F);

  // For each pixel with significant gradient
  int rows = image.rows;
  int cols = image.cols;

  // Define search range based on image size
  int max_range = std::min(rows, cols) / 4;

  for (int y = 0; y < rows; ++y)
  {
    for (int x = 0; x < cols; ++x)
    {
      float mag = magnitude.at<float>(y, x);
      if (mag < mag_threshold)
        continue;

      float angle = orientation.at<float>(y, x);

      // For vertical symmetry: look for matching gradient on opposite side
      // The gradient should point toward or away from the symmetry axis
      if (vertical)
      {
        // Search horizontally for mirrored gradient
        for (int offset = 1; offset < max_range && x + offset < cols && x - offset >= 0; ++offset)
        {
          int x_left = x - offset;
          int x_right = x + offset;

          float mag_left = magnitude.at<float>(y, x_left);
          float mag_right = magnitude.at<float>(y, x_right);

          if (mag_left < mag_threshold || mag_right < mag_threshold)
            continue;

          float angle_left = orientation.at<float>(y, x_left);
          float angle_right = orientation.at<float>(y, x_right);

          // Check if gradients are mirror-symmetric
          // For vertical symmetry, x-component should be opposite, y-component same
          float expected_angle_right = CV_PI - angle_left;
          float angle_diff = std::abs(angle_right - expected_angle_right);
          if (angle_diff > CV_PI)
            angle_diff = 2 * CV_PI - angle_diff;

          // If angles match (mirror symmetry)
          if (angle_diff < 0.5f) // ~30 degrees tolerance
          {
            float contribution = (mag_left * mag_right) / (max_mag * max_mag) / static_cast<float>(offset);
            symmetry.at<float>(y, x) += contribution;
            symmetry.at<float>(y, x_left) += contribution;
            symmetry.at<float>(y, x_right) += contribution;
          }
        }
      }
      else
      {
        // Horizontal symmetry: search vertically
        for (int offset = 1; offset < max_range && y + offset < rows && y - offset >= 0; ++offset)
        {
          int y_top = y - offset;
          int y_bottom = y + offset;

          float mag_top = magnitude.at<float>(y_top, x);
          float mag_bottom = magnitude.at<float>(y_bottom, x);

          if (mag_top < mag_threshold || mag_bottom < mag_threshold)
            continue;

          float angle_top = orientation.at<float>(y_top, x);
          float angle_bottom = orientation.at<float>(y_bottom, x);

          // For horizontal symmetry, angles should be vertically mirrored
          float expected_angle_bottom = -angle_top;
          if (expected_angle_bottom < 0)
            expected_angle_bottom += 2 * CV_PI;

          float angle_diff = std::abs(angle_bottom - expected_angle_bottom);
          if (angle_diff > CV_PI)
            angle_diff = 2 * CV_PI - angle_diff;

          if (angle_diff < 0.5f)
          {
            float contribution = (mag_top * mag_bottom) / (max_mag * max_mag) / static_cast<float>(offset);
            symmetry.at<float>(y, x) += contribution;
            symmetry.at<float>(y_top, x) += contribution;
            symmetry.at<float>(y_bottom, x) += contribution;
          }
        }
      }
    }
  }

  // Normalize
  cv::normalize(symmetry, symmetry, 0.0, 1.0, cv::NORM_MINMAX);

  return symmetry;
}

std::vector<cv::Mat> SymmetryFeature::create_pyramid(const cv::Mat& input, int levels) const
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

cv::Mat SymmetryFeature::combine_scales(const std::vector<cv::Mat>& symmetry_maps) const
{
  if (symmetry_maps.empty())
  {
    return cv::Mat();
  }

  // Use scale 2 as target (coarser than full resolution for efficiency)
  int target_scale = std::min(2, static_cast<int>(symmetry_maps.size()) - 1);
  cv::Size target_size = symmetry_maps[target_scale].size();

  cv::Mat combined = cv::Mat::zeros(target_size, CV_32F);

  for (const auto& sym_map : symmetry_maps)
  {
    cv::Mat resized;
    cv::resize(sym_map, resized, target_size, 0, 0, cv::INTER_LINEAR);
    combined += resized;
  }

  // Average across scales
  combined /= static_cast<float>(symmetry_maps.size());

  // Normalize
  cv::normalize(combined, combined, 0.0, 1.0, cv::NORM_MINMAX);

  return combined;
}

cv::Mat SymmetryFeature::normalize_and_resize(const cv::Mat& feature, const cv::Size& target_size) const
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
