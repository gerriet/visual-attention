#include "attention/features/symmetry_feature.h"
#include <cmath>
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

  // Use precomputed grayscale pyramid from frame
  if (!frame.pyramids_computed || frame.gray_pyramid.empty())
  {
    throw std::runtime_error("SymmetryFeature: Grayscale pyramid not computed. Call frame.compute_pyramids() first.");
  }

  cv::Mat result;

  // If compute_at_scale is specified, only compute at that scale
  if (config_.compute_at_scale > 0)
  {
    int scale_index = std::min(config_.compute_at_scale, static_cast<int>(frame.gray_pyramid.size()) - 1);
    const auto& level = frame.gray_pyramid[scale_index];
    cv::Mat sym = compute_radial_symmetry(level);

    // Resize to original size and normalize
    result = normalize_and_resize(sym, frame.size());
  }
  else
  {
    // Default: compute radial symmetry at each scale of the cached pyramid
    std::vector<cv::Mat> symmetry_maps;
    for (const auto& level : frame.gray_pyramid)
    {
      cv::Mat sym = compute_radial_symmetry(level);
      symmetry_maps.push_back(sym);
    }

    // Combine scales
    cv::Mat combined_symmetry = combine_scales(symmetry_maps);

    // Resize to original size and normalize
    result = normalize_and_resize(combined_symmetry, frame.size());
  }

  return core::FeatureMap("symmetry", result, 1.0f);
}

cv::Mat SymmetryFeature::compute_radial_symmetry(const cv::Mat& image) const
{
  if (image.empty())
  {
    return cv::Mat();
  }

  // Compute gradients using Sobel
  cv::Mat grad_x, grad_y;
  cv::Sobel(image, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(image, grad_y, CV_32F, 0, 1, 3);

  // Compute gradient magnitude
  cv::Mat magnitude;
  cv::magnitude(grad_x, grad_y, magnitude);

  // Find maximum gradient for thresholding
  double max_mag;
  cv::minMaxLoc(magnitude, nullptr, &max_mag);
  float mag_threshold = static_cast<float>(max_mag * config_.gradient_threshold);

  // Initialize accumulator for symmetry votes
  cv::Mat accumulator = cv::Mat::zeros(image.size(), CV_32F);

  int rows = image.rows;
  int cols = image.cols;
  int max_radius = std::min(rows, cols) / config_.max_radius_factor;

  // For each pixel with significant gradient
  for (int y = 0; y < rows; ++y)
  {
    for (int x = 0; x < cols; ++x)
    {
      float mag = magnitude.at<float>(y, x);
      if (mag < mag_threshold)
        continue;

      // Get normalized gradient direction
      float gx = grad_x.at<float>(y, x);
      float gy = grad_y.at<float>(y, x);

      // Normalize
      float norm = std::sqrt(gx * gx + gy * gy);
      if (norm < 1e-6f)
        continue;

      gx /= norm;
      gy /= norm;

      // Vote along gradient direction (and opposite direction for bilateral symmetry)
      for (int sign = -1; sign <= 1; sign += 2)
      {
        float dir_x = sign * gx;
        float dir_y = sign * gy;

        // Cast votes along this direction
        for (int d = 1; d <= max_radius; ++d)
        {
          // Candidate symmetry center
          int cx = static_cast<int>(x + d * dir_x + 0.5f);
          int cy = static_cast<int>(y + d * dir_y + 0.5f);

          // Check bounds
          if (cx < 0 || cx >= cols || cy < 0 || cy >= rows)
            break;

          // Compute vote weight: magnitude * distance_falloff
          float distance_weight = 1.0f / std::pow(static_cast<float>(d), config_.distance_alpha);
          float vote = mag * distance_weight;

          // Accumulate vote
          accumulator.at<float>(cy, cx) += vote;
        }
      }
    }
  }

  // Normalize accumulator
  cv::normalize(accumulator, accumulator, 0.0, 1.0, cv::NORM_MINMAX);

  // Optional: Apply Gaussian smoothing to reduce noise
  cv::GaussianBlur(accumulator, accumulator, cv::Size(5, 5), 1.0);

  return accumulator;
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
