#include "attention/features/symmetry_feature.h"
#include "attention/core/constants.h"
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

  if (!frame.pyramids_computed || frame.gray_pyramid.empty())
  {
    throw std::runtime_error("SymmetryFeature: Grayscale pyramid not computed");
  }

  // Ensure Gabor pyramids are computed with sufficient orientations
  const_cast<core::Frame&>(frame).compute_gabor_pyramids(frame.gray_pyramid.size(), config_.num_orientations,
                                                          config_.wavelength, config_.bandwidth);

  if (!frame.gabor_pyramids_computed || frame.gabor_pyramids.empty())
  {
    throw std::runtime_error("SymmetryFeature: Failed to compute Gabor pyramids");
  }

  // Select the scale to compute at with bounds checking
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

  // Compute symmetry from Gabor responses
  cv::Mat symmetry_map = compute_gabor_symmetry(gabor_level);

  // Resize to original frame size if needed
  cv::Mat result;
  if (scale_index > 0)
  {
    cv::resize(symmetry_map, result, frame.size(), 0, 0, cv::INTER_LINEAR);
  }
  else
  {
    result = symmetry_map;
  }

  // Normalize to [0, 1]
  cv::normalize(result, result, 0.0f, 1.0f, cv::NORM_MINMAX);

  return core::FeatureMap("symmetry", result, 1.0f);
}

cv::Mat SymmetryFeature::compute_gabor_symmetry(const std::vector<cv::Mat>& gabor_responses) const
{
  if (gabor_responses.empty())
  {
    return cv::Mat();
  }

  // Combine bilateral and radial symmetry
  cv::Mat bilateral = compute_bilateral_symmetry(gabor_responses);
  cv::Mat radial = compute_radial_symmetry(gabor_responses);

  // Combine both types of symmetry (weighted sum)
  cv::Mat combined = constants::BILATERAL_SYMMETRY_WEIGHT * bilateral +
                      constants::RADIAL_SYMMETRY_WEIGHT * radial;

  // Apply Gaussian smoothing to reduce noise
  cv::GaussianBlur(combined, combined, cv::Size(5, 5), 1.0);

  return combined;
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

} // namespace features
} // namespace attention
