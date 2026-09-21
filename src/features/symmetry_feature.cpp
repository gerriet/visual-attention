#include "attention/features/symmetry_feature.h"
#include "attention/core/constants.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace attention
{
namespace features
{

namespace
{

// The quadrature pair (even, odd) of one Gabor orientation; same envelope as
// the shared bank (Frame::create_gabor_kernel).
std::pair<cv::Mat, cv::Mat> gabor_pair(double wavelength, double bandwidth, double theta)
{
  int size = static_cast<int>(std::ceil(wavelength * constants::GABOR_KERNEL_SIZE_FACTOR));
  if (size % 2 == 0)
  {
    ++size;
  }
  const double sigma = wavelength * bandwidth / M_PI;
  const double gamma = constants::GABOR_ASPECT_RATIO;
  cv::Mat even = cv::getGaborKernel(cv::Size(size, size), sigma, theta, wavelength, gamma, 0.0, CV_32F);
  cv::Mat odd = cv::getGaborKernel(cv::Size(size, size), sigma, theta, wavelength, gamma, M_PI * 0.5, CV_32F);
  // A uniform area must give no edge energy: remove the even kernel's DC part
  even -= cv::mean(even)[0];
  return {even, odd};
}

cv::Mat quadrature_energy(const cv::Mat& gray, const std::pair<cv::Mat, cv::Mat>& kernels)
{
  cv::Mat even, odd, energy;
  cv::filter2D(gray, even, CV_32F, kernels.first, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::filter2D(gray, odd, CV_32F, kernels.second, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
  cv::magnitude(even, odd, energy);
  return energy;
}

} // namespace

SymmetryFeature::SymmetryFeature(const Config& config) : config_(config)
{
  if (config_.num_orientations < 2)
  {
    throw std::invalid_argument("SymmetryFeature: needs at least 2 orientations");
  }
  if (config_.scales.empty())
  {
    throw std::invalid_argument("SymmetryFeature: needs at least one scale");
  }
  // Absolute scale: the energy of a full-contrast vertical step edge is 1.
  cv::Mat step = cv::Mat::zeros(64, 64, CV_32F);
  step(cv::Rect(32, 0, 32, 64)).setTo(1.0f);
  const cv::Mat response = quadrature_energy(step, gabor_pair(config_.wavelength, config_.bandwidth, 0.0));
  double peak = 0.0;
  cv::minMaxLoc(response(cv::Rect(16, 16, 32, 32)), nullptr, &peak);
  energy_calibration_ = peak > 0.0 ? static_cast<float>(1.0 / peak) : 1.0f;
}

std::vector<cv::Mat> SymmetryFeature::edge_energy(const cv::Mat& gray_float) const
{
  std::vector<cv::Mat> energy(config_.num_orientations);
  const int border = std::min(config_.border_clear, std::min(gray_float.cols, gray_float.rows) / 4);
  for (int o = 0; o < config_.num_orientations; ++o)
  {
    const double theta = o * M_PI / config_.num_orientations;
    cv::Mat e = quadrature_energy(gray_float, gabor_pair(config_.wavelength, config_.bandwidth, theta));
    e *= energy_calibration_ * config_.gabor_gain;
    cv::min(e, 1.0f, e); // the original clips the Gabor magnitude (at 255)
    if (border > 0)
    {
      // Filter responses at the image border are artefacts of the padding
      e.rowRange(0, border).setTo(0.0f);
      e.rowRange(e.rows - border, e.rows).setTo(0.0f);
      e.colRange(0, border).setTo(0.0f);
      e.colRange(e.cols - border, e.cols).setTo(0.0f);
    }
    energy[o] = e;
  }
  return energy;
}

cv::Mat SymmetryFeature::box_kernel(float orientation_degrees, int radius, int step, int width,
                                    float& normalization) const
{
  // Geometry of the original symmetry_intern (note its 180 - alpha): the boxes
  // sit at +-(ax, ay) from the point, box_length along the tangent.
  const float alpha = (180.0f - orientation_degrees) * static_cast<float>(M_PI) / 180.0f;
  const float cos_a = std::cos(alpha);
  const float sin_a = std::sin(alpha);
  const int box_length = static_cast<int>(M_PI * (radius + step) / config_.num_orientations);
  const int box_width = width;
  const int ax = -static_cast<int>(cos_a * (radius + step / 2.0f));
  const int ay = static_cast<int>(sin_a * (radius + step / 2.0f));
  const int area = static_cast<int>(
      std::sqrt(static_cast<float>((box_length + 1) * (box_length + 1) + (box_width + 1) * (box_width + 1))));

  const int reach = std::max(std::abs(ax), std::abs(ay)) + area;
  cv::Mat kernel = cv::Mat::zeros(2 * reach + 1, 2 * reach + 1, CV_32F);
  int count = 0;
  for (int x = -area; x < area; ++x)
  {
    for (int y = -area; y < area; ++y)
    {
      const float w = sin_a * y + cos_a * x;
      const float l = cos_a * y - sin_a * x;
      if (std::abs(2 * w) <= box_width && std::abs(2 * l) <= box_length)
      {
        // One box on each side of the point, point-mirrored
        kernel.at<float>(reach + ay + y, reach + ax + x) += 1.0f;
        kernel.at<float>(reach - ay - y, reach - ax - x) += 1.0f;
        ++count;
      }
    }
  }
  normalization = std::max(1.0f, count * config_.num_orientations / 2.0f);
  return kernel;
}

cv::Mat SymmetryFeature::symmetry_at_scale(const std::vector<cv::Mat>& energy, const ScaleConfig& scale) const
{
  const cv::Size size = energy.front().size();
  const float offset = scale.clip_offset >= 0.0f ? scale.clip_offset : config_.clip_offset;
  const float delta_alpha = 180.0f / config_.num_orientations;

  cv::Mat result = cv::Mat::zeros(size, CV_32F);
  int band_index = 0;
  for (int radius = scale.min_radius; radius <= scale.max_radius;
       radius += std::max(1, scale.radius_step), ++band_index)
  {
    // The boxes must fit into the image for a band to mean anything
    if (2 * (radius + scale.radius_step) >= std::min(size.width, size.height))
    {
      break;
    }
    cv::Mat band = cv::Mat::zeros(size, CV_32F);
    for (int o = 0; o < config_.num_orientations; ++o)
    {
      float normalization = 1.0f;
      const cv::Mat kernel = box_kernel(o * delta_alpha, radius, scale.radius_step, scale.width, normalization);
      cv::Mat summed;
      cv::filter2D(energy[o], summed, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
      band += summed / normalization;
    }
    // Larger radii get a small bonus (original: + index * step), then the
    // offset that removes one-sided responses
    band += band_index * scale.radius_step * config_.band_bonus - offset;
    cv::max(result, band, result);
  }
  cv::min(result, 1.0f, result);
  cv::max(result, 0.0f, result);
  return result;
}

cv::Mat SymmetryFeature::evaluate(const cv::Mat& gray) const
{
  if (gray.empty() || gray.channels() != 1)
  {
    throw std::runtime_error("SymmetryFeature::evaluate: needs a non-empty single-channel image");
  }
  cv::Mat base;
  if (gray.depth() == CV_8U)
  {
    gray.convertTo(base, CV_32F, 1.0 / 255.0);
  }
  else
  {
    gray.convertTo(base, CV_32F);
  }

  cv::Mat combined = cv::Mat::zeros(base.size(), CV_32F);
  const std::size_t scale_count = config_.use_multi_scale ? config_.scales.size() : 1;
  for (std::size_t s = 0; s < scale_count; ++s)
  {
    const ScaleConfig& scale = config_.scales[s];
    if (scale.pyramid_level < 0)
    {
      continue;
    }
    cv::Mat level = base;
    for (int i = 0; i < scale.pyramid_level; ++i)
    {
      cv::Mat half;
      cv::pyrDown(level, half);
      level = half;
    }
    if (std::min(level.cols, level.rows) < 4 * scale.min_radius)
    {
      continue; // too small for even the first radius band
    }
    cv::Mat at_scale = symmetry_at_scale(edge_energy(level), scale);
    if (at_scale.size() != combined.size())
    {
      cv::resize(at_scale, at_scale, combined.size(), 0, 0, cv::INTER_LINEAR);
    }
    // Coarser scales count more (original: factor = 0.5 + 0.5 * which)
    const float weight = config_.use_multi_scale ? 0.5f + 0.5f * scale.pyramid_level : 1.0f;
    at_scale *= weight;
    cv::min(at_scale, 1.0f, at_scale);
    cv::max(combined, at_scale, combined);
  }
  return combined;
}

core::FeatureMap SymmetryFeature::extract(const core::Frame& frame) const
{
  DebugContext no_debug;
  return extract(frame, no_debug);
}

core::FeatureMap SymmetryFeature::extract(const core::Frame& frame, DebugContext& debug) const
{
  const auto start = std::chrono::high_resolution_clock::now();
  if (frame.empty())
  {
    throw std::runtime_error("SymmetryFeature: Cannot extract from empty frame");
  }
  if (!frame.pyramids_computed || frame.gray_pyramid.empty())
  {
    throw std::runtime_error("SymmetryFeature: Grayscale pyramid not computed");
  }

  // The radii are defined on a working image of at most max_working_size
  cv::Mat working = frame.gray_pyramid[0];
  const int long_side = std::max(working.cols, working.rows);
  if (config_.max_working_size > 0 && long_side > config_.max_working_size)
  {
    const double factor = static_cast<double>(config_.max_working_size) / long_side;
    cv::resize(frame.gray_pyramid[0], working, cv::Size(), factor, factor, cv::INTER_AREA);
  }

  cv::Mat symmetry = evaluate(working);
  cv::Mat result;
  if (symmetry.size() != frame.size())
  {
    cv::resize(symmetry, result, frame.size(), 0, 0, cv::INTER_LINEAR);
  }
  else
  {
    result = symmetry;
  }

  if (debug.enabled)
  {
    const auto end = std::chrono::high_resolution_clock::now();
    debug.add_timing("total_time", std::chrono::duration<double, std::milli>(end - start).count());
    debug.add_annotation("num_orientations", std::to_string(config_.num_orientations));
    debug.add_annotation("working_size", std::to_string(working.cols) + "x" + std::to_string(working.rows));
    debug.add_annotation("clip_offset", std::to_string(config_.clip_offset));
    if (debug.is_level(DebugContext::Level::Basic))
    {
      debug.add_image("symmetry_before_resize", symmetry);
    }
  }

  return core::FeatureMap("symmetry", result, 1.0f);
}

} // namespace features
} // namespace attention
