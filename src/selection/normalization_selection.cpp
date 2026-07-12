#include "attention/selection/normalization_selection.h"
#include "attention/core/constants.h"
#include <algorithm>
#include <cmath>

namespace attention
{
namespace selection
{

NormalizationSelection::NormalizationSelection(const SelectionParams& shared, const Params& params)
  : shared_(shared), params_(params)
{
  // Guard user-facing knobs: a non-positive exponent turns the background zeros
  // left by max(saliency,0) into 0^(-k) = +Inf and poisons the whole map; a
  // negative semi-saturation can drive the denominator to zero. Keep both in
  // their meaningful (positive) range.
  params_.exponent = std::max(0.1f, params_.exponent);
  params_.sigma = std::max(0.0f, params_.sigma);
}

std::vector<core::Peak> NormalizationSelection::select(const cv::Mat& saliency, core::RunState& state) const
{
  std::vector<core::Peak> peaks;
  if (saliency.empty())
  {
    return peaks;
  }

  // --- Divisive normalization: scale-invariant contrast competition ---------
  cv::Mat a = cv::max(saliency, 0.0); // the drive cannot be negative
  cv::Mat an;
  cv::pow(a, params_.exponent, an); // a^n
  cv::Mat pool;
  cv::GaussianBlur(an, pool, cv::Size(0, 0), std::max(1, params_.pool_size)); // Gaussian surround pool
  const double mean_an = cv::mean(an)[0];
  const double semi = params_.sigma * mean_an; // semi-saturation, relative to mean activity
  cv::Mat response;
  cv::divide(an, pool + (semi + 1e-12), response);
  // Divisive normalization boosts contrast, which on a solid region concentrates
  // on its edges (a ring). A light smoothing consolidates each region into one
  // central peak, so WTA and inhibition of return act on the object, not a
  // rotating set of edge points. (Too much smoothing would erase the small-blob
  // advantage; the default is deliberately gentle.)
  if (params_.smooth_factor > 0.0f)
  {
    cv::GaussianBlur(response, response, cv::Size(0, 0),
                     std::max(1.0, static_cast<double>(params_.pool_size) * params_.smooth_factor));
  }
  cv::normalize(response, response, 0.0, 1.0, cv::NORM_MINMAX); // [0,1] value/IOR scale

  // --- Absolute detection gate on the RAW fused saliency --------------------
  // Detection stays absolute (as nms/field do): a location competes only if it
  // clears the threshold. Contrast normalization alone would amplify a
  // near-empty frame's noise to [0,1] and hallucinate peaks; the raw gate
  // rejects that. Divisive normalization then decides which of the *detectable*
  // locations win — fairly across regions of different local contrast.
  cv::Mat competed = response.clone();
  competed.setTo(0.0f, a < shared_.threshold);

  // --- Cross-frame, space-based inhibition of return (decaying) --------------
  const bool use_ior = params_.ior_decay > 0.0f;
  cv::Mat& inhibition = state.inhibition_map;
  if (use_ior)
  {
    if (state.frame_index == 0 || inhibition.size() != competed.size() || inhibition.type() != CV_32F)
    {
      inhibition = cv::Mat::zeros(competed.size(), CV_32F);
    }
    competed = cv::max(competed - inhibition, 0.0);
  }

  // --- Sequential winner-take-all readout (the scanpath) --------------------
  cv::Mat working = competed;
  const int radius = std::max(1, shared_.ior_radius);
  const float ksigma = radius / attention::constants::IOR_SIGMA_FACTOR;
  const cv::Rect map_rect(0, 0, working.cols, working.rows);

  for (int i = 0; i < shared_.max_count; ++i)
  {
    double max_val = 0.0;
    cv::Point max_loc;
    cv::minMaxLoc(working, nullptr, &max_val, nullptr, &max_loc);
    if (max_val < shared_.threshold) // the raw gate already zeroed non-detectable regions
    {
      break;
    }
    peaks.emplace_back(max_loc, static_cast<float>(max_val));

    // Gaussian suppression around the winner: within-frame (so the next winner
    // is elsewhere) and, on a stream, deposited into the carried IOR map.
    const cv::Rect kernel_rect(max_loc.x - radius, max_loc.y - radius, 2 * radius + 1, 2 * radius + 1);
    const cv::Rect roi = kernel_rect & map_rect;
    for (int y = roi.y; y < roi.y + roi.height; ++y)
    {
      float* wrow = working.ptr<float>(y);
      float* irow = use_ior ? inhibition.ptr<float>(y) : nullptr;
      for (int x = roi.x; x < roi.x + roi.width; ++x)
      {
        const float dx = static_cast<float>(x - max_loc.x);
        const float dy = static_cast<float>(y - max_loc.y);
        const float g = std::exp(-(dx * dx + dy * dy) / (2.0f * ksigma * ksigma));
        wrow[x] *= (1.0f - shared_.ior_strength * g);
        if (irow != nullptr)
        {
          irow[x] = std::min(1.0f, irow[x] + shared_.ior_strength * g);
        }
      }
    }
  }

  // Decay the carried inhibition for the next frame (thesis §8.3).
  if (use_ior)
  {
    inhibition *= params_.ior_decay;
  }

  return peaks;
}

} // namespace selection
} // namespace attention
