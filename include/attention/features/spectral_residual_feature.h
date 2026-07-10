#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/features/feature_extractor.h"
#include <opencv2/opencv.hpp>

namespace attention
{
namespace features
{

/**
 * SpectralResidualFeature — frequency-domain saliency (not part of the thesis).
 *
 * The log-amplitude spectrum of natural images is statistically smooth; what
 * stands out is what departs from that regularity. This feature computes the
 * "spectral residual" — the log-amplitude spectrum minus its own local average
 * — and reconstructs a saliency map by inverse-transforming that residual with
 * the original phase. It is parameter-light and fast (works on a small,
 * aspect-preserving copy of the image).
 *
 * This is an ALTERNATIVE saliency operator, opt-in via config; the thesis
 * feature set (color, eccentricity, symmetry, ...) remains the default. The
 * Python evaluation layer carries an independent implementation of the same
 * model (eval/attention_eval/models/spectral_residual.py).
 *
 * Reference:
 *   X. Hou, L. Zhang, "Saliency Detection: A Spectral Residual Approach",
 *   CVPR 2007.
 */
class SpectralResidualFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    // Longer side of the working image, in pixels. The model is deliberately
    // computed at low resolution (the paper uses ~64); the map is upscaled back.
    int input_size = 64;
    // Side of the box filter that averages the log-amplitude spectrum (h_n).
    int avg_filter_size = 3;
    // Gaussian sigma (at working resolution) smoothing the reconstructed map.
    double gaussian_sigma = 3.0;
  };

  SpectralResidualFeature() : SpectralResidualFeature(Config{}) {}
  explicit SpectralResidualFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;

  std::string name() const override { return "spectral-residual"; }

 private:
  Config config_;
};

} // namespace features
} // namespace attention
