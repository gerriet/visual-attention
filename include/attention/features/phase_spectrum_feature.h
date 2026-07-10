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
 * PhaseSpectrumFeature — phase-only Fourier saliency (not part of the thesis).
 *
 * Reconstructing an image from its Fourier PHASE alone (unit amplitude)
 * concentrates energy at locations with locally distinct structure — the
 * less periodic / less homogeneous a region, the more it survives. Following
 * the PQFT model this runs on intensity, the two colour-opponent channels
 * (RG, BY), and — when the frame belongs to a temporal stream — a motion
 * channel (absolute intensity change against the previous frame). Channels
 * are phase-only reconstructed independently and their squared responses
 * summed (the paper's PFT formulation, which it reports as performing on par
 * with the full quaternion transform; a true quaternion FFT is not used).
 *
 * The motion channel makes this the first alternative operator that is
 * stream-aware: on `--sequence` / `--live` input, newly appearing or moving
 * structure gains saliency, mirroring the thesis onset feature's role.
 *
 * This is an ALTERNATIVE saliency operator, opt-in via config; the thesis
 * feature set remains the default (see docs/ALTERNATIVE_FEATURES.md).
 *
 * Reference:
 *   C. Guo, Q. Ma, L. Zhang, "Spatio-temporal Saliency Detection Using Phase
 *   Spectrum of Quaternion Fourier Transform", CVPR 2008.
 */
class PhaseSpectrumFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    // Longer side of the working image (the paper resizes to ~64 px).
    int input_size = 64;
    // Gaussian sigma (at working resolution) smoothing the summed response.
    double gaussian_sigma = 3.0;
    // Include the motion channel when the frame carries a previous frame
    // (temporal streams). Static images never have one.
    bool use_motion = true;
  };

  PhaseSpectrumFeature() : PhaseSpectrumFeature(Config{}) {}
  explicit PhaseSpectrumFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;

  std::string name() const override { return "phase-spectrum"; }

 private:
  Config config_;
};

} // namespace features
} // namespace attention
