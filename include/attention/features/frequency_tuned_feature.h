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
 * FrequencyTunedFeature — full-resolution CIELab band-pass saliency (not part
 * of the thesis).
 *
 * Saliency is the Euclidean distance, in CIELab, between each (lightly blurred)
 * pixel and the image's mean Lab colour. The blur removes fine texture/noise
 * (the high-frequency band) while the mean subtraction removes the smoothly
 * varying background (the low-frequency band), leaving a crisp, full-resolution
 * response with well-defined salient boundaries. On grayscale input it degrades
 * to the absolute deviation from the mean intensity.
 *
 * This is an ALTERNATIVE saliency operator, opt-in via config; the thesis
 * feature set remains the default. It can be read as an alternative colour /
 * contrast channel to the thesis ColorFeature.
 *
 * Reference:
 *   R. Achanta, S. Hemami, F. Estrada, S. Susstrunk, "Frequency-tuned Salient
 *   Region Detection", CVPR 2009.
 */
class FrequencyTunedFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    // Side of the Gaussian kernel suppressing fine texture/noise (forced odd).
    int blur_size = 5;
  };

  FrequencyTunedFeature() : FrequencyTunedFeature(Config{}) {}
  explicit FrequencyTunedFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;

  std::string name() const override { return "frequency-tuned"; }

 private:
  Config config_;
};

} // namespace features
} // namespace attention
