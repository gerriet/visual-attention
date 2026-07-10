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
 * BooleanMapFeature — topological figure-ground saliency (not part of the
 * thesis).
 *
 * Each colour channel (CIELab, or intensity on grayscale) is thresholded at a
 * range of levels to produce binary "boolean maps". A region is treated as
 * figure when it is *surrounded* — i.e. its connected component does not touch
 * the image border; border-connected regions are background. Averaging the
 * (L2-normalized) figure maps over all thresholds and channels yields a
 * saliency map driven by enclosure rather than centre-surround contrast, so it
 * is strong on compact foreground objects.
 *
 * This is an ALTERNATIVE saliency operator, opt-in via config; the thesis
 * feature set remains the default.
 *
 * Reference:
 *   J. Zhang, S. Sclaroff, "Saliency Detection: A Boolean Map Approach",
 *   ICCV 2013.
 */
class BooleanMapFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    // Longer side of the working image, in pixels (the model is scale-robust
    // and cheaper at reduced resolution).
    int working_size = 256;
    // Threshold spacing across [0, 255]; smaller = more boolean maps.
    int threshold_step = 20;
    // Gaussian sigma (at working resolution) smoothing the accumulated map.
    double blur_sigma = 3.0;
  };

  BooleanMapFeature() : BooleanMapFeature(Config{}) {}
  explicit BooleanMapFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;

  std::string name() const override { return "boolean-map"; }

 private:
  Config config_;

  // Add the "surrounded" (non-border-connected) foreground of one boolean map,
  // L2-normalized, into the accumulator.
  void accumulate_surrounded(const cv::Mat& boolean_map, cv::Mat& accumulator) const;
};

} // namespace features
} // namespace attention
