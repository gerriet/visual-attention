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
 * OnsetFeature: saliency from the abrupt appearance of structure between
 * consecutive frames (thesis §3.2.5 — attentional capture by onset).
 *
 * The dissertation's onset feature (feature/onset.h) survives only as an
 * empty stub; its implementation was not preserved. This is a principled
 * reconstruction faithful to the concept rather than a line port: onset is
 * the rectified *positive* temporal change in local edge energy — where new
 * structure has appeared since the previous frame — smoothed spatially.
 * Offsets (structure disappearing) are deliberately not salient, matching the
 * onset/offset asymmetry the thesis cites (Folk/Remington; §3.2.5).
 *
 * The feature is temporal: it reads frame.previous_gray, which the pipeline
 * injects from RunState across the frames of a stream. It is inapplicable to
 * the first frame and to independent stills (no previous frame), and the
 * pipeline skips it silently there.
 */
class OnsetFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    bool use_edges = true;    // onset of edge energy (true) vs. raw intensity (false)
    double threshold = 4.0;   // ignore changes below this (noise floor, 0-255 scale)
    int blur_size = 5;        // spatial Gaussian on the change map (odd; 0 = none)
  };

  OnsetFeature() : OnsetFeature(Config{}) {}
  explicit OnsetFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;

  std::string name() const override { return "onset"; }

  // Only runs when a previous frame is available (i.e. inside a stream).
  bool applicable(const core::Frame& frame) const override { return !frame.previous_gray.empty(); }

 private:
  Config config_;

  // Edge energy (gradient magnitude) of a single-channel float image.
  cv::Mat edge_energy(const cv::Mat& gray) const;
};

} // namespace features
} // namespace attention
