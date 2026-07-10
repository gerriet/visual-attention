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
 * MinimumBarrierFeature — boundary-connectivity saliency (not part of the
 * thesis).
 *
 * Salient objects rarely touch the image border. For every pixel, compute the
 * minimum BARRIER distance to the border: over all paths to any border pixel,
 * the smallest (max − min) of intensities along the path. Background regions
 * connect to the border through paths of nearly constant intensity (distance
 * ≈ 0); a distinct object is walled in by its own contrast. Computed with the
 * FastMBD raster-scan approximation (alternating forward/backward sweeps),
 * per CIE Lab channel, summed.
 *
 * This is an ALTERNATIVE saliency operator, opt-in via config; the thesis
 * feature set remains the default (see docs/ALTERNATIVE_FEATURES.md). Only
 * the core MBD transform is implemented — the paper's appearance-based
 * backgroundness cue and post-processing (MB+) are not.
 *
 * Reference:
 *   J. Zhang, S. Sclaroff, Z. Lin, X. Shen, B. Price, R. Mech, "Minimum
 *   Barrier Salient Object Detection at 80 FPS", ICCV 2015.
 */
class MinimumBarrierFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    // Longer side of the working image (the paper caps at ~300 px).
    int working_size = 256;
    // Number of alternating raster sweeps of the FastMBD approximation
    // (the paper uses 3).
    int num_passes = 3;
    // Gaussian sigma (at working resolution) smoothing the summed map.
    double blur_sigma = 3.0;
  };

  MinimumBarrierFeature() : MinimumBarrierFeature(Config{}) {}
  explicit MinimumBarrierFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;

  std::string name() const override { return "minimum-barrier"; }

 private:
  Config config_;
};

} // namespace features
} // namespace attention
