#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/features/feature_extractor.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace attention
{
namespace features
{

/**
 * StereoFeature computes depth-based saliency from a rectified stereo pair,
 * ported from the dissertation's StereoFeature (feature/stereo.C, thesis §5.4).
 *
 * The pipeline is the thesis's non-iterative, correlation-based one:
 *
 *   1. Gabor magnitude responses of the left (frame.image) and right
 *      (frame.stereo_right) images at a few near-vertical orientations —
 *      only vertical-ish edges carry horizontal disparity (§5.4.2, the
 *      aperture problem). Defaults: vertical + ±30°, weights 0.5/0.25/0.25.
 *
 *   2. For each orientation and image row, a windowed normalized
 *      cross-correlation over a disparity range gives a similarity ρ(x,d)
 *      (eq. 5.13). Low-structure windows (variance below a threshold) are
 *      excluded.
 *
 *   3. The per-orientation similarities are accumulated into a confidence
 *      volume conf(x,y,d) weighted by orientation and blurred spatially
 *      (eq. 5.14, the Gaussian neighborhood wσ).
 *
 *   4. Winner-take-all per pixel picks the disparity of maximum confidence
 *      (eq. 5.15); the depth saliency is the normalized disparity magnitude
 *      (eq. 5.16) — near surfaces (large |d|) pop out, textureless regions
 *      with no reliable correspondence stay dark.
 *
 * This is the single-scale core. The original's multi-scale variant
 * (StereoMultiFeature) and multi-hypothesis / exclusivity machinery are not
 * ported; at the project's loose-equivalence bar the single-scale depth map
 * is the faithful behavior. The feature is applicable only when the frame
 * carries a right image; otherwise the pipeline skips it silently.
 *
 * The map doubles as a per-pixel depth cue (produces_depth() == true): the
 * pipeline captures it into RunState::depth_map for the 3D neural field.
 */
class StereoFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    int num_orientations = 3;   // 1, 3, or 5 near-vertical orientations
    int min_disparity = -16;    // disparity search range (px, right vs left);
    int max_disparity = 0;      //   mindisp<0, maxdisp=0 searches leftward
    int window_size = 15;       // correlation window length dl (odd)
    double variance_threshold = 3.0; // min windowed std-dev of the Gabor-magnitude
                                     //   response to correlate — gates out flat,
                                     //   textureless windows (empirical, §5.4.2)
    double gabor_wavelength = 8.0;   // dedicated stereo Gabor wavelength
    double gabor_bandwidth = 1.0;
    int confidence_blur = 3;    // spatial Gaussian on each disparity slice (odd; 0 = none)
    int max_working_size = 256; // downscale so the longer side is ≤ this before correlating
  };

  StereoFeature() : StereoFeature(Config{}) {}
  explicit StereoFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;

  std::string name() const override { return "stereo"; }

  // Only runs when the frame carries a right stereo image.
  bool applicable(const core::Frame& frame) const override { return !frame.stereo_right.empty(); }

  // The depth saliency map is a per-pixel depth cue for the 3D field.
  bool produces_depth() const override { return true; }

 private:
  Config config_;

  // Orientation angles (radians, edge normal) and their confidence weights,
  // derived from num_orientations following the thesis presets.
  void orientation_scheme(std::vector<double>& thetas, std::vector<double>& weights) const;

  // Quadrature Gabor magnitude response of a single-channel float image at the
  // given orientation (phase-invariant edge energy).
  cv::Mat gabor_magnitude(const cv::Mat& gray, double theta) const;
};

} // namespace features
} // namespace attention
