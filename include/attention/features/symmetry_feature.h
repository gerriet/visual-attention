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
 * Symmetry — the dissertation's boundary-based grey-value feature (thesis
 * §5.2.2, after Bollmann; ported from the original `feature/symmetry.C`,
 * single-scale `SymmetryFeature` + `SymmetryMultiFeature`).
 *
 *   1. Edge energy: magnitudes of a quadrature Gabor pair in 12 orientations,
 *      on an *absolute* scale — a full-contrast step edge gives 1 — clipped to
 *      [0, 1], with the image border suppressed.
 *   2. For every point, radius band and orientation alpha: sum the energy of the
 *      edges that lie tangentially to a circle of that radius, in two boxes on
 *      opposite sides of the point (eq. 5.2/5.3, `symmetry_intern`); the bands
 *      adjoin (width = radius step). Normalized by the box area and half the
 *      orientation count.
 *   3. Per point, the maximum over the radius bands of
 *        band + index * bonus - clip_offset,
 *      clipped to [0, 1]. The offset is what makes this a *symmetry* measure:
 *      the sum is additive, so an edge on one side alone already contributes —
 *      about a quarter of what a closed contour does. A fixed offset on an
 *      absolute scale removes those one-sided responses; a relative threshold
 *      cannot (see below).
 *   4. Multi-scale (thesis Tab. 5.1): the same on the working image halved once
 *      and twice; the results are combined by maximum, scale s weighted by
 *      0.5 + 0.5 s.
 *
 * The output is an absolute saliency in [0, 1], not min-max stretched.
 *
 * History: until 2026-09 steps 3–4 normalized every radius band to its own
 * maximum and used relative thresholds plus a "two radii" consistency weight,
 * over three pyramid levels with radii up to 30 px at 1/8 resolution. With no
 * absolute reference a one-sided response was as strong as a true centre: the
 * feature answered in rings *around* objects and its maximum for a lone disk lay
 * beside the disk (docs/replication/REPLICATION_DOSSIER.md, finding A).
 *
 * What the surviving source does not settle: the gain of the original Gabor
 * implementation ("factor = 4 + size/12", commented "Warum??????" there), which
 * fixes what "60 of 255" means. Here the edge energy is calibrated so that a
 * full-contrast step edge is 1 and `gabor_gain` scales it; with gain 1 a closed
 * contour of contrast c sums to about 4c, a one-sided edge to about c.
 */
class SymmetryFeature : public FeatureExtractor
{
 public:
  /// One scale of the multi-scale schedule.
  struct ScaleConfig
  {
    int pyramid_level; // the working image halved this many times (0 = working size)
    int min_radius;    // radius bands min_radius, +step, ... <= max_radius (px at this scale)
    int max_radius;
    int radius_step;   // also the width of a band, so the bands adjoin
    int width;         // width of the summation box across the radius
    float clip_offset; // subtracted from every band; < 0: use Config::clip_offset

    ScaleConfig(int level = 0, int min_r = 6, int max_r = 15, int r_step = 3, int w = 3, float offset = -1.0f)
      : pyramid_level(level), min_radius(min_r), max_radius(max_r), radius_step(r_step), width(w), clip_offset(offset)
    {
    }
  };

  struct Config
  {
    int num_orientations = 12;
    double wavelength = 8.0; // thesis: k0 = 0.75 -> 2 pi / k0 = 8.4 px
    double bandwidth = 1.0;
    // Long side of the image the feature works on (the original ran on 256 x
    // 256); the radii below are pixels at this size and its halvings.
    int max_working_size = 256;
    float gabor_gain = 1.0f;         // edge energy of a full-contrast step edge
    float clip_offset = 60.0f / 255; // the original's clip_threshold
    float band_bonus = 1.0f / 255;   // per radius step and band index (original: i * step)
    int border_clear = 5;            // px of edge energy suppressed at the image border
    bool use_multi_scale = true;
    // Thesis Tab. 5.1: 256 -> radii 6, 9, 12; 128 and 64 -> 6, 9, 12, 15; width 3
    std::vector<ScaleConfig> scales = {ScaleConfig(0, 6, 12, 3, 3), ScaleConfig(1, 6, 15, 3, 3),
                                       ScaleConfig(2, 6, 15, 3, 3)};
  };

  SymmetryFeature() : SymmetryFeature(Config{}) {}
  explicit SymmetryFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;
  std::string name() const override { return "symmetry"; }

  /// Symmetry of an 8-bit or float [0,1] grey image at its own resolution
  /// (all configured scales); CV_32F in [0, 1]. Exposed for tests.
  cv::Mat evaluate(const cv::Mat& gray) const;

 private:
  Config config_;
  float energy_calibration_ = 1.0f; // 1 / (quadrature response to a unit step edge)

  /// Quadrature Gabor energy per orientation, calibrated, clipped, border cleared.
  std::vector<cv::Mat> edge_energy(const cv::Mat& gray_float) const;
  /// Steps 2-3 at one scale.
  cv::Mat symmetry_at_scale(const std::vector<cv::Mat>& energy, const ScaleConfig& scale) const;
  /// The two-box summation kernel for one orientation and radius, and its
  /// normalization (box pixel count * orientations / 2).
  cv::Mat box_kernel(float orientation_degrees, int radius, int step, int width, float& normalization) const;
};

} // namespace features
} // namespace attention
