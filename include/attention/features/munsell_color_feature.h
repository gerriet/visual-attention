#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/features/exclusivity.h"
#include "attention/features/feature_extractor.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace attention
{
namespace features
{

/**
 * Colour contrast — the dissertation's colour feature (thesis §5.3.2, after
 * Bollmann; ported from the original `feature/color.C`, itself "an almost
 * identical copy of the Khoros sources").
 *
 *   1. RGB -> XYZ -> Adams chromatic value -> MTM, the Mathematical Transform
 *      to Munsell (Miyahara & Yoshida 1988), orthogonal form (L, S1, S2)
 *      (eq. 5.7–5.9). The cylindrical form is avoided because its singular
 *      achromatic axis breaks a segmentation.
 *   2. Centroid-linkage region growing in one alternating-direction raster
 *      pass: each pixel joins the most similar of its already-visited
 *      neighbour segments if the Euclidean MTM distance to that segment's
 *      *mean* is below a threshold that rises with the local colour variation
 *      (after Yokoya; eq. 5.10) — otherwise it founds a new segment. The
 *      variance term keeps textured areas from being over-segmented.
 *   3. Saliency of a segment = its mean colour contrast to the neighbouring
 *      segments, weighted by the length of the shared border (eq. 5.11),
 *      optionally divided by the exclusivity of its hue class (12 classes of
 *      30°, §5.5.3), then squashed by a sigmoid that suppresses small and
 *      stresses large contrasts (eq. 5.12). Segments that are too small or too
 *      large get no saliency.
 *
 * The output is an *absolute* saliency in [0, 1] — deliberately not min-max
 * stretched: the thesis's feature integration (§5.5.2) presupposes comparable
 * value ranges, and an image without colour contrast must yield a flat, low
 * map rather than amplified noise.
 *
 * Where the thesis text and the surviving source differ, the source decides
 * the *units* (they fix the meaning of every threshold) and the thesis the
 * *experiment parameters*:
 *   - L is V(Y) = 11.6·(100·Y/Yn)^(1/3) − 1.6 as in the source (the text's
 *     M3 = 0.23·V(Y) is not applied there), on a 0–100 tristimulus scale;
 *   - S2 uses sin(φ), φ = atan2(M2, M1), as in the source and in MTM (the
 *     text prints cos for both);
 *   - the dynamic threshold uses the standard deviation σ in a 3×3 window, as
 *     in the source (the text prints σ²);
 *   - sigmoid slope β = 3 from the text (the source had 4).
 *
 * The Itti–Koch-style opponent-colour feature of the reimplementation remains
 * available as `color` (ColorFeature); this one registers as `color-munsell`.
 */
class MunsellColorFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    // Long side of the image the segmentation runs on (the original worked on
    // 128–256 px maps). The thresholds below are in MTM units and independent
    // of it; the segment-size limits are fractions of the working image.
    int max_working_size = 256;
    float threshold = 8.0f;       // cc_add: constant part of the merge threshold (MTM units)
    float threshold_sigma = 5.0f; // cc_mult: weight of the local colour variation
    // Segments with less contrast or chroma than this (MTM units) take no part
    // in the hue statistics: a hue class is only meaningful for a segment that
    // is both coloured and conspicuous.
    float attribute_threshold = 10.0f;
    float min_segment = 0.0005f; // smallest / largest salient segment,
    float max_segment = 0.15f;   // as fractions of the image area
    float max_contrast = 32.0f;  // MTM contrast mapped to 1 before the sigmoid
    float sigmoid_beta = 3.0f;   // eq. 5.12
    Exclusivity exclusivity;     // off by default; configs/thesis.yaml: 1.1
  };

  /// One colour segment, as grown by the region-growing pass.
  struct Segment
  {
    int pixels = 0;
    double mean[3] = {0.0, 0.0, 0.0}; // L, S1, S2
    int perimeter = 0;                // summed shared-border length
    double contrast = 0.0;            // border-weighted mean contrast (eq. 5.11)
    int hue_class = 0;                // 0..11, 30° each, class 0 centred on 0°
    float saliency = 0.0f;
  };

  /// Segmentation + saliency of one working image; exposed for tests and
  /// debugging. `labels` holds indices into `segments` (CV_32S).
  struct Result
  {
    cv::Mat labels;
    std::vector<Segment> segments;
    cv::Mat saliency; // CV_32F, working size
  };

  MunsellColorFeature() : MunsellColorFeature(Config{}) {}
  explicit MunsellColorFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;
  std::string name() const override { return "color-munsell"; }
  bool applicable(const core::Frame& frame) const override { return frame.channels() == 3; }

  /// BGR (8-bit or float [0,1]) -> MTM (L, S1, S2), CV_32FC3.
  static cv::Mat bgr_to_mtm(const cv::Mat& bgr);

  /// Run segmentation and saliency on an MTM image.
  Result evaluate(const cv::Mat& mtm) const;

 private:
  Config config_;
};

} // namespace features
} // namespace attention
