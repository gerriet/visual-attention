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
 * Eccentricity — the dissertation's area-based grey-value feature (thesis
 * §5.2.3, after Bollmann; ported from the original `feature/eccentricity.C`).
 * Designed as the complement of symmetry: symmetry rates *boundaries*,
 * eccentricity rates the elongation of homogeneous *areas*, so it still
 * responds to objects whose symmetry is broken by occlusion.
 *
 *   1. Homogeneity: Sobel response max(|gx|, |gy|) of the (histogram-
 *      equalised) grey image. The growth threshold is taken from the Sobel
 *      histogram so that a fixed share of all pixels (thesis: 65 %) counts as
 *      area and the rest as boundary.
 *   2. Region growing from every unvisited below-threshold pixel (row-major
 *      seeds; the result does not depend on the seed order).
 *   3. A deliberately small threshold over-segments, so a few passes (thesis:
 *      at most 4) of *merging + dilation* follow: a boundary pixel between
 *      exactly two segments merges them if their mean grey values differ by
 *      at most max_mu (20) and their grey variances are of the same order
 *      (ratio < k, thesis k = 2); any other boundary pixel joins the segment
 *      that dominates its neighbourhood. This is what keeps a large surface
 *      with local structure (lettering) in one piece.
 *   4. Segments too small or too large for attention are dropped.
 *   5. Second-order central moments give each segment an orientation
 *      (eq. 5.5), which sorts it into one of 12 classes of 15° — or a 13th for
 *      segments without a dominant direction — and Jähne's eccentricity
 *        eps = ((m20 - m02)^2 + 4 m11^2) / (m20 + m02)^2          (eq. 5.6)
 *      which is 0 for a round and 1 for a line-shaped segment and is the
 *      saliency, assigned to every pixel of the segment — optionally divided
 *      by the exclusivity of its orientation class (§5.5.3).
 *
 * The output is an *absolute* saliency in [0, 1] and is deliberately not
 * min-max stretched: an image of round blobs must give a flat, near-zero map
 * (thesis Abb. 5.10), not amplified residue.
 *
 * Defaults follow the thesis text where it and the surviving source disagree
 * (growth share 0.65 vs 0.75, variance ratio 2 vs 1.12, no saliency offset vs
 * 0.2); all are configurable. One deliberate deviation: the variance ratio is
 * taken on (variance + 1), so two perfectly flat segments — common in
 * synthetic stimuli, and 0/0 in the original — count as alike.
 */
class EccentricityFeature : public FeatureExtractor
{
 public:
  /// Number of orientation classes (15° each); class kNoOrientation holds the
  /// segments without a dominant direction.
  static constexpr int kOrientationClasses = 12;
  static constexpr int kNoOrientation = 12;

  struct Config
  {
    float edge_threshold = 0.65f;    // share of pixels below the growth threshold (0-1)
    int merge_mean_difference = 20;  // max_mu: largest mean-grey difference for a merge (0-255)
    float variance_threshold = 2.0f; // k: largest grey-variance ratio for a merge
    int merge_iterations = 4;        // merging + dilation passes
    float min_area = 0.05f;          // smallest / largest salient segment,
    float max_area = 30.0f;          // in percent of the image area
    int connectivity = 4;            // 4 or 8, for growth and dilation
    bool equalize = true;            // histogram-equalise first, as the original does
    float saliency_offset = 0.0f;    // eccentricities below it give no saliency (source: 0.2)
    float min_oriented = 0.05f;      // below this eccentricity a segment has no orientation
    Exclusivity exclusivity;         // off by default; configs/thesis.yaml: 1.1
    int compute_at_scale = 0;        // pyramid level (0 = full resolution, -1 = auto:
                                     // quarter resolution for images larger than 640px)
  };

  /// One grey-value segment with the statistics merging and moments need.
  struct Segment
  {
    int pixels = 0;
    double sum_gray = 0.0, sum_gray_sq = 0.0;
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_yy = 0.0, sum_xy = 0.0;
    float eccentricity = 0.0f;
    float angle = 0.0f; // [0, pi), meaningful only when oriented
    int orientation_class = kNoOrientation;
    float saliency = 0.0f;

    double mean_gray() const { return pixels > 0 ? sum_gray / pixels : 0.0; }
    double variance() const
    {
      return pixels > 0 ? sum_gray_sq / pixels - (sum_gray / pixels) * (sum_gray / pixels) : 0.0;
    }
  };

  /// Segmentation + saliency of one grey image; exposed for tests and debugging.
  /// `labels` (CV_32S) holds indices into `segments`; 0 = boundary / dropped.
  struct Result
  {
    cv::Mat sobel;  // CV_32F, max(|gx|, |gy|)
    cv::Mat labels; // CV_32S
    std::vector<Segment> segments;
    cv::Mat saliency; // CV_32F
    int initial_segments = 0;
    int growth_threshold = 0;
  };

  EccentricityFeature() : EccentricityFeature(Config{}) {}
  explicit EccentricityFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;
  std::string name() const override { return "eccentricity"; }

  /// Run the feature on an 8-bit grey image.
  Result evaluate(const cv::Mat& gray_8u) const;

  /// Jähne's eccentricity (eq. 5.6) from second-order central moments.
  static float eccentricity(double mu20, double mu02, double mu11);

 private:
  Config config_;
};

} // namespace features
} // namespace attention
