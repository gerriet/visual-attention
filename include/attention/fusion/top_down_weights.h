#pragma once

#include "attention/core/feature_map.h"
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace attention
{
namespace fusion
{

/**
 * Learned top-down feature weights (M20, H8) — the VOCUS rule.
 *
 * The priority map's existing top-down slot (`PriorityConfig`, M17) takes a
 * *dense* relevance map: a target colour we name, or a file some adapter wrote.
 * This is the other kind of top-down signal, and the one the thesis's model has
 * no dial for at all: a weight per feature *channel*, derived from examples
 * rather than chosen.
 *
 * The rule (Frintrop, Backer & Rome, KI 2005, §2.2 — VOCUS):
 *
 *     w_i = mean of feature map i inside the target region
 *         / mean of feature map i outside it
 *
 * A channel therefore counts to the degree that it **separates** the target
 * from its background, not to the degree that it merely fires on the target —
 * a feature that is strong on the target and equally strong everywhere else
 * gets w ≈ 1 and drops out. That is the whole idea, and it is why the weights
 * from several examples are combined by the **geometric** mean: they are
 * ratios, so a channel that is 2 in one image and 0.5 in another averages to 1
 * and correctly disappears.
 *
 * In search mode the weights become an excitation minus an inhibition:
 *
 *     E = Σ_{w_i > 1}  w_i · X_i
 *     I = Σ_{w_i < 1} (1/w_i) · X_i
 *     S_td = E − I
 *
 * and the map actually searched is `S = (1 − t)·S_bu + t·S_td` for a single
 * top-down factor t ∈ [0, 1]. t = 0 is the pure bottom-up map, bit-identical
 * to what the pipeline computed before — the default, so nothing changes unless
 * asked.
 *
 * Deliberately *not* a replacement for the M17 channels: a dense semantic map
 * and a channel weight vector answer different halves of "what does the task
 * want", and the study runs them against and with each other.
 */
struct TopDownWeightConfig
{
  /// Per-feature multipliers, keyed by feature name. A feature missing from
  /// the map is treated as w = 1 (neutral) and contributes to neither term.
  std::map<std::string, float> weights;

  /// Mixing factor t ∈ [0, 1]: 0 = pure bottom-up (default), 1 = pure top-down.
  float factor = 0.0f;

  bool active() const { return factor > 0.0f && !weights.empty(); }
};

/**
 * Applies a learned weight vector to the feature maps at fusion time.
 *
 * Needs the individual feature maps, not the fused map — which is why this
 * lives beside the fusion call in the pipeline rather than inside
 * `TopDownChannel`, whose input is already fused.
 */
class TopDownWeights
{
 public:
  explicit TopDownWeights(const TopDownWeightConfig& config);

  /**
   * Mix the bottom-up map with the top-down map built from `features`.
   * @param bottom_up fused saliency, CV_32F in [0, 1]
   * @param features the same feature maps that produced it
   * @return `(1 − t)·S_bu + t·S_td`, renormalized to [0, 1]; `bottom_up`
   *         itself (same cv::Mat) when the config is inactive or no weighted
   *         feature is present.
   * @pre every feature map is CV_32F and the size of `bottom_up`
   */
  cv::Mat apply(const cv::Mat& bottom_up, const std::vector<core::FeatureMap>& features) const;

  /**
   * The top-down map `S_td` alone, normalized to [0, 1] — for debugging, for
   * figures, and for the study's "what did it actually weight" column. Empty
   * when no weighted feature is present.
   */
  cv::Mat top_down_map(const cv::Mat& bottom_up, const std::vector<core::FeatureMap>& features) const;

 private:
  TopDownWeightConfig config_;
};

/**
 * Learn one weight per feature from a single example, by the rule above.
 *
 * @param features the feature maps computed on the training frame
 * @param target_mask CV_8U, non-zero inside the target region, the size of the
 *        feature maps
 * @return w_i per feature name. A feature whose outside-mean is ~0 would give
 *         an infinite ratio; it is clamped to `max_weight` instead, since an
 *         unbounded weight would let one channel swamp the sum.
 * @pre the mask has at least one non-zero and one zero pixel
 * @throws std::runtime_error if that precondition fails — a mask covering the
 *         whole frame, or none of it, carries no separation information and
 *         silently returning w = 1 would hide the mistake
 */
std::map<std::string, float> learn_top_down_weights(const std::vector<core::FeatureMap>& features,
                                                    const cv::Mat& target_mask, float max_weight = 10.0f);

/**
 * Combine weight vectors from several examples by the **geometric** mean,
 * because they are ratios (see the class comment). Features absent from a
 * vector are skipped for that example rather than counted as 1, so one
 * example that lacked a channel does not drag the channel toward neutral.
 * @return the combined vector; empty input gives an empty result.
 */
std::map<std::string, float> combine_top_down_weights(const std::vector<std::map<std::string, float>>& examples);

/**
 * The target region inside a box, decided the way VOCUS decides it: the most
 * salient connected region *within* the box, rather than the whole rectangle.
 * The point is that a bounding box contains background, and averaging over it
 * dilutes exactly the contrast the rule is trying to measure — so the method
 * picks out what in the box is the object before it measures anything.
 *
 * @param saliency the bottom-up map, CV_32F in [0, 1]
 * @param box the annotated target box, clipped to the map
 * @param rel_threshold fraction of the in-box maximum that still counts as
 *        target (0.5 = VOCUS's "at least half as salient as the peak")
 * @return CV_8U mask, non-zero on the chosen region. Never empty: if the box
 *         is flat, the whole (clipped) box is returned, which degrades to the
 *         naive behaviour rather than to nothing.
 */
cv::Mat salient_region_in_box(const cv::Mat& saliency, const cv::Rect& box, float rel_threshold = 0.5f);

} // namespace fusion
} // namespace attention
