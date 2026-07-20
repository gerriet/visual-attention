#pragma once

#include "attention/system/object_file.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace attention
{
namespace fusion
{

/**
 * Priority-map configuration (M17): generalizes the fused "master saliency
 * map" into a priority map — bottom-up salience + top-down task relevance +
 * selection history / value (Wolfe's Guided Search 6.0; Awh, Belopolsky &
 * Theeuwes). Every added term is an opt-in weighted channel; with all weights
 * at zero the map is exactly the thesis's bottom-up map (bit-identical, the
 * default).
 *
 * Feature-level top-down (Guided Search's weight modulation) needs no channel:
 * the per-feature fusion weights are already config-driven — a target spec is
 * expressed directly as boosted weights. The channels here are the *dense*
 * terms the weight vector cannot express.
 */
struct PriorityConfig
{
  // --- Top-down task relevance (pipeline level; works on stills) ---
  float top_down_weight = 0.0f; // weight of the dense top-down channel
  // External relevance map (any grayscale image, resized to the frame): the
  // interchange-boundary slot — a CLIP/LLM/detector adapter writes it
  // Python-side; the core neither knows nor cares what produced it.
  std::string top_down_map_path;
  // Native target-colour channel: relevance = colour similarity to a target
  // ("#rrggbb" or a named colour) in Lab space. The zero-dependency dense
  // channel that makes find-my-object work out of the box.
  std::string target_color;
  float target_color_sigma = 25.0f; // Lab-distance falloff (higher = laxer)

  // --- Selection history / value (stage 2; streams) ---
  // Facilitation, explicitly distinct from IOR (which only suppresses):
  // previously selected / rewarded things get a priority *boost*.
  float object_value_weight = 0.0f;        // value rides on object files (moves with them)
  float location_history_weight = 0.0f;    // classic decaying map of attended locations
  float location_history_decay = 0.95f;    // per-frame decay of the location map
  float location_history_radius = 40.0f;   // Gaussian splat radius (px)
  float object_value_per_selection = 0.2f; // value accrued by being selected
  float object_value_decay = 0.98f;        // per-frame leak of object value

  bool top_down_active() const
  {
    return top_down_weight > 0.0f && (!top_down_map_path.empty() || !target_color.empty());
  }
  bool history_active() const { return object_value_weight > 0.0f || location_history_weight > 0.0f; }
};

/**
 * The pipeline-level top-down combination: bottom_up + w_td * top_down,
 * renormalized to [0, 1]. Returns bottom_up untouched (same cv::Mat) when no
 * top-down channel is active — the default path stays bit-identical.
 */
class TopDownChannel
{
 public:
  explicit TopDownChannel(const PriorityConfig& config);

  /**
   * Combine the fused bottom-up map with the configured top-down relevance.
   * @param bottom_up fused saliency, CV_32F in [0, 1]
   * @param frame_bgr the frame (8U BGR), for the target-colour channel
   */
  cv::Mat apply(const cv::Mat& bottom_up, const cv::Mat& frame_bgr) const;

  // The dense relevance map alone (for debugging / visualization); empty if
  // no channel is configured.
  cv::Mat relevance(const cv::Size& size, const cv::Mat& frame_bgr) const;

 private:
  cv::Mat colour_similarity(const cv::Mat& frame_bgr) const;

  PriorityConfig config_;
  cv::Mat file_map_; // loaded once from top_down_map_path (grayscale float)
  cv::Vec3f target_lab_ = cv::Vec3f(0, 0, 0);
  bool have_target_colour_ = false;
};

/**
 * The stage-2 history channels: a decaying location-history map plus an
 * object-value map projected from the object files' accrued value at their
 * *current* positions (value moves with the object — the account a static
 * location map cannot give). Owned by the AttentionSystem, updated per frame.
 */
class HistoryChannels
{
 public:
  explicit HistoryChannels(const PriorityConfig& config);

  void reset();

  // Per-frame update: decay the location map, then splat the selected focus.
  void decay_and_record(const cv::Point& focus, const cv::Size& frame_size, bool has_focus);

  /**
   * Combine the (already top-down-adjusted) saliency with the history terms:
   * priority = saliency + w_obj * object_value + w_loc * location_history,
   * renormalized to [0, 1]. Returns saliency untouched when inactive.
   */
  cv::Mat apply(const cv::Mat& saliency, const std::vector<system::ObjectFile>& objects) const;

  const cv::Mat& location_map() const { return location_; }

 private:
  PriorityConfig config_;
  cv::Mat location_; // CV_32F, frame-sized, decayed each frame
};

/**
 * Parse a colour spec: "#rrggbb" or a named colour (red, green, blue, yellow,
 * orange, purple, cyan, magenta, white, black, gray). Returns BGR.
 * @throws std::runtime_error for unparseable specs.
 */
cv::Vec3b parse_colour(const std::string& spec);

} // namespace fusion
} // namespace attention
