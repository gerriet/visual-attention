#pragma once

#include "attention/core/run_state.h"
#include "attention/core/saliency_map.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace YAML
{
class Node;
}

namespace attention
{
namespace selection
{

/**
 * Parameters for the built-in selection strategies. A strategy uses the
 * subset that applies to it.
 */
struct SelectionParams
{
  int min_distance = 30;     // NMS: minimum distance between peaks (pixels)
  float threshold = 0.3f;    // Minimum saliency value for a peak
  int max_count = 10;        // Maximum number of peaks
  int ior_radius = 50;       // IOR: radius of inhibition disk (pixels)
  float ior_strength = 0.8f; // IOR: inhibition strength (0-1)
};

/**
 * SelectionStrategy turns a saliency map into an ordered fixation sequence.
 *
 * Implementations are selected by name via config ("pipeline: selection:").
 * RunState is passed so future strategies can carry selection state across
 * frames of a stream (cross-frame IOR, neural-field activation in M3).
 */
class SelectionStrategy
{
 public:
  virtual ~SelectionStrategy() = default;

  virtual std::string name() const = 0;

  /**
   * Select peaks from a saliency map, in scanpath order.
   * @param saliency Saliency map (CV_32F, values [0, 1])
   * @param state Per-run state (read/write)
   */
  virtual std::vector<core::Peak> select(const cv::Mat& saliency, core::RunState& state) const = 0;
};

/**
 * Create a selection strategy by name.
 * @param name "nms" (dilation-based non-maximum suppression),
 *             "ior" (sequential winner-take-all with Gaussian inhibition), or
 *             "neural-field" (2D Amari field dynamics, ported from the
 *             dissertation's nf2d)
 * @param params Shared selection parameters
 * @param strategy_params Strategy-specific YAML params (may be a null node);
 *                        see each strategy's header for its keys
 * @throws std::runtime_error for unknown names
 */
std::unique_ptr<SelectionStrategy> create_selection_strategy(const std::string& name, const SelectionParams& params,
                                                             const YAML::Node& strategy_params);

} // namespace selection
} // namespace attention
