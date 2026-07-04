#pragma once

#include <opencv2/opencv.hpp>

namespace attention
{
namespace core
{

/**
 * RunState carries per-run state across the frames of a stream.
 *
 * The v2 pipeline is stream-oriented: motion/onset features, neural-field
 * dynamics, cross-frame inhibition of return, and ESAB2 object tracking (M6)
 * all need state that outlives a single frame. This struct is their home; it
 * grows as those milestones land.
 */
struct RunState
{
  int frame_index = 0;

  // Neural-field activity at field resolution (empty until the neural-field
  // selection strategy first runs). Persists across stream frames so the
  // field dynamics carry over instead of restarting per frame.
  cv::Mat field_activity;

  // Space-based inhibition-of-return map at field resolution: selected
  // regions are inhibited in subsequent frames, decaying over time
  // (thesis §8.3: static inhibition map, ~20% decay per frame).
  cv::Mat inhibition_map;

  void reset() { *this = RunState(); }
};

} // namespace core
} // namespace attention
