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
 * dynamics, cross-frame inhibition of return, and the AttentionSystem's object
 * tracking (M6) all need state that outlives a single frame. This struct is
 * their home; it grows as those milestones land.
 */
struct RunState
{
  int frame_index = 0;

  // Neural-field activity at field resolution (empty until the neural-field
  // selection strategy first runs). Persists across stream frames so the
  // field dynamics carry over instead of restarting per frame.
  cv::Mat field_activity;

  // The 3D neural field's activity volume, one CV_32F plane per depth layer,
  // at field resolution (empty until the 3D selection strategy first runs).
  // Persists across stream frames for the same reason field_activity does: the
  // thesis's §6.4 extends the field into depth *in order to* select and track
  // through temporary occlusion, and its Abb. 6.14 measures how long tracking
  // survives — neither means anything if the volume restarts every frame. The
  // original kept one NeuralField3D for the life of the run
  // (docs/replication/REPLICATION_DOSSIER.md, finding C).
  std::vector<cv::Mat> field_volume;

  // Space-based inhibition-of-return map at field resolution: selected
  // regions are inhibited in subsequent frames, decaying over time
  // (thesis §8.3: static inhibition map, ~20% decay per frame).
  cv::Mat inhibition_map;

  // Previous frame's grayscale (CV_8U), carried across the frames of a
  // temporal stream so the onset/motion feature can compute a temporal
  // difference. Empty on the first frame and for independent stills.
  cv::Mat previous_gray;

  // Per-pixel depth cue at full resolution (CV_32F, [0,1]) published by a
  // depth-producing feature (StereoFeature) during extraction; the 3D
  // neural-field selection (M5) lifts the 2D saliency into a depth volume
  // with it. Empty when no depth feature ran.
  cv::Mat depth_map;

  void reset() { *this = RunState(); }
};

} // namespace core
} // namespace attention
