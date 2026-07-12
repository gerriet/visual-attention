#pragma once

#include "attention/selection/selection_strategy.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace attention
{
namespace selection
{

/**
 * KalmanMotSelection — multi-blob selection by detection + tracking, an
 * alternative to the neural field (see docs/SELECTION_BACKENDS.md, option B).
 *
 * The Amari field is powerful but hard to parametrize because it fuses five
 * jobs — detect, compete, persist, inhibit, repel — into one dynamical system
 * with ~13 coupled, input-scale-dependent knobs. This strategy decouples them:
 *
 *   1. Detect — threshold the fused saliency, connected-component blobs.
 *   2. Track  — one constant-velocity Kalman filter per blob, greedy gated
 *               data association; unmatched tracks *coast* (predicted forward)
 *               for up to max_age frames, which absorbs brief occlusions.
 *   3. Inhibit— object-based inhibition of return: a selected track is
 *               suppressed for ior_frames frames. Because the tag rides the
 *               *object*, the inhibition follows it as it moves — the property
 *               space-based IOR lacks in dynamic scenes (thesis stage-2 claim).
 *
 * Every knob is semantic and monotone (blob size, association gate, coast
 * frames, refractory window), so it is tunable by inspection, not by search.
 * Depth-aware: when a depth cue is present (RunState::depth_map), data
 * association also gates on depth, so objects that overlap in the image but sit
 * at different disparities stay separate tracks (the 3D field's job).
 *
 * Per-stream state (the track set) lives in mutable members and resets when
 * RunState::frame_index is 0 (stream start; batch mode resets stills too).
 */
class KalmanMotSelection : public SelectionStrategy
{
 public:
  struct Params
  {
    // Detection
    int min_blob_size = 9; // minimum connected-component area (px) to be a blob
    // Data association
    double max_assoc_dist = 40.0;     // gating radius, detection -> track (px)
    double depth_assoc_weight = 60.0; // px-equivalent weight of a unit depth diff
    bool use_depth = true;            // gate on RunState::depth_map when present
    // Track lifecycle
    int max_age = 15; // frames a track may coast unmatched before it dies
    // Kalman noise
    double process_noise = 1.0;     // Q scale (constant-velocity model)
    double measurement_noise = 4.0; // R scale (blob-centroid measurement)
    // Object-based inhibition of return
    bool object_ior = true; // suppress a just-selected track for ior_frames
    int ior_frames = 6;     // refractory window (frames)
  };

  KalmanMotSelection() : KalmanMotSelection(SelectionParams{}, Params{}) {}
  KalmanMotSelection(const SelectionParams& shared, const Params& params);

  std::string name() const override { return "kalman-mot"; }

  std::vector<core::Peak> select(const cv::Mat& saliency, core::RunState& state) const override;

 private:
  // One tracked blob: constant-velocity Kalman state x = [px, py, vx, vy]^T.
  struct Track
  {
    cv::Mat x;                    // 4x1 state (CV_64F)
    cv::Mat P;                    // 4x4 covariance (CV_64F)
    int id = 0;                   // stable identity
    int age = 0;                  // frames since creation
    int time_since_update = 0;    // frames coasted without a measurement
    float saliency = 0.0f;        // last measured blob peak saliency
    float depth = -1.0f;          // EMA of the depth cue (<0 = unknown)
    int last_selected = -1000000; // frame index when last selected (object IOR)
    bool measured_this_frame = false;
  };

  void predict(Track& t) const;                       // constant-velocity KF predict
  void correct(Track& t, const cv::Point2f& z) const; // KF update with a centroid

  SelectionParams shared_;
  Params params_;
  mutable std::vector<Track> tracks_;
  mutable int next_id_ = 1;
};

} // namespace selection
} // namespace attention
