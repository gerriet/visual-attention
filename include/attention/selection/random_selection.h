#pragma once

#include "attention/selection/selection_strategy.h"
#include <random>

namespace attention
{
namespace selection
{

/**
 * RandomSelection: fixations at uniformly random locations, ignoring the
 * saliency map entirely.
 *
 * This is not a model of anything. It is the **floor** every claim of the form
 * "attending here is better than attending anywhere" has to clear, and the
 * project kept needing it and not having it:
 *
 * - H2's control (`docs/HYPOTHESIS_CLOSURE_PLAN.md`): recognition gated on
 *   random ROIs at the same pixel budget. Without it, "51% of detections at
 *   5.8% of pixels" cannot be told apart from "a tenth of the frame happens to
 *   contain most of the people".
 * - H6's `fovea-random` floor, for the same reason one scale up.
 *
 * Peaks are spaced by `min_distance` like the other strategies, so the budget
 * (count and spread) matches the arm it is a control for; only *where* they go
 * is random. Saliency is still reported at each location, so downstream code
 * that reads `Peak::value` keeps working.
 *
 * Determinism: the generator is seeded from `seed` and advanced across frames
 * of a stream, so a run is reproducible but consecutive frames are not
 * identical. Note that the sequence is *not* portable between standard
 * libraries — `std::uniform_int_distribution` is implementation-defined, the
 * same trap the world model hit (see `src/examples/world_model.cpp`). For a
 * control arm this does not matter: what must be reproducible is the
 * distribution, not the exact pixels. Anything compared across platforms should
 * not use this strategy as a golden.
 */
class RandomSelection : public SelectionStrategy
{
 public:
  struct Params
  {
    unsigned int seed = 0;
    /// Keep a drawn location only if the map is at least this value there.
    /// 0 (the default) means "truly anywhere", which is the honest floor;
    /// a small value gives the "random *among plausible* places" variant.
    float min_value = 0.0f;
    /// Give up after this many draws per peak, so a nearly empty frame or a
    /// high min_value cannot spin forever.
    int max_attempts = 200;
  };

  RandomSelection(const SelectionParams& shared, const Params& params);

  std::string name() const override { return "random"; }

  std::vector<core::Peak> select(const cv::Mat& saliency, core::RunState& state) const override;

 private:
  SelectionParams shared_;
  Params params_;
  mutable std::mt19937 rng_;
};

} // namespace selection
} // namespace attention
