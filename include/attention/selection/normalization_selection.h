#pragma once

#include "attention/selection/selection_strategy.h"
#include <opencv2/opencv.hpp>

namespace attention
{
namespace selection
{

/**
 * NormalizationSelection — divisive-normalization competition, an alternative to
 * the neural field (see docs/SELECTION_BACKENDS.md, option D).
 *
 * The neural field's single worst tuning problem is that its ignition depends on
 * the *absolute* scale of the saliency input, so a tuning that works on one
 * image class fails on another. Divisive normalization (the canonical cortical
 * computation; Reynolds & Heeger 2009, Carandini & Heeger 2012) removes that
 * dependence by construction:
 *
 *     r_i = a_i^n / ( sigma * mean(a^n) + pool(a^n)_i )
 *
 * where pool is a Gaussian-weighted surround. Because numerator and pool scale
 * together, the response is contrast-invariant — scaling the whole input leaves
 * the competition (which locations win) unchanged. It does soft winner-take-all
 * and count-limiting in a single pass, with no convergence loop and three
 * semantic, monotone knobs: the exponent n (contrast gain), the pool width, and
 * sigma (semi-saturation, expressed relative to the mean activity so it too is
 * scale-free).
 *
 * Detection stays absolute — a location must clear the shared threshold on the
 * raw fused saliency to compete (so a near-empty frame is rejected, not
 * amplified); divisive normalization then governs which of the detectable
 * locations win. Peaks are read out by sequential winner-take-all with Gaussian
 * inhibition. On a temporal stream a decaying, space-based inhibition-of-return
 * map is carried in RunState (thesis §8.3), so the strategy also serves as a
 * contrast-normalized "space-IOR" baseline for the dynamic-IOR study.
 */
class NormalizationSelection : public SelectionStrategy
{
 public:
  struct Params
  {
    float exponent = 2.0f;      // n: contrast gain of the divisive normalization
    int pool_size = 25;         // surround normalization-pool width (Gaussian sigma, px)
    float sigma = 0.5f;         // semi-saturation, relative to mean activity (scale-free)
    float smooth_factor = 0.2f; // response smoothing = pool_size * this (0 = none)
    float ior_decay = 0.8f;     // cross-frame IOR decay per frame (0 = per-frame only)
  };

  NormalizationSelection() : NormalizationSelection(SelectionParams{}, Params{}) {}
  NormalizationSelection(const SelectionParams& shared, const Params& params);

  std::string name() const override { return "normalization"; }

  std::vector<core::Peak> select(const cv::Mat& saliency, core::RunState& state) const override;

 private:
  SelectionParams shared_;
  Params params_;
};

} // namespace selection
} // namespace attention
