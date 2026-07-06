#pragma once

#include "attention/selection/selection_strategy.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace attention
{
namespace selection
{

/**
 * NeuralFieldSelection: sequential attention selection via 2D dynamic neural
 * field (Amari-style), ported from the dissertation's NeuralField2D (nf2d.h).
 *
 * Field dynamics per cycle (the original relaxation update):
 *
 *   u_new = alpha * ( resting - global_mult * mean(sig(u))   // global inhib.
 *                     + (sig(u) * K)                          // lateral
 *                     + border_suppression
 *                     + input_mult * S )                      // saliency in
 *           + (1 - alpha) * u_old
 *
 * with sig(u) = 1 / (1 + exp(-beta * u)) and the "Backer version" lateral
 * kernel K(r) = k * exp(-r^2/s^2) - (k/3) * exp(-r^2/(10 s^2)). Cycles run
 * until mean |du| < change_thresh (min 3 cycles).
 *
 * Two-stage readout, following the thesis (ch. 6/7): the locally inhibiting
 * DoG field settles into one activation cluster per attention-worthy region
 * (stage 1); each cluster becomes an "Objectfile-lite" (centroid, size, mean
 * saliency) and fixations are the cluster centroids in priority order by
 * mean saliency (stage 2, minimal form — the full symbolic stage with
 * behavior model and object tracking is roadmap M6). The fixation count is
 * decided by the field: only active clusters are reported.
 *
 * Streams: field activity persists across frames (RunState::field_activity)
 * and selected regions feed a decaying space-based IOR map
 * (RunState::inhibition_map, thesis §8.3) subtracted from the next frame's
 * input.
 *
 * The field runs at reduced resolution (max side field_max_size).
 *
 * YAML parameters (pipeline: selection_params:), defaults from the original
 * code where it specifies them:
 *   alpha: 0.5            # relaxation rate
 *   beta: 30.0            # sigmoid steepness
 *   resting: -0.25        # resting level
 *   global_mult: 1.0      # global inhibition strength
 *   input_mult: 1.0       # input gain
 *   kernel_s: 5.0         # lateral kernel spread ("Backer version")
 *   kernel_k: 0.06        # lateral kernel amplitude
 *   kernel_size: 41       # lateral kernel support (odd)
 *   max_cycles: 50        # thesis experiments: cutoff 55, ~10 typical
 *   change_thresh: 0.01   # mean |du| convergence (old code 0.01, thesis 0.02)
 *   field_max_size: 128   # field resolution (max side, downscale only)
 *   border_margin: 9      # border suppression margin (px, original hack)
 *   min_cluster_size: 2   # ignore smaller activation clusters (field px)
 *   ior_decay: 0.8        # per-frame retention of the IOR map (thesis: -20%)
 */
class NeuralFieldSelection : public SelectionStrategy
{
 public:
  struct Params
  {
    float alpha = 0.5f;
    float beta = 30.0f;
    float resting = -0.25f;
    float global_mult = 1.0f;
    float input_mult = 1.0f;
    float kernel_s = 5.0f;
    float kernel_k = 0.06f;
    int kernel_size = 41;
    int max_cycles = 50;
    int cycles_per_frame = 0; // >0: run exactly this many cycles per frame (no
                              // convergence check), carrying field state across
                              // frames — for live streams, so attention shifts
                              // emerge from continuous dynamics (roadmap M8)
    float change_thresh = 0.01f;
    int field_max_size = 128;
    int border_margin = 9;
    int min_cluster_size = 2;
    float ior_decay = 0.8f;
  };

  NeuralFieldSelection(const SelectionParams& shared, const Params& params);

  std::string name() const override { return "neural-field"; }

  std::vector<core::Peak> select(const cv::Mat& saliency, core::RunState& state) const override;

 private:
  /**
   * Run field cycles until convergence (mean |du| < change_thresh, min 3
   * cycles, max max_cycles). Mutates activity in place.
   */
  void run_to_convergence(cv::Mat& activity, const cv::Mat& input, const cv::Mat& border) const;

  cv::Mat sigmoid(const cv::Mat& activity) const;
  cv::Mat make_border_suppression(const cv::Size& size) const;

  SelectionParams shared_;
  Params params_;
  cv::Mat kernel_; // lateral interaction kernel (Backer version)
};

} // namespace selection
} // namespace attention
