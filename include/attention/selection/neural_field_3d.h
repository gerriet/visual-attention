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
 * NeuralField3D: a stack of 2D dynamic neural fields coupled in depth, ported
 * from the dissertation's NeuralField3D (reference/old_code/nf3d.h, thesis
 * §6.4). Each depth plane runs the same Amari relaxation and "Backer version"
 * lateral kernel as the 2D field, with two additions that make depth compete:
 *
 *   - cross-depth inhibition: every plane is inhibited by the mean sigmoid
 *     activity across all planes at the same (x,y) (the depth_accu/zsize term),
 *     so at one location a single depth wins;
 *   - a per-plane global inhibition proportional to that plane's mean activity.
 *
 * Update per cycle, for plane z (sig = logistic sigmoid, K = lateral kernel):
 *
 *   u_z' = alpha * ( resting - global_mult * Σ_z mean(sig(u_z))     // global
 *                    + sig(u_z) * K                                  // lateral
 *                    - (Σ_z sig(u_z)) / zsize                        // depth
 *                    - global_mult * mean(sig(u_z)) * plane_inhib    // plane
 *                    + input_mult * S_z )                            // input
 *          + (1 - alpha) * u_z
 *
 * This is the concrete float port used for depth-aware selection; it is a
 * standalone component (unit-tested independently of the pipeline).
 */
class NeuralField3D
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
    float plane_inhibition = 5.0f; // per-plane inhibition factor (nf3d.h: *5)
    int max_cycles = 50;
    float change_thresh = 0.01f;   // mean |du| per neuron convergence
  };

  NeuralField3D(const cv::Size& plane_size, int depth, const Params& params);

  // Reset every plane to the resting level.
  void initialize();

  /**
   * Run the field to convergence on the given input volume (one CV_32F plane
   * per depth, all the field's plane size). Returns the number of cycles run.
   */
  int update(const std::vector<cv::Mat>& input);

  // Collapsed 2D activation: the maximum activity across depth per pixel.
  cv::Mat collapsed_activation() const;

  // Winning depth index per pixel (argmax over planes), CV_32S.
  cv::Mat winning_depth() const;

  const std::vector<cv::Mat>& activity() const { return activity_; }
  int depth() const { return depth_; }

 private:
  cv::Mat sigmoid(const cv::Mat& u) const;

  Params params_;
  cv::Size size_;
  int depth_;
  cv::Mat kernel_;
  std::vector<cv::Mat> activity_; // one CV_32F plane per depth
};

/**
 * NeuralField3DSelection: depth-aware sequential selection. It lifts the fused
 * 2D saliency into a depth volume using the per-pixel depth cue published by
 * the stereo feature (RunState::depth_map, values in [0,1]) — each pixel's
 * saliency is placed at plane round(depth*(depth_layers-1)), so a higher depth
 * cue (a nearer surface) indexes a higher plane — then runs the 3D field so that
 * spatially-overlapping structures at different depths compete. Readout mirrors
 * the 2D field: the collapsed activation is segmented into clusters and their
 * centroids are fixated in priority order, with a decaying space-based IOR map.
 *
 * With no depth cue (no stereo feature ran) all saliency lands in the middle
 * plane and the field reduces to the 2D case in one plane.
 *
 * YAML parameters (pipeline: selection_params:), beyond the 2D field's keys:
 *   depth_layers: 11        # number of depth planes (thesis: 11)
 *   plane_inhibition: 5.0   # per-plane inhibition factor
 */
class NeuralField3DSelection : public SelectionStrategy
{
 public:
  struct Params
  {
    NeuralField3D::Params field;
    int depth_layers = 11;
    int field_max_size = 96; // 3D is zsize× the work of 2D — a smaller field
    int border_margin = 9;
    int min_cluster_size = 2;
    float ior_decay = 0.8f;
  };

  NeuralField3DSelection(const SelectionParams& shared, const Params& params);

  std::string name() const override { return "neural-field-3d"; }

  std::vector<core::Peak> select(const cv::Mat& saliency, core::RunState& state) const override;

 private:
  SelectionParams shared_;
  Params params_;
};

} // namespace selection
} // namespace attention
