// Selection strategies: turn a saliency map into an ordered fixation
// sequence. NMS was moved verbatim from core::SaliencyMap (v1) and is locked
// by the characterization tests. IOR deliberately deviates from v1: the v1
// inhibition kernel was normalized to unit sum, which made the suppression
// negligible and re-selected the same peak repeatedly; behavior here is
// covered by test_selection.cpp instead.

#include "attention/config/yaml_reader.h"
#include "attention/core/constants.h"
#include "attention/selection/kalman_mot_selection.h"
#include "attention/selection/neural_field_3d.h"
#include "attention/selection/neural_field_selection.h"
#include "attention/selection/selection_strategy.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace attention
{
namespace selection
{

namespace
{

/**
 * Dilation-based non-maximum suppression with adaptive downsampling for
 * large maps. Peaks come out sorted by value, descending.
 */
class NmsSelection : public SelectionStrategy
{
 public:
  explicit NmsSelection(const SelectionParams& params) : params_(params) {}

  std::string name() const override { return "nms"; }

  std::vector<core::Peak> select(const cv::Mat& map, core::RunState& /*state*/) const override
  {
    std::vector<core::Peak> peaks;
    if (map.empty())
    {
      return peaks;
    }

    const int min_distance = params_.min_distance;
    const float threshold = params_.threshold;
    const int max_peaks = params_.max_count;

    // For large images, downsample for peak detection performance.
    // Detection works on coarse scale, peaks are mapped back to full resolution
    cv::Mat detection_map = map;
    float scale_factor = 1.0f;
    int scaled_min_distance = min_distance;

    // Adaptive downsampling: use half resolution if larger than 640px
    if (map.cols > 640 || map.rows > 640)
    {
      scale_factor = 0.5f;
      cv::resize(map, detection_map, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
      scaled_min_distance = static_cast<int>(min_distance * scale_factor + 0.5f);
    }

    // Apply threshold on detection map
    cv::Mat thresholded;
    cv::threshold(detection_map, thresholded, threshold, 255.0, cv::THRESH_BINARY);
    thresholded.convertTo(thresholded, CV_8U);

    // Find local maxima using dilation
    cv::Mat dilated;
    int kernel_size = scaled_min_distance / 2 + 1;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size * 2 + 1, kernel_size * 2 + 1));
    cv::dilate(detection_map, dilated, kernel);

    // Local maxima are where original equals dilated
    cv::Mat local_max;
    cv::compare(detection_map, dilated, local_max, cv::CMP_GE);

    // Combine with threshold
    cv::Mat peak_mask;
    cv::bitwise_and(local_max, thresholded, peak_mask);

    // Extract peak locations and values from detection map
    std::vector<core::Peak> candidate_peaks;
    for (int y = 0; y < peak_mask.rows; ++y)
    {
      for (int x = 0; x < peak_mask.cols; ++x)
      {
        if (peak_mask.at<uchar>(y, x) > 0)
        {
          // Scale coordinates back to original resolution
          int orig_x = static_cast<int>(x / scale_factor + 0.5f);
          int orig_y = static_cast<int>(y / scale_factor + 0.5f);

          orig_x = std::min(orig_x, map.cols - 1);
          orig_y = std::min(orig_y, map.rows - 1);

          float value = map.at<float>(orig_y, orig_x);
          candidate_peaks.push_back(core::Peak(cv::Point(orig_x, orig_y), value));
        }
      }
    }

    // Sort by value (descending)
    std::sort(candidate_peaks.begin(), candidate_peaks.end());

    // Apply non-maximum suppression to enforce minimum distance
    for (const auto& candidate : candidate_peaks)
    {
      bool too_close = false;
      for (const auto& existing_peak : peaks)
      {
        float dist = cv::norm(candidate.location - existing_peak.location);
        if (dist < min_distance)
        {
          too_close = true;
          break;
        }
      }

      if (!too_close)
      {
        peaks.push_back(candidate);
        if (max_peaks > 0 && static_cast<int>(peaks.size()) >= max_peaks)
        {
          break;
        }
      }
    }

    return peaks;
  }

 private:
  SelectionParams params_;
};

/**
 * Sequential winner-take-all with Gaussian inhibition of return: each
 * selected peak inhibits its surround before the next selection. Peaks come
 * out in selection order (the scanpath).
 *
 * Note: inhibition is currently per-frame; carrying it across stream frames
 * via RunState is future work (roadmap M5/M6).
 */
class IorSelection : public SelectionStrategy
{
 public:
  explicit IorSelection(const SelectionParams& params) : params_(params)
  {
    // Pre-compute the attenuation mask once: (1 - strength * gauss(dist)),
    // gaussian 1.0 at the center so ior_strength is the actual suppression
    // applied at the selected peak. (The v1 code normalized the kernel to
    // unit sum, which made the inhibition negligible — the same peak won
    // repeatedly.)
    const int radius = params_.ior_radius;
    const int kernel_size = radius * 2 + 1;
    const float sigma = radius / attention::constants::IOR_SIGMA_FACTOR;

    attenuation_ = cv::Mat(kernel_size, kernel_size, CV_32F);
    for (int y = 0; y < kernel_size; ++y)
    {
      for (int x = 0; x < kernel_size; ++x)
      {
        int dx = x - radius;
        int dy = y - radius;
        float dist_sq = dx * dx + dy * dy;
        float gauss = std::exp(-dist_sq / (2.0f * sigma * sigma));
        attenuation_.at<float>(y, x) = 1.0f - params_.ior_strength * gauss;
      }
    }
  }

  std::string name() const override { return "ior"; }

  std::vector<core::Peak> select(const cv::Mat& map, core::RunState& /*state*/) const override
  {
    std::vector<core::Peak> peaks;
    if (map.empty())
    {
      return peaks;
    }

    const float threshold = params_.threshold;
    const int max_peaks = params_.max_count;
    const int ior_radius = params_.ior_radius;

    // Working copy of the saliency map for sequential inhibition
    cv::Mat working_map = map.clone();
    const cv::Rect map_rect(0, 0, working_map.cols, working_map.rows);

    // Sequential peak detection with IOR
    for (int i = 0; i < max_peaks; ++i)
    {
      cv::Point max_loc;
      double max_val;
      cv::minMaxLoc(working_map, nullptr, &max_val, nullptr, &max_loc);

      if (max_val < threshold)
      {
        break; // No more significant peaks
      }

      peaks.push_back(core::Peak(max_loc, static_cast<float>(max_val)));

      // Inhibit region around the detected peak: multiply the map ROI by the
      // matching sub-rect of the precomputed attenuation mask
      cv::Rect kernel_rect(max_loc.x - ior_radius, max_loc.y - ior_radius, attenuation_.cols, attenuation_.rows);
      cv::Rect roi = kernel_rect & map_rect;
      if (roi.empty())
      {
        continue;
      }
      cv::Rect kernel_roi(roi.x - kernel_rect.x, roi.y - kernel_rect.y, roi.width, roi.height);
      cv::multiply(working_map(roi), attenuation_(kernel_roi), working_map(roi));
    }

    return peaks;
  }

 private:
  SelectionParams params_;
  cv::Mat attenuation_; // (2r+1)^2 mask: 1 - strength * gaussian
};

} // namespace

using config::read_param;

std::unique_ptr<SelectionStrategy> create_selection_strategy(const std::string& name, const SelectionParams& params,
                                                             const YAML::Node& strategy_params)
{
  if (name == "nms")
  {
    return std::make_unique<NmsSelection>(params);
  }
  if (name == "ior")
  {
    return std::make_unique<IorSelection>(params);
  }
  if (name == "neural-field")
  {
    NeuralFieldSelection::Params nf;
    read_param(strategy_params, "alpha", nf.alpha);
    read_param(strategy_params, "beta", nf.beta);
    read_param(strategy_params, "resting", nf.resting);
    read_param(strategy_params, "global_mult", nf.global_mult);
    read_param(strategy_params, "input_mult", nf.input_mult);
    read_param(strategy_params, "kernel_s", nf.kernel_s);
    read_param(strategy_params, "kernel_k", nf.kernel_k);
    read_param(strategy_params, "kernel_size", nf.kernel_size);
    read_param(strategy_params, "max_cycles", nf.max_cycles);
    read_param(strategy_params, "cycles_per_frame", nf.cycles_per_frame);
    read_param(strategy_params, "change_thresh", nf.change_thresh);
    read_param(strategy_params, "field_max_size", nf.field_max_size);
    read_param(strategy_params, "border_margin", nf.border_margin);
    read_param(strategy_params, "min_cluster_size", nf.min_cluster_size);
    read_param(strategy_params, "ior_decay", nf.ior_decay);
    return std::make_unique<NeuralFieldSelection>(params, nf);
  }
  if (name == "neural-field-3d")
  {
    NeuralField3DSelection::Params nf;
    read_param(strategy_params, "alpha", nf.field.alpha);
    read_param(strategy_params, "beta", nf.field.beta);
    read_param(strategy_params, "resting", nf.field.resting);
    read_param(strategy_params, "global_mult", nf.field.global_mult);
    read_param(strategy_params, "input_mult", nf.field.input_mult);
    read_param(strategy_params, "kernel_s", nf.field.kernel_s);
    read_param(strategy_params, "kernel_k", nf.field.kernel_k);
    read_param(strategy_params, "kernel_size", nf.field.kernel_size);
    read_param(strategy_params, "plane_inhibition", nf.field.plane_inhibition);
    read_param(strategy_params, "max_cycles", nf.field.max_cycles);
    read_param(strategy_params, "change_thresh", nf.field.change_thresh);
    read_param(strategy_params, "depth_layers", nf.depth_layers);
    read_param(strategy_params, "field_max_size", nf.field_max_size);
    read_param(strategy_params, "border_margin", nf.border_margin);
    read_param(strategy_params, "min_cluster_size", nf.min_cluster_size);
    read_param(strategy_params, "ior_decay", nf.ior_decay);
    return std::make_unique<NeuralField3DSelection>(params, nf);
  }
  if (name == "kalman-mot")
  {
    KalmanMotSelection::Params km;
    read_param(strategy_params, "min_blob_size", km.min_blob_size);
    read_param(strategy_params, "max_assoc_dist", km.max_assoc_dist);
    read_param(strategy_params, "depth_assoc_weight", km.depth_assoc_weight);
    read_param(strategy_params, "use_depth", km.use_depth);
    read_param(strategy_params, "max_age", km.max_age);
    read_param(strategy_params, "process_noise", km.process_noise);
    read_param(strategy_params, "measurement_noise", km.measurement_noise);
    read_param(strategy_params, "object_ior", km.object_ior);
    read_param(strategy_params, "ior_frames", km.ior_frames);
    return std::make_unique<KalmanMotSelection>(params, km);
  }
  throw std::runtime_error("Unknown selection strategy '" + name +
                           "'. Available: nms, ior, neural-field, neural-field-3d, kalman-mot");
}

} // namespace selection
} // namespace attention
