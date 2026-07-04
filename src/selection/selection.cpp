// Selection strategies: turn a saliency map into an ordered fixation
// sequence. The algorithms were moved verbatim from core::SaliencyMap
// (v1) so behavior is locked by the characterization tests.

#include "attention/selection/selection_strategy.h"
#include "attention/core/constants.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

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
  explicit IorSelection(const SelectionParams& params) : params_(params) {}

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
    const float ior_strength = params_.ior_strength;

    // Working copy of the saliency map for sequential inhibition
    cv::Mat working_map = map.clone();

    // Pre-compute Gaussian inhibition kernel: 1.0 at the peak center falling
    // off with distance, so ior_strength is the actual suppression applied at
    // the selected peak. (The v1 code normalized this kernel to unit sum,
    // which made the inhibition negligible — the same peak won repeatedly.)
    int kernel_size = ior_radius * 2 + 1;
    cv::Mat ior_kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32F);

    float sigma = ior_radius / attention::constants::IOR_SIGMA_FACTOR;
    for (int y = 0; y < kernel_size; ++y)
    {
      for (int x = 0; x < kernel_size; ++x)
      {
        int dx = x - ior_radius;
        int dy = y - ior_radius;
        float dist_sq = dx * dx + dy * dy;
        ior_kernel.at<float>(y, x) = std::exp(-dist_sq / (2.0f * sigma * sigma));
      }
    }

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

      // Inhibit region around the detected peak
      for (int dy = -ior_radius; dy <= ior_radius; ++dy)
      {
        for (int dx = -ior_radius; dx <= ior_radius; ++dx)
        {
          int x = max_loc.x + dx;
          int y = max_loc.y + dy;

          if (x >= 0 && x < working_map.cols && y >= 0 && y < working_map.rows)
          {
            int kx = dx + ior_radius;
            int ky = dy + ior_radius;
            float inhibition = ior_kernel.at<float>(ky, kx) * ior_strength;
            working_map.at<float>(y, x) *= (1.0f - inhibition);
          }
        }
      }
    }

    return peaks;
  }

 private:
  SelectionParams params_;
};

} // namespace

std::unique_ptr<SelectionStrategy> create_selection_strategy(const std::string& name, const SelectionParams& params)
{
  if (name == "nms")
  {
    return std::make_unique<NmsSelection>(params);
  }
  if (name == "ior")
  {
    return std::make_unique<IorSelection>(params);
  }
  throw std::runtime_error("Unknown selection strategy '" + name + "'. Available: nms, ior");
}

} // namespace selection
} // namespace attention
