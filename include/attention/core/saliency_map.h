#pragma once

#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>

namespace attention
{
namespace core
{

/**
 * Peak represents a salient location in the saliency map.
 */
struct Peak
{
  cv::Point location;
  float value;

  Peak() : location(0, 0), value(0.0f) {}
  Peak(const cv::Point& loc, float val) : location(loc), value(val) {}

  // For sorting peaks by value (descending)
  bool operator<(const Peak& other) const
  {
    return value > other.value; // Note: reversed for descending order
  }
};

/**
 * SaliencyMap represents the integrated attention map with detected peaks.
 */
struct SaliencyMap
{
  // Saliency data - normalized values [0, 1]
  // Type: CV_32F (float), single channel
  cv::Mat map;

  // Detected salient locations (peaks in the saliency map)
  std::vector<Peak> peaks;

  // Default constructor
  SaliencyMap() = default;

  // Constructor from cv::Mat
  explicit SaliencyMap(const cv::Mat& saliency_data) : map(saliency_data) {}

  // Move semantics
  SaliencyMap(SaliencyMap&& other) noexcept = default;
  SaliencyMap& operator=(SaliencyMap&& other) noexcept = default;

  // Copy semantics
  SaliencyMap(const SaliencyMap& other) = default;
  SaliencyMap& operator=(const SaliencyMap& other) = default;

  // Query methods
  bool empty() const { return map.empty(); }
  cv::Size size() const { return map.size(); }
  bool is_valid() const { return !map.empty() && map.type() == CV_32F && map.channels() == 1; }

  // Get the global maximum
  Peak get_global_max() const
  {
    if (map.empty())
    {
      return Peak();
    }

    cv::Point max_loc;
    double max_val;
    cv::minMaxLoc(map, nullptr, &max_val, nullptr, &max_loc);
    return Peak(max_loc, static_cast<float>(max_val));
  }

  // Get top N peaks (sorted by value, descending)
  std::vector<Peak> get_top_peaks(int n) const
  {
    if (peaks.empty())
    {
      return std::vector<Peak>();
    }

    std::vector<Peak> sorted_peaks = peaks;
    std::sort(sorted_peaks.begin(), sorted_peaks.end());

    int count = std::min(n, static_cast<int>(sorted_peaks.size()));
    return std::vector<Peak>(sorted_peaks.begin(), sorted_peaks.begin() + count);
  }

  // Normalize the saliency map to [0, 1]
  void normalize()
  {
    if (!map.empty())
    {
      cv::normalize(map, map, 0.0, 1.0, cv::NORM_MINMAX);
    }
  }

  /**
   * Detect local maxima peaks in the saliency map using non-maximum suppression.
   * @param min_distance Minimum distance between peaks (in pixels)
   * @param threshold Minimum saliency value for a peak (0-1)
   * @param max_peaks Maximum number of peaks to return (0 = unlimited)
   */
  void detect_peaks(int min_distance = 20, float threshold = 0.3f, int max_peaks = 10)
  {
    peaks.clear();

    if (map.empty())
    {
      return;
    }

    // For large images, downsample for peak detection performance
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
    // A pixel is a local maximum if it equals the dilated image at that location
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
    std::vector<Peak> candidate_peaks;
    for (int y = 0; y < peak_mask.rows; ++y)
    {
      for (int x = 0; x < peak_mask.cols; ++x)
      {
        if (peak_mask.at<uchar>(y, x) > 0)
        {
          // Scale coordinates back to original resolution
          int orig_x = static_cast<int>(x / scale_factor + 0.5f);
          int orig_y = static_cast<int>(y / scale_factor + 0.5f);

          // Clamp to map bounds
          orig_x = std::min(orig_x, map.cols - 1);
          orig_y = std::min(orig_y, map.rows - 1);

          // Get value from original resolution map
          float value = map.at<float>(orig_y, orig_x);
          candidate_peaks.push_back(Peak(cv::Point(orig_x, orig_y), value));
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
  }
};

} // namespace core
} // namespace attention
