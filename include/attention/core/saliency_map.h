#pragma once

#include "attention/core/constants.h"
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
      cv::normalize(map, map, 0.0f, 1.0f, cv::NORM_MINMAX);
    }
  }

  /**
   * Detect local maxima peaks in the saliency map using non-maximum suppression.
   * @param min_distance Minimum distance between peaks (in pixels)
   * @param threshold Minimum saliency value for a peak (0-1)
   * @param max_peaks Maximum number of peaks to return (0 = unlimited)
   * @param enable_ior Enable Inhibition of Return (sequential inhibition)
   * @param ior_radius Radius of IOR inhibition disk (only used if enable_ior = true)
   * @param ior_strength Strength of IOR inhibition 0-1 (only used if enable_ior = true)
   */
  void detect_peaks(int min_distance = 20, float threshold = 0.3f, int max_peaks = 10, bool enable_ior = false,
                    int ior_radius = 50, float ior_strength = 0.8f)
  {
    peaks.clear();

    if (map.empty())
    {
      return;
    }

    if (enable_ior)
    {
      // IOR-based sequential peak detection
      detect_peaks_with_ior(threshold, max_peaks, ior_radius, ior_strength);
    }
    else
    {
      // Traditional non-maximum suppression approach
      detect_peaks_nms(min_distance, threshold, max_peaks);
    }
  }

private:
  /**
   * Traditional peak detection using non-maximum suppression.
   */
  void detect_peaks_nms(int min_distance, float threshold, int max_peaks)
  {
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

  /**
   * IOR-based sequential peak detection.
   * Each detected peak inhibits surrounding region for subsequent peaks.
   */
  void detect_peaks_with_ior(float threshold, int max_peaks, int ior_radius, float ior_strength)
  {
    // Create working copy of saliency map for sequential inhibition
    cv::Mat working_map = map.clone();

    // Pre-compute Gaussian inhibition kernel
    int kernel_size = ior_radius * 2 + 1;
    cv::Mat ior_kernel = cv::Mat::zeros(kernel_size, kernel_size, CV_32F);

    // Create Gaussian inhibition disk
    float sigma = ior_radius / constants::IOR_SIGMA_FACTOR;  // Standard Gaussian spread
    float sum = 0.0f;
    for (int y = 0; y < kernel_size; ++y)
    {
      for (int x = 0; x < kernel_size; ++x)
      {
        int dx = x - ior_radius;
        int dy = y - ior_radius;
        float dist_sq = dx * dx + dy * dy;
        float gauss = std::exp(-dist_sq / (2.0f * sigma * sigma));
        ior_kernel.at<float>(y, x) = gauss;
        sum += gauss;
      }
    }

    // Normalize kernel
    ior_kernel /= sum;

    // Sequential peak detection with IOR
    for (int i = 0; i < max_peaks; ++i)
    {
      // Find global maximum in working map
      cv::Point max_loc;
      double max_val;
      cv::minMaxLoc(working_map, nullptr, &max_val, nullptr, &max_loc);

      // Check if peak exceeds threshold
      if (max_val < threshold)
      {
        break; // No more significant peaks
      }

      // Add peak to results
      peaks.push_back(Peak(max_loc, static_cast<float>(max_val)));

      // Apply IOR: inhibit region around detected peak
      // Create inhibition mask
      for (int dy = -ior_radius; dy <= ior_radius; ++dy)
      {
        for (int dx = -ior_radius; dx <= ior_radius; ++dx)
        {
          int x = max_loc.x + dx;
          int y = max_loc.y + dy;

          // Check bounds
          if (x >= 0 && x < working_map.cols && y >= 0 && y < working_map.rows)
          {
            int kx = dx + ior_radius;
            int ky = dy + ior_radius;
            float inhibition = ior_kernel.at<float>(ky, kx) * ior_strength;

            // Apply inhibition: multiply by (1 - inhibition)
            working_map.at<float>(y, x) *= (1.0f - inhibition);
          }
        }
      }
    }
  }

public:
};

} // namespace core
} // namespace attention
