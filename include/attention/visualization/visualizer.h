#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/saliency_map.h"
#include <opencv2/opencv.hpp>
#include <string>

namespace attention
{
namespace visualization
{

/**
 * Simple visualization utilities for debugging and development.
 * Will be enhanced in Week 3 with heatmap overlays and peak markers.
 */

/**
 * Visualize a feature map as a grayscale image.
 * @param feature The feature map to visualize
 * @param window_name Optional window name (defaults to feature name)
 * @param wait_key If true, wait for key press after displaying
 * @return Visualization as 8-bit grayscale image
 */
inline cv::Mat visualize_feature_map(const core::FeatureMap& feature, const std::string& window_name = "",
                                     bool wait_key = false)
{
  if (feature.empty() || !feature.is_valid())
  {
    std::cerr << "Warning: Cannot visualize empty or invalid feature map" << std::endl;
    return cv::Mat();
  }

  // Convert normalized float [0,1] to 8-bit [0,255] for display
  cv::Mat vis;
  feature.data.convertTo(vis, CV_8U, 255.0);

  // Display if window name provided
  std::string win_name = window_name.empty() ? feature.name : window_name;
  if (!win_name.empty())
  {
    cv::imshow(win_name, vis);
    if (wait_key)
    {
      cv::waitKey(0);
    }
  }

  return vis;
}

/**
 * Visualize a saliency map with optional peak markers.
 * @param saliency The saliency map to visualize
 * @param original Optional original image for overlay
 * @param window_name Window name for display
 * @param mark_peaks If true, mark detected peaks with circles
 * @param wait_key If true, wait for key press after displaying
 * @return Visualization as BGR image (heatmap or overlay)
 */
inline cv::Mat visualize_saliency_map(const core::SaliencyMap& saliency, const cv::Mat& original = cv::Mat(),
                                      const std::string& window_name = "Saliency Map", bool mark_peaks = false,
                                      bool wait_key = false)
{
  if (saliency.empty() || !saliency.is_valid())
  {
    std::cerr << "Warning: Cannot visualize empty or invalid saliency map" << std::endl;
    return cv::Mat();
  }

  // Convert normalized float [0,1] to 8-bit [0,255]
  cv::Mat sal_8u;
  saliency.map.convertTo(sal_8u, CV_8U, 255.0);

  // Apply color map for better visualization
  cv::Mat heatmap;
  cv::applyColorMap(sal_8u, heatmap, cv::COLORMAP_JET);

  cv::Mat vis;
  if (!original.empty())
  {
    // Overlay heatmap on original image with alpha blending
    cv::Mat original_resized;
    if (original.size() != saliency.size())
    {
      cv::resize(original, original_resized, saliency.size());
    }
    else
    {
      original_resized = original;
    }

    // Convert to BGR if needed
    if (original_resized.channels() == 1)
    {
      cv::cvtColor(original_resized, original_resized, cv::COLOR_GRAY2BGR);
    }

    // Alpha blend: 60% heatmap, 40% original
    cv::addWeighted(heatmap, 0.6, original_resized, 0.4, 0, vis);
  }
  else
  {
    vis = heatmap;
  }

  // Mark peaks if requested
  if (mark_peaks && !saliency.peaks.empty())
  {
    for (const auto& peak : saliency.peaks)
    {
      // Draw circle at peak location
      cv::circle(vis, peak.location, 8, cv::Scalar(255, 255, 255), 2);
      cv::circle(vis, peak.location, 7, cv::Scalar(0, 0, 0), 2);
    }
  }

  // Display
  if (!window_name.empty())
  {
    cv::imshow(window_name, vis);
    if (wait_key)
    {
      cv::waitKey(0);
    }
  }

  return vis;
}

/**
 * Create a side-by-side comparison of multiple visualizations.
 * @param images Vector of images to display side-by-side
 * @param labels Optional labels for each image
 * @param window_name Window name for display
 * @param wait_key If true, wait for key press after displaying
 * @return Combined visualization
 */
inline cv::Mat visualize_side_by_side(const std::vector<cv::Mat>& images, const std::vector<std::string>& labels = {},
                                      const std::string& window_name = "Comparison", bool wait_key = false)
{
  if (images.empty())
  {
    return cv::Mat();
  }

  // Ensure all images are the same height and type
  int max_height = 0;
  int total_width = 0;
  int channels = images[0].channels();

  for (const auto& img : images)
  {
    max_height = std::max(max_height, img.rows);
    total_width += img.cols;
  }

  // Create canvas
  cv::Mat canvas(max_height, total_width, CV_8UC3, cv::Scalar(0, 0, 0));

  // Copy images side by side
  int x_offset = 0;
  for (size_t i = 0; i < images.size(); ++i)
  {
    cv::Mat img_display = images[i];

    // Convert to BGR if needed
    if (img_display.channels() == 1)
    {
      cv::cvtColor(img_display, img_display, cv::COLOR_GRAY2BGR);
    }

    // Ensure 8-bit
    if (img_display.depth() != CV_8U)
    {
      img_display.convertTo(img_display, CV_8U, 255.0);
    }

    // Copy to canvas
    cv::Rect roi(x_offset, 0, img_display.cols, img_display.rows);
    img_display.copyTo(canvas(roi));

    // Add label if provided
    if (i < labels.size() && !labels[i].empty())
    {
      cv::putText(canvas, labels[i], cv::Point(x_offset + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(255, 255, 255), 2);
    }

    x_offset += img_display.cols;
  }

  // Display
  if (!window_name.empty())
  {
    cv::imshow(window_name, canvas);
    if (wait_key)
    {
      cv::waitKey(0);
    }
  }

  return canvas;
}

/**
 * Save visualization to file.
 * @param image Image to save
 * @param filepath Output file path
 * @return true if saved successfully
 */
inline bool save_visualization(const cv::Mat& image, const std::string& filepath)
{
  if (image.empty())
  {
    std::cerr << "Warning: Cannot save empty image" << std::endl;
    return false;
  }

  bool success = cv::imwrite(filepath, image);
  if (success)
  {
    std::cout << "Saved visualization: " << filepath << std::endl;
  }
  else
  {
    std::cerr << "Error: Failed to save " << filepath << std::endl;
  }
  return success;
}

} // namespace visualization
} // namespace attention
