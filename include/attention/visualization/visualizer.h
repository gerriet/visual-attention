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
cv::Mat visualize_feature_map(const core::FeatureMap& feature, const std::string& window_name = "",
                              bool wait_key = false);

/**
 * Visualize a saliency map with optional peak markers.
 * @param saliency The saliency map to visualize
 * @param original Optional original image for overlay
 * @param window_name Window name for display
 * @param mark_peaks If true, mark detected peaks with circles
 * @param wait_key If true, wait for key press after displaying
 * @return Visualization as BGR image (heatmap or overlay)
 */
cv::Mat visualize_saliency_map(const core::SaliencyMap& saliency, const cv::Mat& original = cv::Mat(),
                               const std::string& window_name = "", bool mark_peaks = false,
                               bool wait_key = false);

/**
 * Create a side-by-side comparison of multiple visualizations.
 * @param images Vector of images to display side-by-side
 * @param labels Optional labels for each image
 * @param window_name Window name for display
 * @param wait_key If true, wait for key press after displaying
 * @return Combined visualization
 */
cv::Mat visualize_side_by_side(const std::vector<cv::Mat>& images, const std::vector<std::string>& labels = {},
                               const std::string& window_name = "", bool wait_key = false);

/**
 * Save visualization to file.
 * @param image Image to save
 * @param filepath Output file path
 * @return true if saved successfully
 */
bool save_visualization(const cv::Mat& image, const std::string& filepath);

/**
 * Visualize scan path showing sequential attention shifts.
 * Draws numbered peaks with arrows connecting them in temporal order.
 * @param saliency The saliency map with detected peaks
 * @param original The original image for overlay
 * @param window_name Window name for display
 * @param wait_key If true, wait for key press after displaying
 * @return Visualization with scan path overlay
 */
cv::Mat visualize_scan_path(const core::SaliencyMap& saliency, const cv::Mat& original,
                            const std::string& window_name = "", bool wait_key = false);

} // namespace visualization
} // namespace attention
