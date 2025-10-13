#include "attention/visualization/visualizer.h"
#include <iostream>

namespace attention
{
namespace visualization
{

cv::Mat visualize_feature_map(const core::FeatureMap& feature, const std::string& window_name, bool wait_key)
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

cv::Mat visualize_saliency_map(const core::SaliencyMap& saliency, const cv::Mat& original,
                               const std::string& window_name, bool mark_peaks, bool wait_key)
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

cv::Mat visualize_side_by_side(const std::vector<cv::Mat>& images, const std::vector<std::string>& labels,
                               const std::string& window_name, bool wait_key)
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

bool save_visualization(const cv::Mat& image, const std::string& filepath)
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

cv::Mat visualize_scan_path(const core::SaliencyMap& saliency, const cv::Mat& original, const std::string& window_name,
                            bool wait_key)
{
  if (saliency.empty() || !saliency.is_valid() || original.empty())
  {
    std::cerr << "Warning: Cannot visualize scan path with empty inputs" << std::endl;
    return cv::Mat();
  }

  if (saliency.peaks.empty())
  {
    std::cerr << "Warning: No peaks to visualize in scan path" << std::endl;
    return original.clone();
  }

  // Start with original image
  cv::Mat vis;
  if (original.channels() == 1)
  {
    cv::cvtColor(original, vis, cv::COLOR_GRAY2BGR);
  }
  else
  {
    vis = original.clone();
  }

  // Ensure image is resized to saliency map size if needed
  if (vis.size() != saliency.size())
  {
    cv::resize(vis, vis, saliency.size());
  }

  // Draw scan path: arrows connecting peaks in order
  const int arrow_thickness = 2;
  const double arrow_tip_length = 0.2;

  for (size_t i = 0; i < saliency.peaks.size(); ++i)
  {
    const auto& peak = saliency.peaks[i];

    // Draw circle at peak location
    // White outer circle
    cv::circle(vis, peak.location, 12, cv::Scalar(255, 255, 255), 2);
    // Black inner circle
    cv::circle(vis, peak.location, 10, cv::Scalar(0, 0, 0), 2);
    // Yellow fill
    cv::circle(vis, peak.location, 8, cv::Scalar(0, 255, 255), -1);

    // Draw sequence number
    std::string number = std::to_string(i + 1);
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(number, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
    cv::Point text_pos(peak.location.x - text_size.width / 2, peak.location.y + text_size.height / 2);
    cv::putText(vis, number, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);

    // Draw arrow to next peak
    if (i + 1 < saliency.peaks.size())
    {
      const auto& next_peak = saliency.peaks[i + 1];
      // Draw arrow with white outline for visibility
      cv::arrowedLine(vis, peak.location, next_peak.location, cv::Scalar(255, 255, 255), arrow_thickness + 2,
                      cv::LINE_AA, 0, arrow_tip_length);
      // Draw arrow with cyan color
      cv::arrowedLine(vis, peak.location, next_peak.location, cv::Scalar(255, 255, 0), arrow_thickness, cv::LINE_AA, 0,
                      arrow_tip_length);
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

} // namespace visualization
} // namespace attention
