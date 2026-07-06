#include "attention/features/debug_context.h"
#include <filesystem>
#include <iomanip>
#include <iostream>

namespace attention
{
namespace features
{

void DebugVisualizer::save_debug_images(const DebugContext& context, const std::string& output_dir,
                                        const std::string& prefix)
{
  if (context.empty())
  {
    return;
  }

  // Create output directory if it doesn't exist
  std::filesystem::create_directories(output_dir);

  // Save individual images
  int idx = 0;
  for (const auto& [name, image] : context.images)
  {
    std::string filename = output_dir + "/" + prefix + "_" + std::to_string(idx) + "_" + name + ".png";

    // Convert to 8-bit for saving if needed
    cv::Mat save_img;
    if (image.depth() == CV_32F || image.depth() == CV_64F)
    {
      cv::normalize(image, save_img, 0, 255, cv::NORM_MINMAX);
      save_img.convertTo(save_img, CV_8U);
    }
    else
    {
      save_img = image;
    }

    cv::imwrite(filename, save_img);
    std::cout << "  Saved debug image: " << filename << std::endl;
    idx++;
  }

  // Save pyramids
  for (const auto& [name, pyramid] : context.pyramids)
  {
    cv::Mat pyramid_vis = visualize_pyramid(pyramid);
    std::string filename = output_dir + "/" + prefix + "_pyramid_" + name + ".png";
    cv::imwrite(filename, pyramid_vis);
    std::cout << "  Saved debug pyramid: " << filename << std::endl;
  }
}

cv::Mat DebugVisualizer::create_debug_visualization(const DebugContext& context)
{
  if (context.images.empty())
  {
    return cv::Mat();
  }

  // Calculate grid dimensions
  int num_images = static_cast<int>(context.images.size());
  int cols = std::min(3, num_images);
  int rows = (num_images + cols - 1) / cols;

  // Find maximum image size
  int max_width = 0;
  int max_height = 0;
  for (const auto& [name, image] : context.images)
  {
    max_width = std::max(max_width, image.cols);
    max_height = std::max(max_height, image.rows);
  }

  // Create canvas
  const int padding = 10;
  const int label_height = 30;
  int canvas_width = cols * (max_width + padding) + padding;
  int canvas_height = rows * (max_height + label_height + padding) + padding;
  cv::Mat canvas = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
  canvas.setTo(cv::Scalar(50, 50, 50));

  // Place images on canvas
  int idx = 0;
  for (const auto& [name, image] : context.images)
  {
    int row = idx / cols;
    int col = idx % cols;

    int x = col * (max_width + padding) + padding;
    int y = row * (max_height + label_height + padding) + padding;

    // Convert and resize image
    cv::Mat display_img;
    if (image.depth() == CV_32F || image.depth() == CV_64F)
    {
      cv::normalize(image, display_img, 0, 255, cv::NORM_MINMAX);
      display_img.convertTo(display_img, CV_8U);
    }
    else
    {
      display_img = image;
    }

    if (display_img.channels() == 1)
    {
      cv::cvtColor(display_img, display_img, cv::COLOR_GRAY2BGR);
    }

    // Resize to fit if needed
    if (display_img.cols > max_width || display_img.rows > max_height)
    {
      double scale = std::min(static_cast<double>(max_width) / display_img.cols,
                              static_cast<double>(max_height) / display_img.rows);
      cv::resize(display_img, display_img, cv::Size(), scale, scale);
    }

    // Copy to canvas
    cv::Rect roi(x, y, display_img.cols, display_img.rows);
    display_img.copyTo(canvas(roi));

    // Add label
    cv::putText(canvas, name, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    idx++;
  }

  return canvas;
}

cv::Mat DebugVisualizer::visualize_pyramid(const std::vector<cv::Mat>& pyramid, int max_levels)
{
  if (pyramid.empty())
  {
    return cv::Mat();
  }

  int levels = std::min(max_levels, static_cast<int>(pyramid.size()));

  // Calculate total canvas size
  int total_width = 0;
  int max_height = pyramid[0].rows;
  for (int i = 0; i < levels; ++i)
  {
    total_width += pyramid[i].cols + 5;
  }

  // Create canvas
  cv::Mat canvas = cv::Mat::zeros(max_height + 30, total_width, CV_8UC3);
  canvas.setTo(cv::Scalar(30, 30, 30));

  // Place pyramid levels
  int x_offset = 0;
  for (int i = 0; i < levels; ++i)
  {
    // Convert to displayable format
    cv::Mat level_img;
    if (pyramid[i].depth() == CV_32F || pyramid[i].depth() == CV_64F)
    {
      cv::normalize(pyramid[i], level_img, 0, 255, cv::NORM_MINMAX);
      level_img.convertTo(level_img, CV_8U);
    }
    else
    {
      level_img = pyramid[i];
    }

    if (level_img.channels() == 1)
    {
      cv::cvtColor(level_img, level_img, cv::COLOR_GRAY2BGR);
    }

    // Copy to canvas
    cv::Rect roi(x_offset, 0, level_img.cols, level_img.rows);
    level_img.copyTo(canvas(roi));

    // Add level label
    std::string label = "L" + std::to_string(i);
    cv::putText(canvas, label, cv::Point(x_offset + 5, level_img.rows + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);

    x_offset += level_img.cols + 5;
  }

  return canvas;
}

void DebugVisualizer::print_debug_info(const DebugContext& context, const std::string& feature_name)
{
  if (context.empty())
  {
    return;
  }

  std::cout << "\n=== Debug Info: " << feature_name << " ===" << std::endl;

  // Print annotations
  if (!context.annotations.empty())
  {
    std::cout << "\nAnnotations:" << std::endl;
    for (const auto& [name, value] : context.annotations)
    {
      std::cout << "  " << name << ": " << value << std::endl;
    }
  }

  // Print timings
  if (!context.timings.empty())
  {
    std::cout << "\nTimings:" << std::endl;
    for (const auto& [name, ms] : context.timings)
    {
      std::cout << "  " << std::setw(30) << std::left << name << ": " << std::fixed << std::setprecision(2) << ms
                << " ms" << std::endl;
    }
  }

  // Print captured data summary
  std::cout << "\nCaptured Data:" << std::endl;
  std::cout << "  Images: " << context.images.size() << std::endl;
  std::cout << "  Pyramids: " << context.pyramids.size() << std::endl;

  if (!context.images.empty())
  {
    std::cout << "\n  Image details:" << std::endl;
    for (const auto& [name, image] : context.images)
    {
      std::cout << "    " << std::setw(30) << std::left << name << ": " << image.cols << "x" << image.rows
                << " channels=" << image.channels() << std::endl;
    }
  }

  if (!context.pyramids.empty())
  {
    std::cout << "\n  Pyramid details:" << std::endl;
    for (const auto& [name, pyramid] : context.pyramids)
    {
      std::cout << "    " << std::setw(30) << std::left << name << ": " << pyramid.size() << " levels" << std::endl;
    }
  }

  std::cout << "===========================" << std::endl << std::endl;
}

} // namespace features
} // namespace attention
