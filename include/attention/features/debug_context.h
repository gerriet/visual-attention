#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>
#include <memory>

namespace attention
{
namespace features
{

/**
 * DebugContext captures intermediate computation results during feature extraction.
 *
 * This allows inspection of internal processing steps, which is useful for:
 * - Understanding algorithm behavior
 * - Debugging extraction issues
 * - Visualizing multi-scale processing
 * - Academic presentations and publications
 */
struct DebugContext
{
  /**
   * DebugLevel controls verbosity of debug output.
   */
  enum class Level
  {
    None = 0,        // No debug output
    Basic = 1,       // Only final intermediate results (pyramids, opponent colors)
    Detailed = 2,    // All intermediate steps (center-surround differences, etc.)
    Verbose = 3      // Maximum detail (every computation step)
  };

  Level level = Level::None;
  bool enabled = false;

  // Captured intermediate results (name -> image)
  std::map<std::string, cv::Mat> images;

  // Captured multi-scale data (name -> pyramid)
  std::map<std::string, std::vector<cv::Mat>> pyramids;

  // Text annotations (name -> value)
  std::map<std::string, std::string> annotations;

  // Timing information (name -> milliseconds)
  std::map<std::string, double> timings;

  DebugContext() = default;
  explicit DebugContext(Level level_) : level(level_), enabled(level_ != Level::None) {}

  // Add an image to debug output
  void add_image(const std::string& name, const cv::Mat& image)
  {
    if (enabled && !image.empty())
    {
      images[name] = image.clone();
    }
  }

  // Add a pyramid to debug output
  void add_pyramid(const std::string& name, const std::vector<cv::Mat>& pyramid)
  {
    if (enabled && !pyramid.empty())
    {
      pyramids[name] = pyramid;
    }
  }

  // Add a text annotation
  void add_annotation(const std::string& name, const std::string& value)
  {
    if (enabled)
    {
      annotations[name] = value;
    }
  }

  // Add timing information
  void add_timing(const std::string& name, double milliseconds)
  {
    if (enabled)
    {
      timings[name] = milliseconds;
    }
  }

  // Clear all captured data
  void clear()
  {
    images.clear();
    pyramids.clear();
    annotations.clear();
    timings.clear();
  }

  // Check if debug level meets threshold
  bool is_level(Level threshold) const
  {
    return enabled && (level >= threshold);
  }

  // Get total number of captured items
  size_t size() const
  {
    return images.size() + pyramids.size() + annotations.size() + timings.size();
  }

  bool empty() const
  {
    return size() == 0;
  }
};

/**
 * DebugVisualizer provides utilities for visualizing debug context data.
 */
class DebugVisualizer
{
public:
  /**
   * Save all debug images to a directory.
   * @param context Debug context with captured data
   * @param output_dir Directory to save images
   * @param prefix Filename prefix
   */
  static void save_debug_images(const DebugContext& context,
                                const std::string& output_dir,
                                const std::string& prefix);

  /**
   * Create a combined visualization showing all intermediate steps.
   * @param context Debug context with captured data
   * @return Combined visualization image
   */
  static cv::Mat create_debug_visualization(const DebugContext& context);

  /**
   * Visualize a pyramid as a grid of images.
   * @param pyramid Multi-scale pyramid
   * @param max_levels Maximum number of levels to show
   * @return Grid visualization
   */
  static cv::Mat visualize_pyramid(const std::vector<cv::Mat>& pyramid, int max_levels = 6);

  /**
   * Print debug annotations and timings to console.
   * @param context Debug context with captured data
   * @param feature_name Name of feature being debugged
   */
  static void print_debug_info(const DebugContext& context, const std::string& feature_name);
};

} // namespace features
} // namespace attention
