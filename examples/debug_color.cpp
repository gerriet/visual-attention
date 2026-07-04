/**
 * Simple test program to demonstrate feature debugging capabilities.
 *
 * This is a minimal example showing how to use DebugContext to inspect
 * intermediate computation steps during feature extraction.
 */

#include "attention/core/frame.h"
#include "attention/features/color_feature.h"
#include "attention/features/debug_context.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
    std::cout << "\nThis program demonstrates feature extraction debugging." << std::endl;
    std::cout << "It will extract color features and save intermediate results." << std::endl;
    return 1;
  }

  std::string image_path = argv[1];

  // Load image
  std::cout << "Loading image: " << image_path << std::endl;
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty())
  {
    std::cerr << "Error: Failed to load image: " << image_path << std::endl;
    return 1;
  }

  std::cout << "  Image size: " << image.cols << "x" << image.rows << std::endl;
  std::cout << "  Channels: " << image.channels() << std::endl;

  // Create frame and compute pyramids
  std::cout << "\nComputing pyramids..." << std::endl;
  attention::core::Frame frame(image);
  frame.compute_pyramids(9);  // Compute 9-level pyramid

  // Create debug context with Detailed level
  attention::features::DebugContext debug(attention::features::DebugContext::Level::Detailed);
  std::cout << "Debug level: Detailed" << std::endl;

  // Extract color feature with debugging enabled
  std::cout << "\nExtracting color feature with debugging..." << std::endl;
  attention::features::ColorFeature color_extractor;
  attention::core::FeatureMap color_feature = color_extractor.extract(frame, debug);

  // Print debug information
  attention::features::DebugVisualizer::print_debug_info(debug, "ColorFeature");

  // Save debug images
  std::string output_dir = "debug_output";
  std::cout << "\nSaving debug images to: " << output_dir << std::endl;
  attention::features::DebugVisualizer::save_debug_images(debug, output_dir, "color");

  // Create and save combined visualization
  cv::Mat viz = attention::features::DebugVisualizer::create_debug_visualization(debug);
  if (!viz.empty())
  {
    cv::imwrite(output_dir + "/color_debug_combined.png", viz);
    std::cout << "  Saved combined visualization: " << output_dir << "/color_debug_combined.png" << std::endl;
  }

  // Display final result
  cv::Mat result_8u;
  color_feature.data.convertTo(result_8u, CV_8U, 255.0);
  cv::imshow("Color Feature Result", result_8u);
  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return 0;
}
