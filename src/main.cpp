// Main entry point for the Attention Framework
// Phase 1: Minimal working system

#include "attention/pipeline/attention_pipeline.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char** argv)
{
  // Check command line arguments
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <image_path> [--no-display]" << std::endl;
    std::cerr << "Example: " << argv[0] << " ../data/test_images/input.png" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --no-display  Process without displaying windows (saves to results/)" << std::endl;
    return 1;
  }

  std::string image_path = argv[1];
  bool display = true;

  // Check for --no-display flag
  if (argc >= 3 && std::string(argv[2]) == "--no-display")
  {
    display = false;
  }

  try
  {
    // Create attention pipeline
    attention::pipeline::AttentionPipeline pipeline;

    // Load image
    std::cout << "Loading image: " << image_path << std::endl;
    pipeline.load_image(image_path);

    const auto& frame = pipeline.get_frame();
    std::cout << "  Size: " << frame.width() << "x" << frame.height() << std::endl;
    std::cout << "  Channels: " << frame.channels() << std::endl;

    // Process through attention pipeline
    std::cout << "\nProcessing..." << std::endl;
    pipeline.process();

    std::cout << "  Features extracted: " << pipeline.get_features().size() << std::endl;
    std::cout << "  Peaks detected: " << pipeline.get_saliency_map().peaks.size() << std::endl;
    std::cout << "✓ Processing complete!" << std::endl;

    // Visualize results
    std::cout << "\nGenerating visualization..." << std::endl;
    cv::Mat visualization = pipeline.visualize();

    if (display)
    {
      // Show in window
      const std::string window_name = "Attention Framework - Pipeline Results";
      cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
      cv::imshow(window_name, visualization);

      std::cout << "✓ Visualization displayed" << std::endl;
      std::cout << "\nPress any key in the window to exit..." << std::endl;
      cv::waitKey(0);
      cv::destroyAllWindows();
    }
    else
    {
      // Save to file
      std::string output_path = "../results/pipeline_output.png";
      cv::imwrite(output_path, visualization);
      std::cout << "✓ Saved visualization: " << output_path << std::endl;
    }

    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "\nError: " << e.what() << std::endl;
    return 1;
  }
}
