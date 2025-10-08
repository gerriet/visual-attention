// Main entry point for the Attention Framework
// Phase 1: Minimal working system

#include "attention/config/config_loader.h"
#include "attention/pipeline/attention_pipeline.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

void print_usage(const char* program_name)
{
  std::cerr << "Usage:" << std::endl;
  std::cerr << "  " << program_name << " <image_path> [--no-display]" << std::endl;
  std::cerr << "  " << program_name << " --config <config.yaml>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Examples:" << std::endl;
  std::cerr << "  " << program_name << " data/test_images/input.png" << std::endl;
  std::cerr << "  " << program_name << " data/test_images/input.png --no-display" << std::endl;
  std::cerr << "  " << program_name << " --config config.yaml" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "  --no-display  Process without displaying windows (saves to results/)" << std::endl;
  std::cerr << "  --config      Load configuration from YAML file" << std::endl;
}

int main(int argc, char** argv)
{
  // Check command line arguments
  if (argc < 2)
  {
    print_usage(argv[0]);
    return 1;
  }

  try
  {
    attention::config::ConfigLoader::Config config;
    bool use_config_file = false;

    // Parse command line arguments
    if (std::string(argv[1]) == "--config")
    {
      if (argc < 3)
      {
        std::cerr << "Error: --config requires a YAML file path" << std::endl;
        print_usage(argv[0]);
        return 1;
      }

      std::string config_path = argv[2];
      std::cout << "Loading configuration from: " << config_path << std::endl;
      config = attention::config::ConfigLoader::load(config_path);
      use_config_file = true;

      if (config.input_image.empty())
      {
        std::cerr << "Error: No input image specified in config file" << std::endl;
        return 1;
      }
    }
    else
    {
      // Command-line mode (backward compatible)
      config = attention::config::ConfigLoader::create_default();
      config.input_image = argv[1];
      config.display = true;

      // Check for --no-display flag
      if (argc >= 3 && std::string(argv[2]) == "--no-display")
      {
        config.display = false;
      }
    }

    // Create attention pipeline with configuration
    attention::pipeline::AttentionPipeline pipeline(config.pipeline);

    // Load image
    std::cout << "Loading image: " << config.input_image << std::endl;
    pipeline.load_image(config.input_image);

    const auto& frame = pipeline.get_frame();
    std::cout << "  Size: " << frame.width() << "x" << frame.height() << std::endl;
    std::cout << "  Channels: " << frame.channels() << std::endl;

    // Process through attention pipeline
    std::cout << "\nProcessing..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    pipeline.process();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "  Features extracted: " << pipeline.get_features().size() << std::endl;
    std::cout << "  Peaks detected: " << pipeline.get_saliency_map().peaks.size() << std::endl;
    std::cout << "  Processing time: " << duration.count() << " ms" << std::endl;
    std::cout << "✓ Processing complete!" << std::endl;

    // Visualize results
    std::cout << "\nGenerating visualization..." << std::endl;
    cv::Mat visualization = pipeline.visualize(config.save_features);

    if (config.display)
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
      std::string output_path = config.output_dir + "pipeline_output.png";
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
