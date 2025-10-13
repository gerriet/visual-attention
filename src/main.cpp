// Main entry point for the Attention Framework
// Phase 1: Minimal working system

#include "attention/config/config_loader.h"
#include "attention/pipeline/attention_pipeline.h"
#include "attention/visualization/visualizer.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void process_batch(const std::string& directory, const attention::pipeline::PipelineConfig& config,
                   const std::string& output_base = "")
{
  std::vector<std::string> image_paths;

  // Collect all image files
  for (const auto& entry : fs::directory_iterator(directory))
  {
    if (entry.is_regular_file())
    {
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
      {
        image_paths.push_back(entry.path().string());
      }
    }
  }

  std::sort(image_paths.begin(), image_paths.end());

  std::cout << "Found " << image_paths.size() << " images in " << directory << std::endl;
  std::cout << "Processing batch..." << std::endl;
  std::cout << std::endl;

  attention::pipeline::AttentionPipeline pipeline(config);

  // Statistics tracking
  struct Stats
  {
    long min = LONG_MAX;
    long max = 0;
    long sum = 0;
    int count = 0;

    void add(long value)
    {
      if (value < min)
        min = value;
      if (value > max)
        max = value;
      sum += value;
      count++;
    }

    double mean() const { return count > 0 ? static_cast<double>(sum) / count : 0.0; }
  };

  Stats pyramid_stats, integration_stats, peak_stats, total_stats;
  std::map<std::string, Stats> feature_stats;

  for (size_t i = 0; i < image_paths.size(); ++i)
  {
    const std::string& image_path = image_paths[i];
    fs::path input_path(image_path);
    std::string filename = input_path.stem().string();

    std::cout << "[" << (i + 1) << "/" << image_paths.size() << "] Processing: " << filename << std::endl;

    try
    {
      pipeline.load_image(image_path);
      const auto& frame = pipeline.get_frame();
      std::cout << "  Size: " << frame.width() << "x" << frame.height() << std::endl;

      auto start = std::chrono::high_resolution_clock::now();
      pipeline.process();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

      std::cout << "  Time: " << duration.count() << " ms" << std::endl;
      std::cout << "  Features: " << pipeline.get_features().size()
                << ", Peaks: " << pipeline.get_saliency_map().peaks.size() << std::endl;

      // Determine output directory
      fs::path output_dir;
      if (!output_base.empty())
      {
        output_dir = fs::path(output_base) / filename;
      }
      else
      {
        output_dir = input_path.parent_path() / "results_batch" / filename;
      }
      fs::create_directories(output_dir);

      // Save original
      cv::imwrite((output_dir / "00_original.png").string(), frame.image);

      // Save each feature
      int idx = 1;
      for (const auto& feature : pipeline.get_features())
      {
        cv::Mat feature_vis = attention::visualization::visualize_feature_map(feature);
        std::string feature_filename = "0" + std::to_string(idx) + "_" + feature.name + ".png";
        cv::imwrite((output_dir / feature_filename).string(), feature_vis);
        idx++;
      }

      // Save saliency map
      cv::Mat saliency_vis =
          attention::visualization::visualize_saliency_map(pipeline.get_saliency_map(), frame.image, "", true, false);
      cv::imwrite((output_dir / "99_saliency.png").string(), saliency_vis);

      // Save scan path visualization
      cv::Mat scan_path_vis = attention::visualization::visualize_scan_path(pipeline.get_saliency_map(), frame.image);
      cv::imwrite((output_dir / "98_scan_path.png").string(), scan_path_vis);

      // Save combined visualization
      cv::Mat combined = pipeline.visualize(false);
      cv::imwrite((output_dir / "combined.png").string(), combined);

      // Save timing information
      const auto& timing = pipeline.get_timing();
      std::ofstream timing_file((output_dir / "timing.txt").string());
      timing_file << "Performance Timing (ms)" << std::endl;
      timing_file << "======================" << std::endl;
      timing_file << "Image size: " << frame.width() << "x" << frame.height() << std::endl;
      timing_file << std::endl;
      timing_file << "Pyramid computation: " << timing.pyramid_ms << " ms" << std::endl;
      for (const auto& pair : timing.feature_ms)
      {
        timing_file << "Feature '" << pair.first << "': " << pair.second << " ms" << std::endl;
      }
      timing_file << "Integration: " << timing.integration_ms << " ms" << std::endl;
      timing_file << "Peak detection: " << timing.peak_detection_ms << " ms" << std::endl;
      timing_file << std::endl;
      timing_file << "Total: " << timing.total_ms() << " ms" << std::endl;
      timing_file.close();

      // Collect statistics
      pyramid_stats.add(timing.pyramid_ms);
      integration_stats.add(timing.integration_ms);
      peak_stats.add(timing.peak_detection_ms);
      total_stats.add(timing.total_ms());
      for (const auto& pair : timing.feature_ms)
      {
        feature_stats[pair.first].add(pair.second);
      }

      std::cout << "  ✓ Saved to: " << output_dir.string() << std::endl;
    }
    catch (const std::exception& e)
    {
      std::cerr << "  ✗ Error: " << e.what() << std::endl;
    }

    std::cout << std::endl;
  }

  std::cout << "Batch processing complete!" << std::endl;
  std::cout << std::endl;

  // Print timing statistics
  if (total_stats.count > 0)
  {
    std::cout << "Timing Statistics (ms) - " << total_stats.count << " images processed" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(1);

    auto print_stats = [](const std::string& label, const Stats& stats)
    {
      std::cout << std::left << std::setw(20) << label << "  min: " << std::setw(6) << std::right << stats.min
                << "  max: " << std::setw(6) << stats.max << "  mean: " << std::setw(7) << stats.mean() << std::endl;
    };

    print_stats("Pyramid:", pyramid_stats);
    for (const auto& pair : feature_stats)
    {
      print_stats("Feature '" + pair.first + "':", pair.second);
    }
    print_stats("Integration:", integration_stats);
    print_stats("Peak detection:", peak_stats);
    std::cout << "-------------------------------------------------------" << std::endl;
    print_stats("Total:", total_stats);
  }
}

void print_usage(const char* program_name)
{
  std::cerr << "Usage:" << std::endl;
  std::cerr << "  " << program_name << " <image_path> [--no-display]" << std::endl;
  std::cerr << "  " << program_name << " --config <config.yaml>" << std::endl;
  std::cerr << "  " << program_name << " --batch <directory> [--output <output_dir>]" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Examples:" << std::endl;
  std::cerr << "  " << program_name << " data/test_images/input.png" << std::endl;
  std::cerr << "  " << program_name << " data/test_images/input.png --no-display" << std::endl;
  std::cerr << "  " << program_name << " --config config.yaml" << std::endl;
  std::cerr << "  " << program_name << " --batch data/test_images/" << std::endl;
  std::cerr << "  " << program_name << " --batch data/test_images/ --output results/" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "  --no-display  Process without displaying windows (saves to results/)" << std::endl;
  std::cerr << "  --config      Load configuration from YAML file" << std::endl;
  std::cerr << "  --batch       Process all images in directory, save features separately" << std::endl;
  std::cerr << "  --output      Specify output directory for batch mode (default: input_dir/results_batch)"
            << std::endl;
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
    if (std::string(argv[1]) == "--batch")
    {
      if (argc < 3)
      {
        std::cerr << "Error: --batch requires a directory path" << std::endl;
        print_usage(argv[0]);
        return 1;
      }

      std::string directory = argv[2];
      std::string output_dir = "";

      // Check for optional --output flag
      if (argc >= 5 && std::string(argv[3]) == "--output")
      {
        output_dir = argv[4];
      }

      config = attention::config::ConfigLoader::create_default();
      process_batch(directory, config.pipeline, output_dir);
      return 0;
    }
    else if (std::string(argv[1]) == "--config")
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

    // Generate scan path visualization
    cv::Mat scan_path = attention::visualization::visualize_scan_path(pipeline.get_saliency_map(), frame.image);

    if (config.display)
    {
      // Show in window
      const std::string window_name = "Attention Framework - Pipeline Results";
      cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
      cv::imshow(window_name, visualization);

      // Show scan path in separate window
      const std::string scan_path_window = "Scan Path";
      cv::namedWindow(scan_path_window, cv::WINDOW_AUTOSIZE);
      cv::imshow(scan_path_window, scan_path);

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

      // Save scan path
      std::string scan_path_output = config.output_dir + "scan_path.png";
      cv::imwrite(scan_path_output, scan_path);
      std::cout << "✓ Saved scan path: " << scan_path_output << std::endl;
    }

    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "\nError: " << e.what() << std::endl;
    return 1;
  }
}
