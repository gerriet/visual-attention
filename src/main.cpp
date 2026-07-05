// Main entry point for the Attention Framework
// Phase 1: Minimal working system

#include "attention/config/config_loader.h"
#include "attention/io/result_writer.h"
#include "attention/pipeline/attention_pipeline.h"
#include "attention/visualization/visualizer.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Enable a feature in the config so the CLI modes can switch on stereo/onset
// without a config file. If the feature is already configured, only enable it
// (a user-set weight from --config is preserved); otherwise append it with the
// given default weight.
void enable_feature(attention::pipeline::PipelineConfig& config, const std::string& type, float weight)
{
  for (auto& spec : config.features)
  {
    if (spec.type == type)
    {
      spec.enabled = true; // keep the user's weight
      return;
    }
  }
  attention::pipeline::FeatureSpec spec(type, weight);
  config.features.push_back(spec);
}

// Save the per-frame outputs of a stream (saliency, scan path) into a
// per-frame subdirectory. Shared by batch and sequence modes.
void save_frame_outputs(attention::pipeline::AttentionPipeline& p, const fs::path& output_dir)
{
  fs::create_directories(output_dir);
  const auto& frame = p.get_frame();
  cv::imwrite((output_dir / "00_original.png").string(), frame.image);
  int idx = 1;
  for (const auto& feature : p.get_features())
  {
    cv::Mat feature_vis = attention::visualization::visualize_feature_map(feature);
    cv::imwrite((output_dir / ("0" + std::to_string(idx) + "_" + feature.name + ".png")).string(), feature_vis);
    idx++;
  }
  cv::Mat saliency_vis =
      attention::visualization::visualize_saliency_map(p.get_saliency_map(), frame.image, "", true, false);
  cv::imwrite((output_dir / "99_saliency.png").string(), saliency_vis);
  cv::Mat scan_path_vis = attention::visualization::visualize_scan_path(p.get_saliency_map(), frame.image);
  cv::imwrite((output_dir / "98_scan_path.png").string(), scan_path_vis);
}

// Process a temporal sequence (image directory or video file). Unlike batch
// mode, RunState is NOT reset between frames — onset/motion and neural-field
// dynamics carry across the stream (M5).
void process_sequence(const std::string& path, attention::pipeline::PipelineConfig& config,
                      const std::string& output_base)
{
  enable_feature(config, "onset", 1.0f);
  attention::pipeline::AttentionPipeline pipeline(config);

  std::unique_ptr<attention::pipeline::FrameSource> source;
  std::vector<std::string> image_paths;
  if (fs::is_directory(path))
  {
    image_paths = attention::pipeline::collect_image_paths(path);
    std::cout << "Sequence: " << image_paths.size() << " frames from directory " << path << std::endl;
    source = std::make_unique<attention::pipeline::ImageListSource>(image_paths);
  }
  else
  {
    std::cout << "Sequence: video " << path << std::endl;
    source = std::make_unique<attention::pipeline::VideoFrameSource>(path);
  }

  const std::string out_base = output_base.empty() ? "results/sequence" : output_base;
  size_t processed = 0;
  pipeline.process_stream(
      *source,
      [&](attention::pipeline::AttentionPipeline& p)
      {
        std::ostringstream name;
        name << "frame_" << std::setw(4) << std::setfill('0') << p.get_frame().frame_number;
        fs::path output_dir = fs::path(out_base) / name.str();
        save_frame_outputs(p, output_dir);
        ++processed;
        std::cout << "  [" << processed << "] " << name.str() << ": " << p.get_saliency_map().peaks.size()
                  << " peaks -> " << output_dir.string() << std::endl;
        // NOTE: no reset_state() — this is a temporal stream.
      },
      [](const std::exception& e)
      {
        std::cerr << "  ✗ Error: " << e.what() << std::endl;
        return true;
      });

  std::cout << "Sequence processing complete (" << processed << " frames)." << std::endl;
}

void process_batch(const std::string& directory, const attention::pipeline::PipelineConfig& config,
                   const std::string& output_base = "")
{
  std::vector<std::string> image_paths = attention::pipeline::collect_image_paths(directory);

  std::cout << "Found " << image_paths.size() << " images in " << directory << std::endl;
  std::cout << "Processing batch..." << std::endl;
  std::cout << std::endl;

  attention::pipeline::AttentionPipeline pipeline(config);

  // Statistics tracking with overflow prevention
  struct Stats
  {
    long min = LONG_MAX;
    long max = 0;
    double sum = 0.0;  // Use double to prevent overflow
    int count = 0;

    void add(long value)
    {
      if (value < min)
        min = value;
      if (value > max)
        max = value;
      sum += static_cast<double>(value);
      count++;
    }

    double mean() const { return count > 0 ? sum / count : 0.0; }
  };

  Stats pyramid_stats, integration_stats, peak_stats, total_stats;
  std::map<std::string, Stats> feature_stats;

  // Process the directory as a stream; per-image output happens in the
  // callback after each frame
  attention::pipeline::ImageListSource source(image_paths);
  size_t processed_count = 0;

  pipeline.process_stream(
      source,
      [&](attention::pipeline::AttentionPipeline& p)
      {
        const auto& frame = p.get_frame();
        fs::path input_path(frame.source_path);
        std::string filename = input_path.stem().string();
        ++processed_count;

        const auto& timing = p.get_timing();
        std::cout << "[" << processed_count << "/" << image_paths.size() << "] " << filename << ": " << frame.width()
                  << "x" << frame.height() << ", " << p.get_features().size() << " features, "
                  << p.get_saliency_map().peaks.size() << " peaks, " << timing.total_ms() << " ms" << std::endl;

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
        for (const auto& feature : p.get_features())
        {
          cv::Mat feature_vis = attention::visualization::visualize_feature_map(feature);
          std::string feature_filename = "0" + std::to_string(idx) + "_" + feature.name + ".png";
          cv::imwrite((output_dir / feature_filename).string(), feature_vis);
          idx++;
        }

        // Save saliency map
        cv::Mat saliency_vis =
            attention::visualization::visualize_saliency_map(p.get_saliency_map(), frame.image, "", true, false);
        cv::imwrite((output_dir / "99_saliency.png").string(), saliency_vis);

        // Save scan path visualization
        cv::Mat scan_path_vis = attention::visualization::visualize_scan_path(p.get_saliency_map(), frame.image);
        cv::imwrite((output_dir / "98_scan_path.png").string(), scan_path_vis);

        // Save combined visualization
        cv::Mat combined = p.visualize(false);
        cv::imwrite((output_dir / "combined.png").string(), combined);

        // Save timing information
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
        std::cout << std::endl;

        // Batch images are independent stills, not a temporal stream: clear
        // per-run state (neural-field activity, IOR map) between images
        p.reset_state();
      },
      [](const std::exception& e)
      {
        // Log and continue with the remaining images (v1 batch semantics)
        std::cerr << "  ✗ Error: " << e.what() << std::endl << std::endl;
        return true;
      });

  std::cout << "Batch processing complete!" << std::endl;
  std::cout << std::endl;

  // Print timing statistics
  if (total_stats.count > 0)
  {
    std::cout << "Timing Statistics (ms) - " << total_stats.count << " images processed" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(1);

    // Find longest label for alignment
    size_t max_label_length = 0;
    max_label_length = std::max(max_label_length, std::string("Pyramid:").length());
    max_label_length = std::max(max_label_length, std::string("Integration:").length());
    max_label_length = std::max(max_label_length, std::string("Peak detection:").length());
    for (const auto& pair : feature_stats)
    {
      max_label_length = std::max(max_label_length, std::string("Feature '" + pair.first + "':").length());
    }

    auto print_stats = [max_label_length](const std::string& label, const Stats& stats)
    {
      std::cout << std::left << std::setw(max_label_length) << label << "  min: " << std::setw(6) << std::right << stats.min
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

attention::features::DebugContext::Level parse_debug_level(const std::string& level_str)
{
  std::string lower = level_str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower == "basic")
    return attention::features::DebugContext::Level::Basic;
  else if (lower == "detailed")
    return attention::features::DebugContext::Level::Detailed;
  else if (lower == "verbose")
    return attention::features::DebugContext::Level::Verbose;
  else
    return attention::features::DebugContext::Level::Basic; // Default
}

void print_usage(const char* program_name)
{
  std::cerr << "Usage:" << std::endl;
  std::cerr << "  " << program_name << " <image_path> [options]" << std::endl;
  std::cerr << "  " << program_name << " --config <config.yaml> [image] [options]" << std::endl;
  std::cerr << "  " << program_name << " --batch <directory> [options]" << std::endl;
  std::cerr << "  " << program_name << " --stereo <left> <right> [options]" << std::endl;
  std::cerr << "  " << program_name << " --sequence <dir|video> [--output dir] [--config yaml]" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Examples:" << std::endl;
  std::cerr << "  " << program_name << " data/test_images/input.png" << std::endl;
  std::cerr << "  " << program_name << " data/test_images/input.png --no-display" << std::endl;
  std::cerr << "  " << program_name << " data/test_images/input.png --debug" << std::endl;
  std::cerr << "  " << program_name << " data/test_images/input.png --debug=detailed --debug-print" << std::endl;
  std::cerr << "  " << program_name << " --config config.yaml" << std::endl;
  std::cerr << "  " << program_name << " --batch data/test_images/" << std::endl;
  std::cerr << "  " << program_name << " --batch data/test_images/ --output results/" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "  --no-display         Process without displaying windows (saves to results/)" << std::endl;
  std::cerr << "  --config             Load configuration from YAML file" << std::endl;
  std::cerr << "  --batch              Process all images in directory, save features separately" << std::endl;
  std::cerr << "  --stereo <l> <r>     Process a stereo pair (adds a disparity/depth channel)" << std::endl;
  std::cerr << "  --sequence <path>    Process a directory or video as a temporal stream (onset/motion)" << std::endl;
  std::cerr << "  --output <dir>       Specify output directory for batch/sequence mode (default: input_dir/results_batch)" << std::endl;
  std::cerr << "  --emit-json <path>   Write result JSON + saliency map in the interchange format" << std::endl;
  std::cerr << "                       (see docs/INTERCHANGE_FORMAT.md; single-image and config modes)" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Debug Options:" << std::endl;
  std::cerr << "  --debug[=LEVEL]      Enable debugging (levels: basic, detailed, verbose)" << std::endl;
  std::cerr << "                       Default: basic if no level specified" << std::endl;
  std::cerr << "  --debug-output <dir> Debug output directory (default: debug_output/)" << std::endl;
  std::cerr << "  --debug-print        Print debug info to console" << std::endl;
  std::cerr << "  --no-debug-save      Don't save debug images to disk" << std::endl;
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
    std::string emit_json_path;

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
    else if (std::string(argv[1]) == "--sequence")
    {
      // Temporal stream (image directory or video): onset/motion + carried
      // neural-field/IOR state, no per-frame reset.
      if (argc < 3)
      {
        std::cerr << "Error: --sequence requires a directory or video path" << std::endl;
        print_usage(argv[0]);
        return 1;
      }
      std::string seq_path = argv[2];
      std::string output_dir = "";
      config = attention::config::ConfigLoader::create_default();
      for (int i = 3; i < argc; ++i)
      {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc)
        {
          output_dir = argv[++i];
        }
        else if (arg == "--config" && i + 1 < argc)
        {
          config = attention::config::ConfigLoader::load(argv[++i]);
        }
      }
      process_sequence(seq_path, config.pipeline, output_dir);
      return 0;
    }
    else if (std::string(argv[1]) == "--stereo")
    {
      // Single stereo pair: left + right, stereo feature adds a depth channel.
      if (argc < 4)
      {
        std::cerr << "Error: --stereo requires a left and a right image path" << std::endl;
        print_usage(argv[0]);
        return 1;
      }
      std::string left_path = argv[2];
      std::string right_path = argv[3];
      bool have_config = false;
      config = attention::config::ConfigLoader::create_default();
      config.display = false;
      for (int i = 4; i < argc; ++i)
      {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
        {
          config = attention::config::ConfigLoader::load(argv[++i]);
          have_config = true;
        }
        else if (arg == "--emit-json" && i + 1 < argc)
        {
          emit_json_path = argv[++i];
        }
        else if (arg == "--no-display")
        {
          config.display = false;
        }
      }
      if (!have_config)
      {
        enable_feature(config.pipeline, "stereo", 1.5f);
      }

      attention::pipeline::AttentionPipeline pipeline(config.pipeline);
      std::cout << "Loading stereo pair: " << left_path << " | " << right_path << std::endl;
      pipeline.load_stereo(left_path, right_path);
      pipeline.process();
      std::cout << "  Peaks detected: " << pipeline.get_saliency_map().peaks.size() << std::endl;
      if (!emit_json_path.empty())
      {
        attention::io::ResultWriter::write(pipeline, emit_json_path);
        std::cout << "✓ Saved result JSON: " << emit_json_path << std::endl;
      }
      fs::create_directories(config.output_dir);
      cv::Mat vis = pipeline.visualize(false);
      cv::imwrite(config.output_dir + "stereo_output.png", vis);
      std::cout << "✓ Saved visualization: " << config.output_dir << "stereo_output.png" << std::endl;
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

      for (int i = 3; i < argc; ++i)
      {
        std::string arg = argv[i];
        if (arg == "--emit-json" && i + 1 < argc)
        {
          emit_json_path = argv[++i];
        }
        else if (arg == "--no-display")
        {
          config.display = false;
        }
        else if (arg.rfind("--", 0) != 0)
        {
          // Positional image overrides input.image, so profile configs
          // (configs/thesis.yaml, configs/modern.yaml) work on any image
          config.input_image = arg;
        }
      }

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

      // Parse optional flags
      for (int i = 2; i < argc; ++i)
      {
        std::string arg = argv[i];

        if (arg == "--no-display")
        {
          config.display = false;
        }
        else if (arg == "--debug" || arg.rfind("--debug=", 0) == 0)
        {
          // Parse debug level
          if (arg == "--debug")
          {
            config.pipeline.debug_level = attention::features::DebugContext::Level::Basic;
          }
          else
          {
            std::string level_str = arg.substr(8); // Skip "--debug="
            config.pipeline.debug_level = parse_debug_level(level_str);
          }
        }
        else if (arg == "--debug-output" && i + 1 < argc)
        {
          config.pipeline.debug_output_dir = argv[++i];
        }
        else if (arg == "--debug-print")
        {
          config.pipeline.debug_print_info = true;
        }
        else if (arg == "--no-debug-save")
        {
          config.pipeline.debug_save_images = false;
        }
        else if (arg == "--emit-json" && i + 1 < argc)
        {
          emit_json_path = argv[++i];
        }
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

    if (!emit_json_path.empty())
    {
      attention::io::ResultWriter::write(pipeline, emit_json_path);
      std::cout << "✓ Saved result JSON: " << emit_json_path << std::endl;
    }

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
      fs::create_directories(config.output_dir);
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
