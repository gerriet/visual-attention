// Main entry point for the Attention Framework
// Phase 1: Minimal working system

#include "attention/config/config_loader.h"
#include "attention/io/result_writer.h"
#include "attention/io/scanpath_writer.h"
#include "attention/pipeline/attention_pipeline.h"
#include "attention/system/attention_system.h"
#include "attention/system/live_demonstrator.h"
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

// Draw the object files annotated as in the thesis figures: white = current
// focus, red = never selected, blue = previously selected; plus the focus box.
cv::Mat annotate_objects(const attention::system::AttentionSystem& sys)
{
  const auto& frame = sys.pipeline().get_frame();
  cv::Mat vis;
  if (frame.image.channels() == 1)
  {
    cv::cvtColor(frame.image, vis, cv::COLOR_GRAY2BGR);
  }
  else
  {
    vis = frame.image.clone();
  }
  const auto* focus = sys.current_focus();
  for (const auto& obj : sys.active_files())
  {
    cv::Scalar color = obj.ever_selected() ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255); // BGR: blue / red
    if (focus && obj.label == focus->label)
    {
      color = cv::Scalar(255, 255, 255); // white = current focus
    }
    cv::rectangle(vis, obj.bbox, color, 2);
    cv::putText(vis, std::to_string(obj.label), obj.centroid, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }
  return vis;
}

// Run the AttentionSystem (second stage + behavior) over a temporal sequence
// (image directory or video) and emit its scanpath.
void process_attend(const std::string& path, attention::pipeline::PipelineConfig& pipeline_config,
                    const std::string& output_base, const std::string& emit_scanpath, bool user_config,
                    const std::string& behavior)
{
  if (!user_config)
  {
    enable_feature(pipeline_config, "onset", 1.0f); // motion helps track objects
  }

  attention::system::AttentionSystem::Config cfg;
  cfg.pipeline = pipeline_config;
  if (!behavior.empty())
  {
    cfg.behavior = behavior; // dynamic-IOR ablation: greedy / spatial-ior / object-ior / exploration
  }
  attention::system::AttentionSystem sys(cfg);

  std::unique_ptr<attention::pipeline::FrameSource> source;
  if (fs::is_directory(path))
  {
    std::vector<std::string> frames = attention::pipeline::collect_image_paths(path);
    std::cout << "Attend: " << frames.size() << " frames from directory " << path << std::endl;
    source = std::make_unique<attention::pipeline::ImageListSource>(frames);
  }
  else
  {
    std::cout << "Attend: video " << path << std::endl;
    source = std::make_unique<attention::pipeline::VideoFrameSource>(path);
  }

  const std::string out_base = output_base.empty() ? "results/attend" : output_base;
  sys.process_stream(*source,
                     [&](attention::system::AttentionSystem& s)
                     {
                       std::ostringstream name;
                       name << "frame_" << std::setw(4) << std::setfill('0') << s.frame_index();
                       fs::path dir = fs::path(out_base) / name.str();
                       fs::create_directories(dir);
                       cv::imwrite((dir / "objects.png").string(), annotate_objects(s));
                       const auto* focus = s.current_focus();
                       std::cout << "  " << name.str() << ": " << s.active_files().size() << " objects";
                       if (focus)
                       {
                         std::cout << ", focus #" << focus->label << " at (" << focus->location.x << ","
                                   << focus->location.y << ")";
                       }
                       std::cout << std::endl;
                     });

  std::cout << "Attend complete: " << sys.scanpath().size() << " foci over " << sys.frame_index() << " frames."
            << std::endl;

  if (!emit_scanpath.empty())
  {
    attention::io::ScanpathWriter::write(sys, emit_scanpath);
    std::cout << "✓ Saved scanpath: " << emit_scanpath << std::endl;
  }
}

std::vector<std::string> split_csv(const std::string& s)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
  {
    if (!item.empty())
    {
      out.push_back(item);
    }
  }
  return out;
}

// Live demonstrator (M8): attention on a webcam / video / frame directory with
// object-file plugins running on attended ROIs, drawn as an overlay. Interactive
// (imshow, ESC to quit) or headless (save annotated frames).
void process_live(const std::string& source, attention::system::LiveDemonstrator::Config cfg, bool display,
                  int max_frames, const std::string& output_dir)
{
  std::unique_ptr<attention::pipeline::FrameSource> src;
  if (fs::is_directory(source))
  {
    src = std::make_unique<attention::pipeline::ImageListSource>(attention::pipeline::collect_image_paths(source));
    std::cout << "Live: frame directory " << source << std::endl;
  }
  else
  {
    src = std::make_unique<attention::pipeline::VideoSource>(source);
    std::cout << "Live: " << source << std::endl;
  }

  attention::system::LiveDemonstrator demo(cfg);
  demo.reset();

  const std::string window = "Attention (live) - ESC to quit";
  if (display)
  {
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
  }
  const std::string out_base = output_dir.empty() ? "results/live" : output_dir;

  attention::core::Frame frame;
  int count = 0;
  while (src->next(frame))
  {
    cv::Mat annotated = demo.process(frame.image);
    const auto* focus = demo.system().current_focus();

    if (display)
    {
      cv::imshow(window, annotated);
      if (cv::waitKey(1) == 27) // ESC
      {
        break;
      }
    }
    else
    {
      fs::create_directories(out_base);
      std::ostringstream name;
      name << "frame_" << std::setw(4) << std::setfill('0') << count << ".png";
      cv::imwrite((fs::path(out_base) / name.str()).string(), annotated);
    }

    std::cout << "  frame " << count << ": " << demo.system().active_files().size() << " objects, "
              << demo.annotations().size() << " annotations";
    if (focus)
    {
      std::cout << ", focus #" << focus->label;
    }
    std::cout << std::endl;

    ++count;
    if (max_frames > 0 && count >= max_frames)
    {
      break;
    }
  }
  if (display)
  {
    cv::destroyAllWindows();
  }
  std::cout << "Live complete: " << count << " frames." << std::endl;
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
    double sum = 0.0; // Use double to prevent overflow
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
      std::cout << std::left << std::setw(max_label_length) << label << "  min: " << std::setw(6) << std::right
                << stats.min << "  max: " << std::setw(6) << stats.max << "  mean: " << std::setw(7) << stats.mean()
                << std::endl;
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

void print_usage(const char* program_name, std::ostream& out = std::cerr)
{
  out << "attention - neural-field visual attention pipeline (v2)" << std::endl;
  out << std::endl;
  out << "Usage:" << std::endl;
  out << "  " << program_name << " <image_path> [options]" << std::endl;
  out << "  " << program_name << " --config <config.yaml> [image] [options]" << std::endl;
  out << "  " << program_name << " --batch <directory> [options]" << std::endl;
  out << "  " << program_name << " --stereo <left> <right> [options]" << std::endl;
  out << "  " << program_name << " --sequence <dir|video> [--output dir] [--config yaml]" << std::endl;
  out << "  " << program_name
      << " --attend <dir|video> [--output dir] [--emit-scanpath path] [--config yaml] [--behavior name]" << std::endl;
  out << "  " << program_name << " --live <camera|video|dir> [--config configs/live.yaml] [--processors a,b]"
      << std::endl;
  out << "  " << program_name << " --help" << std::endl;
  out << std::endl;
  out << "Examples:" << std::endl;
  out << "  " << program_name << " data/test_images/input.png" << std::endl;
  out << "  " << program_name << " data/test_images/input.png --no-display" << std::endl;
  out << "  " << program_name << " data/test_images/input.png --debug=detailed --debug-print" << std::endl;
  out << "  " << program_name << " --config configs/thesis.yaml data/test_images/inputc.png --no-display" << std::endl;
  out << "  " << program_name << " --batch data/test_images/ --output results/" << std::endl;
  out << "  " << program_name << " --live 0 --config configs/live.yaml" << std::endl;
  out << "  " << program_name << " --live video.mp4 --processors roi-probe,region-descriptor" << std::endl;
  out << std::endl;
  out << "Options:" << std::endl;
  out << "  --no-display         Process without displaying windows (saves to results/)" << std::endl;
  out << "  --config             Load configuration from YAML file" << std::endl;
  out << "  --batch              Process all images in directory, save features separately" << std::endl;
  out << "  --stereo <l> <r>     Process a stereo pair (adds a disparity/depth channel)" << std::endl;
  out << "  --sequence <path>    Process a directory or video as a temporal stream (onset/motion)" << std::endl;
  out << "  --attend <path>      Run the full attention system (object files + behavior) over a stream" << std::endl;
  out << "  --emit-scanpath <p>  Write the scanpath JSON (with --attend)" << std::endl;
  out << "  --live <src>         Live demo: attention + object-file plugins on camera/video/dir (ESC quits)"
      << std::endl;
  out << "  --processors <a,b>   Object-file plugins for --live (default: region-descriptor)" << std::endl;
  out << "  --process-size <N>   Max side of the downscaled processing frame for --live (default: 480)" << std::endl;
  out << "  --frames <N>         Stop after N frames (--live headless with --no-display)" << std::endl;
  out << "  --output <dir>       Specify output directory for batch/sequence mode (default: input_dir/results_batch)"
      << std::endl;
  out << "  --emit-json <path>   Write result JSON + saliency map in the interchange format" << std::endl;
  out << "                       (see docs/INTERCHANGE_FORMAT.md; single-image and config modes)" << std::endl;
  out << "  --help, -h           Show this help and exit" << std::endl;
  out << std::endl;
  out << "Debug Options:" << std::endl;
  out << "  --debug[=LEVEL]      Enable debugging (levels: basic, detailed, verbose)" << std::endl;
  out << "                       Default: basic if no level specified" << std::endl;
  out << "  --debug-output <dir> Debug output directory (default: debug_output/)" << std::endl;
  out << "  --debug-print        Print debug info to console" << std::endl;
  out << "  --no-debug-save      Don't save debug images to disk" << std::endl;
}

int main(int argc, char** argv)
{
  // Check command line arguments
  if (argc < 2)
  {
    print_usage(argv[0]);
    return 1;
  }
  if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")
  {
    print_usage(argv[0], std::cout);
    return 0;
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
    else if (std::string(argv[1]) == "--live")
    {
      // Live demonstrator: <camera-index | video-file | frame-directory>.
      if (argc < 3)
      {
        std::cerr << "Error: --live requires a camera index, video file, or frame directory" << std::endl;
        print_usage(argv[0]);
        return 1;
      }
      std::string source = argv[2];
      std::string output_dir;
      bool display = true;
      int max_frames = 0;
      attention::system::LiveDemonstrator::Config live_cfg;
      // Built-in defaults unless --config is given (configs/live.yaml is the
      // tuned live profile: quarter-res symmetry, fixed field iterations).
      config = attention::config::ConfigLoader::create_default();
      for (int i = 3; i < argc; ++i)
      {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc)
        {
          config = attention::config::ConfigLoader::load(argv[++i]);
        }
        else if (arg == "--processors" && i + 1 < argc)
        {
          live_cfg.processors = split_csv(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc)
        {
          output_dir = argv[++i];
        }
        else if (arg == "--frames" && i + 1 < argc)
        {
          max_frames = std::stoi(argv[++i]);
        }
        else if (arg == "--process-size" && i + 1 < argc)
        {
          live_cfg.process_max_side = std::stoi(argv[++i]);
        }
        else if (arg == "--no-display")
        {
          display = false;
        }
      }
      live_cfg.system.pipeline = config.pipeline;
      process_live(source, live_cfg, display, max_frames, output_dir);
      return 0;
    }
    else if (std::string(argv[1]) == "--attend")
    {
      // Full active-vision system: second stage (object files) + behavior
      // (Exploration) over a temporal stream, emitting a scanpath.
      if (argc < 3)
      {
        std::cerr << "Error: --attend requires a directory or video path" << std::endl;
        print_usage(argv[0]);
        return 1;
      }
      std::string seq_path = argv[2];
      std::string output_dir;
      std::string scanpath_path;
      std::string behavior;
      bool user_config = false;
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
          user_config = true;
        }
        else if (arg == "--emit-scanpath" && i + 1 < argc)
        {
          scanpath_path = argv[++i];
        }
        else if (arg == "--behavior" && i + 1 < argc)
        {
          behavior = argv[++i];
        }
      }
      process_attend(seq_path, config.pipeline, output_dir, scanpath_path, user_config, behavior);
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
      // Command-line mode (backward compatible): argv[1] is an image path.
      // Reject unknown flags so a typo doesn't get loaded as an image.
      if (std::string(argv[1]).rfind("--", 0) == 0)
      {
        std::cerr << "Error: unknown option '" << argv[1] << "'" << std::endl << std::endl;
        print_usage(argv[0]);
        return 1;
      }
      // --config may appear anywhere in the trailing flags, so both
      // "attention --config X image" and "attention image --config X" work.
      // Load it FIRST (pre-pass) so the remaining flags — debug, --no-display —
      // apply on top regardless of order; the positional image always wins over
      // the config's input.image, and display defaults on only without a config.
      config = attention::config::ConfigLoader::create_default();
      bool config_loaded = false;
      for (int i = 2; i < argc; ++i)
      {
        if (std::string(argv[i]) == "--config")
        {
          if (i + 1 >= argc)
          {
            std::cerr << "Error: --config requires a YAML file path" << std::endl;
            print_usage(argv[0]);
            return 1;
          }
          config = attention::config::ConfigLoader::load(argv[i + 1]);
          config_loaded = true;
          break;
        }
      }
      config.input_image = argv[1]; // positional image overrides config input
      if (!config_loaded)
      {
        config.display = true; // interactive default for a bare single-image run
      }

      // Parse the remaining optional flags on top of the (possibly loaded) config.
      for (int i = 2; i < argc; ++i)
      {
        std::string arg = argv[i];

        if (arg == "--config")
        {
          ++i; // path already consumed by the pre-pass above
        }
        else if (arg == "--no-display")
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
