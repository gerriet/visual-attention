#pragma once

#include "attention/pipeline/attention_pipeline.h"
#include <string>

// Forward declare YAML::Node instead of using void*
namespace YAML
{
class Node;
}

namespace attention
{
namespace config
{

/**
 * ConfigLoader loads pipeline configuration from YAML files.
 *
 * Example YAML format:
 * ```yaml
 * input:
 *   image: "test.jpg"
 *
 * features:
 *   color:
 *     weight: 1.2
 *   intensity:
 *     weight: 0.8
 *   symmetry:
 *     weight: 1.0
 *
 * peaks:
 *   min_distance: 30
 *   threshold: 0.3
 *   max_count: 10
 *   enable_ior: true
 *   ior_radius: 50
 *   ior_strength: 0.8
 *
 * output:
 *   save_features: true
 *   save_saliency: true
 *   output_dir: "results/"
 *   display: false
 * ```
 */
class ConfigLoader
{
 public:
  /**
   * Complete configuration structure.
   */
  struct Config
  {
    // Input configuration
    std::string input_image;

    // Pipeline configuration
    pipeline::PipelineConfig pipeline;

    // Output configuration
    bool save_features = true;
    bool save_saliency = true;
    std::string output_dir = "results/";
    bool display = false;
  };

  /**
   * Load configuration from YAML file.
   * @param yaml_path Path to YAML configuration file
   * @return Loaded configuration
   * @throws std::runtime_error if file cannot be loaded or parsed
   */
  static Config load(const std::string& yaml_path);

  /**
   * Create default configuration.
   * @return Default configuration
   */
  static Config create_default();

 private:
  static void load_features(const YAML::Node& yaml_node, pipeline::PipelineConfig& config);
  static void load_peaks(const YAML::Node& yaml_node, pipeline::PipelineConfig& config);
  static void load_output(const YAML::Node& yaml_node, Config& config);
};

} // namespace config
} // namespace attention
