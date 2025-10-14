#include "attention/config/config_loader.h"
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace attention
{
namespace config
{

ConfigLoader::Config ConfigLoader::load(const std::string& yaml_path)
{
  try
  {
    YAML::Node yaml = YAML::LoadFile(yaml_path);
    Config config = create_default();

    // Load input configuration
    if (yaml["input"])
    {
      if (yaml["input"]["image"])
      {
        config.input_image = yaml["input"]["image"].as<std::string>();
      }
    }

    // Load feature weights
    if (yaml["features"])
    {
      YAML::Node features = yaml["features"];
      load_features(&features, config.pipeline);
    }

    // Load peak detection parameters
    if (yaml["peaks"])
    {
      YAML::Node peaks = yaml["peaks"];
      load_peaks(&peaks, config.pipeline);
    }

    // Load output configuration
    if (yaml["output"])
    {
      YAML::Node output = yaml["output"];
      load_output(&output, config);
    }

    return config;
  }
  catch (const YAML::Exception& e)
  {
    throw std::runtime_error("Failed to load config from '" + yaml_path + "': " + e.what());
  }
}

ConfigLoader::Config ConfigLoader::create_default()
{
  Config config;
  config.input_image = "";
  config.pipeline = pipeline::PipelineConfig();
  config.save_features = true;
  config.save_saliency = true;
  config.output_dir = "results/";
  config.display = false;
  return config;
}

void ConfigLoader::load_features(const void* yaml_node_ptr, pipeline::PipelineConfig& config)
{
  const YAML::Node& features = *static_cast<const YAML::Node*>(yaml_node_ptr);

  // Load individual feature weights
  if (features["color"] && features["color"]["weight"])
  {
    config.feature_weights["color"] = features["color"]["weight"].as<float>();
  }

  if (features["intensity"] && features["intensity"]["weight"])
  {
    config.feature_weights["intensity"] = features["intensity"]["weight"].as<float>();
  }

  if (features["symmetry"] && features["symmetry"]["weight"])
  {
    config.feature_weights["symmetry"] = features["symmetry"]["weight"].as<float>();
  }
}

void ConfigLoader::load_peaks(const void* yaml_node_ptr, pipeline::PipelineConfig& config)
{
  const YAML::Node& peaks = *static_cast<const YAML::Node*>(yaml_node_ptr);

  if (peaks["min_distance"])
  {
    config.peak_min_distance = peaks["min_distance"].as<int>();
  }

  if (peaks["threshold"])
  {
    config.peak_threshold = peaks["threshold"].as<float>();
  }

  if (peaks["max_count"])
  {
    config.peak_max_count = peaks["max_count"].as<int>();
  }

  // IOR parameters
  if (peaks["enable_ior"])
  {
    config.enable_ior = peaks["enable_ior"].as<bool>();
  }

  if (peaks["ior_radius"])
  {
    config.ior_radius = peaks["ior_radius"].as<int>();
  }

  if (peaks["ior_strength"])
  {
    config.ior_strength = peaks["ior_strength"].as<float>();
  }
}

void ConfigLoader::load_output(const void* yaml_node_ptr, Config& config)
{
  const YAML::Node& output = *static_cast<const YAML::Node*>(yaml_node_ptr);

  if (output["save_features"])
  {
    config.save_features = output["save_features"].as<bool>();
  }

  if (output["save_saliency"])
  {
    config.save_saliency = output["save_saliency"].as<bool>();
  }

  if (output["output_dir"])
  {
    config.output_dir = output["output_dir"].as<std::string>();
  }

  if (output["display"])
  {
    config.display = output["display"].as<bool>();
  }
}

} // namespace config
} // namespace attention
