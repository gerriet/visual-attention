#include "attention/config/config_loader.h"
#include "attention/features/feature_registry.h"
#include <sstream>
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

    // Load pipeline strategy selection
    if (yaml["pipeline"])
    {
      load_pipeline(yaml["pipeline"], config.pipeline);
    }

    // Load feature set (overrides/extends the default set)
    if (yaml["features"])
    {
      load_features(yaml["features"], config.pipeline);
    }

    // Load peak detection parameters
    if (yaml["peaks"])
    {
      load_peaks(yaml["peaks"], config.pipeline);
    }

    // Load priority-map channels (M17); absent = pure bottom-up (thesis map)
    if (yaml["priority"])
    {
      load_priority(yaml["priority"], config.pipeline.priority);
    }

    // Load output configuration
    if (yaml["output"])
    {
      load_output(yaml["output"], config);
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

void ConfigLoader::load_priority(const YAML::Node& node, fusion::PriorityConfig& config)
{
  if (node["top_down_weight"])
  {
    config.top_down_weight = node["top_down_weight"].as<float>();
  }
  if (node["top_down_map"])
  {
    config.top_down_map_path = node["top_down_map"].as<std::string>();
  }
  if (node["target_color"])
  {
    config.target_color = node["target_color"].as<std::string>();
  }
  if (node["target_color_sigma"])
  {
    config.target_color_sigma = node["target_color_sigma"].as<float>();
  }
  if (node["object_value_weight"])
  {
    config.object_value_weight = node["object_value_weight"].as<float>();
  }
  if (node["location_history_weight"])
  {
    config.location_history_weight = node["location_history_weight"].as<float>();
  }
  if (node["location_history_decay"])
  {
    config.location_history_decay = node["location_history_decay"].as<float>();
  }
  if (node["location_history_radius"])
  {
    config.location_history_radius = node["location_history_radius"].as<float>();
  }
  if (node["object_value_per_selection"])
  {
    config.object_value_per_selection = node["object_value_per_selection"].as<float>();
  }
  if (node["object_value_decay"])
  {
    config.object_value_decay = node["object_value_decay"].as<float>();
  }
}

void ConfigLoader::load_pipeline(const YAML::Node& node, pipeline::PipelineConfig& config)
{
  if (node["fusion"])
  {
    config.fusion = node["fusion"].as<std::string>();
  }

  if (node["selection"])
  {
    config.selection = node["selection"].as<std::string>();
  }

  if (node["selection_params"])
  {
    config.selection_params_yaml = YAML::Dump(node["selection_params"]);
  }
}

void ConfigLoader::load_features(const YAML::Node& features, pipeline::PipelineConfig& config)
{
  features::register_builtin_features();
  const auto& registry = features::FeatureRegistry::instance();

  if (!features.IsMap())
  {
    throw std::runtime_error(
        "'features' must be a map of feature-name -> settings "
        "(e.g. \"features:\\n  color:\\n    weight: 1.0\"), not a list");
  }

  // Entries override the matching default spec; listed features that are not
  // in the default set are appended. Features stay enabled unless a config
  // says `enabled: false` — this keeps v1 config files (weight-only
  // overrides) working unchanged.
  for (const auto& entry : features)
  {
    const std::string type = entry.first.as<std::string>();
    const YAML::Node& node = entry.second;

    if (!registry.has(type))
    {
      std::ostringstream msg;
      msg << "Unknown feature '" << type << "' in config. Available:";
      for (const auto& name : registry.available())
      {
        msg << " " << name;
      }
      throw std::runtime_error(msg.str());
    }

    pipeline::FeatureSpec* spec = nullptr;
    for (auto& existing : config.features)
    {
      if (existing.type == type)
      {
        spec = &existing;
        break;
      }
    }
    if (spec == nullptr)
    {
      config.features.emplace_back(type);
      spec = &config.features.back();
    }

    if (node["enabled"])
    {
      spec->enabled = node["enabled"].as<bool>();
    }
    if (node["weight"])
    {
      spec->weight = node["weight"].as<float>();
    }
    if (node["params"])
    {
      spec->params_yaml = YAML::Dump(node["params"]);
    }
  }
}

void ConfigLoader::load_peaks(const YAML::Node& peaks, pipeline::PipelineConfig& config)
{
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

  // IOR parameters (enable_ior is the legacy alias for selection: ior)
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

void ConfigLoader::load_output(const YAML::Node& output, Config& config)
{
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
