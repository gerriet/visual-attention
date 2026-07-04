#include "attention/features/feature_registry.h"
#include "attention/features/color_feature.h"
#include "attention/features/eccentricity_feature.h"
#include "attention/features/intensity_feature.h"
#include "attention/features/orientation_feature.h"
#include "attention/features/symmetry_feature.h"
#include <sstream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace attention
{
namespace features
{

FeatureRegistry& FeatureRegistry::instance()
{
  static FeatureRegistry registry;
  return registry;
}

void FeatureRegistry::add(const std::string& type, Factory factory)
{
  factories_[type] = std::move(factory);
}

std::unique_ptr<FeatureExtractor> FeatureRegistry::create(const std::string& type, const YAML::Node& params) const
{
  auto it = factories_.find(type);
  if (it == factories_.end())
  {
    std::ostringstream msg;
    msg << "Unknown feature type '" << type << "'. Available features:";
    for (const auto& name : available())
    {
      msg << " " << name;
    }
    throw std::runtime_error(msg.str());
  }
  return it->second(params);
}

std::vector<std::string> FeatureRegistry::available() const
{
  std::vector<std::string> names;
  for (const auto& pair : factories_)
  {
    names.push_back(pair.first);
  }
  return names;
}

namespace
{

// Read a scalar param if present, else keep the default
template <typename T>
void read(const YAML::Node& params, const char* key, T& value)
{
  if (params && params[key])
  {
    value = params[key].as<T>();
  }
}

} // namespace

void register_builtin_features()
{
  static bool registered = false;
  if (registered)
  {
    return;
  }
  registered = true;

  auto& registry = FeatureRegistry::instance();

  registry.add("color",
               [](const YAML::Node& params)
               {
                 ColorFeature::Config config;
                 read(params, "pyramid_levels", config.pyramid_levels);
                 read(params, "normalize_channels", config.normalize_channels);
                 return std::make_unique<ColorFeature>(config);
               });

  registry.add("intensity",
               [](const YAML::Node& params)
               {
                 IntensityFeature::Config config;
                 read(params, "pyramid_levels", config.pyramid_levels);
                 return std::make_unique<IntensityFeature>(config);
               });

  registry.add("orientation",
               [](const YAML::Node& params)
               {
                 OrientationFeature::Config config;
                 read(params, "num_orientations", config.num_orientations);
                 read(params, "wavelength", config.wavelength);
                 read(params, "bandwidth", config.bandwidth);
                 read(params, "compute_at_scale", config.compute_at_scale);
                 return std::make_unique<OrientationFeature>(config);
               });

  registry.add("eccentricity",
               [](const YAML::Node& params)
               {
                 EccentricityFeature::Config config;
                 // Auto scale selection by image size, as the pipeline
                 // previously hardcoded (full res <= 640px, else quarter res)
                 config.compute_at_scale = -1;
                 read(params, "edge_threshold", config.edge_threshold);
                 read(params, "min_area", config.min_area);
                 read(params, "max_area", config.max_area);
                 read(params, "variance_threshold", config.variance_threshold);
                 read(params, "compute_at_scale", config.compute_at_scale);
                 return std::make_unique<EccentricityFeature>(config);
               });

  registry.add("symmetry",
               [](const YAML::Node& params)
               {
                 // Defaults reproduce the previously hardcoded pipeline setup:
                 // 12 orientations, multi-scale, size-adaptive scale schedule
                 SymmetryFeature::Config config;
                 config.num_orientations = 12;
                 config.wavelength = 8.0;
                 config.bandwidth = 1.0;
                 config.use_multi_scale = true;
                 config.auto_scale_schedule = true;
                 config.scales.clear();

                 read(params, "num_orientations", config.num_orientations);
                 read(params, "wavelength", config.wavelength);
                 read(params, "bandwidth", config.bandwidth);
                 read(params, "use_multi_scale", config.use_multi_scale);

                 if (params && params["scales"])
                 {
                   config.auto_scale_schedule = false;
                   for (const auto& scale : params["scales"])
                   {
                     SymmetryFeature::ScaleConfig sc;
                     sc.pyramid_level = scale["level"].as<int>();
                     if (scale["min_radius"])
                       sc.min_radius = scale["min_radius"].as<int>();
                     if (scale["max_radius"])
                       sc.max_radius = scale["max_radius"].as<int>();
                     if (scale["radius_step"])
                       sc.radius_step = scale["radius_step"].as<int>();
                     if (scale["width"])
                       sc.width = scale["width"].as<int>();
                     if (scale["threshold"])
                       sc.symmetry_threshold = scale["threshold"].as<float>();
                     config.scales.push_back(sc);
                   }
                 }

                 return std::make_unique<SymmetryFeature>(config);
               });
}

} // namespace features
} // namespace attention
