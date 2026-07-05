#pragma once

#include <yaml-cpp/yaml.h>

namespace attention
{
namespace config
{

/**
 * Read a scalar YAML param if present, else keep the default. The shared
 * optional-key idiom for factories and loaders.
 */
template <typename T>
void read_param(const YAML::Node& params, const char* key, T& value)
{
  if (params && params[key])
  {
    value = params[key].as<T>();
  }
}

} // namespace config
} // namespace attention
