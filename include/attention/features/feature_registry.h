#pragma once

#include "attention/features/feature_extractor.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace YAML
{
class Node;
}

namespace attention
{
namespace features
{

/**
 * FeatureRegistry maps feature type names to factories so the feature set is
 * fully config-driven: a YAML config names the features to run and passes
 * their parameters; nothing about the set is hardcoded in the pipeline.
 *
 * Registration is explicit via register_builtin_features() — deliberate, since
 * self-registering translation units get dropped by the linker in static
 * libraries. New features (stereo, onset, ...) add themselves there.
 */
class FeatureRegistry
{
 public:
  /**
   * Factory function: builds an extractor from feature-specific YAML params.
   * The node may be a null/empty node, meaning "all defaults".
   */
  using Factory = std::function<std::unique_ptr<FeatureExtractor>(const YAML::Node& params)>;

  static FeatureRegistry& instance();

  void add(const std::string& type, Factory factory);

  /**
   * Create an extractor by type name.
   * @param type Registry key (e.g. "color", "symmetry")
   * @param params Feature-specific parameters (empty node = defaults)
   * @throws std::runtime_error for unknown types, listing what is available
   */
  std::unique_ptr<FeatureExtractor> create(const std::string& type, const YAML::Node& params) const;

  bool has(const std::string& type) const { return factories_.count(type) > 0; }
  std::vector<std::string> available() const;

 private:
  std::map<std::string, Factory> factories_;
};

/**
 * Register all built-in features (color, intensity, orientation,
 * eccentricity, symmetry). Idempotent; called by the pipeline and the config
 * loader before any registry lookup. Factory defaults reproduce the
 * pre-registry pipeline behavior exactly.
 */
void register_builtin_features();

} // namespace features
} // namespace attention
