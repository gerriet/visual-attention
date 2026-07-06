// Config system tests: the YAML loader drives the full pipeline composition
// (feature set, weights, params, fusion/selection strategies).

#include "attention/config/config_loader.h"
#include "attention/pipeline/attention_pipeline.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using attention::config::ConfigLoader;
using attention::pipeline::FeatureSpec;

namespace
{

fs::path source_dir()
{
  return fs::path(ATTENTION_SOURCE_DIR);
}

const FeatureSpec* find_spec(const std::vector<FeatureSpec>& specs, const std::string& type)
{
  for (const auto& spec : specs)
  {
    if (spec.type == type)
    {
      return &spec;
    }
  }
  return nullptr;
}

int enabled_count(const std::vector<FeatureSpec>& specs)
{
  int count = 0;
  for (const auto& spec : specs)
  {
    if (spec.enabled)
    {
      count++;
    }
  }
  return count;
}

// Write a temp YAML file and return its path
fs::path write_temp_config(const std::string& contents)
{
  fs::path path = fs::temp_directory_path() / "attention_test_config.yaml";
  std::ofstream out(path);
  out << contents;
  return path;
}

} // namespace

TEST_CASE("default config runs all five features with NMS", "[config]")
{
  auto config = ConfigLoader::create_default();
  CHECK(enabled_count(config.pipeline.features) == 5);
  CHECK(config.pipeline.effective_selection() == "nms");
  CHECK(config.pipeline.fusion == "weighted-sum");
}

TEST_CASE("thesis profile enables the dissertation feature set with neural-field selection", "[config]")
{
  auto config = ConfigLoader::load((source_dir() / "configs" / "thesis.yaml").string());

  CHECK(enabled_count(config.pipeline.features) == 3);
  CHECK(find_spec(config.pipeline.features, "color")->enabled);
  CHECK(find_spec(config.pipeline.features, "eccentricity")->enabled);
  CHECK(find_spec(config.pipeline.features, "symmetry")->enabled);
  CHECK_FALSE(find_spec(config.pipeline.features, "intensity")->enabled);
  CHECK_FALSE(find_spec(config.pipeline.features, "orientation")->enabled);
  CHECK(config.pipeline.effective_selection() == "neural-field");
  CHECK_FALSE(config.pipeline.selection_params_yaml.empty());

  // The profile must actually construct
  attention::pipeline::AttentionPipeline pipeline(config.pipeline);
}

TEST_CASE("modern profile enables all features", "[config]")
{
  auto config = ConfigLoader::load((source_dir() / "configs" / "modern.yaml").string());
  CHECK(enabled_count(config.pipeline.features) == 5);
  CHECK(config.pipeline.effective_selection() == "nms");
}

TEST_CASE("every shipped config loads and constructs a pipeline", "[config]")
{
  for (const auto& entry : fs::directory_iterator(source_dir() / "configs"))
  {
    if (entry.path().extension() != ".yaml")
    {
      continue;
    }
    DYNAMIC_SECTION("config: " << entry.path().filename().string())
    {
      auto config = ConfigLoader::load(entry.path().string());
      attention::pipeline::AttentionPipeline pipeline(config.pipeline);
    }
  }
}

TEST_CASE("all feature weights are parsed, none silently dropped", "[config]")
{
  auto path = write_temp_config(R"(
features:
  color: { weight: 0.5 }
  intensity: { weight: 0.6 }
  orientation: { weight: 0.7 }
  eccentricity: { weight: 0.8 }
  symmetry: { weight: 0.9 }
)");
  auto config = ConfigLoader::load(path.string());

  CHECK(find_spec(config.pipeline.features, "color")->weight == 0.5f);
  CHECK(find_spec(config.pipeline.features, "intensity")->weight == 0.6f);
  CHECK(find_spec(config.pipeline.features, "orientation")->weight == 0.7f);
  CHECK(find_spec(config.pipeline.features, "eccentricity")->weight == 0.8f);
  CHECK(find_spec(config.pipeline.features, "symmetry")->weight == 0.9f);
  std::remove(path.string().c_str());
}

TEST_CASE("feature params reach the extractor factory", "[config]")
{
  auto path = write_temp_config(R"(
features:
  symmetry:
    params:
      num_orientations: 8
      scales:
        - { level: 1, min_radius: 4, max_radius: 12, threshold: 0.4 }
)");
  auto config = ConfigLoader::load(path.string());

  const auto* spec = find_spec(config.pipeline.features, "symmetry");
  REQUIRE(spec != nullptr);
  CHECK_FALSE(spec->params_yaml.empty());

  // Must construct an extractor without throwing
  attention::pipeline::AttentionPipeline pipeline(config.pipeline);
  std::remove(path.string().c_str());
}

TEST_CASE("unknown feature and strategy names are rejected with clear errors", "[config]")
{
  auto path = write_temp_config("features:\n  warp_drive: { weight: 1.0 }\n");
  CHECK_THROWS_WITH(ConfigLoader::load(path.string()), Catch::Matchers::ContainsSubstring("warp_drive") &&
                                                           Catch::Matchers::ContainsSubstring("Available"));
  std::remove(path.string().c_str());

  attention::pipeline::PipelineConfig bad_selection;
  bad_selection.selection = "quantum";
  CHECK_THROWS(attention::pipeline::AttentionPipeline(bad_selection));

  attention::pipeline::PipelineConfig bad_fusion;
  bad_fusion.fusion = "psychic";
  CHECK_THROWS(attention::pipeline::AttentionPipeline(bad_fusion));
}
