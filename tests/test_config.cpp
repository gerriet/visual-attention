// Config system tests: the YAML loader drives the full pipeline composition
// (feature set, weights, params, fusion/selection strategies).

#include "attention/config/config_loader.h"
#include "attention/pipeline/attention_pipeline.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
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

// `instance` disambiguates when one type appears several times under its own
// names; empty matches the first spec of that type, as before.
const FeatureSpec* find_spec(const std::vector<FeatureSpec>& specs, const std::string& type,
                             const std::string& instance = "")
{
  for (const auto& spec : specs)
  {
    if (spec.type != type)
    {
      continue;
    }
    if (instance.empty() || spec.name == instance)
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
  auto config = ConfigLoader::load((source_dir() / "configs" / "thesis" / "thesis.yaml").string());

  CHECK(enabled_count(config.pipeline.features) == 3);
  // The thesis's own colour feature, not the Itti-Koch-style `color`
  CHECK(find_spec(config.pipeline.features, "color-munsell")->enabled);
  CHECK_FALSE(find_spec(config.pipeline.features, "color")->enabled);
  CHECK(find_spec(config.pipeline.features, "eccentricity")->enabled);
  CHECK(find_spec(config.pipeline.features, "symmetry")->enabled);
  CHECK_FALSE(find_spec(config.pipeline.features, "intensity")->enabled);
  CHECK_FALSE(find_spec(config.pipeline.features, "orientation")->enabled);
  CHECK(config.pipeline.effective_selection() == "neural-field");
  CHECK_FALSE(config.pipeline.selection_params_yaml.empty());

  // The profile must actually construct
  attention::pipeline::AttentionPipeline pipeline(config.pipeline);
}

TEST_CASE("modern profile: the best measured stage 1 (docs/adr/0005)", "[config]")
{
  auto config = ConfigLoader::load((source_dir() / "configs" / "modern.yaml").string());
  // What the 2026-09 ablation chose; this case changes whenever a measurement
  // changes the profile — it documents the choice, it does not protect it.
  // Today that is the dissertation's feature set (validated on two benchmarks).
  CHECK(enabled_count(config.pipeline.features) == 3);
  CHECK(find_spec(config.pipeline.features, "color-munsell")->enabled);
  CHECK(find_spec(config.pipeline.features, "eccentricity")->enabled);
  CHECK(find_spec(config.pipeline.features, "symmetry")->enabled);
  CHECK_FALSE(find_spec(config.pipeline.features, "color")->enabled);
  CHECK(config.pipeline.effective_selection() == "neural-field");
}

TEST_CASE("thesis-extended profile enables the reimplementation's original five features", "[config]")
{
  auto config = ConfigLoader::load((source_dir() / "configs" / "thesis-extended.yaml").string());
  CHECK(enabled_count(config.pipeline.features) == 5);
  CHECK(config.pipeline.effective_selection() == "nms");
}

TEST_CASE("thesis-track configs enable only components of the dissertation", "[config][thesis-track]")
{
  // docs/adr/0005: configs/thesis/ is the replication track. A modern component
  // (an Itti-Koch channel, an alternative operator, a non-thesis selection
  // backend) never enters it — it gets a sibling profile outside instead.
  const std::set<std::string> thesis_features = {"color-munsell", "eccentricity", "symmetry", "stereo", "onset"};
  // nms is allowed: under --attend the second stage selects, the pipeline's own
  // selection is unused (configs/thesis/attend.yaml)
  const std::set<std::string> thesis_selection = {"neural-field", "neural-field-3d", "nms"};

  int profiles = 0;
  for (const auto& entry : fs::directory_iterator(source_dir() / "configs" / "thesis"))
  {
    if (entry.path().extension() != ".yaml")
    {
      continue;
    }
    ++profiles;
    DYNAMIC_SECTION("config: " << entry.path().filename().string())
    {
      auto config = ConfigLoader::load(entry.path().string());
      for (const auto& spec : config.pipeline.features)
      {
        if (spec.enabled)
        {
          INFO("feature: " << spec.type);
          CHECK(thesis_features.count(spec.type) == 1);
        }
      }
      CHECK(thesis_selection.count(config.pipeline.effective_selection()) == 1);
    }
  }
  CHECK(profiles >= 3); // thesis, stereo, attend
}

TEST_CASE("every shipped config loads and constructs a pipeline", "[config]")
{
  // Recursive: the thesis track and the ablation arms live in subdirectories
  for (const auto& entry : fs::recursive_directory_iterator(source_dir() / "configs"))
  {
    if (entry.path().extension() != ".yaml")
    {
      continue;
    }
    DYNAMIC_SECTION("config: " << fs::relative(entry.path(), source_dir() / "configs").string())
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

TEST_CASE("a feature type can be instantiated several times under its own names", "[config]")
{
  // M20 needs this: a learned weight vector can only say "this target is
  // horizontal" if horizontal is a channel of its own, so the same extractor
  // has to be able to appear four times with different parameters.
  auto path = write_temp_config(
      "features:\n"
      "  orientation: { enabled: false }\n"
      "  ori_0: { type: orientation, params: { num_orientations: 4, "
      "only_orientation: 0 }, weight: 0.5 }\n"
      "  ori_1: { type: orientation, params: { num_orientations: 4, "
      "only_orientation: 1 }, weight: 1.5 }\n");
  const auto config = ConfigLoader::load(path.string());
  std::remove(path.string().c_str());

  const auto* zero = find_spec(config.pipeline.features, "orientation", "ori_0");
  const auto* one = find_spec(config.pipeline.features, "orientation", "ori_1");
  REQUIRE(zero != nullptr);
  REQUIRE(one != nullptr);
  CHECK(zero->weight == 0.5f);
  CHECK(one->weight == 1.5f);
  // Distinct instances, not one spec overwritten by the other.
  CHECK(zero != one);

  attention::pipeline::AttentionPipeline pipeline(config.pipeline);
  cv::Mat frame(60, 80, CV_8UC3, cv::Scalar(30, 30, 30));
  cv::rectangle(frame, cv::Rect(20, 20, 24, 8), cv::Scalar(230, 230, 230), cv::FILLED);
  pipeline.load_image(frame);
  pipeline.process();

  std::set<std::string> names;
  for (const auto& feature : pipeline.get_features())
  {
    names.insert(feature.name);
  }
  // Both instances reach the fused map under their own names; without the
  // instance name they would collide on the extractor's "orientation".
  CHECK(names.count("ori_0") == 1);
  CHECK(names.count("ori_1") == 1);
  CHECK(names.count("orientation") == 0);
}

TEST_CASE("a config without instance names behaves exactly as before", "[config]")
{
  auto path = write_temp_config("features:\n  orientation: { weight: 2.0 }\n");
  const auto config = ConfigLoader::load(path.string());
  std::remove(path.string().c_str());

  const auto* spec = find_spec(config.pipeline.features, "orientation");
  REQUIRE(spec != nullptr);
  CHECK(spec->weight == 2.0f);
  CHECK(spec->name.empty()); // no alias -> the extractor's own name is used
}
