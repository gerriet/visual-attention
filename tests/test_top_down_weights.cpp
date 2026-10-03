// M20 tests: learned top-down feature weights (the VOCUS rule) — the learning
// rule's defining property, the geometric combination, the excitation/
// inhibition split, and the guarantee that an inactive config leaves the
// bottom-up map bit-identical.

#include "attention/config/config_loader.h"
#include "attention/fusion/top_down_weights.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <fstream>

using namespace attention;

namespace
{

// A feature map that is `inside` within `region` and `outside` elsewhere.
core::FeatureMap block_feature(const std::string& name, const cv::Rect& region, float inside, float outside)
{
  cv::Mat data(40, 60, CV_32F, cv::Scalar(outside));
  data(region).setTo(inside);
  return core::FeatureMap(name, data);
}

cv::Mat block_mask(const cv::Rect& region)
{
  cv::Mat mask = cv::Mat::zeros(40, 60, CV_8U);
  mask(region).setTo(255);
  return mask;
}

std::string write_temp_yaml(const std::string& content)
{
  const std::string path = "test_top_down_weights_config.yaml";
  std::ofstream out(path);
  out << content;
  out.close();
  return path;
}

} // namespace

TEST_CASE("the weight is the inside/outside ratio, not the inside level", "[top_down_weights]")
{
  const cv::Rect target(10, 10, 10, 10);

  // The defining property of the rule: `separating` is weak on the target in
  // absolute terms (0.4) but far weaker off it (0.1), so it separates — while
  // `everywhere` is twice as strong on the target (0.8) and just as strong off
  // it, so it says nothing about where the target is.
  std::vector<core::FeatureMap> features;
  features.push_back(block_feature("separating", target, 0.4f, 0.1f));
  features.push_back(block_feature("everywhere", target, 0.8f, 0.8f));

  const auto weights = fusion::learn_top_down_weights(features, block_mask(target));

  REQUIRE(weights.at("separating") > 3.0f);
  REQUIRE(weights.at("everywhere") == Catch::Approx(1.0f).margin(1e-3));
}

TEST_CASE("a feature quieter on the target than off it becomes inhibitory", "[top_down_weights]")
{
  const cv::Rect target(10, 10, 10, 10);
  std::vector<core::FeatureMap> features;
  features.push_back(block_feature("quiet_on_target", target, 0.1f, 0.5f));

  const auto weights = fusion::learn_top_down_weights(features, block_mask(target));
  REQUIRE(weights.at("quiet_on_target") < 1.0f);
}

TEST_CASE("weights are clamped, so one channel cannot swamp the sum", "[top_down_weights]")
{
  const cv::Rect target(10, 10, 10, 10);
  std::vector<core::FeatureMap> features;
  features.push_back(block_feature("only_on_target", target, 0.9f, 0.0f));

  const auto weights = fusion::learn_top_down_weights(features, block_mask(target), 10.0f);
  REQUIRE(weights.at("only_on_target") == Catch::Approx(10.0f));
}

TEST_CASE("a mask with no background is an error, not a silent neutral weight", "[top_down_weights]")
{
  std::vector<core::FeatureMap> features;
  features.push_back(block_feature("f", cv::Rect(0, 0, 60, 40), 1.0f, 1.0f));

  cv::Mat all = cv::Mat::ones(40, 60, CV_8U) * 255;
  REQUIRE_THROWS_AS(fusion::learn_top_down_weights(features, all), std::runtime_error);
  cv::Mat none = cv::Mat::zeros(40, 60, CV_8U);
  REQUIRE_THROWS_AS(fusion::learn_top_down_weights(features, none), std::runtime_error);
}

TEST_CASE("examples combine by the geometric mean, so a contradicted feature drops out", "[top_down_weights]")
{
  // The point of the geometric mean over an arithmetic one: these are ratios.
  // A feature that is 2x in one example and 0.5x in another carries no
  // consistent evidence and must come out neutral — the arithmetic mean would
  // make it 1.25 and keep it excitatory.
  std::vector<std::map<std::string, float>> examples = {
      {{"contradicted", 2.0f}, {"consistent", 4.0f}},
      {{"contradicted", 0.5f}, {"consistent", 1.0f}},
  };
  const auto combined = fusion::combine_top_down_weights(examples);

  REQUIRE(combined.at("contradicted") == Catch::Approx(1.0f).margin(1e-5));
  REQUIRE(combined.at("consistent") == Catch::Approx(2.0f).margin(1e-5)); // sqrt(4 * 1)
}

TEST_CASE("an inactive config returns the bottom-up map untouched", "[top_down_weights]")
{
  cv::Mat bottom_up(40, 60, CV_32F, cv::Scalar(0.3f));
  bottom_up.at<float>(5, 5) = 1.0f;
  std::vector<core::FeatureMap> features;
  features.push_back(block_feature("f", cv::Rect(10, 10, 10, 10), 0.9f, 0.1f));

  SECTION("factor zero")
  {
    fusion::TopDownWeightConfig config;
    config.weights = {{"f", 5.0f}};
    config.factor = 0.0f;
    const cv::Mat out = fusion::TopDownWeights(config).apply(bottom_up, features);
    REQUIRE(out.data == bottom_up.data); // the same cv::Mat, not merely equal
  }

  SECTION("no weights")
  {
    fusion::TopDownWeightConfig config;
    config.factor = 1.0f;
    const cv::Mat out = fusion::TopDownWeights(config).apply(bottom_up, features);
    REQUIRE(out.data == bottom_up.data);
  }

  SECTION("weights naming no present feature")
  {
    fusion::TopDownWeightConfig config;
    config.weights = {{"not_a_feature", 5.0f}};
    config.factor = 1.0f;
    const cv::Mat out = fusion::TopDownWeights(config).apply(bottom_up, features);
    REQUIRE(out.data == bottom_up.data);
  }
}

TEST_CASE("the top-down map prefers the excited feature over the inhibited one", "[top_down_weights]")
{
  const cv::Rect excited_region(5, 5, 10, 10);
  const cv::Rect inhibited_region(40, 20, 10, 10);
  std::vector<core::FeatureMap> features;
  features.push_back(block_feature("wanted", excited_region, 1.0f, 0.0f));
  features.push_back(block_feature("unwanted", inhibited_region, 1.0f, 0.0f));

  fusion::TopDownWeightConfig config;
  config.weights = {{"wanted", 4.0f}, {"unwanted", 0.25f}};
  config.factor = 1.0f;

  cv::Mat bottom_up = cv::Mat::zeros(40, 60, CV_32F);
  const cv::Mat top_down = fusion::TopDownWeights(config).top_down_map(bottom_up, features);

  REQUIRE_FALSE(top_down.empty());
  const float at_wanted = top_down.at<float>(excited_region.y + 5, excited_region.x + 5);
  const float at_unwanted = top_down.at<float>(inhibited_region.y + 5, inhibited_region.x + 5);
  REQUIRE(at_wanted > at_unwanted);
  // The inhibited feature's region must end up below the untouched background,
  // or "inhibition" is only a weaker excitation.
  const float at_background = top_down.at<float>(35, 30);
  REQUIRE(at_unwanted < at_background);
}

TEST_CASE("the target region inside a box is the salient part, not the rectangle", "[top_down_weights]")
{
  cv::Mat saliency = cv::Mat::zeros(40, 60, CV_32F);
  saliency(cv::Rect(12, 12, 6, 6)).setTo(1.0f); // the object
  const cv::Rect box(10, 10, 20, 20);           // a box with plenty of background

  const cv::Mat mask = fusion::salient_region_in_box(saliency, box);
  REQUIRE(cv::countNonZero(mask) == 36);
  REQUIRE(mask.at<uchar>(14, 14) != 0);
  REQUIRE(mask.at<uchar>(26, 26) == 0); // background inside the box is excluded
}

TEST_CASE("a second bright thing in the box does not join the target region", "[top_down_weights]")
{
  cv::Mat saliency = cv::Mat::zeros(40, 60, CV_32F);
  saliency(cv::Rect(12, 12, 6, 6)).setTo(1.0f); // the peak
  saliency(cv::Rect(24, 24, 4, 4)).setTo(0.9f); // a distractor, also above threshold
  const cv::Mat mask = fusion::salient_region_in_box(saliency, cv::Rect(10, 10, 20, 20));

  REQUIRE(mask.at<uchar>(14, 14) != 0);
  REQUIRE(mask.at<uchar>(26, 26) == 0);
}

TEST_CASE("a flat box falls back to the whole rectangle rather than to nothing", "[top_down_weights]")
{
  cv::Mat saliency(40, 60, CV_32F, cv::Scalar(0.5f));
  const cv::Mat mask = fusion::salient_region_in_box(saliency, cv::Rect(10, 10, 20, 20));
  REQUIRE(cv::countNonZero(mask) == 400);
}

TEST_CASE("config loads the weights and the factor", "[top_down_weights]")
{
  const std::string path = write_temp_yaml(
      "pipeline:\n"
      "  fusion: weighted-sum\n"
      "priority:\n"
      "  top_down_factor: 0.25\n"
      "  top_down_weights:\n"
      "    color: 2.5\n"
      "    symmetry: 0.4\n");
  const auto config = config::ConfigLoader::load(path);
  std::remove(path.c_str());

  REQUIRE(config.pipeline.top_down_weights.factor == Catch::Approx(0.25f));
  REQUIRE(config.pipeline.top_down_weights.weights.at("color") == Catch::Approx(2.5f));
  REQUIRE(config.pipeline.top_down_weights.weights.at("symmetry") == Catch::Approx(0.4f));
  REQUIRE(config.pipeline.top_down_weights.active());
}

TEST_CASE("a config without the block leaves the channel inactive", "[top_down_weights]")
{
  const std::string path = write_temp_yaml("pipeline:\n  fusion: weighted-sum\n");
  const auto config = config::ConfigLoader::load(path);
  std::remove(path.c_str());

  REQUIRE_FALSE(config.pipeline.top_down_weights.active());
  REQUIRE(config.pipeline.top_down_weights.factor == Catch::Approx(0.0f));
}
