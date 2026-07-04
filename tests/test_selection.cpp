// Selection strategy tests on synthetic saliency maps.

#include "attention/selection/selection_strategy.h"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using namespace attention;

namespace
{

// A map with two Gaussian-ish blobs: strong at (50, 50), weaker at (150, 150)
cv::Mat two_blob_map()
{
  cv::Mat map = cv::Mat::zeros(200, 200, CV_32F);
  cv::circle(map, cv::Point(50, 50), 6, cv::Scalar(1.0f), -1);
  cv::circle(map, cv::Point(150, 150), 6, cv::Scalar(0.8f), -1);
  cv::GaussianBlur(map, map, cv::Size(15, 15), 4.0);
  cv::normalize(map, map, 0.0f, 1.0f, cv::NORM_MINMAX);
  return map;
}

} // namespace

TEST_CASE("IOR selection visits distinct locations in decreasing order", "[selection]")
{
  selection::SelectionParams params;
  params.threshold = 0.2f;
  params.max_count = 5;
  params.ior_radius = 25;
  params.ior_strength = 0.9f;

  auto strategy = selection::create_selection_strategy("ior", params, YAML::Node());
  core::RunState state;
  auto peaks = strategy->select(two_blob_map(), state);

  REQUIRE(peaks.size() >= 2);
  // First fixation on the strong blob, second on the weak one — not a repeat
  CHECK(cv::norm(peaks[0].location - cv::Point(50, 50)) < 10);
  CHECK(cv::norm(peaks[1].location - cv::Point(150, 150)) < 10);
  CHECK(cv::norm(peaks[0].location - peaks[1].location) > params.ior_radius);
  CHECK(peaks[0].value >= peaks[1].value);
}

TEST_CASE("NMS selection enforces minimum distance", "[selection]")
{
  selection::SelectionParams params;
  params.threshold = 0.2f;
  params.max_count = 10;

  core::RunState state;

  SECTION("distant blobs give two peaks")
  {
    params.min_distance = 30;
    auto peaks = selection::create_selection_strategy("nms", params, YAML::Node())->select(two_blob_map(), state);
    REQUIRE(peaks.size() == 2);
    CHECK(peaks[0].value >= peaks[1].value);
  }

  SECTION("min_distance larger than blob spacing suppresses the weaker peak")
  {
    params.min_distance = 180;
    auto peaks = selection::create_selection_strategy("nms", params, YAML::Node())->select(two_blob_map(), state);
    REQUIRE(peaks.size() == 1);
    CHECK(cv::norm(peaks[0].location - cv::Point(50, 50)) < 10);
  }
}

TEST_CASE("neural-field selection settles on blobs in salience order", "[selection][neural-field]")
{
  selection::SelectionParams params;
  params.max_count = 5;
  params.ior_radius = 50;
  params.ior_strength = 0.95f;

  auto strategy = selection::create_selection_strategy("neural-field", params, YAML::Node());
  core::RunState state;
  auto peaks = strategy->select(two_blob_map(), state);

  REQUIRE(peaks.size() >= 2);
  // Field dynamics should settle first on the strong blob, then (after
  // inhibition of return) on the weak one
  CHECK(cv::norm(peaks[0].location - cv::Point(50, 50)) < 15);
  CHECK(cv::norm(peaks[1].location - cv::Point(150, 150)) < 15);
  // Field state persisted for stream continuation
  CHECK_FALSE(state.field_activity.empty());
}

TEST_CASE("neural-field selection stays quiet on an empty map", "[selection][neural-field]")
{
  selection::SelectionParams params;
  params.max_count = 5;

  auto strategy = selection::create_selection_strategy("neural-field", params, YAML::Node());
  core::RunState state;
  auto peaks = strategy->select(cv::Mat::zeros(200, 200, CV_32F), state);

  CHECK(peaks.empty());
}
