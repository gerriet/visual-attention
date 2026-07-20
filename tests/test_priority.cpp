// M17 tests: the priority map — top-down and history/value channels over the
// bottom-up master map, and the guarantee that inactive channels leave the
// thesis map bit-identical.

#include "attention/config/config_loader.h"
#include "attention/fusion/priority_map.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdio>
#include <fstream>
#include <string>

using namespace attention;

namespace
{
// A frame with a red block on the left, a blue block on the right.
cv::Mat two_colour_frame()
{
  cv::Mat frame(80, 120, CV_8UC3, cv::Scalar(20, 20, 20));
  cv::rectangle(frame, cv::Rect(10, 30, 20, 20), cv::Scalar(40, 40, 220), cv::FILLED); // red (BGR)
  cv::rectangle(frame, cv::Rect(90, 30, 20, 20), cv::Scalar(220, 60, 40), cv::FILLED); // blue
  return frame;
}

std::string write_temp_yaml(const std::string& content)
{
  const std::string path = "test_priority_config.yaml";
  std::ofstream out(path);
  out << content;
  out.close();
  return path;
}
} // namespace

TEST_CASE("parse_colour: named, hex, and rejection", "[priority]")
{
  CHECK(fusion::parse_colour("red") == cv::Vec3b(0, 0, 220));
  CHECK(fusion::parse_colour("#0080ff") == cv::Vec3b(255, 128, 0)); // BGR
  CHECK_THROWS_AS(fusion::parse_colour("not-a-colour"), std::runtime_error);
  // A #-prefixed non-hex spec still surfaces the friendly error, not a raw
  // stoi std::invalid_argument.
  CHECK_THROWS_WITH(fusion::parse_colour("#gggggg"), Catch::Matchers::ContainsSubstring("Unparseable"));
}

TEST_CASE("inactive top-down channel ignores a stale/missing map path", "[priority]")
{
  // weight 0 = channel off: a leftover (here missing) map path must not throw
  // at construction.
  fusion::PriorityConfig config;
  config.top_down_map_path = "no/such/map.png";
  CHECK_NOTHROW(fusion::TopDownChannel(config));
}

TEST_CASE("location history term is size-guarded against a resolution change", "[priority]")
{
  fusion::PriorityConfig config;
  config.location_history_weight = 1.0f;
  fusion::HistoryChannels history(config);
  history.decay_and_record(cv::Point(10, 10), cv::Size(120, 80), true);

  // A later frame at a different resolution must not throw a size mismatch.
  cv::Mat smaller(40, 60, CV_32F, cv::Scalar(0.5f));
  CHECK_NOTHROW(history.apply(smaller, {}));
}

TEST_CASE("inactive channels leave the map untouched (bit-identical default)", "[priority]")
{
  fusion::PriorityConfig config; // all weights 0
  const fusion::TopDownChannel top_down(config);
  const fusion::HistoryChannels history(config);

  cv::Mat map(80, 120, CV_32F, cv::Scalar(0.5f));
  const cv::Mat frame = two_colour_frame();

  // Same underlying data, not merely equal values: the default path must not
  // even copy.
  CHECK(top_down.apply(map, frame).data == map.data);
  CHECK(history.apply(map, {}).data == map.data);
}

TEST_CASE("target-colour channel boosts the target's region", "[priority]")
{
  fusion::PriorityConfig config;
  config.top_down_weight = 2.0f;
  config.target_color = "red";
  const fusion::TopDownChannel top_down(config);

  const cv::Mat frame = two_colour_frame();
  cv::Mat flat(frame.size(), CV_32F, cv::Scalar(0.5f)); // no bottom-up preference
  const cv::Mat priority = top_down.apply(flat, frame);

  const float at_red = priority.at<float>(40, 20);
  const float at_blue = priority.at<float>(40, 100);
  CHECK(at_red > at_blue + 0.3f); // decisively higher, not merely above
}

TEST_CASE("external top-down map file is loaded and fused", "[priority]")
{
  // A relevance map bright only in the top-left quadrant.
  cv::Mat relevance(40, 60, CV_8U, cv::Scalar(0));
  cv::rectangle(relevance, cv::Rect(0, 0, 30, 20), cv::Scalar(255), cv::FILLED);
  const std::string map_path = "test_priority_topdown.png";
  cv::imwrite(map_path, relevance);

  fusion::PriorityConfig config;
  config.top_down_weight = 2.0f;
  config.top_down_map_path = map_path;
  const fusion::TopDownChannel top_down(config);

  cv::Mat flat(80, 120, CV_32F, cv::Scalar(0.5f));
  const cv::Mat priority = top_down.apply(flat, two_colour_frame());
  CHECK(priority.at<float>(10, 10) > priority.at<float>(70, 110) + 0.3f);

  std::remove(map_path.c_str());

  // A missing file fails fast, not silently.
  fusion::PriorityConfig missing;
  missing.top_down_weight = 1.0f;
  missing.top_down_map_path = "no/such/map.png";
  CHECK_THROWS_AS(fusion::TopDownChannel(missing), std::runtime_error);
}

TEST_CASE("history channels: location decay and object value facilitation", "[priority]")
{
  fusion::PriorityConfig config;
  config.object_value_weight = 1.0f;
  config.location_history_weight = 1.0f;
  config.location_history_radius = 6.0f;
  fusion::HistoryChannels history(config);

  const cv::Size size(120, 80);
  history.decay_and_record(cv::Point(30, 40), size, true);
  const float just_recorded = history.location_map().at<float>(40, 30);
  CHECK(just_recorded > 0.9f);
  history.decay_and_record(cv::Point(), size, false); // no focus: decay only
  CHECK(history.location_map().at<float>(40, 30) < just_recorded);

  // An object file with accrued value lifts priority at its current position.
  system::ObjectFile object;
  object.label = 1;
  object.centroid = cv::Point(90, 40);
  object.bbox = cv::Rect(80, 30, 20, 20);
  object.value = 1.0f;

  cv::Mat flat(size, CV_32F, cv::Scalar(0.5f));
  const cv::Mat priority = history.apply(flat, {object});
  CHECK(priority.at<float>(40, 90) > priority.at<float>(10, 10));
}

TEST_CASE("priority YAML block reaches the pipeline config", "[priority][config]")
{
  const std::string path = write_temp_yaml(
      "priority:\n"
      "  top_down_weight: 1.5\n"
      "  target_color: red\n"
      "  object_value_weight: 0.6\n"
      "  location_history_weight: 0.2\n");
  const auto config = config::ConfigLoader::load(path);
  std::remove(path.c_str());

  CHECK(config.pipeline.priority.top_down_weight == 1.5f);
  CHECK(config.pipeline.priority.target_color == "red");
  CHECK(config.pipeline.priority.object_value_weight == 0.6f);
  CHECK(config.pipeline.priority.location_history_weight == 0.2f);
  CHECK(config.pipeline.priority.top_down_active());
  CHECK(config.pipeline.priority.history_active());
}
