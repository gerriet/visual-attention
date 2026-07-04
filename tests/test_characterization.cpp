// Characterization tests: snapshot the current pipeline behavior on the
// reference images and fail when it drifts. These are refactor tripwires,
// not ground truth — when an intentional algorithm change lands, regenerate
// the golden data with:
//
//   ATTENTION_UPDATE_GOLDEN=1 ./tests/characterization_tests
//
// and commit the changes under tests/golden/ after reviewing them.

#include "attention/io/result_writer.h"
#include "attention/pipeline/attention_pipeline.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

namespace fs = std::filesystem;
using attention::pipeline::AttentionPipeline;

namespace
{

// Small thesis-era reference images; kept small so the suite stays fast
const char* const kImages[] = {"input", "inputc", "lena"};

// Tolerances leave headroom for reordered float arithmetic in refactors
// while still catching real algorithmic drift
constexpr double kMeanTolerance = 0.005;
constexpr double kMaxTolerance = 0.05;

fs::path source_dir()
{
  return fs::path(ATTENTION_SOURCE_DIR);
}

bool update_mode()
{
  return std::getenv("ATTENTION_UPDATE_GOLDEN") != nullptr;
}

void write_golden_map(const cv::Mat& map, const fs::path& path)
{
  cv::Mat map16;
  map.convertTo(map16, CV_16U, 65535.0);
  REQUIRE(cv::imwrite(path.string(), map16));
}

cv::Mat load_golden_map(const fs::path& path)
{
  cv::Mat map16 = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
  REQUIRE(!map16.empty());
  REQUIRE(map16.type() == CV_16U);
  cv::Mat map;
  map16.convertTo(map, CV_32F, 1.0 / 65535.0);
  return map;
}

void check_map(const cv::Mat& actual, const fs::path& golden_path)
{
  if (update_mode())
  {
    write_golden_map(actual, golden_path);
    return;
  }

  INFO("golden file: " << golden_path.string()
                       << " (regenerate with ATTENTION_UPDATE_GOLDEN=1 after intentional changes)");
  REQUIRE(fs::exists(golden_path));

  cv::Mat expected = load_golden_map(golden_path);
  REQUIRE(expected.size() == actual.size());

  cv::Mat diff;
  cv::absdiff(expected, actual, diff);
  double max_diff;
  cv::minMaxLoc(diff, nullptr, &max_diff);
  double mean_diff = cv::mean(diff)[0];

  INFO("mean abs diff: " << mean_diff << ", max abs diff: " << max_diff);
  CHECK(mean_diff < kMeanTolerance);
  CHECK(max_diff < kMaxTolerance);
}

} // namespace

TEST_CASE("pipeline characterization on reference images", "[characterization]")
{
  for (const char* stem : kImages)
  {
    DYNAMIC_SECTION("image: " << stem)
    {
      fs::path image_path = source_dir() / "data" / "test_images" / (std::string(stem) + ".png");
      REQUIRE(fs::exists(image_path));

      AttentionPipeline pipeline; // default configuration
      pipeline.load_image(image_path.string());
      pipeline.process();

      fs::path golden = source_dir() / "tests" / "golden" / stem;
      if (update_mode())
      {
        fs::create_directories(golden);
      }

      check_map(pipeline.get_saliency_map().map, golden / "saliency.png");
      for (const auto& feature : pipeline.get_features())
      {
        check_map(feature.data, golden / ("feature_" + feature.name + ".png"));
      }

      if (update_mode())
      {
        // Golden interchange result, consumed by the behavioral scanpath tests
        attention::io::ResultWriter::write(pipeline, (golden / "result.json").string());
      }
      else
      {
        CHECK(!pipeline.get_saliency_map().peaks.empty());
      }
    }
  }
}
