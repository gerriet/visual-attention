// M8 tests: object-file processors (plugins) and the LiveDemonstrator layer —
// the registry, the built-in processors on real/empty ROIs, and the
// native-vs-processing resolution split with scaled-back annotations.

#include "attention/system/live_demonstrator.h"
#include "attention/system/processor.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace attention;
using Catch::Matchers::ContainsSubstring;

namespace
{
system::ObjectFile file_at(int label, int x, int y)
{
  system::ObjectFile f;
  f.label = label;
  f.centroid = cv::Point(x, y);
  f.bbox = cv::Rect(x - 10, y - 10, 20, 20);
  f.size = 400;
  f.saliency = 0.8f;
  return f;
}
} // namespace

TEST_CASE("built-in processors are registered and creatable", "[system][processor]")
{
  system::register_builtin_processors();
  auto& registry = system::ProcessorRegistry::instance();
  CHECK(registry.has("roi-probe"));
  CHECK(registry.has("region-descriptor"));

  auto probe = system::create_processor("roi-probe");
  REQUIRE(probe);
  CHECK(probe->name() == "roi-probe");
}

TEST_CASE("unknown processor names are rejected listing the alternatives", "[system][processor]")
{
  CHECK_THROWS_WITH(system::create_processor("no-such-processor"),
                    ContainsSubstring("no-such-processor") && ContainsSubstring("region-descriptor"));
}

TEST_CASE("processors annotate an attended region and tolerate an empty ROI", "[system][processor]")
{
  const auto object = file_at(3, 32, 32);
  cv::Mat roi(20, 20, CV_8UC3, cv::Scalar(0, 0, 255)); // saturated red

  for (const char* name : {"roi-probe", "region-descriptor"})
  {
    auto processor = system::create_processor(name);
    const auto annotation = processor->process(object, roi);
    CHECK(annotation.processor == name);
    // The label carries the object identity for the overlay.
    CHECK_THAT(annotation.label, ContainsSubstring("#3"));

    // An off-frame bbox yields an empty ROI: degrade gracefully, never crash.
    const auto empty = processor->process(object, cv::Mat());
    CHECK_FALSE(empty.label.empty());
  }

  auto descriptor = system::create_processor("region-descriptor");
  CHECK_THAT(descriptor->process(object, roi).detail, ContainsSubstring("red"));
}

TEST_CASE("LiveDemonstrator annotates native frames from a downscaled analysis", "[system][live]")
{
  system::LiveDemonstrator::Config cfg;
  cfg.processors = {"roi-probe", "region-descriptor"};
  // Native 640x480 input, processing capped at 240 -> a real downscale, so
  // this exercises the coordinate scaling between the two resolutions.
  cfg.process_max_side = 240;
  system::LiveDemonstrator demo(cfg);
  demo.reset();

  const fs::path dir = fs::path(ATTENTION_SOURCE_DIR) / "data" / "test_images" / "motion_seq";
  const cv::Size native(640, 480);
  int annotated_frames = 0;
  for (const char* name : {"f00.png", "f01.png", "f02.png"})
  {
    cv::Mat frame = cv::imread((dir / name).string(), cv::IMREAD_COLOR);
    REQUIRE_FALSE(frame.empty());
    cv::resize(frame, frame, native, 0, 0, cv::INTER_NEAREST);

    cv::Mat annotated = demo.process(frame);
    REQUIRE(annotated.size() == native); // overlay lives at native resolution
    if (!demo.annotations().empty())
    {
      ++annotated_frames;
    }
  }

  // The bright moving square becomes an object file with annotations.
  CHECK(annotated_frames > 0);
  CHECK_FALSE(demo.system().active_files().empty());
  CHECK(demo.frame_index() == 3);

  // Object files live at processing resolution (240x180 for 640x480 capped
  // at 240); the scaling to native happens at draw/ROI time.
  for (const auto& f : demo.system().active_files())
  {
    CHECK(cv::Rect(0, 0, 240, 180).contains(f.centroid));
  }

  // reset() starts a fresh stream.
  demo.reset();
  CHECK(demo.system().active_files().empty());
}
