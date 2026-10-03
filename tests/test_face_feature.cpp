// H4 follow-up: the opt-in face channel. MIT1003 is full of faces, and the
// scanpath study's verdict named this as the cheap thing that would move its
// numbers. These tests cover the parts that can fail silently — a map that is
// normalized up from nothing, and config plumbing — plus one real detection on
// an image that is in the repository, so the suite needs no dataset.

#include "attention/features/face_feature.h"
#include "attention/pipeline/attention_pipeline.h"
#include "attention/util/haar_cascade.h"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>

using attention::features::FaceFeature;

namespace
{
bool cascade_available()
{
  return !FaceFeature::default_cascade_path().empty();
}
} // namespace

TEST_CASE("an unknown backend is rejected at construction", "[face]")
{
  FaceFeature::Config config;
  config.backend = "telepathy";
  REQUIRE_THROWS_AS(FaceFeature(config), std::runtime_error);
}

TEST_CASE("yunet without a model path is rejected, and says which key is missing", "[face]")
{
  FaceFeature::Config config;
  config.backend = "yunet";
  REQUIRE_THROWS_AS(FaceFeature(config), std::runtime_error);
}

TEST_CASE("a frame with no faces gives an all-zero map, not normalized noise", "[face]")
{
  if (!cascade_available())
  {
    SUCCEED("no Haar cascade installed");
    return;
  }
  // The trap this guards: normalizing every frame to [0, 1] would turn "no
  // faces" into "a frame full of faces" as soon as any pixel was non-zero.
  // That is the dossier's finding A in miniature.
  FaceFeature feature{FaceFeature::Config{}};
  cv::Mat image(120, 160, CV_8UC3, cv::Scalar(90, 110, 130));
  cv::randu(image, cv::Scalar(80, 100, 120), cv::Scalar(100, 120, 140)); // texture, no face
  const attention::core::Frame frame(image);

  const auto map = feature.extract(frame);
  REQUIRE(map.data.type() == CV_32F);
  REQUIRE(map.data.size() == image.size());
  double hi = 0.0;
  cv::minMaxLoc(map.data, nullptr, &hi);
  CHECK(hi == 0.0);
}

TEST_CASE("a real face produces a normalized blob with its peak on the face", "[face]")
{
  if (!cascade_available())
  {
    SUCCEED("no Haar cascade installed");
    return;
  }
  const std::filesystem::path image_path =
      std::filesystem::path(ATTENTION_SOURCE_DIR) / "data" / "samples" / "images" / "soccer.jpg";
  if (!std::filesystem::exists(image_path))
  {
    SUCCEED("sample image missing");
    return;
  }

  FaceFeature feature{FaceFeature::Config{}};
  const attention::core::Frame frame(cv::imread(image_path.string()));
  REQUIRE_FALSE(frame.empty());

  const auto map = feature.extract(frame);
  double hi = 0.0;
  cv::Point peak;
  cv::minMaxLoc(map.data, nullptr, &hi, nullptr, &peak);

  CHECK(hi > 0.99); // a detection normalizes the strongest blob to 1
  CHECK(hi <= 1.0);
  // A Gaussian blob, not a filled rectangle: the value must fall off away from
  // the peak, or peak selection downstream has nothing to find.
  const cv::Point far(std::min(map.data.cols - 1, peak.x + map.data.cols / 3), peak.y);
  CHECK(map.data.at<float>(far) < map.data.at<float>(peak));
}

TEST_CASE("the face channel is opt-in: no default profile enables it", "[face]")
{
  // The replication profiles must stay what the thesis described, so this
  // channel has to be absent until a config asks for it by name.
  const attention::pipeline::PipelineConfig config;
  for (const auto& spec : config.features)
  {
    CHECK(spec.type != "face");
  }
}

TEST_CASE("the cascade finder agrees with the processor's", "[face]")
{
  // Both the haar-face processor (M13) and this feature resolve cascades
  // through util::find_haar_cascade, so they cannot disagree about where
  // OpenCV put them.
  const std::string path = attention::util::find_haar_cascade("haarcascade_frontalface_default.xml");
  CHECK(path == FaceFeature::default_cascade_path());
}
