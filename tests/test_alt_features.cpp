// M9 tests: alternative (non-thesis) saliency features — spectral residual,
// phase-spectrum, frequency-tuned, image-signature, boolean-map, and
// minimum-barrier. Behavioral checks: each localizes a planted salient region,
// produces a valid normalized map, and is reachable through the feature
// registry (the config-driven plug-in path).

#include "attention/core/frame.h"
#include "attention/features/boolean_map_feature.h"
#include "attention/features/feature_registry.h"
#include "attention/features/frequency_tuned_feature.h"
#include "attention/features/image_signature_feature.h"
#include "attention/features/minimum_barrier_feature.h"
#include "attention/features/phase_spectrum_feature.h"
#include "attention/features/spectral_residual_feature.h"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using namespace attention;

TEST_CASE("spectral-residual highlights a structure that breaks background regularity", "[alt][spectral-residual]")
{
  // Spectral residual keys on departures from the (statistically regular)
  // background spectrum, so it needs a textured background: a smooth object
  // embedded in fine random texture is the canonical pop-out. (A single blob on
  // a perfectly flat field is degenerate for this model — the whole spectrum is
  // near-DC.) cv::RNG is deterministic across platforms, so this is repeatable.
  cv::Mat img(200, 200, CV_8UC3);
  cv::RNG rng(42);
  rng.fill(img, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));
  cv::rectangle(img, cv::Rect(80, 80, 40, 40), cv::Scalar(210, 210, 210), -1);

  core::Frame frame(img);
  features::SpectralResidualFeature sr;
  REQUIRE(sr.applicable(frame));

  core::FeatureMap map = sr.extract(frame);
  REQUIRE(map.data.type() == CV_32F);
  REQUIRE(map.data.size() == img.size());

  double mn, mx;
  cv::minMaxLoc(map.data, &mn, &mx);
  CHECK(mx > 0.99); // normalized to [0, 1]

  const double at_object = cv::mean(map.data(cv::Rect(80, 80, 40, 40)))[0];
  const double background = cv::mean(map.data(cv::Rect(10, 10, 30, 30)))[0];
  CHECK(at_object > background + 0.2);
}

TEST_CASE("frequency-tuned highlights a colour-contrasted region", "[alt][frequency-tuned]")
{
  cv::Mat img(200, 200, CV_8UC3, cv::Scalar(120, 120, 120));               // neutral gray
  cv::rectangle(img, cv::Rect(70, 70, 50, 50), cv::Scalar(0, 0, 220), -1); // red (BGR)

  core::Frame frame(img);
  features::FrequencyTunedFeature ft;
  core::FeatureMap map = ft.extract(frame);
  REQUIRE(map.data.type() == CV_32F);
  REQUIRE(map.data.size() == img.size());

  const double at_square = cv::mean(map.data(cv::Rect(70, 70, 50, 50)))[0];
  const double background = cv::mean(map.data(cv::Rect(0, 0, 30, 30)))[0];
  CHECK(at_square > background + 0.2);
}

TEST_CASE("frequency-tuned degrades gracefully on grayscale input", "[alt][frequency-tuned]")
{
  cv::Mat img(120, 120, CV_8UC1, cv::Scalar(60));
  cv::rectangle(img, cv::Rect(45, 45, 30, 30), cv::Scalar(200), -1);

  core::Frame frame(img);
  features::FrequencyTunedFeature ft;
  REQUIRE(ft.applicable(frame));
  core::FeatureMap map = ft.extract(frame);
  REQUIRE(map.data.size() == img.size());

  const double at_square = cv::mean(map.data(cv::Rect(45, 45, 30, 30)))[0];
  const double background = cv::mean(map.data(cv::Rect(0, 0, 20, 20)))[0];
  CHECK(at_square > background);
}

TEST_CASE("boolean-map favours a surrounded figure over the border", "[alt][boolean-map]")
{
  cv::Mat img(200, 200, CV_8UC1, cv::Scalar(30));                    // dark background
  cv::rectangle(img, cv::Rect(80, 80, 40, 40), cv::Scalar(220), -1); // bright, surrounded

  core::Frame frame(img);
  features::BooleanMapFeature bms;
  core::FeatureMap map = bms.extract(frame);
  REQUIRE(map.data.type() == CV_32F);
  REQUIRE(map.data.size() == img.size());

  const double figure = cv::mean(map.data(cv::Rect(85, 85, 30, 30)))[0];
  const double border = cv::mean(map.data(cv::Rect(0, 0, 200, 8)))[0]; // top strip
  CHECK(figure > border);
}

TEST_CASE("image-signature highlights a sparse foreground on a plain background", "[alt][image-signature]")
{
  // The signature theory's canonical case: a spatially sparse foreground on a
  // spectrally sparse (here: flat) background.
  cv::Mat img(200, 200, CV_8UC3, cv::Scalar(110, 110, 110));
  cv::rectangle(img, cv::Rect(80, 80, 40, 40), cv::Scalar(30, 200, 60), -1);

  core::Frame frame(img);
  features::ImageSignatureFeature is;
  REQUIRE(is.applicable(frame));

  core::FeatureMap map = is.extract(frame);
  REQUIRE(map.data.type() == CV_32F);
  REQUIRE(map.data.size() == img.size());

  double mn, mx;
  cv::minMaxLoc(map.data, &mn, &mx);
  CHECK(mx > 0.99); // normalized to [0, 1]

  const double at_object = cv::mean(map.data(cv::Rect(80, 80, 40, 40)))[0];
  const double background = cv::mean(map.data(cv::Rect(10, 10, 30, 30)))[0];
  CHECK(at_object > background + 0.2);
}

TEST_CASE("phase-spectrum highlights distinct structure", "[alt][phase-spectrum]")
{
  cv::Mat img(200, 200, CV_8UC3, cv::Scalar(120, 120, 120));
  cv::rectangle(img, cv::Rect(75, 75, 40, 40), cv::Scalar(0, 0, 220), -1); // red (BGR)

  core::Frame frame(img);
  features::PhaseSpectrumFeature ps;
  core::FeatureMap map = ps.extract(frame);
  REQUIRE(map.data.type() == CV_32F);
  REQUIRE(map.data.size() == img.size());

  const double at_object = cv::mean(map.data(cv::Rect(75, 75, 40, 40)))[0];
  const double background = cv::mean(map.data(cv::Rect(5, 5, 30, 30)))[0];
  CHECK(at_object > background + 0.1);
}

TEST_CASE("phase-spectrum motion channel favours the newly appeared object", "[alt][phase-spectrum]")
{
  // Two identical squares; only one is new relative to the previous frame.
  // Spatially they are equally salient — the motion channel breaks the tie.
  cv::Mat img(200, 200, CV_8UC3, cv::Scalar(120, 120, 120));
  cv::rectangle(img, cv::Rect(30, 80, 40, 40), cv::Scalar(220, 220, 220), -1);  // static
  cv::rectangle(img, cv::Rect(130, 80, 40, 40), cv::Scalar(220, 220, 220), -1); // appears now

  cv::Mat prev(200, 200, CV_8UC1, cv::Scalar(120));
  cv::rectangle(prev, cv::Rect(30, 80, 40, 40), cv::Scalar(220), -1); // only the static one

  core::Frame frame(img);
  frame.previous_gray = prev;

  features::PhaseSpectrumFeature ps;
  core::FeatureMap map = ps.extract(frame);

  const double at_new = cv::mean(map.data(cv::Rect(130, 80, 40, 40)))[0];
  const double at_static = cv::mean(map.data(cv::Rect(30, 80, 40, 40)))[0];
  CHECK(at_new > at_static);
}

TEST_CASE("minimum-barrier favours a contrast-walled figure over the background", "[alt][minimum-barrier]")
{
  cv::Mat img(200, 200, CV_8UC1, cv::Scalar(40));                    // background touches the border
  cv::rectangle(img, cv::Rect(80, 80, 40, 40), cv::Scalar(210), -1); // walled in by its own contrast

  core::Frame frame(img);
  features::MinimumBarrierFeature mb;
  REQUIRE(mb.applicable(frame));
  core::FeatureMap map = mb.extract(frame);
  REQUIRE(map.data.type() == CV_32F);
  REQUIRE(map.data.size() == img.size());

  const double figure = cv::mean(map.data(cv::Rect(85, 85, 30, 30)))[0];
  const double background = cv::mean(map.data(cv::Rect(10, 10, 30, 30)))[0];
  CHECK(figure > background + 0.2);
}

TEST_CASE("alternative features are reachable through the registry", "[alt][registry]")
{
  features::register_builtin_features();
  auto& registry = features::FeatureRegistry::instance();

  for (const char* name :
       {"spectral-residual", "phase-spectrum", "frequency-tuned", "image-signature", "boolean-map", "minimum-barrier"})
  {
    REQUIRE(registry.has(name));
    auto extractor = registry.create(name, YAML::Node());
    REQUIRE(extractor != nullptr);
    CHECK(extractor->name() == name);
  }
}
