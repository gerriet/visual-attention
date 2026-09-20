// Behavioural tests for the dissertation's static features as ported from the
// original sources: eccentricity (§5.2.3), colour contrast in MTM/Munsell space
// (§5.3.2), and the exclusivity weighting (§5.5.3). The characterization
// goldens only detect *change*; these check that the features respond to the
// property they are named after (thesis Abb. 5.10, 5.13, 5.20, 5.34).

#include "attention/core/frame.h"
#include "attention/features/eccentricity_feature.h"
#include "attention/features/exclusivity.h"
#include "attention/features/feature_registry.h"
#include "attention/features/munsell_color_feature.h"
#include "attention/features/stereo_feature.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <set>
#include <yaml-cpp/yaml.h>

using namespace attention;
using Catch::Approx;
using features::EccentricityFeature;
using features::Exclusivity;
using features::MunsellColorFeature;

namespace
{

cv::Mat gray_canvas(int value = 128)
{
  return cv::Mat(300, 400, CV_8UC1, cv::Scalar(value));
}

// Saliency of the segment under a point
float saliency_at(const EccentricityFeature::Result& r, cv::Point p)
{
  return r.saliency.at<float>(p);
}

// A scene of `n` vertical bars plus one horizontal bar at the bottom
cv::Mat bars_scene(int n)
{
  cv::Mat image = gray_canvas();
  for (int i = 0; i < n; ++i)
  {
    cv::rectangle(image, cv::Rect(30 + i * 60, 30, 14, 110), cv::Scalar(230), cv::FILLED);
  }
  cv::rectangle(image, cv::Rect(100, 220, 160, 16), cv::Scalar(230), cv::FILLED);
  return image;
}

cv::Mat colour_canvas(cv::Scalar bgr = cv::Scalar(120, 120, 120))
{
  return cv::Mat(300, 400, CV_8UC3, bgr);
}

float segment_saliency_at(const MunsellColorFeature::Result& r, cv::Point p)
{
  return r.segments[r.labels.at<int>(p)].saliency;
}

} // namespace

TEST_CASE("exclusivity divisor: thesis c^n, source n^p, and off by default", "[exclusivity]")
{
  Exclusivity off;
  CHECK_FALSE(off.enabled());
  CHECK(off.divisor(7) == Approx(1.0f));

  Exclusivity thesis;
  thesis.strength = 1.1f;
  CHECK(thesis.enabled());
  CHECK(thesis.divisor(0) == Approx(1.0f));
  CHECK(thesis.divisor(1) == Approx(1.1f));
  CHECK(thesis.divisor(5) == Approx(std::pow(1.1f, 5.0f)));

  Exclusivity source;
  source.mode = Exclusivity::Mode::Power;
  source.strength = 0.5f;
  CHECK(source.divisor(1) == Approx(1.0f)); // a unique segment is left alone
  CHECK(source.divisor(4) == Approx(2.0f));

  CHECK(Exclusivity::parse_mode("thesis") == Exclusivity::Mode::Exponential);
  CHECK_THROWS(Exclusivity::parse_mode("sometimes"));
}

TEST_CASE("Jaehne eccentricity: 0 for round, 1 for a line, 0.36 for a 2:1 ellipse", "[eccentricity]")
{
  CHECK(EccentricityFeature::eccentricity(10.0, 10.0, 0.0) == Approx(0.0f));
  CHECK(EccentricityFeature::eccentricity(10.0, 0.0, 0.0) == Approx(1.0f));
  // Axes 2:1 -> variances 4:1 -> ((4-1)/(4+1))^2 (thesis eq. 5.6; the
  // eigenvalue form sqrt(1 - l2/l1) used before the port gives 0.87)
  CHECK(EccentricityFeature::eccentricity(4.0, 1.0, 0.0) == Approx(0.36f));
  // Rotation invariance: the same ellipse at 45 degrees
  CHECK(EccentricityFeature::eccentricity(2.5, 2.5, 1.5) == Approx(0.36f));
  CHECK(EccentricityFeature::eccentricity(0.0, 0.0, 0.0) == Approx(0.0f));
}

TEST_CASE("eccentricity rates elongation: bar > ellipse > disk = square = 0 (Abb. 5.10)", "[eccentricity]")
{
  cv::Mat image = gray_canvas();
  cv::circle(image, cv::Point(70, 70), 40, cv::Scalar(200), cv::FILLED);
  cv::ellipse(image, cv::Point(230, 80), cv::Size(80, 40), 0, 0, 360, cv::Scalar(60), cv::FILLED);
  cv::rectangle(image, cv::Rect(40, 180, 160, 20), cv::Scalar(220), cv::FILLED);
  cv::rectangle(image, cv::Rect(260, 170, 80, 80), cv::Scalar(40), cv::FILLED);

  const auto result = EccentricityFeature().evaluate(image);
  const float disk = saliency_at(result, {70, 70});
  const float ellipse = saliency_at(result, {230, 80});
  const float bar = saliency_at(result, {120, 190});
  const float square = saliency_at(result, {300, 210});

  CHECK(disk < 0.02f);
  CHECK(square < 0.02f);
  CHECK(ellipse == Approx(0.36f).margin(0.05f));
  CHECK(bar > 0.85f);
  // The map is an absolute saliency: the background (too large a segment) is 0
  CHECK(saliency_at(result, {380, 150}) == 0.0f);
}

TEST_CASE("eccentricity segments follow the image, not a marker grid", "[eccentricity]")
{
  // Regression: the pre-port implementation handed cv::watershed a [0,1] float
  // image converted to 8 bit without scaling — an all-black image — so its
  // segments were Voronoi cells of the markers and ignored the content.
  cv::Mat image = gray_canvas();
  const cv::Rect bar(60, 120, 240, 30);
  cv::rectangle(image, bar, cv::Scalar(225), cv::FILLED);

  const auto result = EccentricityFeature().evaluate(image);
  const int label = result.labels.at<int>(135, 180);
  REQUIRE(label != 0);

  // One segment covers the bar's interior ...
  const cv::Rect interior(bar.x + 3, bar.y + 3, bar.width - 6, bar.height - 6);
  CHECK(cv::countNonZero(result.labels(interior) != label) == 0);
  // ... and ends where the bar ends
  CHECK(result.labels.at<int>(135, 20) != label);
  CHECK(result.labels.at<int>(60, 180) != label);
  CHECK(result.segments[label].pixels == Approx(bar.area()).epsilon(0.15));
}

TEST_CASE("eccentricity through a Frame: float pyramid in, absolute map out", "[eccentricity]")
{
  cv::Mat image = gray_canvas();
  cv::rectangle(image, cv::Rect(40, 140, 300, 24), cv::Scalar(230), cv::FILLED);
  core::Frame frame(image);
  frame.compute_pyramids(4);

  const core::FeatureMap map = EccentricityFeature().extract(frame);
  REQUIRE(map.data.type() == CV_32F);
  REQUIRE(map.data.size() == image.size());
  CHECK(map.data.at<float>(152, 190) > 0.85f);
  CHECK(map.data.at<float>(40, 40) == 0.0f);

  // No elongated structure -> a flat zero map, not min-max-stretched residue
  cv::Mat blank = gray_canvas();
  cv::circle(blank, cv::Point(200, 150), 50, cv::Scalar(30), cv::FILLED);
  core::Frame round(blank);
  round.compute_pyramids(4);
  double max_value = 0.0;
  cv::minMaxLoc(EccentricityFeature().extract(round).data, nullptr, &max_value);
  CHECK(max_value < 0.05);
}

TEST_CASE("exclusivity makes the odd orientation pop out (Abb. 5.34)", "[eccentricity][exclusivity]")
{
  const cv::Mat image = bars_scene(5);
  const cv::Point vertical(37, 80), horizontal(180, 228);

  const auto plain = EccentricityFeature().evaluate(image);
  REQUIRE(saliency_at(plain, vertical) > 0.8f);
  // Without the weighting both orientations are about equally salient
  CHECK(saliency_at(plain, horizontal) == Approx(saliency_at(plain, vertical)).margin(0.08f));
  CHECK(plain.segments[plain.labels.at<int>(vertical)].orientation_class !=
        plain.segments[plain.labels.at<int>(horizontal)].orientation_class);

  EccentricityFeature::Config config;
  config.exclusivity.strength = 1.1f;
  const auto weighted = EccentricityFeature(config).evaluate(image);
  // Five share the vertical class (/1.1^5), one holds the horizontal (/1.1)
  const float ratio = saliency_at(weighted, horizontal) / saliency_at(weighted, vertical);
  CHECK(ratio == Approx(std::pow(1.1f, 4.0f)).epsilon(0.1));
}

TEST_CASE("MTM transform: greys are achromatic, opponent colours lie far apart", "[color-munsell]")
{
  cv::Mat patches(1, 5, CV_8UC3);
  patches.at<cv::Vec3b>(0, 0) = {128, 128, 128}; // grey
  patches.at<cv::Vec3b>(0, 1) = {255, 255, 255}; // white
  patches.at<cv::Vec3b>(0, 2) = {30, 30, 220};   // red (BGR)
  patches.at<cv::Vec3b>(0, 3) = {60, 160, 40};   // green
  patches.at<cv::Vec3b>(0, 4) = {40, 40, 200};   // a second, similar red

  const cv::Mat mtm = MunsellColorFeature::bgr_to_mtm(patches);
  REQUIRE(mtm.type() == CV_32FC3);
  auto at = [&](int i) { return mtm.at<cv::Vec3f>(0, i); };
  auto chroma = [&](int i) { return std::hypot(at(i)[1], at(i)[2]); };

  CHECK(chroma(0) < 1.0f);
  CHECK(chroma(1) < 1.0f);
  CHECK(at(1)[0] > at(0)[0]); // L rises with luminance
  CHECK(chroma(2) > 20.0f);
  CHECK(cv::norm(at(2) - at(3)) > 4.0 * cv::norm(at(2) - at(4)));

  // Float [0,1] input is the same colour as its 8-bit form
  cv::Mat as_float;
  patches.convertTo(as_float, CV_32F, 1.0 / 255.0);
  CHECK(cv::norm(MunsellColorFeature::bgr_to_mtm(as_float), mtm, cv::NORM_INF) < 0.05);
}

TEST_CASE("colour contrast: salience sits on the coloured object and grows with contrast (Abb. 5.20)",
          "[color-munsell]")
{
  auto blob_saliency = [](cv::Scalar blob_bgr)
  {
    cv::Mat image = colour_canvas(cv::Scalar(70, 150, 60)); // green ground
    cv::circle(image, cv::Point(200, 150), 45, blob_bgr, cv::FILLED);
    const auto result = MunsellColorFeature().evaluate(MunsellColorFeature::bgr_to_mtm(image));
    // The ground is one background-sized segment and gets no saliency
    CHECK(segment_saliency_at(result, {20, 20}) == 0.0f);
    return segment_saliency_at(result, {200, 150});
  };

  const float similar = blob_saliency(cv::Scalar(70, 170, 95));  // a slightly different green
  const float medium = blob_saliency(cv::Scalar(60, 150, 170));  // ochre
  const float opponent = blob_saliency(cv::Scalar(40, 40, 215)); // red on green
  CHECK(similar < medium);
  CHECK(medium < opponent);
  CHECK(opponent > 0.9f);
}

TEST_CASE("colour contrast through a Frame: no colour contrast -> flat, low map", "[color-munsell]")
{
  cv::Mat image = colour_canvas();
  core::Frame frame(image);
  MunsellColorFeature feature;
  REQUIRE(feature.applicable(frame));
  const core::FeatureMap map = feature.extract(frame);
  REQUIRE(map.name == "color-munsell");
  REQUIRE(map.data.size() == image.size());
  double max_value = 0.0;
  cv::minMaxLoc(map.data, nullptr, &max_value);
  CHECK(max_value < 0.06); // nothing is stretched to 1 (the Itti-style `color` does that)

  core::Frame gray(gray_canvas());
  CHECK_FALSE(feature.applicable(gray));
}

TEST_CASE("colour segmentation: one segment per flat object, textured ground stays whole", "[color-munsell]")
{
  cv::Mat image = colour_canvas(cv::Scalar(90, 90, 90));
  cv::RNG rng(7);
  cv::Mat noise(image.size(), CV_8UC3);
  rng.fill(noise, cv::RNG::UNIFORM, 0, 14); // fine texture: would shatter a fixed threshold
  image += noise;
  cv::circle(image, cv::Point(120, 150), 40, cv::Scalar(40, 40, 210), cv::FILLED);
  cv::circle(image, cv::Point(290, 150), 40, cv::Scalar(200, 120, 30), cv::FILLED);

  const auto result = MunsellColorFeature().evaluate(MunsellColorFeature::bgr_to_mtm(image));
  const int red = result.labels.at<int>(150, 120);
  const int blue = result.labels.at<int>(150, 290);
  const int ground = result.labels.at<int>(20, 20);
  CHECK(std::set<int>{red, blue, ground}.size() == 3);
  CHECK(result.labels.at<int>(280, 380) == ground);
  CHECK(result.segments[red].pixels == Approx(CV_PI * 40 * 40).epsilon(0.1));
  CHECK(result.segments[red].hue_class != result.segments[blue].hue_class);
}

TEST_CASE("exclusivity makes the odd colour pop out", "[color-munsell][exclusivity]")
{
  // Muted colours keep the sigmoid (eq. 5.12) out of saturation
  cv::Mat image = colour_canvas(cv::Scalar(120, 120, 120));
  const cv::Scalar green(95, 140, 100), red(100, 100, 150);
  const cv::Point greens[] = {{230, 70}, {330, 90}, {210, 210}, {320, 230}, {90, 220}};
  for (const auto& p : greens)
  {
    cv::circle(image, p, 30, green, cv::FILLED);
  }
  const cv::Point odd(80, 80);
  cv::circle(image, odd, 30, red, cv::FILLED);
  const cv::Mat mtm = MunsellColorFeature::bgr_to_mtm(image);

  MunsellColorFeature::Config lenient;
  lenient.attribute_threshold = 2.0f; // let the muted blobs count as coloured
  const auto plain = MunsellColorFeature(lenient).evaluate(mtm);
  const float plain_gap = segment_saliency_at(plain, odd) - segment_saliency_at(plain, greens[0]);

  MunsellColorFeature::Config weighted_config = lenient;
  weighted_config.exclusivity.strength = 1.1f;
  const auto weighted = MunsellColorFeature(weighted_config).evaluate(mtm);
  const float weighted_gap = segment_saliency_at(weighted, odd) - segment_saliency_at(weighted, greens[0]);

  REQUIRE(segment_saliency_at(plain, greens[0]) > 0.06f); // the blobs are salient at all
  CHECK(weighted_gap > plain_gap + 0.02f);
  // All five greens are suppressed alike
  CHECK(segment_saliency_at(weighted, greens[3]) == Approx(segment_saliency_at(weighted, greens[0])).margin(0.02f));
}

TEST_CASE("stereo exclusivity dims the crowded depth plane", "[stereo][exclusivity]")
{
  const std::filesystem::path dir = std::filesystem::path(ATTENTION_SOURCE_DIR) / "data" / "test_images" / "stereo";
  cv::Mat left = cv::imread((dir / "left.png").string(), cv::IMREAD_GRAYSCALE);
  cv::Mat right = cv::imread((dir / "right.png").string(), cv::IMREAD_GRAYSCALE);
  REQUIRE_FALSE(left.empty());
  core::Frame frame(left);
  frame.stereo_right = right;

  const cv::Rect foreground(80, 64, 96, 112); // disparity 10, see test_stereo_onset.cpp
  const double plain = cv::mean(features::StereoFeature().extract(frame).data(foreground))[0];

  features::StereoFeature::Config config;
  config.exclusivity.strength = 1.1f;
  const double weighted = cv::mean(features::StereoFeature(config).extract(frame).data(foreground))[0];
  // The foreground plane holds far more than an average level's share of the
  // matched pixels, so it is divided by more than c^1
  CHECK(weighted < plain / 1.1);
  CHECK(weighted > 0.0);
}

TEST_CASE("registry: color-munsell and the exclusivity params are configurable", "[color-munsell][config]")
{
  features::register_builtin_features();
  const auto& registry = features::FeatureRegistry::instance();
  REQUIRE(registry.has("color-munsell"));
  REQUIRE(registry.has("color")); // the Itti-Koch-style feature stays available

  const YAML::Node params = YAML::Load("{exclusivity: 1.1, threshold: 6.0, sigmoid_beta: 4.0}");
  CHECK(registry.create("color-munsell", params)->name() == "color-munsell");
  CHECK(registry.create("eccentricity", YAML::Load("{exclusivity_mode: power, exclusivity: 0.5}")) != nullptr);
  CHECK_THROWS(registry.create("eccentricity", YAML::Load("{exclusivity_mode: bogus}")));
  CHECK_THROWS(registry.create("eccentricity", YAML::Load("{connectivity: 6}")));
}
