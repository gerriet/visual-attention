// M5 tests: stereo disparity feature, onset/motion feature, and the 3D neural
// field (core dynamics + depth-aware selection).

#include "attention/core/frame.h"
#include "attention/features/onset_feature.h"
#include "attention/features/stereo_feature.h"
#include "attention/selection/neural_field_3d.h"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;
using namespace attention;

namespace
{
fs::path source_dir()
{
  return fs::path(ATTENTION_SOURCE_DIR);
}
} // namespace

TEST_CASE("stereo feature lights up the near foreground, not the zero-disparity background", "[stereo]")
{
  const fs::path dir = source_dir() / "data" / "test_images" / "stereo";
  cv::Mat left = cv::imread((dir / "left.png").string(), cv::IMREAD_GRAYSCALE);
  cv::Mat right = cv::imread((dir / "right.png").string(), cv::IMREAD_GRAYSCALE);
  REQUIRE_FALSE(left.empty());
  REQUIRE_FALSE(right.empty());

  core::Frame frame(left);
  frame.stereo_right = right;

  features::StereoFeature stereo; // defaults: 3 orientations, disparity [-16, 0]
  REQUIRE(stereo.applicable(frame));

  core::FeatureMap map = stereo.extract(frame);
  REQUIRE(map.data.type() == CV_32F);
  REQUIRE(map.data.size() == left.size());

  // Foreground rect (cols 80..176, rows 64..176) is at disparity 10 (near);
  // the left strip is at disparity 0 (background).
  const double fg = cv::mean(map.data(cv::Rect(80, 64, 96, 112)))[0];
  const double bg = cv::mean(map.data(cv::Rect(0, 0, 60, 256)))[0];

  CHECK(fg > 0.4);            // near surface pops out (10/16 ~ 0.63)
  CHECK(bg < 0.05);           // zero-disparity background stays dark
  CHECK(fg > bg + 0.3);
}

TEST_CASE("stereo feature is inapplicable without a right image", "[stereo]")
{
  core::Frame frame(cv::Mat::zeros(64, 64, CV_8U));
  features::StereoFeature stereo;
  CHECK_FALSE(stereo.applicable(frame));
}

TEST_CASE("stereo feature degrades to a zero map on a too-small pair (no throw)", "[stereo]")
{
  // Smaller than the correlation window — no disparity can be recovered. The
  // feature must return a flat map, not throw (a throw in the parallel
  // extraction path would abort the process).
  cv::Mat tiny(8, 8, CV_8U, cv::Scalar(120));
  core::Frame frame(tiny);
  frame.stereo_right = tiny.clone();

  features::StereoFeature stereo;
  REQUIRE(stereo.applicable(frame));
  core::FeatureMap map;
  REQUIRE_NOTHROW(map = stereo.extract(frame));
  REQUIRE(map.data.size() == tiny.size());
  CHECK(cv::countNonZero(map.data) == 0);
}

TEST_CASE("onset feature responds to appearing structure only", "[onset]")
{
  cv::Mat prev = cv::Mat::zeros(120, 120, CV_8U);
  cv::randu(prev, 40, 80); // textured but static background

  // Current frame = previous + a bright textured square that just appeared.
  cv::Mat cur = prev.clone();
  cv::Mat square(30, 30, CV_8U);
  cv::randu(square, 180, 255);
  square.copyTo(cur(cv::Rect(70, 45, 30, 30)));

  core::Frame frame(cur);
  frame.previous_gray = prev;

  features::OnsetFeature onset;
  REQUIRE(onset.applicable(frame));
  core::FeatureMap map = onset.extract(frame);
  REQUIRE(map.data.type() == CV_32F);

  const double at_onset = cv::mean(map.data(cv::Rect(70, 45, 30, 30)))[0];
  const double elsewhere = cv::mean(map.data(cv::Rect(0, 0, 40, 40)))[0];
  CHECK(at_onset > elsewhere);
  CHECK(at_onset > 0.1);
}

TEST_CASE("onset feature is inapplicable on the first frame", "[onset]")
{
  core::Frame frame(cv::Mat::zeros(64, 64, CV_8U)); // no previous_gray
  features::OnsetFeature onset;
  CHECK_FALSE(onset.applicable(frame));
}

TEST_CASE("3D neural field converges and localizes a single blob in its plane", "[neural-field-3d]")
{
  const int Z = 5;
  const cv::Size sz(48, 48);
  selection::NeuralField3D::Params params;
  selection::NeuralField3D field(sz, Z, params);

  // A single input blob at plane 2, centred.
  std::vector<cv::Mat> volume(Z);
  for (int z = 0; z < Z; ++z)
  {
    volume[z] = cv::Mat::zeros(sz, CV_32F);
  }
  cv::circle(volume[2], cv::Point(24, 24), 4, cv::Scalar(1.0f), -1);

  int cycles = field.update(volume);
  CHECK(cycles >= 3);
  CHECK(cycles <= params.max_cycles);

  cv::Mat collapsed = field.collapsed_activation();
  double mn, mx;
  cv::Point max_loc;
  cv::minMaxLoc(collapsed, &mn, &mx, nullptr, &max_loc);
  CHECK(mx > 0.0);                                     // sustained activation
  CHECK(cv::norm(max_loc - cv::Point(24, 24)) < 8);    // localized at the blob

  // The winning depth at the blob centre is its input plane.
  CHECK(field.winning_depth().at<int>(24, 24) == 2);
}

TEST_CASE("3D neural field keeps two separated blobs at different depths", "[neural-field-3d]")
{
  const int Z = 6;
  const cv::Size sz(64, 64);
  selection::NeuralField3D::Params params;
  selection::NeuralField3D field(sz, Z, params);

  std::vector<cv::Mat> volume(Z);
  for (int z = 0; z < Z; ++z)
  {
    volume[z] = cv::Mat::zeros(sz, CV_32F);
  }
  cv::circle(volume[1], cv::Point(16, 16), 4, cv::Scalar(1.0f), -1); // far, top-left
  cv::circle(volume[4], cv::Point(48, 48), 4, cv::Scalar(1.0f), -1); // near, bottom-right

  field.update(volume);
  cv::Mat collapsed = field.collapsed_activation();
  cv::Mat winning = field.winning_depth();

  // Both regions stay active (spatially separated -> no competition), each at
  // its own depth.
  CHECK(collapsed.at<float>(16, 16) > 0.0f);
  CHECK(collapsed.at<float>(48, 48) > 0.0f);
  CHECK(winning.at<int>(16, 16) == 1);
  CHECK(winning.at<int>(48, 48) == 4);
}

TEST_CASE("neural-field-3d selection fixates the salient cluster with a depth cue", "[neural-field-3d][selection]")
{
  cv::Mat saliency = cv::Mat::zeros(120, 120, CV_32F);
  cv::circle(saliency, cv::Point(80, 40), 8, cv::Scalar(1.0f), -1);
  cv::GaussianBlur(saliency, saliency, cv::Size(11, 11), 3.0);
  cv::normalize(saliency, saliency, 0.0f, 1.0f, cv::NORM_MINMAX);

  // Depth cue: the blob is near (high depth value), the rest is far.
  cv::Mat depth = cv::Mat::zeros(120, 120, CV_32F);
  cv::circle(depth, cv::Point(80, 40), 8, cv::Scalar(0.9f), -1);

  selection::SelectionParams shared;
  shared.max_count = 5;
  shared.ior_strength = 0.9f;
  auto strategy = selection::create_selection_strategy("neural-field-3d", shared, YAML::Node());

  core::RunState state;
  state.depth_map = depth;
  auto peaks = strategy->select(saliency, state);

  REQUIRE_FALSE(peaks.empty());
  CHECK(cv::norm(peaks[0].location - cv::Point(80, 40)) < 15);
  CHECK_FALSE(state.field_activity.empty());
}

TEST_CASE("neural-field-3d selection stays quiet on an empty map", "[neural-field-3d][selection]")
{
  selection::SelectionParams shared;
  auto strategy = selection::create_selection_strategy("neural-field-3d", shared, YAML::Node());
  core::RunState state;
  auto peaks = strategy->select(cv::Mat::zeros(96, 96, CV_32F), state);
  CHECK(peaks.empty());
}
