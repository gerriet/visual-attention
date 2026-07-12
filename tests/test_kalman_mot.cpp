// Tests for the kalman-mot selection backend (docs/SELECTION_BACKENDS.md, B):
// detection + Kalman tracking as an alternative to the neural field. Behavioral
// checks — still-image detection, moving-blob tracking, object-based IOR, and
// occlusion coasting — plus the registry path.

#include "attention/core/run_state.h"
#include "attention/selection/kalman_mot_selection.h"
#include "attention/selection/selection_strategy.h"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using namespace attention;

namespace
{
// A normalized saliency map (CV_32F, [0,1]) with a soft blob at each center.
cv::Mat blob_map(cv::Size size, const std::vector<cv::Point>& centers, int radius = 12)
{
  cv::Mat m = cv::Mat::zeros(size, CV_32F);
  for (const auto& c : centers)
  {
    cv::circle(m, c, radius, cv::Scalar(1.0), -1);
  }
  if (!centers.empty())
  {
    cv::GaussianBlur(m, m, cv::Size(0, 0), radius / 2.0);
    cv::normalize(m, m, 0.0, 1.0, cv::NORM_MINMAX);
  }
  return m;
}

bool has_peak_near(const std::vector<core::Peak>& peaks, cv::Point p, double tol = 18.0)
{
  for (const auto& peak : peaks)
  {
    if (cv::norm(peak.location - p) <= tol)
    {
      return true;
    }
  }
  return false;
}
} // namespace

TEST_CASE("kalman-mot detects multiple blobs on a still frame", "[selection][kalman-mot]")
{
  cv::Mat m = blob_map({200, 200}, {{40, 40}, {160, 40}, {100, 160}});
  selection::KalmanMotSelection sel;
  core::RunState state; // frame_index == 0

  auto peaks = sel.select(m, state);
  CHECK(peaks.size() == 3);
  CHECK(has_peak_near(peaks, {40, 40}));
  CHECK(has_peak_near(peaks, {160, 40}));
  CHECK(has_peak_near(peaks, {100, 160}));
}

TEST_CASE("kalman-mot returns nothing on an empty map", "[selection][kalman-mot]")
{
  selection::KalmanMotSelection sel;
  core::RunState state;
  CHECK(sel.select(cv::Mat::zeros(100, 100, CV_32F), state).empty());
}

TEST_CASE("kalman-mot follows a moving blob across frames", "[selection][kalman-mot]")
{
  selection::KalmanMotSelection::Params p;
  p.object_ior = false; // keep the blob selectable every frame for the check
  selection::KalmanMotSelection sel(selection::SelectionParams{}, p);
  core::RunState state;

  for (int f = 0; f < 5; ++f)
  {
    const int cx = 30 + 8 * f; // moves right 8 px/frame
    cv::Mat m = blob_map({200, 120}, {{cx, 60}});
    state.frame_index = f;
    auto peaks = sel.select(m, state);
    REQUIRE(peaks.size() == 1);
    CHECK(std::abs(peaks[0].location.x - cx) <= 12);
    CHECK(std::abs(peaks[0].location.y - 60) <= 12);
  }
}

TEST_CASE("kalman-mot object-based IOR suppresses revisits to a static object", "[selection][kalman-mot]")
{
  cv::Mat m = blob_map({120, 120}, {{60, 60}});

  auto count_selections = [&](bool object_ior, int ior_frames)
  {
    selection::KalmanMotSelection::Params p;
    p.object_ior = object_ior;
    p.ior_frames = ior_frames;
    selection::KalmanMotSelection sel(selection::SelectionParams{}, p);
    core::RunState state;
    int selected = 0;
    for (int f = 0; f < 6; ++f)
    {
      state.frame_index = f;
      if (!sel.select(m, state).empty())
      {
        ++selected;
      }
    }
    return selected;
  };

  const int without_ior = count_selections(false, 3);
  const int with_ior = count_selections(true, 3);

  CHECK(without_ior == 6);       // no IOR: fixated every frame
  CHECK(with_ior < without_ior); // object IOR: revisits suppressed
  CHECK(with_ior >= 2);          // and it recovers (frames 0 and 3)
}

TEST_CASE("kalman-mot coasts through a brief occlusion (track survives)", "[selection][kalman-mot]")
{
  // The object is selected on frame 0, then disappears for two frames, then
  // returns. If the track survived the gap (coasting), it keeps its IOR tag and
  // stays suppressed on return; if it had died and been reborn, it would be
  // selected again. So "not selected on return" proves the track coasted.
  selection::KalmanMotSelection::Params p;
  p.object_ior = true;
  p.ior_frames = 6;
  p.max_age = 15;
  selection::KalmanMotSelection sel(selection::SelectionParams{}, p);
  core::RunState state;

  cv::Mat present = blob_map({120, 120}, {{60, 60}});
  cv::Mat empty = cv::Mat::zeros(120, 120, CV_32F);

  state.frame_index = 0;
  CHECK(sel.select(present, state).size() == 1); // selected, IOR tag set to 0
  state.frame_index = 1;
  sel.select(empty, state); // occluded — coast
  state.frame_index = 2;
  sel.select(empty, state); // occluded — coast
  state.frame_index = 3;
  auto on_return = sel.select(present, state); // within the refractory window
  CHECK(on_return.empty());                    // still inhibited -> the same track re-acquired
}

TEST_CASE("kalman-mot is reachable through the selection registry", "[selection][kalman-mot][registry]")
{
  auto strategy = selection::create_selection_strategy("kalman-mot", selection::SelectionParams{}, YAML::Node());
  REQUIRE(strategy != nullptr);
  CHECK(strategy->name() == "kalman-mot");
}
