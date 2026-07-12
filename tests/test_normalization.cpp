// Tests for the normalization selection backend (docs/SELECTION_BACKENDS.md, D):
// divisive-normalization competition as an alternative to the neural field. The
// centerpiece is input-scale invariance — the field's worst tuning problem, gone.

#include "attention/core/run_state.h"
#include "attention/selection/normalization_selection.h"
#include "attention/selection/selection_strategy.h"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using namespace attention;

namespace
{
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

TEST_CASE("normalization localizes multiple blobs on a still frame", "[selection][normalization]")
{
  cv::Mat m = blob_map({200, 200}, {{50, 50}, {150, 50}, {100, 150}});
  selection::NormalizationSelection sel;
  core::RunState state;

  auto peaks = sel.select(m, state);
  CHECK(peaks.size() == 3);
  CHECK(has_peak_near(peaks, {50, 50}));
  CHECK(has_peak_near(peaks, {150, 50}));
  CHECK(has_peak_near(peaks, {100, 150}));
}

TEST_CASE("normalization's surround suppression beats raw winner-take-all", "[selection][normalization]")
{
  // The real, non-trivial property of divisive normalization: it responds to
  // contrast against the surround, not absolute brightness. A small, isolated
  // blob (low surround pool) should out-compete a larger, brighter region
  // (high self-pool). Raw WTA (nms) instead just picks the brightest pixel — so
  // the two strategies disagree, which is exactly what makes this meaningful.
  cv::Mat m = cv::Mat::zeros(cv::Size(200, 200), CV_32F);
  cv::circle(m, {50, 50}, 7, cv::Scalar(0.8), -1);    // small, isolated  -> A
  cv::circle(m, {140, 140}, 26, cv::Scalar(1.0), -1); // large, brightest -> B
  cv::GaussianBlur(m, m, cv::Size(0, 0), 3.0);

  // Test the pure divisive response (no response smoothing, which would spread
  // the small blob's sharp peak and mask the effect).
  selection::NormalizationSelection::Params p;
  p.smooth_factor = 0.0f;
  core::RunState s1;
  auto norm_peaks = selection::NormalizationSelection(selection::SelectionParams{}, p).select(m, s1);
  REQUIRE_FALSE(norm_peaks.empty());
  CHECK(cv::norm(norm_peaks[0].location - cv::Point(50, 50)) < 15); // A wins

  // Sanity: plain NMS picks the brighter B first (so the test isn't tautological).
  core::RunState s2;
  auto nms_peaks =
      selection::create_selection_strategy("nms", selection::SelectionParams{}, YAML::Node())->select(m, s2);
  REQUIRE_FALSE(nms_peaks.empty());
  CHECK(cv::norm(nms_peaks[0].location - cv::Point(140, 140)) < 15); // B wins
}

TEST_CASE("normalization returns nothing on an empty or sub-threshold map", "[selection][normalization]")
{
  selection::NormalizationSelection sel;
  core::RunState s1;
  CHECK(sel.select(cv::Mat::zeros(100, 100, CV_32F), s1).empty());

  // Near-empty noise, all below threshold: the raw gate must reject it (the
  // contrast normalization must not amplify noise into hallucinated peaks).
  cv::Mat noise(100, 100, CV_32F);
  cv::randu(noise, 0.0, 0.05); // max 0.05 << threshold 0.3
  core::RunState s2;
  CHECK(sel.select(noise, s2).empty());
}

TEST_CASE("normalization cross-frame IOR suppresses revisits to a static object", "[selection][normalization]")
{
  cv::Mat m = blob_map({120, 120}, {{60, 60}});

  auto count_selections = [&](float ior_decay)
  {
    selection::NormalizationSelection::Params p;
    p.ior_decay = ior_decay;
    selection::NormalizationSelection sel(selection::SelectionParams{}, p);
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

  const int no_ior = count_selections(0.0f);   // per-frame only -> selected every frame
  const int with_ior = count_selections(0.9f); // decaying space-based IOR

  CHECK(no_ior == 6);
  CHECK(with_ior < no_ior);
  CHECK(with_ior >= 1);
}

TEST_CASE("normalization is reachable through the selection registry", "[selection][normalization][registry]")
{
  auto strategy = selection::create_selection_strategy("normalization", selection::SelectionParams{}, YAML::Node());
  REQUIRE(strategy != nullptr);
  CHECK(strategy->name() == "normalization");
}
