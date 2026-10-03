// Selection strategy tests on synthetic saliency maps.

#include "attention/selection/neural_field_3d.h"
#include "attention/selection/neural_field_selection.h"
#include "attention/selection/random_selection.h"
#include "attention/selection/selection_strategy.h"
#include <catch2/catch_test_macros.hpp>
#include <cmath>
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

TEST_CASE("NMS selection: a flat segment is one peak, not a grid of them", "[selection]")
{
  // Segment-based features (eccentricity, colour contrast) give
  // piecewise-constant maps. Every pixel of a plateau equals its dilation, so
  // a pixel-wise reading tiles the largest segment with peaks min_distance
  // apart and never reaches the others — which put the default readout at
  // chance on V*Bench (docs/FEATURE_ASSESSMENT.md, "Ablation").
  cv::Mat map = cv::Mat::zeros(240, 320, CV_32F);
  cv::rectangle(map, cv::Rect(20, 20, 200, 120), cv::Scalar(0.9f), cv::FILLED); // large, most salient
  cv::rectangle(map, cv::Rect(250, 170, 40, 40), cv::Scalar(0.7f), cv::FILLED);
  cv::rectangle(map, cv::Rect(40, 180, 50, 30), cv::Scalar(0.5f), cv::FILLED);

  selection::SelectionParams params;
  params.threshold = 0.3f;
  params.min_distance = 30;
  params.max_count = 10;
  core::RunState state;
  auto peaks = selection::create_selection_strategy("nms", params, YAML::Node())->select(map, state);

  REQUIRE(peaks.size() == 3);
  CHECK(cv::Rect(20, 20, 200, 120).contains(peaks[0].location));
  CHECK(cv::Rect(250, 170, 40, 40).contains(peaks[1].location));
  CHECK(cv::Rect(40, 180, 50, 30).contains(peaks[2].location));
  // The peak sits inside the segment, away from its border
  CHECK(cv::Rect(60, 50, 120, 60).contains(peaks[0].location));
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

namespace
{

// The field parameters of the dissertation system (esab2.C, single 2D field)
selection::NeuralFieldSelection dissertation_field()
{
  selection::NeuralFieldSelection::Params p;
  p.alpha = 0.33f;
  p.global_mult = 8.0f;
  p.resting = -0.33f;
  p.kernel_size = 15;
  p.kernel_s = 3.3f;
  p.kernel_k = 0.12f;
  p.kernel_s2 = 14.0f;
  p.kernel_k2 = 0.03f;
  p.max_cycles = 55;
  p.change_thresh = 0.02f;
  return selection::NeuralFieldSelection(selection::SelectionParams{}, p);
}

cv::Mat two_maxima(float distance)
{
  cv::Mat input = cv::Mat::zeros(64, 64, CV_32F);
  for (float cx : {32.0f - distance / 2.0f, 32.0f + distance / 2.0f})
  {
    for (int y = 0; y < 64; ++y)
    {
      for (int x = 0; x < 64; ++x)
      {
        const float d2 = (x - cx) * (x - cx) + (y - 32.0f) * (y - 32.0f);
        input.at<float>(y, x) += 0.8f * std::exp(-d2 / 18.0f);
      }
    }
  }
  cv::min(input, 1.0f, input);
  return input;
}

int active_clusters(const cv::Mat& activity)
{
  cv::Mat labels;
  return cv::connectedComponents(activity > 0.0f, labels, 8, CV_32S) - 1;
}

} // namespace

TEST_CASE("field dynamics: relax() converges in about ten cycles (thesis Abb. 6.8)", "[neural-field][dynamics]")
{
  const auto field = dissertation_field();
  cv::Mat activity = field.resting_field(cv::Size(64, 64));
  const int cycles = field.relax(activity, two_maxima(30.0f));
  CHECK(cycles >= 3);  // the original's minimum
  CHECK(cycles <= 20); // "typically already 10 update cycles"
  CHECK(active_clusters(activity) == 2);

  cv::Mat wrong_size = field.resting_field(cv::Size(32, 32));
  CHECK_THROWS(field.relax(wrong_size, two_maxima(30.0f)));
}

TEST_CASE("field dynamics: two maxima coexist when apart and merge when close (Abb. 6.5 / 6.10)",
          "[neural-field][dynamics]")
{
  const auto field = dissertation_field();
  auto clusters_at = [&](float distance)
  {
    cv::Mat activity = field.resting_field(cv::Size(64, 64));
    field.relax(activity, two_maxima(distance));
    return active_clusters(activity);
  };
  CHECK(clusters_at(24.0f) == 2); // beyond the interaction range: coexistence
  CHECK(clusters_at(3.0f) == 1);  // below x_a: one cluster

  // Hysteresis of the transition: approaching maxima stay two clusters at a
  // distance where maxima that are being separated are still one
  cv::Mat approaching = field.resting_field(cv::Size(64, 64));
  for (float d = 30.0f; d >= 9.0f; d -= 1.0f)
  {
    field.relax(approaching, two_maxima(d));
  }
  cv::Mat separating = field.resting_field(cv::Size(64, 64));
  for (float d = 0.0f; d <= 9.0f; d += 1.0f)
  {
    field.relax(separating, two_maxima(d));
  }
  CHECK(active_clusters(approaching) == 2);
  CHECK(active_clusters(separating) == 1);
}

TEST_CASE("field kernel: the DoG parameters default to the Backer kernel", "[neural-field]")
{
  // kernel_k2 / kernel_s2 generalize the lateral kernel; left at their defaults
  // the field must behave exactly as before (k/3 and s*sqrt(10))
  selection::NeuralFieldSelection::Params defaults, explicit_backer;
  explicit_backer.kernel_k2 = defaults.kernel_k / 3.0f;
  explicit_backer.kernel_s2 = defaults.kernel_s * std::sqrt(10.0f);
  const selection::NeuralFieldSelection a(selection::SelectionParams{}, defaults);
  const selection::NeuralFieldSelection b(selection::SelectionParams{}, explicit_backer);
  cv::Mat ua = a.resting_field(cv::Size(64, 64)), ub = b.resting_field(cv::Size(64, 64));
  a.relax(ua, two_maxima(20.0f));
  b.relax(ub, two_maxima(20.0f));
  CHECK(cv::norm(ua, ub, cv::NORM_INF) < 1e-4);
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

TEST_CASE("neural field: activity clusters follow their input from frame to frame", "[selection][neural-field][track]")
{
  // The dissertation system's single 2D field (configs/thesis/thesis.yaml), a
  // fixed 20 cycles per frame as that system ran it
  selection::NeuralFieldSelection::Params params;
  params.alpha = 0.33f;
  params.global_mult = 8.0f;
  params.resting = -0.33f;
  params.kernel_size = 15;
  params.kernel_s = 3.3f;
  params.kernel_k = 0.12f;
  params.kernel_s2 = 14.0f;
  params.kernel_k2 = 0.03f;
  params.cycles_per_frame = 20;
  params.field_max_size = 96;
  const selection::NeuralFieldSelection field(selection::SelectionParams{}, params);

  cv::Mat activity;
  int last_x = 0;
  for (int frame = 0; frame < 12; ++frame)
  {
    cv::Mat map = cv::Mat::zeros(96, 96, CV_32F);
    map(cv::Rect(20 + 2 * frame, 40, 5, 5)) = 1.0f; // moves 2 px per frame
    map(cv::Rect(70, 70, 5, 5)) = 1.0f;             // stays
    const auto clusters = field.track(map, activity);
    REQUIRE(clusters.size() == 2); // no cluster is left behind, none appears
    for (const auto& cluster : clusters)
    {
      if (cluster.centroid.y < 55.0f)
      {
        CHECK(std::abs(cluster.centroid.x - (22.0f + 2.0f * frame)) < 3.0f);
        last_x = static_cast<int>(cluster.centroid.x);
      }
      else
      {
        CHECK(std::abs(cluster.centroid.x - 72.0f) < 1.5f);
        CHECK(std::abs(cluster.centroid.y - 72.0f) < 1.5f);
      }
      CHECK(cluster.mask.size() == map.size());
      CHECK(cluster.size == cv::countNonZero(cluster.mask));
    }
  }
  CHECK(last_x > 38); // it has travelled with its input
}

TEST_CASE("3D neural field: the volume survives across frames and through a gap", "[selection][neural-field-3d][time]")
{
  // Thesis §6.4 extends the field into depth *in order to* track through
  // temporary occlusion, and Abb. 6.14 measures how long tracking survives.
  // Before 2026-09-23 this port rebuilt the field on every frame, so nothing
  // survived anything (dossier, finding C).
  selection::NeuralField3D::Params params;
  params.cycles_per_frame = 20;
  const cv::Size size(48, 48);
  const int depth = 5;

  auto input_with_blob = [&](bool present)
  {
    std::vector<cv::Mat> volume(depth);
    for (int z = 0; z < depth; ++z)
    {
      volume[z] = cv::Mat::zeros(size, CV_32F);
    }
    if (present)
    {
      cv::circle(volume[2], cv::Point(24, 24), 5, cv::Scalar(1.0f), -1);
    }
    return volume;
  };

  selection::NeuralField3D field(size, depth, params);
  field.initialize();

  // Drive it while the stimulus is present, then take the stimulus away.
  for (int t = 0; t < 4; ++t)
  {
    field.update(input_with_blob(true));
  }
  const double driven = cv::sum(field.collapsed_activation() > 0.0f)[0];
  REQUIRE(driven > 0.0); // the field found it at all

  field.update(input_with_blob(false));
  const double after_gap = cv::sum(field.collapsed_activation() > 0.0f)[0];
  CHECK(after_gap > 0.0); // ... and still holds it one frame later

  // A field that starts from rest on the empty frame holds nothing: this is
  // exactly what the port did before, and what the assertion above rules out.
  selection::NeuralField3D fresh(size, depth, params);
  fresh.initialize();
  fresh.update(input_with_blob(false));
  CHECK(cv::sum(fresh.collapsed_activation() > 0.0f)[0] == 0.0);

  // set_activity continues from a given volume; a mismatched one is refused.
  selection::NeuralField3D continued(size, depth, params);
  continued.initialize();
  REQUIRE(continued.set_activity(field.activity()));
  CHECK(cv::norm(continued.activity()[2], field.activity()[2], cv::NORM_INF) == 0.0);
  CHECK_FALSE(continued.set_activity(std::vector<cv::Mat>(depth, cv::Mat::zeros(size, CV_8U))));
  CHECK_FALSE(continued.set_activity(std::vector<cv::Mat>(depth + 1, cv::Mat::zeros(size, CV_32F))));
}

TEST_CASE("3D neural-field selection carries its volume in RunState", "[selection][neural-field-3d][time]")
{
  YAML::Node params;
  params["depth_layers"] = 5;
  params["field_max_size"] = 48;
  params["cycles_per_frame"] = 20;
  auto strategy = selection::create_selection_strategy("neural-field-3d", selection::SelectionParams{}, params);

  cv::Mat saliency = cv::Mat::zeros(48, 48, CV_32F);
  cv::circle(saliency, cv::Point(24, 24), 5, cv::Scalar(1.0f), -1);

  core::RunState state;
  CHECK(state.field_volume.empty());
  strategy->select(saliency, state);
  REQUIRE(state.field_volume.size() == 5);
  CHECK(state.field_volume[0].size() == cv::Size(48, 48));

  // The next frame continues from it: driving the same input again must not
  // reproduce the first frame's activity exactly (the field has moved on).
  const cv::Mat first = state.field_volume[2].clone();
  strategy->select(saliency, state);
  CHECK(cv::norm(first, state.field_volume[2], cv::NORM_INF) > 0.0);

  state.reset();
  CHECK(state.field_volume.empty());
}

TEST_CASE("random selection is the floor: it ignores saliency but keeps the budget", "[selection][random]")
{
  // The control arm H2 and H6 both need. What must hold is not where it looks
  // but that it looks the same *number* of times, spread the same way, so a
  // comparison against it is about placement and not about budget.
  cv::Mat saliency = cv::Mat::zeros(200, 300, CV_32F);
  cv::circle(saliency, cv::Point(250, 50), 10, cv::Scalar(1.0f), -1); // one bright spot

  attention::selection::SelectionParams shared;
  shared.max_count = 6;
  shared.min_distance = 20;
  attention::selection::RandomSelection::Params params;
  params.seed = 7;

  attention::core::RunState state;
  attention::selection::RandomSelection strategy(shared, params);
  const auto peaks = strategy.select(saliency, state);

  REQUIRE(peaks.size() == 6);
  for (const auto& peak : peaks)
  {
    CHECK(peak.location.x >= 0);
    CHECK(peak.location.x < saliency.cols);
    CHECK(peak.location.y >= 0);
    CHECK(peak.location.y < saliency.rows);
  }
  // Spacing is honoured, so this arm is not denser than the arm it controls.
  for (size_t i = 0; i < peaks.size(); ++i)
  {
    for (size_t j = i + 1; j < peaks.size(); ++j)
    {
      const double dx = peaks[i].location.x - peaks[j].location.x;
      const double dy = peaks[i].location.y - peaks[j].location.y;
      CHECK(std::sqrt(dx * dx + dy * dy) >= shared.min_distance);
    }
  }
  // And it really does ignore the map: with one bright spot in 60000 pixels,
  // landing on it would be a coincidence, not a selection.
  int on_the_spot = 0;
  for (const auto& peak : peaks)
  {
    if (cv::norm(peak.location - cv::Point(250, 50)) < 12)
    {
      ++on_the_spot;
    }
  }
  CHECK(on_the_spot == 0);
}

TEST_CASE("random selection is reproducible for a seed and differs between seeds", "[selection][random]")
{
  cv::Mat saliency(120, 160, CV_32F, cv::Scalar(0.5f));
  attention::selection::SelectionParams shared;
  shared.max_count = 5;
  shared.min_distance = 10;

  auto run = [&](unsigned int seed)
  {
    attention::selection::RandomSelection::Params params;
    params.seed = seed;
    attention::core::RunState state;
    return attention::selection::RandomSelection(shared, params).select(saliency, state);
  };

  const auto a = run(42);
  const auto b = run(42);
  const auto c = run(43);
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i)
  {
    CHECK(a[i].location == b[i].location);
  }
  CHECK(a[0].location != c[0].location);
}

TEST_CASE("random selection returns fewer peaks rather than crowding them", "[selection][random]")
{
  // A frame too small to hold max_count well-spaced points must come back
  // short. Relaxing the spacing instead would make the control arm denser than
  // the arm it controls — the one thing it must not do.
  cv::Mat saliency(40, 40, CV_32F, cv::Scalar(0.5f));
  attention::selection::SelectionParams shared;
  shared.max_count = 50;
  shared.min_distance = 30;
  attention::selection::RandomSelection::Params params;
  params.seed = 1;

  attention::core::RunState state;
  const auto peaks = attention::selection::RandomSelection(shared, params).select(saliency, state);
  CHECK(peaks.size() < 50);
  CHECK_FALSE(peaks.empty());
}

TEST_CASE("random selection is reachable from config by name", "[selection][random]")
{
  const auto strategy =
      attention::selection::create_selection_strategy("random", attention::selection::SelectionParams{}, YAML::Node());
  REQUIRE(strategy != nullptr);
  CHECK(strategy->name() == "random");
}
