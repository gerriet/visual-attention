// Field dynamics harness (replication dossier, roadmap M10): drives the 2D
// dynamic neural field with synthetic activation — no image, no features — and
// re-runs the dissertation's field experiments (ch. 6.2/6.3):
//
//   hysteresis   Abb. 6.4   two input peaks of amplitude a and 1-a; a swept up and
//                           down: the cluster switches late, and at different a
//   bifurcation  Abb. 6.5   one maximum split into two: from a certain distance two
//                           activation clusters
//   noise        Abb. 6.6   a rectangular pulse plus uniform noise: activation
//                           inside vs outside the pulse
//   convergence  Abb. 6.8   cycles from rest to a stable state (mean |du| < 0.02)
//   tracking     Abb. 6.9   a target moving through Gaussian noise: update cycles
//                           per frame needed to keep it, by speed and amplitude
//   approach     Abb. 6.10  two maxima approaching: repulsion, then merging
//
// Output: one JSON document on stdout (read by eval/replication.py). Field
// parameters are those of configs/thesis/thesis.yaml. The thesis does not give
// the shape of its input peaks; here they are Gaussian blobs (sigma 3 px) on a
// 64 x 64 field, which is stated wherever a number depends on it.

#include "attention/selection/neural_field_selection.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using attention::selection::NeuralFieldSelection;
using attention::selection::SelectionParams;

namespace
{

constexpr int kField = 64;
constexpr float kBlobSigma = 3.0f;

// Which parameter set the experiments run on (--params):
//   port   the reimplementation's defaults = the default arguments of the
//          original's setkernels()/setparameters()
//   esab2  what the dissertation system itself configured for its single 2D field
//          (esab2.C): setparameters(0.33, 8, -0.33, 30, ...),
//          set_DoG_kernels(15, 15, 3.3, 0.12, 14, 0.03)
std::string g_parameter_set = "port";

NeuralFieldSelection make_field(int cycles_per_frame = 0)
{
  NeuralFieldSelection::Params params;
  if (g_parameter_set == "esab2")
  {
    params.alpha = 0.33f;
    params.global_mult = 8.0f;
    params.resting = -0.33f;
    params.kernel_size = 15;
    params.kernel_s = 3.3f;
    params.kernel_k = 0.12f;
    params.kernel_s2 = 14.0f;
    params.kernel_k2 = 0.03f;
  }
  params.max_cycles = 55;       // thesis: experiments stopped after 55 cycles
  params.change_thresh = 0.02f; // thesis: mean |du| < 0.02
  params.cycles_per_frame = cycles_per_frame;
  return NeuralFieldSelection(SelectionParams{}, params);
}

void add_blob(cv::Mat& map, float cx, float cy, float amplitude, float sigma = kBlobSigma)
{
  for (int y = 0; y < map.rows; ++y)
  {
    for (int x = 0; x < map.cols; ++x)
    {
      const float d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      map.at<float>(y, x) += amplitude * std::exp(-d2 / (2.0f * sigma * sigma));
    }
  }
}

struct Cluster
{
  double x = 0.0, y = 0.0;
  int size = 0;
};

// Activation clusters R(u) = {x | u(x) > 0} (thesis eq. 6.2), largest first
std::vector<Cluster> clusters_of(const cv::Mat& activity, int min_size = 2)
{
  cv::Mat labels, stats, centroids;
  const int count = cv::connectedComponentsWithStats(activity > 0.0f, labels, stats, centroids, 8, CV_32S);
  std::vector<Cluster> out;
  for (int i = 1; i < count; ++i)
  {
    const int size = stats.at<int>(i, cv::CC_STAT_AREA);
    if (size >= min_size)
    {
      out.push_back({centroids.at<double>(i, 0), centroids.at<double>(i, 1), size});
    }
  }
  std::sort(out.begin(), out.end(), [](const Cluster& a, const Cluster& b) { return a.size > b.size; });
  return out;
}

// --- a minimal JSON writer ------------------------------------------------------

std::string num(double v)
{
  std::ostringstream s;
  s << std::setprecision(6) << v;
  return s.str();
}

std::string array_of(const std::vector<double>& values)
{
  std::string s = "[";
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    s += (i ? ", " : "") + num(values[i]);
  }
  return s + "]";
}

// --- experiments ------------------------------------------------------------------

// Abb. 6.4. Peaks at x = 16 and x = 48; amplitudes a and 1 - a, scaled by `gain`
// so that a single full peak clearly exceeds the resting level. The field state
// is carried from step to step — that is what makes it a hysteresis experiment.
std::string hysteresis()
{
  const auto field = make_field();
  const float gain = 1.0f;
  std::vector<double> alphas, position_up, position_down;
  for (int i = 0; i <= 20; ++i)
  {
    alphas.push_back(i / 20.0);
  }
  cv::Mat activity = field.resting_field(cv::Size(kField, kField));

  // Position of the (largest) activation cluster, or -1 if the field is silent
  auto step = [&](double alpha)
  {
    cv::Mat input = cv::Mat::zeros(kField, kField, CV_32F);
    add_blob(input, 16, 32, gain * static_cast<float>(alpha));
    add_blob(input, 48, 32, gain * static_cast<float>(1.0 - alpha));
    field.relax(activity, input);
    const auto clusters = clusters_of(activity);
    return clusters.empty() ? -1.0 : clusters.front().x;
  };
  for (double alpha : alphas)
  {
    position_up.push_back(step(alpha));
  }
  for (auto it = alphas.rbegin(); it != alphas.rend(); ++it)
  {
    position_down.push_back(step(*it));
  }
  std::reverse(position_down.begin(), position_down.end()); // index by alpha again

  // Where the cluster is first found at the lower peak (x = 16) going up, and
  // where it is last found there going down
  double switch_up = -1.0, switch_down = -1.0;
  for (std::size_t i = 0; i < alphas.size(); ++i)
  {
    if (switch_up < 0.0 && position_up[i] >= 0.0 && position_up[i] < 32.0)
    {
      switch_up = alphas[i];
    }
    if (position_down[i] >= 0.0 && position_down[i] < 32.0 && switch_down < 0.0)
    {
      switch_down = alphas[i];
    }
  }
  return "{\"alpha\": " + array_of(alphas) + ", \"position_alpha_rising\": " + array_of(position_up) +
         ", \"position_alpha_falling\": " + array_of(position_down) +
         ", \"switch_to_peak16_rising\": " + num(switch_up) +
         ", \"holds_peak16_down_to_falling\": " + num(switch_down) + "}";
}

// Abb. 6.5 (separate = true: the maxima move apart) and Abb. 6.10 (they approach).
// State carried; per distance: number of clusters and their separation.
std::string two_maxima(bool separate)
{
  const auto field = make_field();
  std::vector<double> distances, counts, separations;
  for (int d = 0; d <= 30; ++d)
  {
    distances.push_back(d);
  }
  if (!separate)
  {
    std::reverse(distances.begin(), distances.end());
  }
  cv::Mat activity = field.resting_field(cv::Size(kField, kField));
  for (double d : distances)
  {
    cv::Mat input = cv::Mat::zeros(kField, kField, CV_32F);
    add_blob(input, 32.0f - static_cast<float>(d) / 2.0f, 32, 0.8f);
    add_blob(input, 32.0f + static_cast<float>(d) / 2.0f, 32, 0.8f);
    cv::min(input, 1.0f, input); // a master map is in [0, 1]
    field.relax(activity, input);
    const auto clusters = clusters_of(activity);
    counts.push_back(static_cast<double>(clusters.size()));
    separations.push_back(clusters.size() >= 2 ? std::abs(clusters[0].x - clusters[1].x) : 0.0);
  }
  double first_two = -1.0, last_two = -1.0, max_excess = 0.0, excess_at = -1.0;
  int changes = 0;
  for (std::size_t i = 0; i < distances.size(); ++i)
  {
    if (counts[i] >= 2)
    {
      if (first_two < 0.0)
      {
        first_two = distances[i];
      }
      last_two = distances[i];
      const double excess = separations[i] - distances[i]; // > 0: pushed apart
      if (excess > max_excess)
      {
        max_excess = excess;
        excess_at = distances[i];
      }
    }
    if (i > 0 && (counts[i] >= 2) != (counts[i - 1] >= 2))
    {
      ++changes; // 1 = one clean transition; more = oscillation
    }
  }
  return "{\"distance\": " + array_of(distances) + ", \"clusters\": " + array_of(counts) +
         ", \"cluster_separation\": " + array_of(separations) + ", \"two_clusters_first_at\": " + num(first_two) +
         ", \"two_clusters_last_at\": " + num(last_two) + ", \"max_repulsion_px\": " + num(max_excess) +
         ", \"max_repulsion_at_distance\": " + num(excess_at) + ", \"transitions\": " + num(changes) + "}";
}

// Abb. 6.6. A 16 x 16 pulse of amplitude 1 plus uniform noise of amplitude n:
// share of active neurons inside and outside the pulse, over several noise
// samples. The thesis does not say whether its noise was zero-mean, so both
// readings are run: added on top ([0, n]) and zero-mean ([-n/2, n/2]).
std::string noise()
{
  const auto field = make_field();
  const cv::Rect pulse(24, 24, 16, 16);
  std::vector<double> levels;
  std::vector<double> inside[2], outside[2];
  for (int i = 0; i <= 12; ++i)
  {
    levels.push_back(i * 0.2);
  }
  for (int zero_mean = 0; zero_mean < 2; ++zero_mean)
  {
    cv::RNG rng(20040521);
    for (double level : levels)
    {
      double in_sum = 0.0, out_sum = 0.0;
      const int samples = 10;
      for (int s = 0; s < samples; ++s)
      {
        cv::Mat input(kField, kField, CV_32F);
        const double lo = zero_mean ? -level / 2.0 : 0.0;
        rng.fill(input, cv::RNG::UNIFORM, lo, lo + std::max(level, 1e-6));
        input(pulse) += 1.0f;
        cv::Mat activity = field.resting_field(input.size());
        field.relax(activity, input);
        const cv::Mat active = activity > 0.0f;
        const double in_count = cv::countNonZero(active(pulse));
        in_sum += in_count / pulse.area();
        out_sum += (cv::countNonZero(active) - in_count) / (kField * kField - pulse.area());
      }
      inside[zero_mean].push_back(in_sum / samples);
      outside[zero_mean].push_back(out_sum / samples);
    }
  }
  return "{\"noise_amplitude\": " + array_of(levels) + ", \"active_inside\": " + array_of(inside[0]) +
         ", \"active_outside\": " + array_of(outside[0]) + ", \"active_inside_zero_mean\": " + array_of(inside[1]) +
         ", \"active_outside_zero_mean\": " + array_of(outside[1]) + "}";
}

// Abb. 6.8. From rest to a stable state on a small "master map" of three blobs.
std::string convergence()
{
  const auto field = make_field();
  cv::Mat input = cv::Mat::zeros(kField, kField, CV_32F);
  add_blob(input, 16, 18, 0.9f);
  add_blob(input, 44, 24, 0.7f);
  add_blob(input, 30, 48, 0.6f);
  cv::Mat activity = field.resting_field(input.size());
  const int cycles = field.relax(activity, input);
  return "{\"cycles_to_stable\": " + num(cycles) + ", \"clusters\": " + num(clusters_of(activity).size()) + "}";
}

// Abb. 6.9. A target blob crosses the field at `speed` px per frame through
// Gaussian noise; the field gets `cycles` update cycles per frame. Tracking is
// correct while an activation cluster covers at least half of the target (the
// thesis's criterion). Reported: the smallest cycle count (<= 55) that keeps
// the target for the whole crossing, per speed and amplitude; 0 = none does.
std::string tracking()
{
  const std::vector<int> speeds = {1, 2, 4, 6, 8, 10, 12, 14, 16};
  const std::vector<double> amplitudes = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  const std::vector<int> budgets = {1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 55};
  const float target_sigma = 3.0f;
  const float noise_sigma = 0.1f;
  const int width = 128; // room to move

  auto tracked = [&](int speed, double amplitude, int cycles)
  {
    const auto field = make_field(cycles);
    cv::RNG rng(7 + speed * 101 + static_cast<int>(amplitude * 10));
    cv::Mat activity = field.resting_field(cv::Size(width, kField));
    const int frames = (width - 40) / speed;
    for (int f = 0; f < frames; ++f)
    {
      const float cx = 20.0f + f * speed;
      cv::Mat input(kField, width, CV_32F);
      rng.fill(input, cv::RNG::NORMAL, 0.0, noise_sigma);
      add_blob(input, cx, kField / 2.0f, static_cast<float>(amplitude), target_sigma);
      field.relax(activity, input);
      if (f < 2)
      {
        continue; // let the cluster form before judging
      }
      // Target = the disc within one sigma-and-a-half of its centre
      cv::Mat target = cv::Mat::zeros(input.size(), CV_8U);
      cv::circle(target, cv::Point(cvRound(cx), kField / 2), cvRound(1.5f * target_sigma), cv::Scalar(255), cv::FILLED);
      const cv::Mat active = activity > 0.0f;
      const double covered = static_cast<double>(cv::countNonZero(active & target)) / cv::countNonZero(target);
      if (covered < 0.5)
      {
        return false;
      }
    }
    return true;
  };

  std::string rows = "[";
  for (std::size_t a = 0; a < amplitudes.size(); ++a)
  {
    std::vector<double> needed;
    for (int speed : speeds)
    {
      int found = 0;
      for (int cycles : budgets)
      {
        if (tracked(speed, amplitudes[a], cycles))
        {
          found = cycles;
          break;
        }
      }
      needed.push_back(found);
    }
    rows += std::string(a ? ", " : "") + "{\"amplitude\": " + num(amplitudes[a]) +
            ", \"cycles_needed\": " + array_of(needed) + "}";
  }
  std::vector<double> speed_values(speeds.begin(), speeds.end());
  return "{\"speed_px_per_frame\": " + array_of(speed_values) + ", \"rows\": " + rows +
         "], \"noise_sigma\": " + num(noise_sigma) + ", \"cycle_cutoff\": 55}";
}

// Not a thesis figure, but what the others hinge on: does a cluster outlive its
// input? A blob is shown, then removed; the field relaxes on zero input.
std::string memory()
{
  const auto field = make_field();
  cv::Mat input = cv::Mat::zeros(kField, kField, CV_32F);
  add_blob(input, 32, 32, 0.8f);
  cv::Mat activity = field.resting_field(input.size());
  field.relax(activity, input);
  const auto with_input = clusters_of(activity);
  std::vector<double> sizes;
  const cv::Mat nothing = cv::Mat::zeros(kField, kField, CV_32F);
  for (int i = 0; i < 10; ++i)
  {
    field.relax(activity, nothing);
    const auto now = clusters_of(activity);
    sizes.push_back(now.empty() ? 0.0 : now.front().size);
  }
  return "{\"cluster_size_with_input\": " + num(with_input.empty() ? 0 : with_input.front().size) +
         ", \"cluster_size_after_input_removed\": " + array_of(sizes) + "}";
}

void print_usage(const char* program, std::ostream& out)
{
  out << "Usage: " << program << " [--params port|esab2] [experiment ...]" << std::endl
      << std::endl
      << "Drives the 2D dynamic neural field with synthetic activation and re-runs the" << std::endl
      << "dissertation's field experiments (ch. 6.2/6.3); prints one JSON document." << std::endl
      << std::endl
      << "  --params port    the reimplementation's field defaults (default)" << std::endl
      << "  --params esab2   the parameters the dissertation system set for its single 2D field" << std::endl
      << std::endl
      << "Experiments (default: all):" << std::endl
      << "  hysteresis   Abb. 6.4   two peaks of amplitude a and 1-a, a swept up and down" << std::endl
      << "  bifurcation  Abb. 6.5   one maximum split into two" << std::endl
      << "  noise        Abb. 6.6   a rectangular pulse plus uniform noise" << std::endl
      << "  convergence  Abb. 6.8   cycles from rest to a stable state" << std::endl
      << "  tracking     Abb. 6.9   cycles per frame needed to keep a moving target" << std::endl
      << "  approach     Abb. 6.10  two maxima approaching: repulsion, then merging" << std::endl
      << "  memory       (no figure) does a cluster outlive its input?" << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
  const std::map<std::string, std::function<std::string()>> experiments = {
      {"hysteresis", hysteresis}, {"bifurcation", [] { return two_maxima(true); }},
      {"noise", noise},           {"convergence", convergence},
      {"tracking", tracking},     {"approach", [] { return two_maxima(false); }},
      {"memory", memory},
  };

  std::vector<std::string> wanted;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
    {
      print_usage(argv[0], std::cout);
      return 0;
    }
    if (arg == "--params" && i + 1 < argc)
    {
      g_parameter_set = argv[++i];
      if (g_parameter_set != "port" && g_parameter_set != "esab2")
      {
        std::cerr << "Error: --params must be port or esab2" << std::endl;
        return 1;
      }
      continue;
    }
    if (experiments.count(arg) == 0)
    {
      std::cerr << "Error: unknown experiment '" << arg << "'" << std::endl << std::endl;
      print_usage(argv[0], std::cerr);
      return 1;
    }
    wanted.push_back(arg);
  }
  if (wanted.empty())
  {
    for (const auto& entry : experiments)
    {
      wanted.push_back(entry.first);
    }
  }

  std::cout << "{\"parameter_set\": \"" << g_parameter_set << "\",\n ";
  for (std::size_t i = 0; i < wanted.size(); ++i)
  {
    std::cout << (i ? ",\n " : "") << "\"" << wanted[i] << "\": " << experiments.at(wanted[i])();
  }
  std::cout << "}" << std::endl;
  return 0;
}
