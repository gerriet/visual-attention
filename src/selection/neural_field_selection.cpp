// 2D dynamic neural field selection, ported from the dissertation's
// NeuralField2D (reference/old_code/nf2d.h) with the two-stage readout of
// thesis chapters 6/7 in minimal form (activation clusters -> Objectfile-lite
// -> priority order by mean saliency). Deliberate deviations from the
// original, within the project's loose-equivalence bar (see V2_ROADMAP.md):
//  - true logistic sigmoid instead of the five-piece linear approximation
//    (the approximation only existed for 2003-era speed)
//  - stage 2 is priority ordering only; the symbolic Objectfile stage with
//    behavior model, dwell time and object tracking is roadmap M6

#include "attention/selection/neural_field_selection.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace attention
{
namespace selection
{

NeuralFieldSelection::NeuralFieldSelection(const SelectionParams& shared, const Params& params)
  : shared_(shared), params_(params)
{
  if (params_.kernel_size % 2 == 0)
  {
    throw std::runtime_error("NeuralFieldSelection: kernel_size must be odd");
  }

  // "Backer version" lateral kernel: k*exp(-d) - (k/3)*exp(-d/10) with
  // d = r^2 / s^2 (nf2d.h setkernels)
  const int size = params_.kernel_size;
  const int center = size / 2;
  const float s2 = params_.kernel_s * params_.kernel_s;
  kernel_ = cv::Mat(size, size, CV_32F);
  for (int y = 0; y < size; ++y)
  {
    for (int x = 0; x < size; ++x)
    {
      float dist = ((x - center) * (x - center) + (y - center) * (y - center)) / s2;
      kernel_.at<float>(y, x) = params_.kernel_k * std::exp(-dist) - params_.kernel_k / 3.0f * std::exp(-dist / 10.0f);
    }
  }
}

cv::Mat NeuralFieldSelection::sigmoid(const cv::Mat& activity) const
{
  // sig(u) = 1 / (1 + exp(-beta * u))
  cv::Mat result;
  cv::exp(activity * -params_.beta, result);
  result = 1.0f / (1.0f + result);
  return result;
}

cv::Mat NeuralFieldSelection::make_border_suppression(const cv::Size& size) const
{
  // Original "quick hack for suppressing wrong border activation (missing
  // inhibition)": -0.5/(dist+2) within border_margin pixels of the border
  cv::Mat border = cv::Mat::zeros(size, CV_32F);
  const int x2 = size.width / 2;
  const int y2 = size.height / 2;
  for (int y = 0; y < size.height; ++y)
  {
    for (int x = 0; x < size.width; ++x)
    {
      int dist = std::min(x2 - std::abs(x - x2), y2 - std::abs(y - y2));
      if (dist < params_.border_margin)
      {
        border.at<float>(y, x) = -0.5f / (dist + 2);
      }
    }
  }
  return border;
}

void NeuralFieldSelection::run_to_convergence(cv::Mat& activity, const cv::Mat& input, const cv::Mat& border) const
{
  const float pixel_count = static_cast<float>(activity.total());
  const float threshold = params_.change_thresh * pixel_count; // original: sum |du| vs 0.01 * N

  // Live mode: run a fixed number of cycles per frame (no convergence break),
  // so with field state carried in RunState attention evolves across frames.
  const bool fixed = params_.cycles_per_frame > 0;
  const int cycles = fixed ? params_.cycles_per_frame : params_.max_cycles;

  cv::Mat sig = sigmoid(activity);

  for (int cycle = 0; cycle < cycles; ++cycle)
  {
    // Lateral interaction: convolution of the sigmoid activity with the
    // kernel; zero border like the original (whose missing border inhibition
    // the border term compensates)
    cv::Mat lateral;
    cv::filter2D(sig, lateral, CV_32F, kernel_, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);

    float sigmean = static_cast<float>(cv::mean(sig)[0]);
    float global_effect = params_.resting - sigmean * params_.global_mult;

    cv::Mat updated = params_.alpha * (global_effect + lateral + border + params_.input_mult * input) +
                      (1.0f - params_.alpha) * activity;

    float change = static_cast<float>(cv::sum(cv::abs(updated - activity))[0]);
    activity = updated;
    sig = sigmoid(activity);

    if (!fixed && change <= threshold && cycle >= 2) // original: minimum 3 cycles
    {
      break;
    }
  }
}

namespace
{

// Objectfile-lite: the symbolic handle for one activation cluster
// (thesis ch. 7: centroid, pixel count, region saliency)
struct ClusterInfo
{
  int label = 0;
  cv::Point centroid;
  int size = 0;
  float mean_saliency = 0.0f;
};

} // namespace

std::vector<core::Peak> NeuralFieldSelection::select(const cv::Mat& saliency, core::RunState& state) const
{
  std::vector<core::Peak> peaks;
  if (saliency.empty())
  {
    return peaks;
  }

  // Field runs at reduced resolution (downscale only)
  float scale = 1.0f;
  int max_side = std::max(saliency.cols, saliency.rows);
  if (max_side > params_.field_max_size)
  {
    scale = static_cast<float>(params_.field_max_size) / max_side;
  }

  cv::Mat input;
  if (scale < 1.0f)
  {
    cv::resize(saliency, input, cv::Size(), scale, scale, cv::INTER_AREA);
  }
  else
  {
    input = saliency.clone();
  }

  // Per-run state persists across stream frames; (re)initialize when absent
  // or the resolution changed
  if (state.field_activity.size() != input.size() || state.field_activity.type() != CV_32F)
  {
    state.field_activity = cv::Mat(input.size(), CV_32F, cv::Scalar(params_.resting));
    state.inhibition_map = cv::Mat::zeros(input.size(), CV_32F);
  }
  cv::Mat activity = state.field_activity;

  // Space-based IOR (thesis §8.3): decay the inhibition map, subtract it
  // from this frame's input
  state.inhibition_map *= params_.ior_decay;
  cv::Mat working_input = input - state.inhibition_map;

  // Stage 1: field dynamics settle into activation clusters
  cv::Mat border = make_border_suppression(input.size());
  run_to_convergence(activity, working_input, border);

  // Stage 2 (minimal): one Objectfile-lite per connected active region,
  // fixations at cluster centroids in priority order by mean saliency
  cv::Mat active_mask = activity > 0.0f; // CV_8U
  cv::Mat labels, stats, centroids;
  int num_labels = cv::connectedComponentsWithStats(active_mask, labels, stats, centroids, 8, CV_32S);

  std::vector<ClusterInfo> clusters;
  for (int label = 1; label < num_labels; ++label) // 0 = background
  {
    ClusterInfo cluster;
    cluster.label = label;
    cluster.size = stats.at<int>(label, cv::CC_STAT_AREA);
    if (cluster.size < params_.min_cluster_size)
    {
      continue;
    }
    cluster.centroid = cv::Point(static_cast<int>(centroids.at<double>(label, 0) + 0.5),
                                 static_cast<int>(centroids.at<double>(label, 1) + 0.5));
    cluster.mean_saliency = static_cast<float>(cv::mean(input, labels == label)[0]);
    clusters.push_back(cluster);
  }

  std::sort(clusters.begin(), clusters.end(),
            [](const ClusterInfo& a, const ClusterInfo& b) { return a.mean_saliency > b.mean_saliency; });

  int count = std::min(static_cast<int>(clusters.size()), shared_.max_count);
  for (int i = 0; i < count; ++i)
  {
    const auto& cluster = clusters[i];
    cv::Point full_loc(std::min(static_cast<int>(cluster.centroid.x / scale + 0.5f), saliency.cols - 1),
                       std::min(static_cast<int>(cluster.centroid.y / scale + 0.5f), saliency.rows - 1));
    peaks.push_back(core::Peak(full_loc, cluster.mean_saliency));

    // Feed the space-based IOR map: selected regions are inhibited in
    // subsequent frames ("corresponding pixels get high activation")
    state.inhibition_map.setTo(shared_.ior_strength, labels == cluster.label);
  }

  state.field_activity = activity;
  return peaks;
}

} // namespace selection
} // namespace attention
