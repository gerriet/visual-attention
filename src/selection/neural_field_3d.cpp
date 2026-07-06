// 3D dynamic neural field, ported from the dissertation's NeuralField3D
// (reference/old_code/nf3d.h, thesis §6.4): a stack of 2D Amari fields coupled
// by cross-depth inhibition so that at each location a single depth wins. Used
// for depth-aware selection (NeuralField3DSelection). Deliberate deviations,
// within the loose-equivalence bar: logistic sigmoid (not the 5-piece
// approximation); the field re-initializes per frame (no cross-frame volume
// persistence yet) but still uses the 2D space-based IOR map for readout.

#include "attention/selection/neural_field_3d.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace attention
{
namespace selection
{

namespace
{
cv::Mat backer_kernel(int size, float s, float k)
{
  const int center = size / 2;
  const float s2 = s * s;
  cv::Mat kernel(size, size, CV_32F);
  for (int y = 0; y < size; ++y)
  {
    for (int x = 0; x < size; ++x)
    {
      const float dist = ((x - center) * (x - center) + (y - center) * (y - center)) / s2;
      kernel.at<float>(y, x) = k * std::exp(-dist) - k / 3.0f * std::exp(-dist / 10.0f);
    }
  }
  return kernel;
}

// Border-suppression bias, the same "quick hack for missing border inhibition"
// the 2D field uses: -0.5/(dist+2) within `margin` pixels of the border, added
// to the field input so the zero-padded DoG kernel does not sustain phantom
// activation hugging the frame edge.
cv::Mat border_suppression(const cv::Size& size, int margin)
{
  cv::Mat border = cv::Mat::zeros(size, CV_32F);
  if (margin <= 0)
  {
    return border;
  }
  const int x2 = size.width / 2;
  const int y2 = size.height / 2;
  for (int y = 0; y < size.height; ++y)
  {
    for (int x = 0; x < size.width; ++x)
    {
      const int dist = std::min(x2 - std::abs(x - x2), y2 - std::abs(y - y2));
      if (dist < margin)
      {
        border.at<float>(y, x) = -0.5f / (dist + 2);
      }
    }
  }
  return border;
}
} // namespace

NeuralField3D::NeuralField3D(const cv::Size& plane_size, int depth, const Params& params)
  : params_(params), size_(plane_size), depth_(depth)
{
  if (params_.kernel_size % 2 == 0)
  {
    throw std::runtime_error("NeuralField3D: kernel_size must be odd");
  }
  if (depth_ < 1)
  {
    throw std::runtime_error("NeuralField3D: depth must be >= 1");
  }
  kernel_ = backer_kernel(params_.kernel_size, params_.kernel_s, params_.kernel_k);
  initialize();
}

void NeuralField3D::initialize()
{
  activity_.assign(depth_, cv::Mat(size_, CV_32F, cv::Scalar(params_.resting)));
  // assign() shares the same Mat header across planes; give each its own buffer
  for (auto& plane : activity_)
  {
    plane = plane.clone();
  }
}

cv::Mat NeuralField3D::sigmoid(const cv::Mat& u) const
{
  cv::Mat result;
  cv::exp(u * -params_.beta, result);
  result = 1.0f / (1.0f + result);
  return result;
}

int NeuralField3D::update(const std::vector<cv::Mat>& input)
{
  if (static_cast<int>(input.size()) != depth_)
  {
    throw std::runtime_error("NeuralField3D::update: input volume depth mismatch");
  }

  const float total = static_cast<float>(size_.area()) * depth_;
  const float threshold = params_.change_thresh * total; // sum |du| vs mean-per-neuron * N

  int cycle = 0;
  for (; cycle < params_.max_cycles; ++cycle)
  {
    // Synchronous update: all planes use the previous cycle's sigmoids for the
    // depth accumulator, lateral term and global term.
    std::vector<cv::Mat> sig(depth_);
    cv::Mat depth_accu = cv::Mat::zeros(size_, CV_32F);
    std::vector<float> plane_mean(depth_);
    float sigmean = 0.0f;
    for (int z = 0; z < depth_; ++z)
    {
      sig[z] = sigmoid(activity_[z]);
      depth_accu += sig[z];
      plane_mean[z] = static_cast<float>(cv::mean(sig[z])[0]);
      sigmean += plane_mean[z];
    }
    const float global_effect = params_.resting - sigmean * params_.global_mult;
    cv::Mat depth_inhib = depth_accu / static_cast<float>(depth_);

    float change = 0.0f;
    for (int z = 0; z < depth_; ++z)
    {
      cv::Mat lateral;
      cv::filter2D(sig[z], lateral, CV_32F, kernel_, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
      const float plane_effect = -params_.global_mult * plane_mean[z] * params_.plane_inhibition;

      cv::Mat updated =
          params_.alpha * (global_effect + lateral - depth_inhib + plane_effect + params_.input_mult * input[z]) +
          (1.0f - params_.alpha) * activity_[z];
      change += static_cast<float>(cv::sum(cv::abs(updated - activity_[z]))[0]);
      activity_[z] = updated;
    }

    if (change <= threshold && cycle >= 2) // minimum 3 cycles, as in the 2D field
    {
      ++cycle;
      break;
    }
  }
  return cycle;
}

cv::Mat NeuralField3D::collapsed_activation() const
{
  cv::Mat collapsed = activity_[0].clone();
  for (int z = 1; z < depth_; ++z)
  {
    collapsed = cv::max(collapsed, activity_[z]);
  }
  return collapsed;
}

cv::Mat NeuralField3D::winning_depth() const
{
  cv::Mat best = activity_[0].clone();
  cv::Mat argmax = cv::Mat::zeros(size_, CV_32S);
  for (int z = 1; z < depth_; ++z)
  {
    cv::Mat mask = activity_[z] > best; // CV_8U
    activity_[z].copyTo(best, mask);
    argmax.setTo(z, mask);
  }
  return argmax;
}

// ---------------------------------------------------------------------------
// NeuralField3DSelection
// ---------------------------------------------------------------------------

namespace
{
struct ClusterInfo
{
  int label = 0;
  cv::Point centroid;
  int size = 0;
  float mean_saliency = 0.0f;
  float mean_depth = 0.0f;
};
} // namespace

NeuralField3DSelection::NeuralField3DSelection(const SelectionParams& shared, const Params& params)
  : shared_(shared), params_(params)
{
  if (params_.depth_layers < 1)
  {
    throw std::runtime_error("NeuralField3DSelection: depth_layers must be >= 1");
  }
}

std::vector<core::Peak> NeuralField3DSelection::select(const cv::Mat& saliency, core::RunState& state) const
{
  std::vector<core::Peak> peaks;
  if (saliency.empty())
  {
    return peaks;
  }

  // Field runs at reduced resolution (3D is depth_layers× the work of 2D).
  float scale = 1.0f;
  const int max_side = std::max(saliency.cols, saliency.rows);
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

  // Per-pixel depth cue (published by the stereo feature), matched to the
  // field resolution. Empty when no depth feature ran.
  cv::Mat depth_small;
  if (!state.depth_map.empty())
  {
    cv::resize(state.depth_map, depth_small, input.size(), 0, 0, cv::INTER_AREA);
  }

  // Space-based IOR (thesis §8.3): decay the map, subtract from this frame.
  if (state.inhibition_map.size() != input.size() || state.inhibition_map.type() != CV_32F)
  {
    state.inhibition_map = cv::Mat::zeros(input.size(), CV_32F);
  }
  state.inhibition_map *= params_.ior_decay;
  cv::Mat working_input = input - state.inhibition_map;

  // Border suppression (as in the 2D field): bias the input negative near the
  // frame edge so the zero-padded lateral kernel cannot sustain phantom edge
  // clusters. Applied to the input, which the field re-adds every cycle.
  working_input += border_suppression(input.size(), params_.border_margin);

  // Lift the 2D saliency into a depth volume: each pixel's saliency lands in
  // the plane its disparity indexes — a higher depth cue (nearer surface, per
  // the stereo feature's normalized-disparity map) indexes a higher plane.
  // With no depth cue everything goes to the middle plane (reduces to a 2D
  // field in one plane).
  const int Z = params_.depth_layers;
  std::vector<cv::Mat> volume(Z);
  for (int z = 0; z < Z; ++z)
  {
    volume[z] = cv::Mat::zeros(input.size(), CV_32F);
  }
  const int mid = Z / 2;
  for (int y = 0; y < input.rows; ++y)
  {
    const float* wrow = working_input.ptr<float>(y);
    const float* drow = depth_small.empty() ? nullptr : depth_small.ptr<float>(y);
    for (int x = 0; x < input.cols; ++x)
    {
      int layer = mid;
      if (drow)
      {
        layer = static_cast<int>(std::lround(drow[x] * (Z - 1)));
        layer = std::max(0, std::min(Z - 1, layer));
      }
      volume[layer].at<float>(y, x) = wrow[x];
    }
  }

  // Run the 3D field to convergence.
  NeuralField3D field(input.size(), Z, params_.field);
  field.update(volume);
  cv::Mat collapsed = field.collapsed_activation();
  cv::Mat winning = field.winning_depth();

  // Readout: one cluster per connected active region of the collapsed field,
  // fixated in priority order by mean saliency (mirrors the 2D field).
  cv::Mat active_mask = collapsed > 0.0f; // CV_8U
  cv::Mat labels, stats, centroids;
  int num_labels = cv::connectedComponentsWithStats(active_mask, labels, stats, centroids, 8, CV_32S);

  std::vector<ClusterInfo> clusters;
  for (int label = 1; label < num_labels; ++label)
  {
    ClusterInfo cluster;
    cluster.label = label;
    cluster.size = stats.at<int>(label, cv::CC_STAT_AREA);
    if (cluster.size < params_.min_cluster_size)
    {
      continue;
    }
    cv::Mat cluster_mask = labels == label;
    cluster.centroid = cv::Point(static_cast<int>(centroids.at<double>(label, 0) + 0.5),
                                 static_cast<int>(centroids.at<double>(label, 1) + 0.5));
    cluster.mean_saliency = static_cast<float>(cv::mean(input, cluster_mask)[0]);
    cluster.mean_depth = static_cast<float>(cv::mean(winning, cluster_mask)[0]);
    clusters.push_back(cluster);
  }

  std::sort(clusters.begin(), clusters.end(),
            [](const ClusterInfo& a, const ClusterInfo& b) { return a.mean_saliency > b.mean_saliency; });

  const int count = std::min(static_cast<int>(clusters.size()), shared_.max_count);
  for (int i = 0; i < count; ++i)
  {
    const auto& cluster = clusters[i];
    cv::Point full_loc(std::min(static_cast<int>(cluster.centroid.x / scale + 0.5f), saliency.cols - 1),
                       std::min(static_cast<int>(cluster.centroid.y / scale + 0.5f), saliency.rows - 1));
    peaks.push_back(core::Peak(full_loc, cluster.mean_saliency));
    state.inhibition_map.setTo(shared_.ior_strength, labels == cluster.label);
  }

  // Keep the collapsed field available for inspection/visualization.
  state.field_activity = collapsed;
  return peaks;
}

} // namespace selection
} // namespace attention
