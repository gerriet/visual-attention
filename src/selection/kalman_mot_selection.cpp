#include "attention/selection/kalman_mot_selection.h"
#include <algorithm>
#include <cmath>

namespace attention
{
namespace selection
{

KalmanMotSelection::KalmanMotSelection(const SelectionParams& shared, const Params& params)
  : shared_(shared), params_(params)
{
}

void KalmanMotSelection::predict(Track& t) const
{
  // Constant-velocity transition, dt = 1 frame.
  cv::Mat F = cv::Mat::eye(4, 4, CV_64F);
  F.at<double>(0, 2) = 1.0;
  F.at<double>(1, 3) = 1.0;

  t.x = F * t.x;
  cv::Mat Q = cv::Mat::eye(4, 4, CV_64F) * params_.process_noise;
  t.P = F * t.P * F.t() + Q;
}

void KalmanMotSelection::correct(Track& t, const cv::Point2f& z) const
{
  // Measure position only.
  cv::Mat H = cv::Mat::zeros(2, 4, CV_64F);
  H.at<double>(0, 0) = 1.0;
  H.at<double>(1, 1) = 1.0;

  cv::Mat R = cv::Mat::eye(2, 2, CV_64F) * params_.measurement_noise;
  cv::Mat measurement = (cv::Mat_<double>(2, 1) << z.x, z.y);

  cv::Mat innovation = measurement - H * t.x;
  cv::Mat S = H * t.P * H.t() + R;
  cv::Mat K = t.P * H.t() * S.inv();

  t.x = t.x + K * innovation;
  cv::Mat I = cv::Mat::eye(4, 4, CV_64F);
  t.P = (I - K * H) * t.P;
}

std::vector<core::Peak> KalmanMotSelection::select(const cv::Mat& saliency, core::RunState& state) const
{
  std::vector<core::Peak> result;
  if (saliency.empty())
  {
    return result;
  }

  // Stream start (also every still in batch mode, which resets RunState).
  if (state.frame_index == 0)
  {
    tracks_.clear();
    next_id_ = 1;
  }

  // --- 1. Detect blobs on the fused saliency map ---------------------------
  double gmin = 0.0, gmax = 0.0;
  cv::minMaxLoc(saliency, &gmin, &gmax);
  // Absolute floor from the shared threshold, but adapt if the map is not
  // normalized to [0, 1] (weighted-sum fusion need not be).
  const float thr = std::max(shared_.threshold, static_cast<float>(0.3 * gmax));

  cv::Mat mask;
  cv::threshold(saliency, mask, thr, 255.0, cv::THRESH_BINARY);
  mask.convertTo(mask, CV_8U);

  cv::Mat labels, stats, centroids;
  const int num = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

  // Per-label peak saliency in one pass over the labelled image.
  std::vector<float> peak_val(std::max(num, 1), 0.0f);
  for (int y = 0; y < labels.rows; ++y)
  {
    const int* lrow = labels.ptr<int>(y);
    const float* srow = saliency.ptr<float>(y);
    for (int x = 0; x < labels.cols; ++x)
    {
      const int l = lrow[x];
      if (l > 0 && srow[x] > peak_val[l])
      {
        peak_val[l] = srow[x];
      }
    }
  }

  const bool have_depth = params_.use_depth && !state.depth_map.empty() && state.depth_map.size() == saliency.size();

  struct Detection
  {
    cv::Point2f pos;
    float saliency;
    float depth;
  };
  std::vector<Detection> detections;
  for (int i = 1; i < num; ++i)
  {
    if (stats.at<int>(i, cv::CC_STAT_AREA) < params_.min_blob_size)
    {
      continue;
    }
    Detection d;
    d.pos = cv::Point2f(static_cast<float>(centroids.at<double>(i, 0)), static_cast<float>(centroids.at<double>(i, 1)));
    d.saliency = peak_val[i];
    d.depth = -1.0f;
    if (have_depth)
    {
      const int cx = std::min(std::max(static_cast<int>(std::lround(d.pos.x)), 0), state.depth_map.cols - 1);
      const int cy = std::min(std::max(static_cast<int>(std::lround(d.pos.y)), 0), state.depth_map.rows - 1);
      d.depth = state.depth_map.at<float>(cy, cx);
    }
    detections.push_back(d);
  }
  // Strongest first: greedy association priority and output order.
  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) { return a.saliency > b.saliency; });

  // --- 2. Predict every track; mark all as unmeasured this frame -----------
  for (auto& t : tracks_)
  {
    predict(t);
    ++t.age;
    ++t.time_since_update;
    t.measured_this_frame = false;
  }

  // --- 3. Greedy data association (detection -> nearest gated track) --------
  std::vector<char> track_used(tracks_.size(), 0);
  for (const auto& d : detections)
  {
    int best = -1;
    double best_dist2 = params_.max_assoc_dist * params_.max_assoc_dist;
    for (size_t k = 0; k < tracks_.size(); ++k)
    {
      if (track_used[k])
      {
        continue;
      }
      const Track& t = tracks_[k];
      const double dx = d.pos.x - t.x.at<double>(0);
      const double dy = d.pos.y - t.x.at<double>(1);
      double dist2 = dx * dx + dy * dy;
      if (have_depth && t.depth >= 0.0f && d.depth >= 0.0f)
      {
        const double dz = static_cast<double>(d.depth - t.depth) * params_.depth_assoc_weight;
        dist2 += dz * dz;
      }
      if (dist2 < best_dist2)
      {
        best_dist2 = dist2;
        best = static_cast<int>(k);
      }
    }

    if (best >= 0)
    {
      Track& t = tracks_[best];
      correct(t, d.pos);
      t.time_since_update = 0;
      t.measured_this_frame = true;
      t.saliency = d.saliency;
      if (d.depth >= 0.0f)
      {
        t.depth = (t.depth < 0.0f) ? d.depth : 0.7f * t.depth + 0.3f * d.depth;
      }
      track_used[best] = 1;
    }
    else
    {
      // Birth: a new track initialized at the detection, velocity unknown.
      Track t;
      t.x = (cv::Mat_<double>(4, 1) << d.pos.x, d.pos.y, 0.0, 0.0);
      t.P = cv::Mat::eye(4, 4, CV_64F);
      t.P.at<double>(2, 2) = 1000.0; // large initial velocity uncertainty
      t.P.at<double>(3, 3) = 1000.0;
      t.id = next_id_++;
      t.saliency = d.saliency;
      t.depth = d.depth;
      t.time_since_update = 0;
      t.measured_this_frame = true;
      tracks_.push_back(std::move(t));
      track_used.push_back(1);
    }
  }

  // --- 4. Cull tracks that coasted longer than allowed (lost) --------------
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track& t) { return t.time_since_update > params_.max_age; }),
                tracks_.end());

  // --- 5. Emit peaks: visible tracks, object-IOR, spatial NMS --------------
  struct Candidate
  {
    cv::Point loc;
    float value;
    size_t index;
  };
  std::vector<Candidate> candidates;
  for (size_t k = 0; k < tracks_.size(); ++k)
  {
    const Track& t = tracks_[k];
    if (!t.measured_this_frame) // do not fixate coasting/occluded tracks
    {
      continue;
    }
    if (params_.object_ior && (state.frame_index - t.last_selected) < params_.ior_frames)
    {
      continue; // still within its refractory window
    }
    if (t.saliency < thr)
    {
      continue;
    }
    Candidate c;
    c.loc.x = std::min(std::max(static_cast<int>(std::lround(t.x.at<double>(0))), 0), saliency.cols - 1);
    c.loc.y = std::min(std::max(static_cast<int>(std::lround(t.x.at<double>(1))), 0), saliency.rows - 1);
    c.value = t.saliency;
    c.index = k;
    candidates.push_back(c);
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.value > b.value; });

  for (const auto& c : candidates)
  {
    if (static_cast<int>(result.size()) >= shared_.max_count)
    {
      break;
    }
    bool too_close = false;
    for (const auto& p : result)
    {
      if (cv::norm(c.loc - p.location) < shared_.min_distance)
      {
        too_close = true;
        break;
      }
    }
    if (too_close)
    {
      continue;
    }
    result.emplace_back(c.loc, c.value);
    tracks_[c.index].last_selected = state.frame_index; // object-based IOR tag
  }

  return result;
}

} // namespace selection
} // namespace attention
