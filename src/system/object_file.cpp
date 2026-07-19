#include "attention/system/object_file.h"
#include <algorithm>

namespace attention
{
namespace system
{

namespace
{
// Where an object file is expected `steps` frames ahead: its last centroid plus
// its trajectory velocity (last step), when motion prediction is enabled and
// there is enough history. Falls back to the last centroid otherwise.
cv::Point2d expected_centroid(const ObjectFile& file, bool use_motion, int steps)
{
  const cv::Point2d last(file.centroid.x, file.centroid.y);
  if (!use_motion || file.trajectory.size() < 2)
  {
    return last;
  }
  const cv::Point& p1 = file.trajectory[file.trajectory.size() - 1];
  const cv::Point& p0 = file.trajectory[file.trajectory.size() - 2];
  const int s = std::max(1, std::min(steps, 8)); // clamp wild extrapolation
  return cv::Point2d(p1.x + s * (p1.x - p0.x), p1.y + s * (p1.y - p0.y));
}
} // namespace

void LabelMemory::add_vote(const std::string& label, float confidence)
{
  Vote& vote = votes[label];
  vote.count += 1;
  vote.confidence_sum += confidence;
}

std::string LabelMemory::best_label() const
{
  std::string best;
  int best_count = 0;
  float best_conf = 0.0f;
  for (const auto& entry : votes)
  {
    if (entry.second.count > best_count ||
        (entry.second.count == best_count && entry.second.confidence_sum > best_conf))
    {
      best = entry.first;
      best_count = entry.second.count;
      best_conf = entry.second.confidence_sum;
    }
  }
  return best;
}

float LabelMemory::best_confidence() const
{
  const std::string best = best_label();
  if (best.empty())
  {
    return 0.0f;
  }
  const Vote& vote = votes.at(best);
  return vote.count > 0 ? vote.confidence_sum / vote.count : 0.0f;
}

int LabelMemory::best_count() const
{
  const std::string best = best_label();
  return best.empty() ? 0 : votes.at(best).count;
}

ObjectFileStore::ObjectFileStore(const Config& config) : config_(config) {}

ObjectFile ObjectFileStore::make_file(const Cluster& cluster, int frame)
{
  ObjectFile file;
  file.label = next_label_++;
  file.centroid = cluster.centroid;
  file.bbox = cluster.bbox;
  file.size = cluster.size;
  file.saliency = cluster.mean_saliency;
  file.avg_saliency = cluster.mean_saliency;
  file.created_frame = frame;
  file.last_seen_frame = frame;
  file.last_selected_frame = -1;
  file.active = true;
  file.trajectory.push_back(cluster.centroid);
  file.appearance = cluster.appearance;
  return file;
}

void ObjectFileStore::update_file(ObjectFile& file, const Cluster& cluster, int frame)
{
  file.centroid = cluster.centroid;
  file.bbox = cluster.bbox;
  file.size = cluster.size;
  file.saliency = cluster.mean_saliency;
  // Leaky integrator: recent frames weigh more, older information decays.
  file.avg_saliency = config_.leaky_alpha * cluster.mean_saliency + (1.0f - config_.leaky_alpha) * file.avg_saliency;
  file.appearance = 0.5f * cluster.appearance + 0.5f * file.appearance; // stabilize the descriptor
  file.last_seen_frame = frame;
  file.active = true;
  file.trajectory.push_back(cluster.centroid);
  while (static_cast<int>(file.trajectory.size()) > config_.trajectory_length)
  {
    file.trajectory.pop_front();
  }
}

void ObjectFileStore::update(const std::vector<Cluster>& clusters, int frame)
{
  const int nf = static_cast<int>(active_.size());
  const int nc = static_cast<int>(clusters.size());
  const double radius = config_.correspondence_radius;

  std::vector<int> file_to_cluster(nf, -1);
  std::vector<int> cluster_to_file(nc, -1);

  // Greedy nearest-first assignment within 2× the correspondence radius
  // (thesis §7.2.3: accept within threshold, minimize summed errors for the
  // rest). All candidate pairs are sorted by distance and assigned closest
  // first — an approximation of the global minimum that is cheap and
  // deterministic given the small number of files and clusters.
  struct Pair
  {
    double cost;
    int file;
    int cluster;
  };
  std::vector<Pair> pairs;
  for (int f = 0; f < nf; ++f)
  {
    for (int c = 0; c < nc; ++c)
    {
      const cv::Point2d predicted = expected_centroid(active_[f], config_.motion_prediction, 1);
      const double d = cv::norm(predicted - cv::Point2d(clusters[c].centroid.x, clusters[c].centroid.y));
      if (d >= 2.0 * radius) // position gate: too far to be the same object
      {
        continue;
      }
      // Appearance folds into the cost (not the gate): a colour mismatch pushes a
      // spatially plausible but wrong-looking cluster down the assignment order,
      // so crossing objects of different colour keep their labels.
      double cost = d;
      if (config_.appearance_matching)
      {
        cost += config_.appearance_weight * cv::norm(active_[f].appearance - clusters[c].appearance);
      }
      pairs.push_back({cost, f, c});
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const Pair& a, const Pair& b)
            {
              if (a.cost != b.cost)
                return a.cost < b.cost;
              if (a.file != b.file)
                return a.file < b.file;
              return a.cluster < b.cluster;
            });
  for (const auto& p : pairs)
  {
    if (file_to_cluster[p.file] == -1 && cluster_to_file[p.cluster] == -1)
    {
      file_to_cluster[p.file] = p.cluster;
      cluster_to_file[p.cluster] = p.file;
    }
  }

  std::vector<ObjectFile> next_active;
  next_active.reserve(active_.size() + clusters.size());

  // Matched files carry over updated; unmatched active files go inactive.
  for (int f = 0; f < nf; ++f)
  {
    if (file_to_cluster[f] != -1)
    {
      update_file(active_[f], clusters[file_to_cluster[f]], frame);
      next_active.push_back(active_[f]);
    }
    else
    {
      active_[f].active = false;
      inactive_.push_back(active_[f]);
    }
  }

  // Unmatched clusters: revive a nearby inactive file if one exists (within the
  // correspondence radius), otherwise create a new file.
  for (int c = 0; c < nc; ++c)
  {
    if (cluster_to_file[c] != -1)
    {
      continue;
    }
    int best = -1;
    double best_dist = radius;
    for (int i = 0; i < static_cast<int>(inactive_.size()); ++i)
    {
      // Extrapolate the inactive file forward over the frames it was gone, so an
      // object that kept moving while occluded is revived at where it should be.
      const int gone = std::max(1, frame - inactive_[i].last_seen_frame);
      const cv::Point2d predicted = expected_centroid(inactive_[i], config_.motion_prediction, gone);
      const double d = cv::norm(predicted - cv::Point2d(clusters[c].centroid.x, clusters[c].centroid.y));
      if (d < best_dist)
      {
        best_dist = d;
        best = i;
      }
    }
    if (best != -1)
    {
      ObjectFile revived = inactive_[best];
      inactive_.erase(inactive_.begin() + best);
      update_file(revived, clusters[c], frame);
      next_active.push_back(revived);
    }
    else
    {
      next_active.push_back(make_file(clusters[c], frame));
    }
  }

  active_ = std::move(next_active);

  // Age out inactive files unseen for too long.
  inactive_.erase(std::remove_if(inactive_.begin(), inactive_.end(), [&](const ObjectFile& f)
                                 { return frame - f.last_seen_frame > config_.max_inactive_age; }),
                  inactive_.end());
}

void ObjectFileStore::mark_selected(int label, int frame)
{
  if (ObjectFile* file = find_active(label))
  {
    file->last_selected_frame = frame;
    file->selection_count += 1;
  }
}

void ObjectFileStore::record_inspection(int label)
{
  if (ObjectFile* file = find_active(label))
  {
    file->labels.add_inspection();
  }
}

void ObjectFileStore::add_label_vote(int label, const std::string& class_label, float confidence)
{
  if (class_label.empty())
  {
    return;
  }
  if (ObjectFile* file = find_active(label))
  {
    file->labels.add_vote(class_label, confidence);
  }
}

ObjectFile* ObjectFileStore::find_active(int label)
{
  for (auto& file : active_)
  {
    if (file.label == label)
    {
      return &file;
    }
  }
  return nullptr;
}

void ObjectFileStore::reset()
{
  active_.clear();
  inactive_.clear();
  next_label_ = 1;
}

} // namespace system
} // namespace attention
