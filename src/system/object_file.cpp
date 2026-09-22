#include "attention/system/object_file.h"
#include <algorithm>
#include <cmath>
#include <limits>

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

// Fold a duplicate file's history into the one that stays: object-based IOR
// keeps the latest selection, the value and age carry over.
void absorb(ObjectFile& kept, const ObjectFile& gone)
{
  kept.last_selected_frame = std::max(kept.last_selected_frame, gone.last_selected_frame);
  kept.selection_count += gone.selection_count;
  kept.value = std::max(kept.value, gone.value);
  kept.created_frame = std::min(kept.created_frame, gone.created_frame);
}

// The file's speed over its last trajectory step (px/frame); 0 without history.
double last_speed(const ObjectFile& file)
{
  if (file.trajectory.size() < 2)
  {
    return 0.0;
  }
  return cv::norm(file.trajectory[file.trajectory.size() - 1] - file.trajectory[file.trajectory.size() - 2]);
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
  file.features = cluster.features;
  file.avg_features = cluster.features;
  file.centroid_exact = cluster.centroid_exact;
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
  file.centroid_exact = cluster.centroid_exact;
  file.features = cluster.features;
  if (file.avg_features.size() != cluster.features.size())
  {
    file.avg_features = cluster.features;
  }
  else
  {
    for (size_t i = 0; i < cluster.features.size(); ++i)
    {
      file.avg_features[i] =
          config_.leaky_alpha * cluster.features[i] + (1.0f - config_.leaky_alpha) * file.avg_features[i];
    }
  }
  file.last_seen_frame = frame;
  file.active = true;
  file.trajectory.push_back(cluster.centroid);
  while (static_cast<int>(file.trajectory.size()) > config_.trajectory_length)
  {
    file.trajectory.pop_front();
  }
}

namespace
{
double position_distance(const ObjectFile& file, const Cluster& cluster)
{
  return cv::norm(cv::Point2d(file.centroid.x, file.centroid.y) - cv::Point2d(cluster.centroid.x, cluster.centroid.y));
}

// L2 over the feature means; 0 when either side has none (the criterion is
// then vacuous and position decides).
double feature_distance(const std::vector<float>& file_features, const std::vector<float>& cluster_features)
{
  if (file_features.empty() || file_features.size() != cluster_features.size())
  {
    return 0.0;
  }
  double sum = 0.0;
  for (size_t i = 0; i < file_features.size(); ++i)
  {
    const double d = file_features[i] - cluster_features[i];
    sum += d * d;
  }
  return std::sqrt(sum);
}
} // namespace

void ObjectFileStore::update_thesis(const std::vector<Cluster>& clusters, int frame)
{
  const int nf = static_cast<int>(active_.size());
  const int nc = static_cast<int>(clusters.size());
  const double radius = config_.correspondence_radius;
  std::vector<int> file_to_cluster(nf, -1);
  std::vector<int> cluster_to_file(nc, -1);

  // 1. Unambiguous within the radius: exactly one file for the cluster, and
  //    that file has no other cluster.
  std::vector<std::vector<int>> near_files(nc), near_clusters(nf);
  for (int f = 0; f < nf; ++f)
  {
    for (int c = 0; c < nc; ++c)
    {
      if (position_distance(active_[f], clusters[c]) < radius)
      {
        near_files[c].push_back(f);
        near_clusters[f].push_back(c);
      }
    }
  }
  // 3 (detected here): two or more files whose only cluster is this one — their
  //    clusters have united. The thesis checks this while updating the field;
  //    the store sees it as several files claiming one cluster and nothing else.
  constexpr int kMerged = -2;
  std::vector<std::vector<int>> united(nc);
  for (int c = 0; c < nc; ++c)
  {
    if (near_files[c].size() == 1 && near_clusters[near_files[c][0]].size() == 1)
    {
      cluster_to_file[c] = near_files[c][0];
      file_to_cluster[near_files[c][0]] = c;
    }
    else if (near_files[c].size() >= 2 && std::all_of(near_files[c].begin(), near_files[c].end(),
                                                      [&](int f) { return near_clusters[f].size() == 1; }))
    {
      std::vector<int> files = near_files[c];
      std::sort(files.begin(), files.end(), [&](int a, int b)
                { return position_distance(active_[a], clusters[c]) < position_distance(active_[b], clusters[c]); });
      files.resize(2); // the file points at two predecessors (thesis: "die beiden")
      united[c] = files;
      cluster_to_file[c] = kMerged;
      for (int f : files)
      {
        file_to_cluster[f] = kMerged;
      }
    }
  }

  // 2. The rest: a cluster gets the file that is minimal on both criteria —
  //    position and feature distance — within twice the radius. Two clusters
  //    that want the same file: the nearer one has it.
  struct Proposal
  {
    double distance;
    int file;
    int cluster;
  };
  std::vector<Proposal> proposals;
  for (int c = 0; c < nc; ++c)
  {
    if (cluster_to_file[c] != -1)
    {
      continue;
    }
    int nearest = -1;
    double nearest_d = 2.0 * radius;
    double min_feature = std::numeric_limits<double>::max();
    for (int f = 0; f < nf; ++f)
    {
      if (file_to_cluster[f] != -1)
      {
        continue;
      }
      const double d = position_distance(active_[f], clusters[c]);
      if (d >= 2.0 * radius)
      {
        continue;
      }
      min_feature = std::min(min_feature, feature_distance(active_[f].avg_features, clusters[c].features));
      if (d < nearest_d)
      {
        nearest_d = d;
        nearest = f;
      }
    }
    if (nearest != -1 && feature_distance(active_[nearest].avg_features, clusters[c].features) <=
                             min_feature + config_.feature_tolerance)
    {
      proposals.push_back({nearest_d, nearest, c});
    }
  }
  std::sort(proposals.begin(), proposals.end(), [](const Proposal& a, const Proposal& b)
            { return a.distance != b.distance ? a.distance < b.distance : a.cluster < b.cluster; });
  for (const auto& p : proposals)
  {
    if (file_to_cluster[p.file] == -1)
    {
      file_to_cluster[p.file] = p.cluster;
      cluster_to_file[p.cluster] = p.file;
    }
  }

  std::vector<ObjectFile> next_active;
  next_active.reserve(active_.size() + clusters.size());
  // Files without a cluster go inactive now, so that a cluster left over below
  // can be recognized as one of them by its features (objects that crossed)
  for (int f = 0; f < nf; ++f)
  {
    if (file_to_cluster[f] >= 0)
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

  for (int c = 0; c < nc; ++c)
  {
    if (cluster_to_file[c] >= 0)
    {
      continue;
    }
    if (cluster_to_file[c] == kMerged)
    {
      // 3. A new file that points at both predecessors until features decide
      ObjectFile merged = make_file(clusters[c], frame);
      merged.merged_from = {active_[united[c][0]].label, active_[united[c][1]].label};
      next_active.push_back(merged);
      continue;
    }
    // 4. An inactive file, primarily by its features; else a new file.
    const int revive = thesis_inactive(clusters[c]);
    if (revive != -1)
    {
      ObjectFile revived = inactive_[revive];
      inactive_.erase(inactive_.begin() + revive);
      update_file(revived, clusters[c], frame);
      next_active.push_back(revived);
    }
    else
    {
      next_active.push_back(make_file(clusters[c], frame));
    }
  }
  active_ = std::move(next_active);
  resolve_merges(frame);

  // 5. Maximum age (§7.2.4). Predecessors of an unresolved merge are kept.
  inactive_.erase(
      std::remove_if(inactive_.begin(), inactive_.end(),
                     [&](const ObjectFile& f)
                     {
                       if (frame - f.last_seen_frame <= config_.max_inactive_age)
                       {
                         return false;
                       }
                       for (const auto& a : active_)
                       {
                         if (std::find(a.merged_from.begin(), a.merged_from.end(), f.label) != a.merged_from.end())
                         {
                           return false;
                         }
                       }
                       return true;
                     }),
      inactive_.end());
}

void ObjectFileStore::resolve_merges(int frame)
{
  for (auto& file : active_)
  {
    if (file.merged_from.size() != 2 || frame - file.created_frame < config_.merge_resolve_frames)
    {
      continue;
    }
    int index[2] = {-1, -1};
    for (int i = 0; i < static_cast<int>(inactive_.size()); ++i)
    {
      for (int k = 0; k < 2; ++k)
      {
        if (inactive_[i].label == file.merged_from[k])
        {
          index[k] = i;
        }
      }
    }
    file.merged_from.clear();
    if (index[0] == -1 || index[1] == -1 || file.avg_features.empty())
    {
      continue; // a predecessor is gone, or there is nothing to compare: the file stays new
    }
    const double d0 = feature_distance(inactive_[index[0]].avg_features, file.avg_features);
    const double d1 = feature_distance(inactive_[index[1]].avg_features, file.avg_features);
    // "Unambiguous": the difference to one is less than half that to the other
    int winner = -1;
    if (d0 < 0.5 * d1)
    {
      winner = 0;
    }
    else if (d1 < 0.5 * d0)
    {
      winner = 1;
    }
    if (winner == -1)
    {
      continue;
    }
    // The merged file *is* that predecessor: it takes over label and history
    // and keeps the current state.
    ObjectFile continued = inactive_[index[winner]];
    continued.centroid = file.centroid;
    continued.centroid_exact = file.centroid_exact;
    continued.bbox = file.bbox;
    continued.size = file.size;
    continued.saliency = file.saliency;
    continued.avg_saliency = file.avg_saliency;
    continued.features = file.features;
    continued.avg_features = file.avg_features;
    continued.appearance = file.appearance;
    continued.last_seen_frame = file.last_seen_frame;
    continued.last_selected_frame = std::max(continued.last_selected_frame, file.last_selected_frame);
    continued.selection_count += file.selection_count;
    continued.active = true;
    for (const auto& point : file.trajectory)
    {
      continued.trajectory.push_back(point);
    }
    while (static_cast<int>(continued.trajectory.size()) > config_.trajectory_length)
    {
      continued.trajectory.pop_front();
    }
    inactive_.erase(inactive_.begin() + index[winner]);
    file = continued;
  }
}

int ObjectFileStore::thesis_inactive(const Cluster& cluster) const
{
  int best = -1;
  double best_feature = std::numeric_limits<double>::max();
  double best_distance = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(inactive_.size()); ++i)
  {
    const ObjectFile& file = inactive_[i];
    const bool held_by_merge = std::any_of(
        active_.begin(), active_.end(), [&](const ObjectFile& a)
        { return std::find(a.merged_from.begin(), a.merged_from.end(), file.label) != a.merged_from.end(); });
    if (held_by_merge)
    {
      continue; // its fate is decided when the merge is resolved
    }
    const double d = position_distance(file, cluster);
    const bool comparable = !file.avg_features.empty() && file.avg_features.size() == cluster.features.size();
    if (!comparable)
    {
      // Nothing to compare by: only a file that was last seen here qualifies
      if (d < config_.correspondence_radius && best_feature == std::numeric_limits<double>::max() && d < best_distance)
      {
        best = i;
        best_distance = d;
      }
      continue;
    }
    const double fd = feature_distance(file.avg_features, cluster.features);
    if (fd >= config_.feature_gate)
    {
      continue;
    }
    // Primarily the features; among equally similar files the nearest
    const bool clearly_better = fd + config_.feature_tolerance < best_feature;
    const bool as_good = fd <= best_feature + config_.feature_tolerance;
    if (clearly_better || (as_good && d < best_distance))
    {
      best = i;
      best_feature = std::min(best_feature, fd);
      best_distance = d;
    }
  }
  return best;
}

void ObjectFileStore::update(const std::vector<Cluster>& clusters, int frame)
{
  if (config_.rule == Rule::Thesis)
  {
    update_thesis(clusters, frame);
    return;
  }
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

  // Unmatched clusters: revive a matching inactive file if one exists,
  // otherwise create a new file.
  for (int c = 0; c < nc; ++c)
  {
    if (cluster_to_file[c] != -1)
    {
      continue;
    }
    const int best =
        config_.persistent_identity ? reidentify(clusters[c], frame) : nearest_inactive(clusters[c], frame);
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
  if (config_.persistent_identity)
  {
    merge_duplicates();
  }

  // Age out inactive files unseen for too long — unless memory is persistent.
  if (!config_.persistent_identity)
  {
    inactive_.erase(std::remove_if(inactive_.begin(), inactive_.end(), [&](const ObjectFile& f)
                                   { return frame - f.last_seen_frame > config_.max_inactive_age; }),
                    inactive_.end());
  }
}

void ObjectFileStore::merge_duplicates()
{
  std::vector<bool> dropped(active_.size(), false);
  for (size_t i = 0; i < active_.size(); ++i)
  {
    for (size_t j = i + 1; j < active_.size() && !dropped[i]; ++j)
    {
      const ObjectFile& a = active_[i];
      const ObjectFile& b = active_[j];
      if (dropped[j] || cv::norm(a.appearance) == 0.0 || cv::norm(b.appearance) == 0.0)
      {
        continue; // no descriptor to judge by
      }
      const double inter = (a.bbox & b.bbox).area();
      const double smaller = std::min(a.bbox.area(), b.bbox.area());
      if (smaller <= 0.0 || inter < 0.5 * smaller || cv::norm(a.appearance - b.appearance) >= config_.reid_colour_gate)
      {
        continue; // not the same place, or doesn't look the same
      }
      const size_t keep = a.label < b.label ? i : j;
      const size_t drop = keep == i ? j : i;
      absorb(active_[keep], active_[drop]);
      dropped[drop] = true;
    }
  }
  std::vector<ObjectFile> merged;
  merged.reserve(active_.size());
  for (size_t i = 0; i < active_.size(); ++i)
  {
    if (!dropped[i])
    {
      merged.push_back(active_[i]);
    }
  }
  active_ = std::move(merged);

  // An inactive look-alike last seen within the correspondence radius of an
  // active file is the same object: fold it in, so a later revival can't flip
  // the object back to the old duplicate's label.
  inactive_.erase(std::remove_if(inactive_.begin(), inactive_.end(),
                                 [&](const ObjectFile& f)
                                 {
                                   if (cv::norm(f.appearance) == 0.0)
                                   {
                                     return false;
                                   }
                                   for (auto& a : active_)
                                   {
                                     const bool alike =
                                         cv::norm(a.appearance) > 0.0 &&
                                         cv::norm(a.appearance - f.appearance) < config_.reid_colour_gate;
                                     const double d = cv::norm(cv::Point2d(a.centroid.x, a.centroid.y) -
                                                               cv::Point2d(f.centroid.x, f.centroid.y));
                                     if (alike && d < config_.correspondence_radius)
                                     {
                                       absorb(a, f);
                                       return true;
                                     }
                                   }
                                   return false;
                                 }),
                  inactive_.end());
}

int ObjectFileStore::nearest_inactive(const Cluster& cluster, int frame) const
{
  int best = -1;
  double best_dist = config_.correspondence_radius;
  for (int i = 0; i < static_cast<int>(inactive_.size()); ++i)
  {
    // Extrapolate the inactive file forward over the frames it was gone, so an
    // object that kept moving while occluded is revived at where it should be.
    const int gone = std::max(1, frame - inactive_[i].last_seen_frame);
    const cv::Point2d predicted = expected_centroid(inactive_[i], config_.motion_prediction, gone);
    const double d = cv::norm(predicted - cv::Point2d(cluster.centroid.x, cluster.centroid.y));
    if (d < best_dist)
    {
      best_dist = d;
      best = i;
    }
  }
  return best;
}

int ObjectFileStore::reidentify(const Cluster& cluster, int frame) const
{
  int best = -1;
  double best_cost = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(inactive_.size()); ++i)
  {
    const ObjectFile& file = inactive_[i];
    const int gone = std::max(1, frame - file.last_seen_frame);
    const cv::Point2d predicted = expected_centroid(file, config_.motion_prediction, gone);
    const double d = cv::norm(predicted - cv::Point2d(cluster.centroid.x, cluster.centroid.y));
    // Appearance rules only when both sides carry a descriptor (a cluster
    // segmented without its frame has none); otherwise the position gate alone.
    const bool compare_look = cv::norm(file.appearance) > 0.0 && cv::norm(cluster.appearance) > 0.0;
    const double colour = compare_look ? cv::norm(file.appearance - cluster.appearance) : 0.0;
    if (compare_look && colour >= config_.reid_colour_veto)
    {
      continue; // looks different: a different object, however close
    }
    // Where it could be by now: the gate widens with the time it was unseen.
    const double gate = config_.correspondence_radius + config_.gate_growth * last_speed(file) * gone;
    const bool look_alike = compare_look && colour < config_.reid_colour_gate;
    if (d >= gate && !look_alike)
    {
      continue;
    }
    const double cost = d + config_.appearance_weight * colour;
    if (cost < best_cost)
    {
      best_cost = cost;
      best = i;
    }
  }
  return best;
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

void ObjectFileStore::add_value(int label, float amount)
{
  if (ObjectFile* file = find_active(label))
  {
    file->value += amount;
  }
}

void ObjectFileStore::decay_values(float factor)
{
  for (auto& file : active_)
  {
    file.value *= factor;
  }
  for (auto& file : inactive_)
  {
    file.value *= factor;
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

void ObjectFileStore::displace(const cv::Point2f& shift)
{
  const cv::Point delta(static_cast<int>(std::lround(shift.x)), static_cast<int>(std::lround(shift.y)));
  auto move = [&](ObjectFile& file)
  {
    file.centroid += delta;
    file.bbox.x += delta.x;
    file.bbox.y += delta.y;
    if (file.centroid_exact.x >= 0.0f)
    {
      file.centroid_exact += shift;
    }
    for (auto& point : file.trajectory)
    {
      point += delta;
    }
  };
  for (auto& file : active_)
  {
    move(file);
  }
  for (auto& file : inactive_)
  {
    move(file);
  }
}

void ObjectFileStore::reset()
{
  active_.clear();
  inactive_.clear();
  next_label_ = 1;
}

} // namespace system
} // namespace attention
