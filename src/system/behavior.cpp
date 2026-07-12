#include "attention/system/behavior.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace attention
{
namespace system
{

namespace
{
// Priority ordering for Exploration: `a` outranks `b` when it should be
// selected first. Never-selected objects lead; among ever-selected ones the
// longest-unselected leads; ties break by mean saliency, then label (for a
// deterministic choice).
bool higher_priority(const ObjectFile& a, const ObjectFile& b, int frame)
{
  const bool a_never = !a.ever_selected();
  const bool b_never = !b.ever_selected();
  if (a_never != b_never)
  {
    return a_never;
  }
  if (!a_never)
  {
    const int a_since = frame - a.last_selected_frame;
    const int b_since = frame - b.last_selected_frame;
    if (a_since != b_since)
    {
      return a_since > b_since;
    }
  }
  if (a.saliency != b.saliency)
  {
    return a.saliency > b.saliency;
  }
  return a.label < b.label;
}
} // namespace

const ObjectFile* Exploration::select_focus(ObjectFileStore& store, int frame)
{
  const auto& active = store.active_files();
  if (active.empty())
  {
    current_focus_label_ = -1;
    dwell_count_ = 0;
    return nullptr;
  }

  // Keep dwelling on the current focus if it is still present and the dwell is
  // not yet complete.
  if (current_focus_label_ >= 0 && dwell_count_ < params_.dwell_frames)
  {
    if (ObjectFile* current = store.find_active(current_focus_label_))
    {
      ++dwell_count_;
      store.mark_selected(current->label, frame);
      return store.find_active(current_focus_label_);
    }
    // The focused object vanished mid-dwell — fall through and re-select.
  }

  // Pick the highest-priority active object file.
  const ObjectFile* best = &active[0];
  for (const auto& file : active)
  {
    if (higher_priority(file, *best, frame))
    {
      best = &file;
    }
  }

  current_focus_label_ = best->label;
  dwell_count_ = 1;
  store.mark_selected(best->label, frame);
  return store.find_active(current_focus_label_);
}

void Exploration::reset()
{
  current_focus_label_ = -1;
  dwell_count_ = 0;
}

const ObjectFile* IorBehavior::select_focus(ObjectFileStore& store, int frame)
{
  const auto& active = store.active_files();
  if (active.empty())
  {
    return nullptr;
  }

  // Pick the active object with the highest (saliency − current inhibition).
  const ObjectFile* best = nullptr;
  float best_score = -std::numeric_limits<float>::max();
  for (const auto& file : active)
  {
    float inhibition = 0.0f;
    if (mode_ == Mode::Spatial)
    {
      for (const auto& spot : spatial_)
      {
        const float dx = static_cast<float>(file.centroid.x - spot.loc.x);
        const float dy = static_cast<float>(file.centroid.y - spot.loc.y);
        inhibition += spot.strength * std::exp(-(dx * dx + dy * dy) / (2.0f * params_.ior_radius * params_.ior_radius));
      }
    }
    else if (mode_ == Mode::Object)
    {
      auto it = object_inhib_.find(file.label);
      if (it != object_inhib_.end())
      {
        inhibition = it->second;
      }
    }

    const float score = file.saliency - inhibition;
    if (best == nullptr || score > best_score || (score == best_score && file.label < best->label))
    {
      best_score = score;
      best = &file;
    }
  }

  const int label = best->label;
  const cv::Point where = best->centroid;

  // Deposit inhibition on the winner, then decay every tag (thesis §8.3).
  if (mode_ == Mode::Spatial)
  {
    spatial_.push_back({where, params_.ior_strength});
    for (auto& spot : spatial_)
    {
      spot.strength *= params_.ior_decay;
    }
    spatial_.erase(std::remove_if(spatial_.begin(), spatial_.end(), [](const Spot& s) { return s.strength < 0.02f; }),
                   spatial_.end());
  }
  else if (mode_ == Mode::Object)
  {
    object_inhib_[label] = std::min(1.0f, object_inhib_[label] + params_.ior_strength);
    for (auto& entry : object_inhib_)
    {
      entry.second *= params_.ior_decay;
    }
  }

  store.mark_selected(label, frame);
  return store.find_active(label);
}

void IorBehavior::reset()
{
  spatial_.clear();
  object_inhib_.clear();
}

std::unique_ptr<Behavior> create_behavior(const std::string& name)
{
  if (name == "exploration")
  {
    return std::make_unique<Exploration>();
  }
  if (name == "greedy")
  {
    return std::make_unique<IorBehavior>(IorBehavior::Mode::None, "greedy");
  }
  if (name == "spatial-ior")
  {
    return std::make_unique<IorBehavior>(IorBehavior::Mode::Spatial, "spatial-ior");
  }
  if (name == "object-ior")
  {
    return std::make_unique<IorBehavior>(IorBehavior::Mode::Object, "object-ior");
  }
  throw std::runtime_error("Unknown behavior '" + name + "'. Available: exploration, greedy, spatial-ior, object-ior");
}

} // namespace system
} // namespace attention
