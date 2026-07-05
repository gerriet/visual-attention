#include "attention/system/behavior.h"
#include <algorithm>
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

std::unique_ptr<Behavior> create_behavior(const std::string& name)
{
  if (name == "exploration")
  {
    return std::make_unique<Exploration>();
  }
  throw std::runtime_error("Unknown behavior '" + name + "'. Available: exploration");
}

} // namespace system
} // namespace attention
