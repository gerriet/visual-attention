#pragma once

#include "attention/system/object_file.h"
#include <memory>
#include <string>

namespace attention
{
namespace system
{

/**
 * Behavior: the task-dependent policy of the second selection stage. It picks
 * the focus object file for a frame using only the symbolic information in the
 * object files (thesis §7.3.1) — nothing about the raw image. The base system
 * ships Exploration; visual search, multi-object tracking, and search-and-track
 * (thesis §8.5) slot in behind the same interface.
 */
class Behavior
{
 public:
  virtual ~Behavior() = default;

  virtual std::string name() const = 0;

  /**
   * Choose the focus object file for this frame, or nullptr if there is no
   * eligible object. May consult and mutate the store (e.g. mark the chosen
   * file selected). The returned pointer is valid until the next
   * ObjectFileStore::update().
   */
  virtual const ObjectFile* select_focus(ObjectFileStore& store, int frame) = 0;

  virtual void reset() = 0;
};

/**
 * Exploration (thesis §8.5.1): the default, parameter-free behavior — gather as
 * much of the scene as possible into an up-to-date world model. Object files
 * are ranked in priority classes by time since last focal selection (never
 * selected first, then longest-unselected), and within a class by mean
 * saliency. The focus dwells on one object for `dwell_frames` frames (the
 * thesis's "attentive computation takes ~3 frames") before switching, which
 * together with the last-selected ordering yields object-based inhibition of
 * return and a scene-covering scanpath. Initiates no camera movement.
 */
class Exploration : public Behavior
{
 public:
  struct Params
  {
    int dwell_frames = 3; // frames the focus stays on one object (thesis: ~3)
  };

  Exploration() : Exploration(Params{}) {}
  explicit Exploration(const Params& params) : params_(params) {}

  std::string name() const override { return "exploration"; }

  const ObjectFile* select_focus(ObjectFileStore& store, int frame) override;

  void reset() override;

  int current_focus_label() const { return current_focus_label_; }

 private:
  Params params_;
  int current_focus_label_ = -1;
  int dwell_count_ = 0;
};

/**
 * Create a behavior by name ("exploration"). Throws for unknown names.
 */
std::unique_ptr<Behavior> create_behavior(const std::string& name);

} // namespace system
} // namespace attention
