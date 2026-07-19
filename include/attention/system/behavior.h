#pragma once

#include "attention/system/object_file.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

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
 * IorBehavior: the inhibition-of-return ablation for the dynamic-IOR study
 * (roadmap M12, docs/DYNAMIC_IOR_STUDY.md). Three modes that are identical
 * except in *what* they inhibit, so a scanpath comparison isolates the effect of
 * the inhibition domain — the thesis's core stage-2 claim (H1):
 *   None    ("greedy")      — no inhibition; always the most salient object, so
 *                             attention perseverates on the strongest one.
 *   Spatial ("spatial-ior") — inhibit recently attended LOCATIONS (decaying); a
 *                             moving object escapes its own inhibited spot, so it
 *                             is re-fixated and empty inhibited regions waste
 *                             revisits — the space-based weakness in motion.
 *   Object  ("object-ior")  — inhibit recently attended OBJECTS (decaying); the
 *                             tag follows the object as it moves.
 * Focus score = object saliency − inhibition; the winner is re-inhibited.
 */
class IorBehavior : public Behavior
{
 public:
  enum class Mode
  {
    None,
    Spatial,
    Object
  };

  struct Params
  {
    float ior_radius = 60.0f;  // Spatial: Gaussian radius of a location tag (px)
    float ior_strength = 1.0f; // inhibition deposited on the selected focus
    float ior_decay = 0.8f;    // per-frame decay (thesis §8.3: ~20%/frame)
  };

  IorBehavior(Mode mode, std::string name) : IorBehavior(mode, std::move(name), Params{}) {}
  IorBehavior(Mode mode, std::string name, const Params& params) : mode_(mode), name_(std::move(name)), params_(params)
  {
  }

  std::string name() const override { return name_; }
  const ObjectFile* select_focus(ObjectFileStore& store, int frame) override;
  void reset() override;

 private:
  struct Spot
  {
    cv::Point loc;
    float strength;
  };

  Mode mode_;
  std::string name_;
  Params params_;
  std::vector<Spot> spatial_;         // Spatial mode: decaying location tags
  std::map<int, float> object_inhib_; // Object mode: decaying inhibition by label
};

/**
 * Identification (M13): curiosity — attention as a drive to *know* the scene,
 * not just to have seen it. Object files without a confident semantic label
 * outrank recognized ones (least-inspected first, so looks spread across the
 * unknowns); once an object is confidently labeled — or has been inspected
 * max_inspections times without yielding a label ("give up") — it drops into
 * the background rotation, which cycles like Exploration. This is
 * recognition-triggered semantic inhibition of return: being identified is
 * what inhibits. Pair with recognition processors (they cast the label votes);
 * without them every object stays unknown and the behavior degenerates to
 * least-inspected-first coverage.
 */
class Identification : public Behavior
{
 public:
  struct Params
  {
    float confidence = 0.5f; // a label at/above this mean confidence counts
    int min_votes = 2;       // votes needed before a label is trusted
    int max_inspections = 6; // stop chasing an unlabelable object after this
    int dwell_frames = 3;    // focus dwell, as in Exploration
  };

  Identification() : Identification(Params{}) {}
  explicit Identification(const Params& params) : params_(params) {}

  std::string name() const override { return "identification"; }

  const ObjectFile* select_focus(ObjectFileStore& store, int frame) override;

  void reset() override;

  // Whether this behavior considers the object identified (or given up on).
  bool is_settled(const ObjectFile& file) const;

 private:
  Params params_;
  int current_focus_label_ = -1;
  int dwell_count_ = 0;
};

/**
 * Create a behavior by name. Available: "exploration" (thesis default), the
 * dynamic-IOR ablation "greedy" / "spatial-ior" / "object-ior", and
 * "identification" (M13 curiosity). The IOR params apply to the ablation
 * behaviors, the identification params to Identification; others ignore them.
 * Throws for unknown names.
 */
std::unique_ptr<Behavior> create_behavior(
    const std::string& name, const IorBehavior::Params& ior_params = IorBehavior::Params{},
    const Identification::Params& identification_params = Identification::Params{});

} // namespace system
} // namespace attention
