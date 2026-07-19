#pragma once

#include <deque>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace attention
{
namespace system
{

/**
 * A candidate object region for one frame — the input to the object-file
 * store. In the thesis these are the neural-field activation clusters; here
 * they are the segmented regions of the (field-processed) saliency map.
 */
struct Cluster
{
  cv::Point centroid;
  cv::Rect bbox;
  int size = 0; // pixel count
  float mean_saliency = 0.0f;
  cv::Vec3f appearance = cv::Vec3f(0, 0, 0); // mean colour (BGR, 0-255) over the region
};

/**
 * LabelMemory: accumulated semantic labels on one object file (M13). Each
 * recognition-processor firing on the object casts a confidence-weighted vote;
 * the majority label with its mean confidence is the object's stable semantic
 * identity ("person #3"). Processor-agnostic: HOG, Haar, DNN, or a future VLM
 * labeler all feed the same votes.
 */
struct LabelMemory
{
  struct Vote
  {
    int count = 0;
    float confidence_sum = 0.0f;
  };

  std::map<std::string, Vote> votes; // class label -> accumulated votes
  int inspections = 0;               // processor firings, including empty-handed ones

  void add_inspection() { ++inspections; }
  void add_vote(const std::string& label, float confidence);

  // The majority label (most votes; confidence sum breaks ties), or "" if no
  // votes yet.
  std::string best_label() const;
  // Mean confidence of the majority label's votes (0 if unlabeled).
  float best_confidence() const;
  // Votes cast for the majority label.
  int best_count() const;
};

/**
 * ObjectFile: the symbolic record of one attended object, persisting across
 * frames (thesis §7.2, Fig. 7.1). It binds object-based and spatial
 * information so the second selection stage can operate on symbols alone.
 *
 * A file is created with a unique `label`, tracks its current position/extent
 * and a leaky-integrated saliency, remembers when it was created / last seen /
 * last focally selected, and keeps a short position trajectory. Files with no
 * corresponding cluster this frame become inactive (kept for re-correspondence
 * and review) and are aged out after a while.
 */
struct ObjectFile
{
  int label = 0;
  cv::Point centroid;
  cv::Rect bbox;
  int size = 0;
  float saliency = 0.0f;     // current-frame mean saliency
  float avg_saliency = 0.0f; // leaky-integrated (older frames weigh less)

  int created_frame = 0;
  int last_seen_frame = 0;
  int last_selected_frame = -1; // -1 == never focally selected

  bool active = true;
  int selection_count = 0; // how many frames it has been the focus (dwell)

  std::deque<cv::Point> trajectory;          // recent centroids (most recent last)
  cv::Vec3f appearance = cv::Vec3f(0, 0, 0); // leaky-integrated mean colour (BGR)
  LabelMemory labels;                        // accumulated semantic identity (M13)

  bool ever_selected() const { return last_selected_frame >= 0; }
};

/**
 * ObjectFileStore: maintains the set of object files across the frames of a
 * stream (thesis §7.2.3–7.2.4). Each frame it corresponds the incoming
 * clusters to existing files (by centroid proximity, via a greedy
 * nearest-first assignment), creates new files for clusters
 * with no match, moves unmatched files to an inactive stack (from which a
 * later cluster can revive them), and ages out long-inactive files.
 */
class ObjectFileStore
{
 public:
  struct Config
  {
    // Correspondence threshold (px): a cluster within this distance of a file's
    // centroid is the same object; beyond 2× a new file is created. Derived in
    // the thesis from the typical activation-cluster radius / local-max window.
    double correspondence_radius = 30.0;
    int max_inactive_age = 30;  // drop inactive files unseen for this many frames
    float leaky_alpha = 0.5f;   // leaky-integrator weight for the current frame
    int trajectory_length = 16; // capped position history per file
    // Match clusters to each file's *predicted* next centroid (last + trajectory
    // velocity) instead of its last centroid. Holds object identity through fast
    // motion and short occlusions, which object-based IOR depends on (M12).
    // Default off: the thesis's simple nearest-centroid correspondence.
    bool motion_prediction = false;
    // Fold an appearance descriptor (mean colour of the region, computed from
    // features the pipeline already produced) into the correspondence cost, so
    // crossing objects of different colour keep their labels — the DeepSORT
    // idea, which is what stops the label-switches that cripple object-based IOR.
    bool appearance_matching = false;
    // Weight of appearance in the cost: cost = position_px + weight * colour_L2.
    // A colour difference of 100 (0-255 scale) at weight 0.3 adds 30 px of cost.
    double appearance_weight = 0.3;
  };

  ObjectFileStore() : ObjectFileStore(Config{}) {}
  explicit ObjectFileStore(const Config& config);

  /**
   * Correspond the frame's clusters to the stored files and update them.
   * @param clusters  candidate object regions for this frame
   * @param frame     current frame index (monotonic)
   */
  void update(const std::vector<Cluster>& clusters, int frame);

  // Mark a file as focally selected on the given frame (drives object-based
  // IOR and the Exploration priority classes).
  void mark_selected(int label, int frame);

  // Label memory (M13): record that a recognition processor inspected the
  // file, and optionally cast a confidence-weighted vote for a class label.
  // No-ops if the label is not active.
  void record_inspection(int label);
  void add_label_vote(int label, const std::string& class_label, float confidence);

  const std::vector<ObjectFile>& active_files() const { return active_; }
  const std::vector<ObjectFile>& inactive_files() const { return inactive_; }

  // Look up an active file by label (nullptr if absent).
  ObjectFile* find_active(int label);

  void reset();

 private:
  ObjectFile make_file(const Cluster& cluster, int frame);
  void update_file(ObjectFile& file, const Cluster& cluster, int frame);

  Config config_;
  std::vector<ObjectFile> active_;
  std::vector<ObjectFile> inactive_; // stack-like: revived by later clusters
  int next_label_ = 1;
};

} // namespace system
} // namespace attention
