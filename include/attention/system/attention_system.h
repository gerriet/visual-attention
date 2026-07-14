#pragma once

#include "attention/pipeline/attention_pipeline.h"
#include "attention/pipeline/frame_source.h"
#include "attention/system/behavior.h"
#include "attention/system/object_file.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace attention
{
namespace system
{

/**
 * One focus of attention over the stream: the object file selected on a frame,
 * with its location and extent (the "focus" is the object's segment, not just
 * a point — thesis §7.3.2).
 */
struct Focus
{
  int frame = 0;
  int label = 0;      // object-file label
  cv::Point location; // centroid in image coordinates
  cv::Rect bbox;
  float saliency = 0.0f;
};

/**
 * AttentionSystem: the top-level active-vision attention system — the second
 * selection stage and behavior layer on top of the feature/fusion pipeline.
 * (In the original dissertation code this class was called ESAB2, after the
 * DFG project "Entwicklung von Systembausteinen der Aktiven Bildanalyse II";
 * the name carried no meaning outside that project.)
 *
 * Per stream frame it:
 *   1. runs the AttentionPipeline (features → fusion → saliency);
 *   2. segments the fused saliency into candidate object clusters (an
 *      approximation of the neural field's activation clusters at this bar);
 *   3. corresponds them to persistent ObjectFiles (create / update / retire);
 *   4. runs the Behavior to pick the focus object file (symbolic, object-based
 *      inhibition of return via the last-selected ordering);
 *   5. appends the focus to the scanpath.
 *
 * Action modes (thesis ch. 8): Feature computes saliency only; Scanpath runs
 * the full second stage and behavior. move_sensor (overt gaze shift with field
 * displacement) needs a controllable image source and is deferred.
 */
class AttentionSystem
{
 public:
  enum class ActionMode
  {
    Feature, // compute saliency only (no object files / behavior)
    Scanpath // full second stage + behavior over the stream
  };

  struct Config
  {
    pipeline::PipelineConfig pipeline;
    ObjectFileStore::Config object_store;
    std::string behavior = "exploration";
    IorBehavior::Params ior_params; // params for the IOR-ablation behaviors (M12)
    ActionMode action_mode = ActionMode::Scanpath;

    // Saliency segmentation into candidate clusters:
    float segment_fraction = 0.35f; // threshold as a fraction of the map's max
    float segment_min = 0.1f;       // absolute threshold floor
    int min_cluster_size = 20;      // ignore clusters smaller than this (px)
  };

  using FocusCallback = std::function<void(AttentionSystem&)>;

  AttentionSystem() : AttentionSystem(Config{}) {}
  explicit AttentionSystem(const Config& config);

  /**
   * Process a stream. Resets object files, behavior, and scanpath, then runs
   * the pipeline frame by frame, performing the second stage after each and
   * invoking on_frame (if given).
   */
  void process_stream(pipeline::FrameSource& source, const FocusCallback& on_frame = {});

  /**
   * Reset all per-run state (pipeline RunState, object files, behavior,
   * scanpath). Call once before driving a stream frame by frame via
   * process_frame() (the live demonstrator does this).
   */
  void reset();

  /**
   * Process a single frame: run the pipeline (stage 1) and the second stage,
   * carrying state from the previous frame. Use with reset() for manual
   * per-frame control (e.g. a live loop that also displays each frame);
   * process_stream() is the convenience form.
   */
  void process_frame(const cv::Mat& image, const std::string& source_name = "");

  const std::vector<Focus>& scanpath() const { return scanpath_; }
  const std::vector<ObjectFile>& active_files() const { return object_store_.active_files(); }
  const ObjectFileStore& object_store() const { return object_store_; }
  const pipeline::AttentionPipeline& pipeline() const { return pipeline_; }

  // The focus chosen on the most recent frame (nullptr if none / Feature mode).
  const Focus* current_focus() const { return has_focus_ ? &current_focus_ : nullptr; }

  int frame_index() const { return frame_index_; }
  const Config& config() const { return config_; }

 private:
  // Segment the fused saliency map into candidate object clusters.
  std::vector<Cluster> segment(const cv::Mat& saliency) const;

  // Run the second stage for the current pipeline frame.
  void process_second_stage();

  // Reset the second-stage state only (object files, behavior, scanpath).
  void reset_stage2();

  Config config_;
  pipeline::AttentionPipeline pipeline_;
  ObjectFileStore object_store_;
  std::unique_ptr<Behavior> behavior_;

  std::vector<Focus> scanpath_;
  Focus current_focus_;
  bool has_focus_ = false;
  int frame_index_ = 0;
};

} // namespace system
} // namespace attention
