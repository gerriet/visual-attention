#pragma once

#include "attention/pipeline/attention_pipeline.h"
#include "attention/pipeline/frame_source.h"
#include "attention/system/behavior.h"
#include "attention/system/object_file.h"
#include "attention/system/processor.h"
#include <functional>
#include <map>
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

  /**
   * When recognition processors fire during a stream (M13, H2). The default is
   * the thesis's story — one attentive computation per ~dwell-length window of
   * focus: fire when the focus settles on an object, and again every
   * process_repeat_frames while it stays held. EveryFrame re-processes the
   * focus each frame (more label votes, more compute). FullFrame is the H2
   * *baseline* arm: processors run over the entire frame every frame, ungated
   * (same code path, so the comparison is honest).
   */
  enum class ProcessorCadence
  {
    PerDwell,
    EveryFrame,
    FullFrame
  };

  // Per-processor compute accounting over a run (for H2's accuracy-vs-compute
  // curves): how often it fired, how many pixels it saw, how long it took.
  struct ProcessorStats
  {
    long long calls = 0;
    long long pixels = 0;
    double ms = 0.0;
  };

  struct Config
  {
    pipeline::PipelineConfig pipeline;
    ObjectFileStore::Config object_store;
    std::string behavior = "exploration";
    IorBehavior::Params ior_params;               // params for the IOR-ablation behaviors (M12)
    Identification::Params identification_params; // params for "identification" (M13)
    ActionMode action_mode = ActionMode::Scanpath;

    // Saliency segmentation into candidate clusters:
    float segment_fraction = 0.35f; // threshold as a fraction of the map's max
    float segment_min = 0.1f;       // absolute threshold floor
    int min_cluster_size = 20;      // ignore clusters smaller than this (px)

    // Recognition processors on attended ROIs (M13). Empty = none. ROIs are
    // taken from the pipeline frame (native resolution in --attend, which does
    // not downscale), expanded by roi_margin for detector context.
    std::vector<std::string> processors;
    ProcessorCadence processor_cadence = ProcessorCadence::PerDwell;
    float roi_margin = 0.25f; // bbox expansion per side, as a fraction of size
    // PerDwell: a focus held continuously on one object re-fires every this
    // many frames (each ~dwell-sized window is one attentive computation).
    // Without this, a behavior that keeps re-selecting the same object — e.g.
    // identification waiting for enough votes to settle — would inspect it
    // exactly once and deadlock unlabeled.
    int process_repeat_frames = 3;
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

  /**
   * Record a processor result: count the inspection and label vote on the
   * annotated object file, log the annotation, and accumulate compute stats.
   * Called internally after each gated processor firing; also the entry point
   * for external processor runs (the live demonstrator's native-ROI loop).
   */
  void record_annotation(const Annotation& annotation);

  const std::vector<Focus>& scanpath() const { return scanpath_; }
  const std::vector<ObjectFile>& active_files() const { return object_store_.active_files(); }
  const ObjectFileStore& object_store() const { return object_store_; }
  const pipeline::AttentionPipeline& pipeline() const { return pipeline_; }

  // All processor annotations of the run, in firing order (frame-stamped).
  const std::vector<Annotation>& annotations() const { return annotations_; }
  // Per-processor compute totals for the run (keyed by processor name).
  const std::map<std::string, ProcessorStats>& processor_stats() const { return processor_stats_; }

  // The focus chosen on the most recent frame (nullptr if none / Feature mode).
  const Focus* current_focus() const { return has_focus_ ? &current_focus_ : nullptr; }

  int frame_index() const { return frame_index_; }
  const Config& config() const { return config_; }

 private:
  // Segment the fused saliency map into candidate object clusters.
  std::vector<Cluster> segment(const cv::Mat& saliency) const;

  // Run the second stage for the current pipeline frame.
  void process_second_stage();

  // Run the configured processors for this frame, honoring the cadence:
  // gated on the focus ROI, or over the whole frame (FullFrame baseline).
  void run_processors();

  // Reset the second-stage state only (object files, behavior, scanpath).
  void reset_stage2();

  Config config_;
  pipeline::AttentionPipeline pipeline_;
  ObjectFileStore object_store_;
  std::unique_ptr<Behavior> behavior_;
  std::vector<std::unique_ptr<Processor>> processors_;

  std::vector<Focus> scanpath_;
  Focus current_focus_;
  bool has_focus_ = false;
  int frame_index_ = 0;

  std::vector<Annotation> annotations_;
  std::map<std::string, ProcessorStats> processor_stats_;
  int last_processed_label_ = -1;  // PerDwell: the focus label last processed
  int frames_since_processed_ = 0; // PerDwell: frames the focus has been held since
};

} // namespace system
} // namespace attention
