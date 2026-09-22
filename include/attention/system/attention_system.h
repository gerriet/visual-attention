#pragma once

#include "attention/fusion/priority_map.h"
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

    // Where the candidate clusters come from.
    //   Saliency (default): threshold and label the fused saliency map — this
    //     system's approximation since M6.
    //   Field: the thesis's first selection stage (ch. 6, §7.2.2) — a neural
    //     field relaxed on the saliency map, carried from frame to frame; its
    //     activity clusters are what object files are created for, so noise
    //     suppression, hysteresis and tracking happen before correspondence.
    //     Takes the field's parameters from pipeline.selection_params (the
    //     pipeline's own selection may stay "nms", so that no second field
    //     runs), its defaults otherwise. The segment_* and proto_* keys below
    //     do not apply.
    enum class ClusterSource
    {
      Saliency,
      Field
    };
    ClusterSource cluster_source = ClusterSource::Saliency;

    // Saliency segmentation into candidate clusters:
    float segment_fraction = 0.35f; // threshold as a fraction of the map's max
    float segment_min = 0.1f;       // absolute threshold floor
    int min_cluster_size = 20;      // ignore clusters smaller than this (px)
    // Opt-in (M19): bridge an object's fragments before labelling — a moving
    // uniform disk, salient only at its leading and trailing edges, falls apart
    // into two crescents that would each get an object file — and drop regions
    // too large to be an object (diffuse background saliency). 0 = off (thesis).
    int segment_close = 0;             // morphological closing radius (px)
    float max_cluster_fraction = 0.0f; // max cluster area, as a fraction of the map
    // Opt-in (M19) proto-objects: turn each salient cluster into the object(s)
    // under it. Seed at the cluster pixel whose colour differs most from the
    // frame's median colour (a figure-ground estimate); a cluster where nothing
    // differs is background — a motion ghost, a symmetry response between
    // objects — and is dropped. Otherwise grow the object from the seed by
    // colour (flood fill in a window around the cluster; a fill that floods
    // the window is background too), re-seed on what's left so a cluster
    // spanning touching objects yields each, and merge results that grew into
    // the same object. A salient but untextured-growth region keeps its
    // cluster. Needs the frame at saliency resolution.
    bool proto_objects = false;
    float proto_min_contrast = 40.0f; // colour L2 (0-255) from the frame median: below = background
    float proto_tolerance = 30.0f;    // flood-fill tolerance per channel (0-255), relative to the seed
    float proto_window = 1.0f;        // growth window: cluster bbox expanded by this × its size per side

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

  /**
   * Apply a config file's `attention_system:` section (raw YAML, as kept by
   * ConfigLoader) to `config`. Keys: segment_fraction, segment_min,
   * cluster_source (saliency | field),
   * min_cluster_size, segment_close, max_cluster_fraction, proto_objects,
   * proto_min_contrast, proto_tolerance, proto_window, and
   * object_files: { correspondence (position | thesis), feature_tolerance,
   * feature_gate, merge_resolve_frames, correspondence_radius,
   * max_inactive_age, motion_prediction, appearance_matching,
   * appearance_weight, persistent_identity, reid_colour_gate,
   * reid_colour_veto, gate_growth }. Absent keys keep their defaults; an
   * unknown key throws std::runtime_error (a typo must not pass silently).
   */
  static void apply_config_yaml(const std::string& yaml, Config& config);

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

  // Segment a fused saliency (priority) map into candidate object clusters —
  // what the second stage does every frame, against the current pipeline
  // frame; the two-argument form takes the frame explicitly (public for tests).
  std::vector<Cluster> segment(const cv::Mat& saliency) const;
  std::vector<Cluster> segment(const cv::Mat& saliency, const cv::Mat& image) const;

  // Config::ClusterSource::Field: relax the field on this frame's map and
  // return its activity clusters (stateful: the field's activity persists).
  std::vector<Cluster> field_clusters(const cv::Mat& saliency);

 private:
  // Mean of every pipeline feature map over a region (Cluster::features).
  std::vector<float> feature_means(const cv::Mat& region) const;

  // The proto-object(s) under one salient cluster (see Config::proto_objects);
  // empty = background.
  std::vector<Cluster> proto_objects(const cv::Mat& region, const cv::Mat& image, const Cluster& cluster,
                                     const cv::Vec3f& ground) const;

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
  fusion::HistoryChannels history_;                     // M17 selection-history / value channels
  std::unique_ptr<selection::SelectionStrategy> field_; // ClusterSource::Field only
  cv::Mat field_activity_;                              //   its state across frames

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
