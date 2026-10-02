#pragma once

#include "attention/system/attention_system.h"
#include <string>

namespace attention
{
namespace io
{

/**
 * TraceWriter records what the system computed on *every* frame of a stream,
 * not just the focus it chose ("attention-trace/v1"). ScanpathWriter answers
 * "where did attention go?"; a trace answers "what did the model see, and what
 * did it know, at each moment" — which is what a visualization, a debugging
 * session, or a figure needs.
 *
 * Layout under `directory`:
 *
 *   trace.json                 index: schema, frames, size, feature names
 *   frame_0000/state.json      focus, every object file (active and inactive)
 *   frame_0000/saliency.png    the fused map, 16-bit, fixed [0, 1] scale
 *   frame_0000/field.png       neural-field activity, fixed [-1, 1] scale
 *   frame_0000/feature_<name>.png   one per feature, fixed [0, 1] scale
 *   ...
 *
 * Every map is written on a *fixed* scale, never per-frame normalized: a
 * brightness change between two frames of a trace means the response changed.
 * (Per-frame normalization is exactly the kind of rescaling that hid a broken
 * feature for two months; see docs/replication/REPLICATION_DOSSIER.md,
 * finding A.)
 */
class TraceWriter
{
 public:
  struct Options
  {
    bool maps = true;             ///< write the PNG maps as well as the JSON
    bool inactive_objects = true; ///< include inactive object files (the model still knows them)
    int trajectory_points = 16;   ///< trajectory samples per object file, most recent last
  };

  explicit TraceWriter(std::string directory);
  TraceWriter(std::string directory, Options options);

  /**
   * Append one frame. Call after the system has processed it — from the
   * per-frame callback of AttentionSystem::process_stream, or after
   * process_frame(). Directories are numbered contiguously from zero in call
   * order; the system's own frame index is recorded inside state.json.
   */
  void write_frame(const system::AttentionSystem& system);

  /// Write trace.json. Call once after the stream.
  void finish(const system::AttentionSystem& system);

  int frames_written() const { return frames_; }

 private:
  std::string directory_;
  Options options_;
  int frames_ = 0;
  std::vector<std::string> feature_names_;
  cv::Size frame_size_;
};

} // namespace io
} // namespace attention
