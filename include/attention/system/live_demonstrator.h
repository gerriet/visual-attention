#pragma once

#include "attention/system/attention_system.h"
#include "attention/system/processor.h"
#include <memory>
#include <string>
#include <vector>

namespace attention
{
namespace system
{

/**
 * LiveDemonstrator: the M8 real-time demo layer on top of AttentionSystem.
 *
 * Processing and display resolution are decoupled — frames are captured at
 * native resolution, attention is computed on a downscaled copy (≤ VGA) for a
 * bounded per-frame cost, and object-file coordinates are scaled back up so
 * annotations and processor ROIs live at native resolution. Enabled processors
 * (object-file plugins) run only on attended regions, and the result is drawn
 * as an overlay: one marker per object file (stable ID from M6 tracking), the
 * current focus highlighted, a scanpath trail, and each processor's label.
 */
class LiveDemonstrator
{
 public:
  struct Config
  {
    AttentionSystem::Config system;
    std::vector<std::string> processors{"region-descriptor"};
    int process_max_side = 480; // process at most this on the longer side (≤ VGA)
    int scanpath_trail = 12;    // draw this many recent foci as a trail
    bool run_on_focus_only = false; // run processors on the focus only vs all active files
  };

  explicit LiveDemonstrator(const Config& config);

  // Start a fresh stream (clears attention + object state).
  void reset();

  /**
   * Process one native-resolution frame and return the annotated native frame.
   * Runs attention on a downscaled copy, then the processors on native ROIs.
   */
  cv::Mat process(const cv::Mat& native_frame);

  const std::vector<Annotation>& annotations() const { return annotations_; }
  const AttentionSystem& system() const { return system_; }
  int frame_index() const { return system_.frame_index(); }

 private:
  cv::Mat draw_overlay(const cv::Mat& native, double sx, double sy) const;

  Config config_;
  AttentionSystem system_;
  std::vector<std::unique_ptr<Processor>> processors_;
  std::vector<Annotation> annotations_; // from the most recent frame
};

} // namespace system
} // namespace attention
