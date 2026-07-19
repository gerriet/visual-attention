#include "attention/system/live_demonstrator.h"
#include <algorithm>

namespace attention
{
namespace system
{

LiveDemonstrator::LiveDemonstrator(const Config& config) : config_(config), system_(config.system)
{
  for (const auto& name : config_.processors)
  {
    processors_.push_back(create_processor(name));
  }
}

void LiveDemonstrator::reset()
{
  system_.reset();
  annotations_.clear();
}

cv::Mat LiveDemonstrator::process(const cv::Mat& native_frame)
{
  annotations_.clear();
  if (native_frame.empty())
  {
    return native_frame;
  }

  // Downscale to the processing resolution (bounded per-frame cost).
  const int longer = std::max(native_frame.cols, native_frame.rows);
  double scale = 1.0;
  if (longer > config_.process_max_side)
  {
    scale = static_cast<double>(config_.process_max_side) / longer;
  }
  cv::Mat proc;
  if (scale < 1.0)
  {
    cv::resize(native_frame, proc, cv::Size(), scale, scale, cv::INTER_AREA);
  }
  else
  {
    proc = native_frame;
  }

  system_.process_frame(proc);

  // Map process-resolution object coordinates back up to native.
  const double sx = static_cast<double>(native_frame.cols) / proc.cols;
  const double sy = static_cast<double>(native_frame.rows) / proc.rows;

  // Run the processors on native ROIs of the attended object files — the
  // attention premise: expensive analysis only where the system is looking.
  const Focus* focus = system_.current_focus();
  for (const auto& obj : system_.active_files())
  {
    if (config_.run_on_focus_only && !(focus && obj.label == focus->label))
    {
      continue;
    }
    cv::Rect native_box(static_cast<int>(obj.bbox.x * sx), static_cast<int>(obj.bbox.y * sy),
                        static_cast<int>(obj.bbox.width * sx), static_cast<int>(obj.bbox.height * sy));
    // Same ROI policy as the --attend recognition path: expand by the margin
    // (detectors want context around the object), then clamp to the frame.
    const int mx = static_cast<int>(native_box.width * config_.system.roi_margin);
    const int my = static_cast<int>(native_box.height * config_.system.roi_margin);
    native_box += cv::Point(-mx, -my);
    native_box += cv::Size(2 * mx, 2 * my);
    native_box &= cv::Rect(0, 0, native_frame.cols, native_frame.rows);
    const cv::Mat roi = native_box.area() > 0 ? native_frame(native_box) : cv::Mat();
    // process_frame() has already advanced frame_index; the frame just
    // analyzed is the previous index (keeps annotation stamps aligned with
    // the scanpath).
    const int current_frame = system_.frame_index() - 1;
    for (const auto& processor : processors_)
    {
      // Timed run + record: feeds the object file's label memory (M13), so the
      // overlay can show the accumulated semantic identity ("person #3").
      Annotation ann = run_processor(*processor, obj, roi, native_box.tl(), current_frame);
      system_.record_annotation(ann);
      annotations_.push_back(std::move(ann));
    }
  }

  return draw_overlay(native_frame, sx, sy);
}

cv::Mat LiveDemonstrator::draw_overlay(const cv::Mat& native, double sx, double sy) const
{
  cv::Mat vis;
  if (native.channels() == 1)
  {
    cv::cvtColor(native, vis, cv::COLOR_GRAY2BGR);
  }
  else
  {
    vis = native.clone();
  }

  const Focus* focus = system_.current_focus();

  // Scanpath trail: the last N foci, scaled to native, connected.
  const auto& path = system_.scanpath();
  const int trail = std::min<int>(config_.scanpath_trail, static_cast<int>(path.size()));
  cv::Point prev(-1, -1);
  for (int i = static_cast<int>(path.size()) - trail; i < static_cast<int>(path.size()); ++i)
  {
    if (i < 0)
    {
      continue;
    }
    cv::Point p(static_cast<int>(path[i].location.x * sx), static_cast<int>(path[i].location.y * sy));
    if (prev.x >= 0)
    {
      cv::line(vis, prev, p, cv::Scalar(0, 220, 220), 1, cv::LINE_AA);
    }
    prev = p;
  }

  // Object files: white = current focus, blue = previously selected, red = never.
  for (const auto& obj : system_.active_files())
  {
    cv::Scalar colour = obj.ever_selected() ? cv::Scalar(255, 128, 0) : cv::Scalar(0, 0, 255);
    int thickness = 1;
    if (focus && obj.label == focus->label)
    {
      colour = cv::Scalar(255, 255, 255);
      thickness = 2;
    }
    cv::Rect box(static_cast<int>(obj.bbox.x * sx), static_cast<int>(obj.bbox.y * sy),
                 static_cast<int>(obj.bbox.width * sx), static_cast<int>(obj.bbox.height * sy));
    box &= cv::Rect(0, 0, vis.cols, vis.rows);
    if (box.area() > 0) // skip fully off-frame objects (box and ID both)
    {
      cv::rectangle(vis, box, colour, thickness);
      cv::Point label_at(box.x, std::max(12, box.y - 4));
      // Stable semantic identity from label memory, when the object has one.
      std::string text = "#" + std::to_string(obj.label);
      const std::string semantic = obj.labels.best_label();
      if (!semantic.empty())
      {
        text = semantic + " " + text;
      }
      cv::putText(vis, text, label_at, cv::FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv::LINE_AA);
    }
  }

  // Processor annotations: draw each label under its object's box.
  for (const auto& obj : system_.active_files())
  {
    int line = 0;
    for (const auto& ann : annotations_)
    {
      // Match on the object label field (not a string prefix, which would let
      // "#1" swallow "#10", "#12", ...).
      if (ann.object_label != obj.label)
      {
        continue;
      }
      cv::Rect box(static_cast<int>(obj.bbox.x * sx), static_cast<int>(obj.bbox.y * sy),
                   static_cast<int>(obj.bbox.width * sx), static_cast<int>(obj.bbox.height * sy));
      box &= cv::Rect(0, 0, vis.cols, vis.rows);
      cv::Point at(box.x, std::min(vis.rows - 4, box.y + box.height + 14 + line * 14));
      cv::putText(vis, ann.label, at, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 128), 1, cv::LINE_AA);
      ++line;
    }
  }

  return vis;
}

} // namespace system
} // namespace attention
