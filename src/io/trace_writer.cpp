#include "attention/io/trace_writer.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace attention
{
namespace io
{

namespace
{

std::string escape_json(const std::string& s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s)
  {
    switch (c)
    {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
    }
  }
  return out;
}

std::string num(double v)
{
  std::ostringstream oss;
  oss.precision(6);
  oss << v;
  return oss.str();
}

std::string frame_dir_name(int index)
{
  std::ostringstream oss;
  oss << "frame_" << std::setw(4) << std::setfill('0') << index;
  return oss.str();
}

// [0, 1] -> 16 bit, saturating. The scale is fixed on purpose (see the header).
void write_unit_map(const cv::Mat& map, const fs::path& path)
{
  if (map.empty())
  {
    return;
  }
  cv::Mat map16;
  map.convertTo(map16, CV_16U, 65535.0);
  if (!cv::imwrite(path.string(), map16))
  {
    throw std::runtime_error("TraceWriter: failed to write " + path.string());
  }
}

// Field activity sits around the resting level and reaches a little past +-1 in
// practice (measured: -1.6 .. +1.6 on the dissertation system's parameters), so
// the fixed encoding is [-2, 2] -> 16 bit. Each frame records its true range in
// state.json, so a reader can see if this ever clipped.
void write_field_map(const cv::Mat& field, const fs::path& path)
{
  if (field.empty())
  {
    return;
  }
  cv::Mat map16;
  field.convertTo(map16, CV_16U, 65535.0 / 4.0, 65535.0 / 2.0);
  if (!cv::imwrite(path.string(), map16))
  {
    throw std::runtime_error("TraceWriter: failed to write " + path.string());
  }
}

void write_object(std::ostream& out, const system::ObjectFile& file, bool active, int trajectory_points)
{
  out << "    {";
  out << "\"label\": " << file.label;
  out << ", \"active\": " << (active ? "true" : "false");
  out << ", \"x\": " << file.centroid.x << ", \"y\": " << file.centroid.y;
  if (file.centroid_exact.x >= 0.0f)
  {
    out << ", \"centroid_exact\": [" << num(file.centroid_exact.x) << ", " << num(file.centroid_exact.y) << "]";
  }
  out << ", \"bbox\": [" << file.bbox.x << ", " << file.bbox.y << ", " << file.bbox.width << ", " << file.bbox.height
      << "]";
  out << ", \"size\": " << file.size;
  out << ", \"saliency\": " << num(file.saliency);
  out << ", \"avg_saliency\": " << num(file.avg_saliency);
  out << ", \"created_frame\": " << file.created_frame;
  out << ", \"last_seen_frame\": " << file.last_seen_frame;
  out << ", \"last_selected_frame\": " << file.last_selected_frame;
  out << ", \"selection_count\": " << file.selection_count;
  out << ", \"appearance\": [" << num(file.appearance[0]) << ", " << num(file.appearance[1]) << ", "
      << num(file.appearance[2]) << "]";
  if (!file.avg_features.empty())
  {
    out << ", \"features\": [";
    for (size_t i = 0; i < file.avg_features.size(); ++i)
    {
      out << (i ? ", " : "") << num(file.avg_features[i]);
    }
    out << "]";
  }
  if (!file.trajectory.empty() && trajectory_points > 0)
  {
    const size_t take = std::min<size_t>(file.trajectory.size(), static_cast<size_t>(trajectory_points));
    const size_t first = file.trajectory.size() - take;
    out << ", \"trajectory\": [";
    for (size_t i = first; i < file.trajectory.size(); ++i)
    {
      out << (i > first ? ", " : "") << "[" << file.trajectory[i].x << ", " << file.trajectory[i].y << "]";
    }
    out << "]";
  }
  const std::string best = file.labels.best_label();
  if (!best.empty())
  {
    out << ", \"class\": \"" << escape_json(best) << "\"";
    out << ", \"class_confidence\": " << num(file.labels.best_confidence());
    out << ", \"class_votes\": " << file.labels.best_count();
  }
  if (file.labels.inspections > 0)
  {
    out << ", \"inspections\": " << file.labels.inspections;
  }
  out << "}";
}

} // namespace

TraceWriter::TraceWriter(std::string directory) : TraceWriter(std::move(directory), Options{}) {}

TraceWriter::TraceWriter(std::string directory, Options options) : directory_(std::move(directory)), options_(options)
{
  fs::create_directories(directory_);
}

void TraceWriter::write_frame(const system::AttentionSystem& sys)
{
  // Directories are numbered by this writer, contiguously from zero, so a trace
  // is predictable however the system was driven; the system's own frame index
  // goes into state.json.
  const fs::path dir = fs::path(directory_) / frame_dir_name(frames_);
  fs::create_directories(dir);

  const auto& pipeline = sys.pipeline();
  const cv::Mat& saliency = pipeline.get_saliency_map().map;
  if (!saliency.empty())
  {
    frame_size_ = saliency.size();
  }
  // Collect names across frames, not just the first: a temporal feature (onset)
  // is not applicable on frame 0 and appears only from frame 1 on.
  for (const auto& feature : pipeline.get_features())
  {
    if (std::find(feature_names_.begin(), feature_names_.end(), feature.name) == feature_names_.end())
    {
      feature_names_.push_back(feature.name);
    }
  }

  // The field: the second stage's own, when it builds object files from field
  // clusters; otherwise the pipeline's selection stage, when that is a field.
  cv::Mat field = sys.field_activity();
  if (field.empty())
  {
    field = pipeline.get_run_state().field_activity;
  }

  if (options_.maps)
  {
    write_unit_map(saliency, dir / "saliency.png");
    write_field_map(field, dir / "field.png");
    for (const auto& feature : pipeline.get_features())
    {
      write_unit_map(feature.data, dir / ("feature_" + feature.name + ".png"));
    }
  }

  std::ofstream out((dir / "state.json").string());
  if (!out)
  {
    throw std::runtime_error("TraceWriter: failed to open " + (dir / "state.json").string());
  }
  out << "{\n";
  out << "  \"schema\": \"attention-trace-frame/v1\",\n";
  out << "  \"frame\": " << sys.frame_index() << ",\n";
  if (!saliency.empty())
  {
    out << "  \"size\": [" << saliency.cols << ", " << saliency.rows << "],\n";
  }
  if (!field.empty())
  {
    double lo = 0.0;
    double hi = 0.0;
    cv::minMaxLoc(field, &lo, &hi);
    out << "  \"field_size\": [" << field.cols << ", " << field.rows << "],\n";
    // The true range before the fixed-scale encoding of field.png, so a reader
    // can tell whether that encoding clipped.
    out << "  \"field_range\": [" << num(lo) << ", " << num(hi) << "],\n";
  }
  out << "  \"features\": [";
  {
    const auto& features = pipeline.get_features();
    for (size_t i = 0; i < features.size(); ++i)
    {
      out << (i ? ", " : "") << "\"" << escape_json(features[i].name) << "\"";
    }
  }
  out << "],\n";
  const cv::Point2f& shift = sys.last_camera_shift();
  if (shift.x != 0.0f || shift.y != 0.0f)
  {
    out << "  \"camera_shift\": [" << num(shift.x) << ", " << num(shift.y) << "],\n";
  }

  const system::Focus* focus = sys.current_focus();
  out << "  \"focus\": ";
  if (focus == nullptr)
  {
    out << "null,\n";
  }
  else
  {
    out << "{\"label\": " << focus->label << ", \"x\": " << focus->location.x << ", \"y\": " << focus->location.y
        << ", \"bbox\": [" << focus->bbox.x << ", " << focus->bbox.y << ", " << focus->bbox.width << ", "
        << focus->bbox.height << "], \"saliency\": " << num(focus->saliency) << "},\n";
  }

  out << "  \"objects\": [\n";
  bool first = true;
  for (const auto& file : sys.active_files())
  {
    if (!first)
    {
      out << ",\n";
    }
    write_object(out, file, true, options_.trajectory_points);
    first = false;
  }
  if (options_.inactive_objects)
  {
    for (const auto& file : sys.object_store().inactive_files())
    {
      if (!first)
      {
        out << ",\n";
      }
      write_object(out, file, false, options_.trajectory_points);
      first = false;
    }
  }
  out << "\n  ]\n}\n";
  ++frames_;
}

void TraceWriter::finish(const system::AttentionSystem& sys)
{
  std::ofstream out((fs::path(directory_) / "trace.json").string());
  if (!out)
  {
    throw std::runtime_error("TraceWriter: failed to open the trace index");
  }
  out << "{\n";
  out << "  \"schema\": \"attention-trace/v1\",\n";
  out << "  \"generator\": {\"name\": \"attention-framework\", \"behavior\": \"" << escape_json(sys.config().behavior)
      << "\"},\n";
  out << "  \"frames\": " << frames_ << ",\n";
  if (frame_size_.width > 0)
  {
    out << "  \"size\": [" << frame_size_.width << ", " << frame_size_.height << "],\n";
  }
  out << "  \"features\": [";
  for (size_t i = 0; i < feature_names_.size(); ++i)
  {
    out << (i ? ", " : "") << "\"" << escape_json(feature_names_[i]) << "\"";
  }
  out << "],\n";
  // How the 16-bit maps decode: value/65535 * (hi - lo) + lo
  out << "  \"map_scales\": {\"saliency\": [0, 1], \"feature\": [0, 1], \"field\": [-2, 2]},\n";
  out << "  \"frame_dir\": \"frame_%04d\"\n";
  out << "}\n";
}

} // namespace io
} // namespace attention
