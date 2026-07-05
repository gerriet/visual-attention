#include "attention/io/scanpath_writer.h"
#include <filesystem>
#include <fstream>
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
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

std::string format_float(float value)
{
  std::ostringstream oss;
  oss.precision(6);
  oss << value;
  return oss.str();
}
} // namespace

void ScanpathWriter::write(const system::AttentionSystem& sys, const std::string& json_path)
{
  fs::path json_file(json_path);
  if (json_file.has_parent_path())
  {
    fs::create_directories(json_file.parent_path());
  }

  std::ofstream out(json_path);
  if (!out)
  {
    throw std::runtime_error("ScanpathWriter: failed to open for writing: " + json_path);
  }

  const auto& scanpath = sys.scanpath();
  const auto& objects = sys.active_files();

  out << "{\n";
  out << "  \"schema\": \"attention-scanpath/v1\",\n";
  out << "  \"generator\": {\n";
  out << "    \"name\": \"attention-framework\",\n";
  out << "    \"behavior\": \"" << escape_json(sys.config().behavior) << "\"\n";
  out << "  },\n";
  out << "  \"frames\": " << sys.frame_index() << ",\n";

  // The scanpath: one focus per frame that produced one, in temporal order.
  out << "  \"scanpath\": [\n";
  for (size_t i = 0; i < scanpath.size(); ++i)
  {
    const auto& f = scanpath[i];
    out << "    {\"frame\": " << f.frame << ", \"label\": " << f.label << ", \"x\": " << f.location.x
        << ", \"y\": " << f.location.y << ", \"saliency\": " << format_float(f.saliency) << ", \"bbox\": ["
        << f.bbox.x << ", " << f.bbox.y << ", " << f.bbox.width << ", " << f.bbox.height << "]}";
    if (i + 1 < scanpath.size())
    {
      out << ",";
    }
    out << "\n";
  }
  out << "  ],\n";

  // Final active object files (the symbolic world model at stream end).
  out << "  \"objects\": [\n";
  for (size_t i = 0; i < objects.size(); ++i)
  {
    const auto& o = objects[i];
    out << "    {\"label\": " << o.label << ", \"x\": " << o.centroid.x << ", \"y\": " << o.centroid.y
        << ", \"size\": " << o.size << ", \"saliency\": " << format_float(o.saliency)
        << ", \"avg_saliency\": " << format_float(o.avg_saliency) << ", \"created_frame\": " << o.created_frame
        << ", \"last_selected_frame\": " << o.last_selected_frame << "}";
    if (i + 1 < objects.size())
    {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";

  if (!out.good())
  {
    throw std::runtime_error("ScanpathWriter: error while writing: " + json_path);
  }
}

} // namespace io
} // namespace attention
