#include "attention/io/result_writer.h"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
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

std::string format_float(float value)
{
  std::ostringstream oss;
  oss.precision(6);
  oss << value;
  return oss.str();
}

} // namespace

void ResultWriter::write(const pipeline::AttentionPipeline& pipeline, const std::string& json_path)
{
  if (!pipeline.is_processed())
  {
    throw std::runtime_error("ResultWriter: pipeline not processed, call process() first");
  }

  fs::path json_file(json_path);
  if (json_file.has_parent_path())
  {
    fs::create_directories(json_file.parent_path());
  }

  // Saliency map as 16-bit grayscale PNG next to the JSON file
  std::string map_name = json_file.stem().string() + "_saliency.png";
  cv::Mat map16;
  pipeline.get_saliency_map().map.convertTo(map16, CV_16U, 65535.0);
  fs::path map_path = json_file.has_parent_path() ? json_file.parent_path() / map_name : fs::path(map_name);
  if (!cv::imwrite(map_path.string(), map16))
  {
    throw std::runtime_error("ResultWriter: failed to write saliency map: " + map_path.string());
  }

  const auto& frame = pipeline.get_frame();
  const auto& config = pipeline.get_config();
  const auto& timing = pipeline.get_timing();

  std::ofstream out(json_path);
  if (!out)
  {
    throw std::runtime_error("ResultWriter: failed to open for writing: " + json_path);
  }

  out << "{\n";
  out << "  \"schema\": \"attention-result/v1\",\n";
  out << "  \"source\": {\n";
  out << "    \"image\": \"" << escape_json(frame.source_path) << "\",\n";
  out << "    \"width\": " << frame.width() << ",\n";
  out << "    \"height\": " << frame.height() << "\n";
  out << "  },\n";
  out << "  \"generator\": {\n";
  out << "    \"name\": \"attention-framework\",\n";
  out << "    \"variant\": \"" << escape_json(config.fusion + "+" + config.effective_selection()) << "\"\n";
  out << "  },\n";

  out << "  \"params\": {\n";
  out << "    \"feature_weights\": {";
  bool first = true;
  for (const auto& spec : config.features)
  {
    if (!spec.enabled)
    {
      continue;
    }
    if (!first)
    {
      out << ", ";
    }
    out << "\"" << escape_json(spec.type) << "\": " << format_float(spec.weight);
    first = false;
  }
  out << "},\n";
  out << "    \"peaks\": {\n";
  out << "      \"min_distance\": " << config.peak_min_distance << ",\n";
  out << "      \"threshold\": " << format_float(config.peak_threshold) << ",\n";
  out << "      \"max_count\": " << config.peak_max_count << ",\n";
  out << "      \"selection\": \"" << escape_json(config.effective_selection()) << "\",\n";
  out << "      \"ior_radius\": " << config.ior_radius << ",\n";
  out << "      \"ior_strength\": " << format_float(config.ior_strength) << "\n";
  out << "    }\n";
  out << "  },\n";

  out << "  \"saliency_map\": \"" << escape_json(map_name) << "\",\n";

  // Fixations in scanpath order (peak storage order: descending salience for
  // NMS, selection sequence for IOR)
  out << "  \"fixations\": [\n";
  const auto& peaks = pipeline.get_saliency_map().peaks;
  for (size_t i = 0; i < peaks.size(); ++i)
  {
    out << "    {\"n\": " << i << ", \"x\": " << peaks[i].location.x << ", \"y\": " << peaks[i].location.y
        << ", \"value\": " << format_float(peaks[i].value) << "}";
    if (i + 1 < peaks.size())
    {
      out << ",";
    }
    out << "\n";
  }
  out << "  ],\n";

  out << "  \"timing_ms\": {\n";
  out << "    \"pyramid\": " << timing.pyramid_ms << ",\n";
  out << "    \"features\": {";
  first = true;
  for (const auto& pair : timing.feature_ms)
  {
    if (!first)
    {
      out << ", ";
    }
    out << "\"" << escape_json(pair.first) << "\": " << pair.second;
    first = false;
  }
  out << "},\n";
  out << "    \"integration\": " << timing.integration_ms << ",\n";
  out << "    \"peak_detection\": " << timing.peak_detection_ms << ",\n";
  out << "    \"total\": " << timing.total_ms() << "\n";
  out << "  }\n";
  out << "}\n";

  if (!out.good())
  {
    throw std::runtime_error("ResultWriter: error while writing: " + json_path);
  }
}

} // namespace io
} // namespace attention
