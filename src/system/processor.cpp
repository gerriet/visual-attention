#include "attention/system/processor.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace attention
{
namespace system
{

namespace
{

std::string fmt(double value, int precision = 2)
{
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

cv::Mat as_bgr(const cv::Mat& roi)
{
  if (roi.channels() == 3)
  {
    return roi;
  }
  cv::Mat bgr;
  cv::cvtColor(roi, bgr, cv::COLOR_GRAY2BGR);
  return bgr;
}

// Coarse colour name for a mean BGR value (via HSV).
std::string colour_name(const cv::Scalar& mean_bgr)
{
  cv::Mat px(1, 1, CV_8UC3, cv::Scalar(mean_bgr[0], mean_bgr[1], mean_bgr[2]));
  cv::cvtColor(px, px, cv::COLOR_BGR2HSV);
  const cv::Vec3b hsv = px.at<cv::Vec3b>(0, 0);
  const int h = hsv[0], s = hsv[1], v = hsv[2];
  if (v < 50)
    return "black";
  if (s < 40)
    return v > 200 ? "white" : "gray";
  if (h < 10 || h >= 160)
    return "red";
  if (h < 25)
    return "orange";
  if (h < 35)
    return "yellow";
  if (h < 85)
    return "green";
  if (h < 100)
    return "cyan";
  if (h < 130)
    return "blue";
  return "purple";
}

// roi-probe: reports the ROI dimensions and the time a trivial pass over it
// takes — proves the plugin mechanism and that work is bounded to the region.
class RoiProbe : public Processor
{
 public:
  std::string name() const override { return "roi-probe"; }

  Annotation process(const ObjectFile& object, const cv::Mat& roi) const override
  {
    Annotation a;
    a.processor = name();
    a.object_label = object.label;
    if (roi.empty())
    {
      a.label = "#" + std::to_string(object.label) + " roi:empty";
      a.detail = "object " + std::to_string(object.label) + " has no in-frame ROI";
      return a;
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    const cv::Scalar mean = cv::mean(roi); // the "analysis" — bounded to the ROI
    const auto t1 = std::chrono::high_resolution_clock::now();
    const long us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    a.label = "#" + std::to_string(object.label) + " " + std::to_string(roi.cols) + "x" + std::to_string(roi.rows);
    a.detail = "roi " + std::to_string(roi.cols) + "x" + std::to_string(roi.rows) + " (" + std::to_string(roi.total()) +
               " px), mean=" + fmt(mean[0]) + ", " + std::to_string(us) + "us";
    return a;
  }
};

// region-descriptor: a real, dependency-free analysis of the attended region —
// colour, brightness, edge density, aspect ratio.
class RegionDescriptor : public Processor
{
 public:
  std::string name() const override { return "region-descriptor"; }

  Annotation process(const ObjectFile& object, const cv::Mat& roi) const override
  {
    Annotation a;
    a.processor = name();
    a.object_label = object.label;
    if (roi.empty())
    {
      a.label = "#" + std::to_string(object.label) + " ?";
      return a;
    }
    const cv::Mat bgr = as_bgr(roi);
    const cv::Scalar mean_bgr = cv::mean(bgr);
    const double lum = 0.114 * mean_bgr[0] + 0.587 * mean_bgr[1] + 0.299 * mean_bgr[2]; // BGR order
    const std::string colour = colour_name(mean_bgr);
    const std::string brightness = lum < 60 ? "dark" : (lum < 170 ? "mid" : "bright");

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    const double edge_density = static_cast<double>(cv::countNonZero(mag > 50.0)) / std::max<size_t>(1, mag.total());
    const double aspect = roi.rows > 0 ? static_cast<double>(roi.cols) / roi.rows : 0.0;

    a.label = "#" + std::to_string(object.label) + " " + brightness + "/" + colour;
    a.detail = "colour=" + colour + " lum=" + fmt(lum, 0) + " edges=" + fmt(edge_density) + " ar=" + fmt(aspect);
    return a;
  }
};

} // namespace

ProcessorRegistry& ProcessorRegistry::instance()
{
  static ProcessorRegistry registry;
  return registry;
}

void ProcessorRegistry::add(const std::string& name, Factory factory)
{
  factories_[name] = std::move(factory);
}

std::unique_ptr<Processor> ProcessorRegistry::create(const std::string& name) const
{
  auto it = factories_.find(name);
  if (it == factories_.end())
  {
    std::ostringstream msg;
    msg << "Unknown processor '" << name << "'. Available:";
    for (const auto& n : available())
    {
      msg << " " << n;
    }
    throw std::runtime_error(msg.str());
  }
  return it->second();
}

std::vector<std::string> ProcessorRegistry::available() const
{
  std::vector<std::string> names;
  for (const auto& pair : factories_)
  {
    names.push_back(pair.first);
  }
  return names;
}

void register_builtin_processors()
{
  static bool registered = false;
  if (registered)
  {
    return;
  }
  registered = true;
  auto& registry = ProcessorRegistry::instance();
  registry.add("roi-probe", [] { return std::make_unique<RoiProbe>(); });
  registry.add("region-descriptor", [] { return std::make_unique<RegionDescriptor>(); });
}

std::unique_ptr<Processor> create_processor(const std::string& name)
{
  register_builtin_processors();
  return ProcessorRegistry::instance().create(name);
}

} // namespace system
} // namespace attention
