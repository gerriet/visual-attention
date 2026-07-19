#include "attention/system/processor.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <opencv2/dnn.hpp>
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

// Monotone squash of an unbounded detector score into [0, 1). Not a calibrated
// probability — good enough for weighting label votes and ranking.
float squash_score(double score, double scale)
{
  if (score <= 0.0)
  {
    return 0.0f;
  }
  return static_cast<float>(score / (score + scale));
}

// hog-person: OpenCV's HOG pedestrian detector (Dalal & Triggs) on the
// attended region. Zero new dependencies — the Tier-1 recognition processor.
class HogPerson : public Processor
{
 public:
  HogPerson() { hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector()); }

  std::string name() const override { return "hog-person"; }

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

    // The detector's window is 64x128; an ROI that frames a person tightly is
    // smaller than that, so upscale small ROIs enough to give the window (plus
    // some slack) room. Detections are mapped back to ROI coordinates.
    const cv::Mat bgr = as_bgr(roi);
    const double up = std::max({1.0, 88.0 / bgr.cols, 176.0 / bgr.rows});
    cv::Mat search = bgr;
    if (up > 1.0)
    {
      cv::resize(bgr, search, cv::Size(), up, up, cv::INTER_LINEAR);
    }

    std::vector<cv::Rect> found;
    std::vector<double> weights;
    hog_.detectMultiScale(search, found, weights, 0.0, cv::Size(8, 8), cv::Size(16, 16), 1.05, 2.0, false);
    // Charge what the detector actually scanned (the upscaled image), not the
    // handed-in ROI — otherwise gated runs under-count their compute.
    a.pixels = static_cast<long long>(search.total());

    float best = 0.0f;
    for (size_t i = 0; i < found.size(); ++i)
    {
      Detection d;
      d.box = cv::Rect(static_cast<int>(found[i].x / up), static_cast<int>(found[i].y / up),
                       static_cast<int>(found[i].width / up), static_cast<int>(found[i].height / up));
      // SVM margin, typically 0..~3.
      d.confidence = squash_score(i < weights.size() ? weights[i] : 0.0, 1.0);
      d.label = "person";
      best = std::max(best, d.confidence);
      a.detections.push_back(d);
    }
    if (!a.detections.empty())
    {
      a.class_label = "person";
      a.confidence = best;
      a.label = "#" + std::to_string(object.label) + " person(" + std::to_string(a.detections.size()) + ")";
      a.detail = std::to_string(a.detections.size()) + " person(s), best conf=" + fmt(best);
    }
    else
    {
      a.label = "#" + std::to_string(object.label) + " no-person";
      a.detail = "no person found";
    }
    return a;
  }

 private:
  cv::HOGDescriptor hog_;
};

// haar-face: OpenCV's Haar-cascade frontal-face detector on the attended
// region. The cascade XML ships with every OpenCV install (not with this
// repo); locate it via $ATTENTION_HAAR_DIR or the usual install locations.
class HaarFace : public Processor
{
 public:
  HaarFace()
  {
    const std::string path = find_cascade("haarcascade_frontalface_default.xml");
    if (path.empty() || !cascade_.load(path))
    {
      throw std::runtime_error(
          "haar-face: haarcascade_frontalface_default.xml not found. Set ATTENTION_HAAR_DIR to "
          "your OpenCV haarcascades directory.");
    }
  }

  std::string name() const override { return "haar-face"; }

  Annotation process(const ObjectFile& object, const cv::Mat& roi) const override
  {
    Annotation a;
    a.processor = name();
    a.object_label = object.label;
    if (roi.empty() || roi.cols < 24 || roi.rows < 24) // below the cascade's window
    {
      a.label = "#" + std::to_string(object.label) + " ?";
      return a;
    }

    cv::Mat gray;
    cv::cvtColor(as_bgr(roi), gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> found;
    std::vector<int> reject_levels;
    std::vector<double> level_weights;
    cascade_.detectMultiScale(gray, found, reject_levels, level_weights, 1.1, 3, 0, cv::Size(24, 24), cv::Size(), true);

    float best = 0.0f;
    for (size_t i = 0; i < found.size(); ++i)
    {
      Detection d;
      d.box = found[i];
      // Final-stage weight, typically 0..~10.
      d.confidence = squash_score(i < level_weights.size() ? level_weights[i] : 0.0, 4.0);
      d.label = "face";
      best = std::max(best, d.confidence);
      a.detections.push_back(d);
    }
    if (!a.detections.empty())
    {
      a.class_label = "face";
      a.confidence = best;
      a.label = "#" + std::to_string(object.label) + " face(" + std::to_string(a.detections.size()) + ")";
      a.detail = std::to_string(a.detections.size()) + " face(s), best conf=" + fmt(best);
    }
    else
    {
      a.label = "#" + std::to_string(object.label) + " no-face";
      a.detail = "no face found";
    }
    return a;
  }

 private:
  static std::string find_cascade(const std::string& file)
  {
    std::vector<std::string> dirs;
    if (const char* env = std::getenv("ATTENTION_HAAR_DIR"))
    {
      dirs.push_back(env);
    }
    dirs.insert(dirs.end(),
                {"/opt/homebrew/opt/opencv/share/opencv4/haarcascades", "/usr/local/share/opencv4/haarcascades",
                 "/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades"});
    for (const auto& dir : dirs)
    {
      const std::filesystem::path candidate = std::filesystem::path(dir) / file;
      if (std::filesystem::exists(candidate))
      {
        return candidate.string();
      }
    }
    return "";
  }

  // detectMultiScale is non-const in OpenCV; processors run single-threaded.
  mutable cv::CascadeClassifier cascade_;
};

// dnn-classify: any ImageNet-style ONNX classifier via cv::dnn (no new link
// dependency). Config: "model.onnx:labels.txt[:input_size]"; defaults expect
// tools/fetch_models.py to have populated models/. Preprocessing is the
// standard ImageNet recipe (RGB, /255, mean/std normalize), which the ONNX
// model-zoo classifiers (SqueezeNet, MobileNetV2, ResNet, ...) share.
class DnnClassify : public Processor
{
 public:
  explicit DnnClassify(const std::string& config)
  {
    std::string model = "models/squeezenet1.1.onnx";
    std::string labels = "models/imagenet_classes.txt";
    std::vector<std::string> parts;
    std::stringstream ss(config);
    std::string item;
    while (std::getline(ss, item, ':'))
    {
      parts.push_back(item);
    }
    if (parts.size() > 0 && !parts[0].empty())
    {
      model = parts[0];
    }
    if (parts.size() > 1 && !parts[1].empty())
    {
      labels = parts[1];
    }
    if (parts.size() > 2 && !parts[2].empty())
    {
      input_size_ = std::stoi(parts[2]);
    }

    if (!std::filesystem::exists(model))
    {
      throw std::runtime_error("dnn-classify: model not found: " + model +
                               " (run tools/fetch_models.py, or pass dnn-classify:<model>:<labels>)");
    }
    net_ = cv::dnn::readNetFromONNX(model);

    std::ifstream in(labels);
    if (!in)
    {
      throw std::runtime_error("dnn-classify: labels file not found: " + labels);
    }
    std::string line;
    while (std::getline(in, line))
    {
      labels_.push_back(line);
    }
  }

  std::string name() const override { return "dnn-classify"; }

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

    // ImageNet preprocessing: RGB, resize to the input square, x/255, then
    // per-channel mean/std normalization.
    const cv::Mat bgr = as_bgr(roi);
    cv::Mat blob = cv::dnn::blobFromImage(bgr, 1.0 / 255.0, cv::Size(input_size_, input_size_),
                                          cv::Scalar(0.485 * 255, 0.456 * 255, 0.406 * 255), true, false);
    const float std_rgb[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c)
    {
      cv::Mat plane(input_size_, input_size_, CV_32F, blob.ptr<float>(0, c));
      plane /= std_rgb[c];
    }

    net_.setInput(blob);
    cv::Mat out = net_.forward();
    out = out.reshape(1, 1); // [1,N] / [1,N,1,1] -> flat row of N scores

    // Softmax over the logits for a [0,1] confidence.
    double max_logit = 0.0;
    cv::Point max_at;
    cv::minMaxLoc(out, nullptr, &max_logit, nullptr, &max_at);
    cv::Mat shifted;
    cv::exp(out - max_logit, shifted);
    const double denominator = cv::sum(shifted)[0];
    const int index = max_at.x;
    const float prob = static_cast<float>(shifted.at<float>(0, index) / denominator);

    const std::string cls =
        index < static_cast<int>(labels_.size()) ? labels_[index] : "class-" + std::to_string(index);
    a.class_label = cls;
    a.confidence = prob;
    a.label = "#" + std::to_string(object.label) + " " + cls;
    a.detail = cls + " p=" + fmt(prob);
    return a;
  }

 private:
  mutable cv::dnn::Net net_; // forward() is non-const; processors run single-threaded
  std::vector<std::string> labels_;
  int input_size_ = 224;
};

} // namespace

Annotation run_processor(const Processor& processor, const ObjectFile& object, const cv::Mat& roi, cv::Point roi_origin,
                         int frame)
{
  const auto t0 = std::chrono::high_resolution_clock::now();
  Annotation a = processor.process(object, roi);
  const auto t1 = std::chrono::high_resolution_clock::now();
  a.frame = frame;
  if (a.pixels == 0) // processors that internally rescale report their own count
  {
    a.pixels = static_cast<long long>(roi.total());
  }
  a.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  for (auto& d : a.detections)
  {
    d.box.x += roi_origin.x;
    d.box.y += roi_origin.y;
  }
  return a;
}

ProcessorRegistry& ProcessorRegistry::instance()
{
  static ProcessorRegistry registry;
  return registry;
}

void ProcessorRegistry::add(const std::string& name, Factory factory)
{
  factories_[name] = std::move(factory);
}

std::unique_ptr<Processor> ProcessorRegistry::create(const std::string& spec) const
{
  // "name" or "name:config" — everything after the first colon goes to the
  // factory verbatim.
  const size_t colon = spec.find(':');
  const std::string name = spec.substr(0, colon);
  const std::string config = colon == std::string::npos ? "" : spec.substr(colon + 1);

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
  return it->second(config);
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
  // Parameter-free processors reject a config string outright — silently
  // ignoring "hog-person:scale=1.05" would run an experiment with different
  // parameters than the user believes.
  auto no_config = [](const std::string& name, const std::string& config)
  {
    if (!config.empty())
    {
      throw std::runtime_error("processor '" + name + "' takes no config (got ':" + config + "')");
    }
  };
  auto& registry = ProcessorRegistry::instance();
  registry.add("roi-probe",
               [no_config](const std::string& config)
               {
                 no_config("roi-probe", config);
                 return std::make_unique<RoiProbe>();
               });
  registry.add("region-descriptor",
               [no_config](const std::string& config)
               {
                 no_config("region-descriptor", config);
                 return std::make_unique<RegionDescriptor>();
               });
  registry.add("hog-person",
               [no_config](const std::string& config)
               {
                 no_config("hog-person", config);
                 return std::make_unique<HogPerson>();
               });
  registry.add("haar-face",
               [no_config](const std::string& config)
               {
                 no_config("haar-face", config);
                 return std::make_unique<HaarFace>();
               });
  registry.add("dnn-classify", [](const std::string& config) { return std::make_unique<DnnClassify>(config); });
}

std::unique_ptr<Processor> create_processor(const std::string& spec)
{
  register_builtin_processors();
  return ProcessorRegistry::instance().create(spec);
}

} // namespace system
} // namespace attention
