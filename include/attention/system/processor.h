#pragma once

#include "attention/system/object_file.h"
#include <functional>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace attention
{
namespace system
{

/**
 * One localized hit inside a processed region (a detector match). `box` is
 * relative to the ROI the processor received; run_processor() translates it
 * into image coordinates before the annotation is stored.
 */
struct Detection
{
  cv::Rect box;
  float confidence = 0.0f; // in [0, 1]
  std::string label;       // class of this hit ("person", "face", ...)
};

/**
 * The result of running a processor on one attended region.
 *
 * `class_label` + `confidence` are the semantic result: what the processor
 * thinks the region is. An empty class_label means "inspected, nothing
 * recognized" — still evidence (the object file counts the inspection).
 * The bookkeeping fields (`frame`, `pixels`, `ms`) are filled by
 * run_processor(), not by the processor itself.
 */
struct Annotation
{
  int object_label = -1;             // the object file this annotates (-1 == whole frame)
  std::string processor;             // which processor produced this
  std::string label;                 // short text drawn on the overlay
  std::string detail;                // longer description (logged / tooltip)
  std::string class_label;           // semantic class, empty if nothing recognized
  float confidence = 0.0f;           // confidence in class_label, in [0, 1]
  std::vector<Detection> detections; // localized hits (image coordinates)

  // Compute accounting for the gated-recognition argument (H2): what was
  // analyzed, and what it cost. frame/ms come from run_processor(). pixels
  // defaults to the handed-in ROI; a processor that internally rescales
  // before analyzing (e.g. hog-person upscaling small ROIs) sets it to the
  // pixel count it actually scanned, and run_processor() keeps that value.
  int frame = -1;       // frame index the processor ran on
  long long pixels = 0; // pixels the processor actually analyzed
  double ms = 0.0;      // wall-clock of the process() call
};

/**
 * Processor: a pluggable analysis that runs on a single attended object file
 * and its image region (ROI, at native resolution). This is the attention
 * premise made concrete — expensive analysis runs only where the system is
 * looking, not over the whole frame. Processors are registry- and
 * config-driven, like features / fusion / selection strategies.
 */
class Processor
{
 public:
  virtual ~Processor() = default;

  virtual std::string name() const = 0;

  /**
   * Analyze one attended region.
   * @param object The object file (position, size, saliency, history)
   * @param roi    The object's image region at native resolution (may be BGR
   *               or single channel; empty if the bbox fell outside the frame)
   */
  virtual Annotation process(const ObjectFile& object, const cv::Mat& roi) const = 0;
};

/**
 * Registry of processor factories (name -> factory). Mirrors FeatureRegistry.
 *
 * A processor spec may carry a config string after the first colon —
 * "dnn-classify:models/net.onnx:models/labels.txt" — which is handed to the
 * factory verbatim (empty for a bare name). Parameter-free processors ignore
 * it.
 */
class ProcessorRegistry
{
 public:
  using Factory = std::function<std::unique_ptr<Processor>(const std::string& config)>;

  static ProcessorRegistry& instance();

  void add(const std::string& name, Factory factory);
  bool has(const std::string& name) const { return factories_.count(name) > 0; }
  std::unique_ptr<Processor> create(const std::string& spec) const;
  std::vector<std::string> available() const;

 private:
  std::map<std::string, Factory> factories_;
};

/**
 * Run a processor on one region and fill in the bookkeeping the processor
 * itself does not know: frame index, pixels processed, wall-clock, and the
 * translation of detection boxes from ROI-relative to image coordinates
 * (`roi_origin` is the ROI's top-left corner in the image).
 */
Annotation run_processor(const Processor& processor, const ObjectFile& object, const cv::Mat& roi, cv::Point roi_origin,
                         int frame);

/**
 * Register the built-in processors ("roi-probe", "region-descriptor", and the
 * M13 recognition tier: "hog-person", "haar-face", "dnn-classify").
 * Idempotent.
 */
void register_builtin_processors();

/**
 * Create a processor from a spec — a name, optionally followed by a
 * colon-separated config string (registers built-ins first).
 * @throws std::runtime_error for unknown names.
 */
std::unique_ptr<Processor> create_processor(const std::string& spec);

} // namespace system
} // namespace attention
