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
 * The result of running a processor on one attended region.
 */
struct Annotation
{
  int object_label = -1; // the object file this annotates (for exact matching)
  std::string processor; // which processor produced this
  std::string label;     // short text drawn on the overlay
  std::string detail;    // longer description (logged / tooltip)
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
 */
class ProcessorRegistry
{
 public:
  using Factory = std::function<std::unique_ptr<Processor>()>;

  static ProcessorRegistry& instance();

  void add(const std::string& name, Factory factory);
  bool has(const std::string& name) const { return factories_.count(name) > 0; }
  std::unique_ptr<Processor> create(const std::string& name) const;
  std::vector<std::string> available() const;

 private:
  std::map<std::string, Factory> factories_;
};

/**
 * Register the built-in processors ("roi-probe", "region-descriptor").
 * Idempotent.
 */
void register_builtin_processors();

/**
 * Create a processor by name (registers built-ins first).
 * @throws std::runtime_error for unknown names.
 */
std::unique_ptr<Processor> create_processor(const std::string& name);

} // namespace system
} // namespace attention
