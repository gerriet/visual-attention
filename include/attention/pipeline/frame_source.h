#pragma once

#include "attention/core/frame.h"
#include <string>
#include <vector>

namespace attention
{
namespace pipeline
{

/**
 * FrameSource yields the frames of a stream. A single image is simply a
 * stream of length one; video and stereo sources (roadmap M5) plug in here.
 */
class FrameSource
{
 public:
  virtual ~FrameSource() = default;

  /**
   * Produce the next frame.
   * @param frame Output frame (frame_number is set by the source)
   * @return true if a frame was produced, false at end of stream
   * @throws std::runtime_error if a frame exists but cannot be loaded
   */
  virtual bool next(core::Frame& frame) = 0;
};

/**
 * ImageListSource yields images from an explicit list of file paths.
 */
class ImageListSource : public FrameSource
{
 public:
  explicit ImageListSource(std::vector<std::string> paths);

  bool next(core::Frame& frame) override;

  const std::vector<std::string>& paths() const { return paths_; }

 private:
  std::vector<std::string> paths_;
  size_t index_ = 0;
};

/**
 * Collect all image files (jpg/jpeg/png/bmp) in a directory, sorted by path.
 * Feed the result to ImageListSource.
 */
std::vector<std::string> collect_image_paths(const std::string& directory);

} // namespace pipeline
} // namespace attention
