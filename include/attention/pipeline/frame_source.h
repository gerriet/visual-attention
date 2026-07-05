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
 * StereoImageSource yields left/right image pairs: each produced frame has its
 * left image in Frame::image and the matching right image in
 * Frame::stereo_right, so the stereo feature runs on a sequence of pairs. The
 * two path lists must be the same length.
 */
class StereoImageSource : public FrameSource
{
 public:
  StereoImageSource(std::vector<std::string> left_paths, std::vector<std::string> right_paths);

  bool next(core::Frame& frame) override;

 private:
  std::vector<std::string> left_paths_;
  std::vector<std::string> right_paths_;
  size_t index_ = 0;
};

/**
 * VideoFrameSource yields the frames of a video file (via cv::VideoCapture) as
 * a temporal stream — the input for onset/motion. The caller keeps RunState
 * across frames (no per-frame reset) so temporal features see the predecessor.
 */
class VideoFrameSource : public FrameSource
{
 public:
  explicit VideoFrameSource(const std::string& path);

  bool next(core::Frame& frame) override;

 private:
  cv::VideoCapture capture_;
  std::string path_;
  int index_ = 0;
};

/**
 * Collect all image files (jpg/jpeg/png/bmp) in a directory, sorted by path.
 * Feed the result to ImageListSource.
 */
std::vector<std::string> collect_image_paths(const std::string& directory);

} // namespace pipeline
} // namespace attention
