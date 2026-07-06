#include "attention/pipeline/frame_source.h"
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace fs = std::filesystem;

namespace attention
{
namespace pipeline
{

ImageListSource::ImageListSource(std::vector<std::string> paths) : paths_(std::move(paths)) {}

bool ImageListSource::next(core::Frame& frame)
{
  if (index_ >= paths_.size())
  {
    return false;
  }

  // Advance before loading so a throwing entry is not retried when the
  // caller's error handler continues the stream
  const std::string path = paths_[index_];
  const int frame_number = static_cast<int>(index_);
  ++index_;

  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty())
  {
    throw std::runtime_error("Failed to load image: " + path);
  }

  frame = core::Frame(image, path);
  frame.frame_number = frame_number;
  return true;
}

StereoImageSource::StereoImageSource(std::vector<std::string> left_paths, std::vector<std::string> right_paths)
  : left_paths_(std::move(left_paths)), right_paths_(std::move(right_paths))
{
  if (left_paths_.size() != right_paths_.size())
  {
    throw std::runtime_error("StereoImageSource: left and right path lists differ in length");
  }
}

bool StereoImageSource::next(core::Frame& frame)
{
  if (index_ >= left_paths_.size())
  {
    return false;
  }

  // Advance before loading so a throwing entry is not retried on continue.
  const std::string left_path = left_paths_[index_];
  const std::string right_path = right_paths_[index_];
  const int frame_number = static_cast<int>(index_);
  ++index_;

  cv::Mat left = cv::imread(left_path, cv::IMREAD_COLOR);
  if (left.empty())
  {
    throw std::runtime_error("Failed to load left image: " + left_path);
  }
  cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);
  if (right.empty())
  {
    throw std::runtime_error("Failed to load right image: " + right_path);
  }

  frame = core::Frame(left, left_path);
  frame.stereo_right = right;
  frame.frame_number = frame_number;
  return true;
}

VideoFrameSource::VideoFrameSource(const std::string& path) : path_(path)
{
  if (!capture_.open(path))
  {
    throw std::runtime_error("Failed to open video: " + path);
  }
}

bool VideoFrameSource::next(core::Frame& frame)
{
  cv::Mat image;
  if (!capture_.read(image) || image.empty())
  {
    return false; // end of stream
  }

  frame = core::Frame(image, path_ + "#" + std::to_string(index_));
  frame.frame_number = index_;
  ++index_;
  return true;
}

VideoSource::VideoSource(const std::string& source) : source_(source)
{
  const bool all_digits = !source.empty() && source.find_first_not_of("0123456789") == std::string::npos;
  bool opened = false;
  if (all_digits)
  {
    is_camera_ = true;
    try
    {
      opened = capture_.open(std::stoi(source));
    }
    catch (const std::exception&)
    {
      opened = false; // implausibly large index -> clean open failure below
    }
  }
  else
  {
    opened = capture_.open(source);
  }
  if (!opened)
  {
    throw std::runtime_error("Failed to open video source: " + source +
                             (all_digits ? " (camera device)" : " (file)"));
  }
}

bool VideoSource::next(core::Frame& frame)
{
  cv::Mat image;
  if (!capture_.read(image) || image.empty())
  {
    return false; // end of file, or camera stopped
  }
  frame = core::Frame(image, source_ + "#" + std::to_string(index_));
  frame.frame_number = index_;
  ++index_;
  return true;
}

std::vector<std::string> collect_image_paths(const std::string& directory)
{
  std::vector<std::string> image_paths;

  for (const auto& entry : fs::directory_iterator(directory))
  {
    if (entry.is_regular_file())
    {
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
      {
        image_paths.push_back(entry.path().string());
      }
    }
  }

  std::sort(image_paths.begin(), image_paths.end());
  return image_paths;
}

} // namespace pipeline
} // namespace attention
