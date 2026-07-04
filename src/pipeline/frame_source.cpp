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

  const std::string& path = paths_[index_];
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty())
  {
    throw std::runtime_error("Failed to load image: " + path);
  }

  frame = core::Frame(image, path);
  frame.frame_number = static_cast<int>(index_);
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
