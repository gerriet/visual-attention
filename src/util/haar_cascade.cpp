#include "attention/util/haar_cascade.h"
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace attention
{
namespace util
{

std::string find_haar_cascade(const std::string& file)
{
  std::vector<std::string> dirs;
  if (const char* env = std::getenv("ATTENTION_HAAR_DIR"))
  {
    dirs.push_back(env);
  }
#ifdef ATTENTION_OPENCV_HAAR_DIR
  dirs.push_back(ATTENTION_OPENCV_HAAR_DIR); // the linked OpenCV's own, baked in by CMake
#endif
  dirs.insert(
      dirs.end(),
      {"/opt/homebrew/opt/opencv/share/opencv4/haarcascades", "/opt/homebrew/opt/opencv@4/share/opencv4/haarcascades",
       "/usr/local/share/opencv4/haarcascades", "/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades"});
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

} // namespace util
} // namespace attention
