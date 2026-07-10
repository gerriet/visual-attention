#include "attention/features/minimum_barrier_feature.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace attention
{
namespace features
{

namespace
{

// Relax pixel (y, x) from an already-swept neighbor (ny, nx): extending the
// neighbor's best path by this pixel gives barrier max(U, I) − min(L, I).
inline void relax(const cv::Mat& img, cv::Mat& dist, cv::Mat& upper, cv::Mat& lower, int y, int x, int ny, int nx)
{
  const float value = img.at<float>(y, x);
  const float u = std::max(upper.at<float>(ny, nx), value);
  const float l = std::min(lower.at<float>(ny, nx), value);
  const float barrier = u - l;
  if (barrier < dist.at<float>(y, x))
  {
    dist.at<float>(y, x) = barrier;
    upper.at<float>(y, x) = u;
    lower.at<float>(y, x) = l;
  }
}

// FastMBD (Zhang et al. 2015, Alg. 2): minimum barrier distance from the image
// border, approximated by alternating forward/backward raster sweeps. Forward
// sweeps consider the up/left neighbors, backward sweeps down/right.
cv::Mat mbd_transform(const cv::Mat& img, int num_passes)
{
  const int rows = img.rows, cols = img.cols;
  cv::Mat dist(img.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
  cv::Mat upper = img.clone();
  cv::Mat lower = img.clone();

  // Seed set: the border pixels (distance 0, path = the pixel itself).
  for (int x = 0; x < cols; ++x)
  {
    dist.at<float>(0, x) = 0.0f;
    dist.at<float>(rows - 1, x) = 0.0f;
  }
  for (int y = 0; y < rows; ++y)
  {
    dist.at<float>(y, 0) = 0.0f;
    dist.at<float>(y, cols - 1) = 0.0f;
  }

  for (int pass = 0; pass < num_passes; ++pass)
  {
    const bool forward = (pass % 2 == 0);
    if (forward)
    {
      for (int y = 0; y < rows; ++y)
      {
        for (int x = 0; x < cols; ++x)
        {
          if (y > 0)
          {
            relax(img, dist, upper, lower, y, x, y - 1, x);
          }
          if (x > 0)
          {
            relax(img, dist, upper, lower, y, x, y, x - 1);
          }
        }
      }
    }
    else
    {
      for (int y = rows - 1; y >= 0; --y)
      {
        for (int x = cols - 1; x >= 0; --x)
        {
          if (y < rows - 1)
          {
            relax(img, dist, upper, lower, y, x, y + 1, x);
          }
          if (x < cols - 1)
          {
            relax(img, dist, upper, lower, y, x, y, x + 1);
          }
        }
      }
    }
  }
  return dist;
}

} // namespace

MinimumBarrierFeature::MinimumBarrierFeature(const Config& config) : config_(config) {}

core::FeatureMap MinimumBarrierFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("MinimumBarrierFeature: cannot extract from empty frame");
  }

  const cv::Size orig = frame.image.size();
  const int longer = std::max(orig.width, orig.height);
  const double scale = static_cast<double>(config_.working_size) / std::max(1, longer);
  const cv::Size work(std::max(8, cvRound(orig.width * scale)), std::max(8, cvRound(orig.height * scale)));
  const int interp = (work.area() < orig.area()) ? cv::INTER_AREA : cv::INTER_LINEAR;

  cv::Mat small;
  cv::resize(frame.image, small, work, 0, 0, interp);

  // Per-channel MBD transform, summed (CIE Lab for colour input, as in the
  // paper; grayscale uses the single channel).
  std::vector<cv::Mat> channels;
  if (small.channels() == 3)
  {
    cv::Mat lab;
    cv::cvtColor(small, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32F, 1.0 / 255.0);
    cv::split(lab, channels);
  }
  else
  {
    cv::Mat gray;
    small.convertTo(gray, CV_32F, 1.0 / 255.0);
    channels.push_back(gray);
  }

  cv::Mat saliency = cv::Mat::zeros(work, CV_32F);
  const int passes = std::max(1, config_.num_passes);
  for (const auto& channel : channels)
  {
    saliency += mbd_transform(channel, passes);
  }

  // Smooth, upscale to the original size, normalize to [0, 1].
  cv::GaussianBlur(saliency, saliency, cv::Size(0, 0), config_.blur_sigma);
  cv::Mat result;
  cv::resize(saliency, result, orig, 0, 0, cv::INTER_LINEAR);
  cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);

  return core::FeatureMap("minimum-barrier", result, 1.0f);
}

} // namespace features
} // namespace attention
