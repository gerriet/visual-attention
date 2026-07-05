#include "attention/features/onset_feature.h"
#include <stdexcept>

namespace attention
{
namespace features
{

OnsetFeature::OnsetFeature(const Config& config) : config_(config) {}

cv::Mat OnsetFeature::edge_energy(const cv::Mat& gray) const
{
  cv::Mat gx, gy, mag;
  cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
  cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
  cv::magnitude(gx, gy, mag);
  return mag;
}

core::FeatureMap OnsetFeature::extract(const core::Frame& frame) const
{
  DebugContext dummy_debug;
  return extract(frame, dummy_debug);
}

core::FeatureMap OnsetFeature::extract(const core::Frame& frame, DebugContext& /*debug*/) const
{
  if (frame.empty())
  {
    throw std::runtime_error("OnsetFeature: cannot extract from empty frame");
  }
  if (frame.previous_gray.empty())
  {
    throw std::runtime_error("OnsetFeature: no previous frame on this frame");
  }

  // Current and previous frame to grayscale float, matched in size.
  cv::Mat cur_gray;
  if (frame.image.channels() == 3)
  {
    cv::cvtColor(frame.image, cur_gray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    cur_gray = frame.image;
  }
  cv::Mat prev_gray = frame.previous_gray;
  if (prev_gray.size() != cur_gray.size())
  {
    cv::resize(prev_gray, prev_gray, cur_gray.size(), 0, 0, cv::INTER_AREA);
  }

  cv::Mat cur_f, prev_f;
  cur_gray.convertTo(cur_f, CV_32F);
  prev_gray.convertTo(prev_f, CV_32F);

  // Rectified positive change: where local structure has *appeared* since the
  // previous frame. Offsets (negative change) are clamped to zero.
  cv::Mat change;
  if (config_.use_edges)
  {
    change = edge_energy(cur_f) - edge_energy(prev_f);
  }
  else
  {
    change = cur_f - prev_f;
  }
  cv::threshold(change, change, config_.threshold, 0.0, cv::THRESH_TOZERO);

  // Spatial smoothing of the onset map.
  if (config_.blur_size >= 3 && config_.blur_size % 2 == 1)
  {
    cv::GaussianBlur(change, change, cv::Size(config_.blur_size, config_.blur_size), 0);
  }

  // Normalize to [0, 1]; an all-zero change map (no onset) stays zero.
  double max_val = 0.0;
  cv::minMaxLoc(change, nullptr, &max_val);
  if (max_val > 1e-6)
  {
    change /= max_val;
  }

  return core::FeatureMap(name(), change);
}

} // namespace features
} // namespace attention
