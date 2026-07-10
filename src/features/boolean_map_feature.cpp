#include "attention/features/boolean_map_feature.h"
#include <stdexcept>
#include <vector>

namespace attention
{
namespace features
{

BooleanMapFeature::BooleanMapFeature(const Config& config) : config_(config) {}

void BooleanMapFeature::accumulate_surrounded(const cv::Mat& boolean_map, cv::Mat& accumulator) const
{
  // Label foreground (255) connected components; a component is "figure" only if
  // it does not touch the image border (i.e. it is fully surrounded).
  cv::Mat labels;
  const int num_labels = cv::connectedComponents(boolean_map, labels, 8, CV_32S);
  if (num_labels <= 1)
  {
    return; // no foreground component
  }

  std::vector<char> touches_border(num_labels, 0);
  for (int x = 0; x < labels.cols; ++x)
  {
    touches_border[labels.at<int>(0, x)] = 1;
    touches_border[labels.at<int>(labels.rows - 1, x)] = 1;
  }
  for (int y = 0; y < labels.rows; ++y)
  {
    touches_border[labels.at<int>(y, 0)] = 1;
    touches_border[labels.at<int>(y, labels.cols - 1)] = 1;
  }

  // Build the figure map (label 0 is the thresholded-out background: skip it).
  cv::Mat figure = cv::Mat::zeros(boolean_map.size(), CV_32F);
  for (int y = 0; y < labels.rows; ++y)
  {
    const int* label_row = labels.ptr<int>(y);
    float* figure_row = figure.ptr<float>(y);
    for (int x = 0; x < labels.cols; ++x)
    {
      const int label = label_row[x];
      if (label != 0 && !touches_border[label])
      {
        figure_row[x] = 1.0f;
      }
    }
  }

  // L2-normalize before accumulating so a boolean map with a small surrounded
  // region is not swamped by one with a large one (Zhang & Sclaroff weighting).
  const double norm = cv::norm(figure, cv::NORM_L2);
  if (norm > 1e-6)
  {
    accumulator += figure / norm;
  }
}

core::FeatureMap BooleanMapFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("BooleanMapFeature: cannot extract from empty frame");
  }

  // Work at a reduced, aspect-preserving resolution.
  const cv::Size orig = frame.image.size();
  const int longer = std::max(orig.width, orig.height);
  const double scale = std::min(1.0, static_cast<double>(config_.working_size) / std::max(1, longer));
  const cv::Size work(std::max(8, cvRound(orig.width * scale)), std::max(8, cvRound(orig.height * scale)));

  cv::Mat img;
  cv::resize(frame.image, img, work, 0, 0, cv::INTER_AREA);

  // Colour images are decomposed in CIELab; grayscale uses intensity alone.
  std::vector<cv::Mat> channels;
  if (img.channels() == 3)
  {
    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab); // 8-bit Lab is enough for thresholding
    cv::split(lab, channels);
  }
  else
  {
    channels.push_back(img);
  }

  const int step = std::max(1, config_.threshold_step);
  cv::Mat accumulator = cv::Mat::zeros(work, CV_32F);
  int contributions = 0;

  for (const cv::Mat& channel : channels)
  {
    for (int t = step; t < 255; t += step)
    {
      // Both polarities: bright-surrounded-on-dark and dark-surrounded-on-bright.
      cv::Mat boolean_map;
      cv::threshold(channel, boolean_map, t, 255, cv::THRESH_BINARY);
      accumulate_surrounded(boolean_map, accumulator);

      cv::threshold(channel, boolean_map, t, 255, cv::THRESH_BINARY_INV);
      accumulate_surrounded(boolean_map, accumulator);

      contributions += 2;
    }
  }

  if (contributions > 0)
  {
    accumulator /= static_cast<float>(contributions);
  }

  cv::GaussianBlur(accumulator, accumulator, cv::Size(0, 0), config_.blur_sigma);
  cv::Mat result;
  cv::resize(accumulator, result, orig, 0, 0, cv::INTER_LINEAR);
  cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);

  return core::FeatureMap("boolean-map", result, 1.0f);
}

} // namespace features
} // namespace attention
