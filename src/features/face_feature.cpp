#include "attention/features/face_feature.h"
#include <stdexcept>

#ifndef ATTENTION_HAARCASCADE_FILE
#define ATTENTION_HAARCASCADE_FILE ""
#endif

namespace attention
{
namespace features
{

std::string FaceFeature::default_cascade_path()
{
  return std::string(ATTENTION_HAARCASCADE_FILE);
}

FaceFeature::FaceFeature(const Config& config) : config_(config)
{
  if (config_.backend == "haar")
  {
    const std::string path = config_.model_path.empty() ? default_cascade_path() : config_.model_path;
    if (path.empty())
    {
      throw std::runtime_error("FaceFeature: no Haar cascade available. CMake found none at configure "
                               "time; set features.face.params.model_path to a "
                               "haarcascade_frontalface_*.xml");
    }
    if (!cascade_.load(path))
    {
      throw std::runtime_error("FaceFeature: failed to load Haar cascade '" + path + "'");
    }
  }
  else if (config_.backend == "yunet")
  {
    if (config_.model_path.empty())
    {
      throw std::runtime_error("FaceFeature: backend 'yunet' needs features.face.params.model_path "
                               "(an ONNX file, e.g. face_detection_yunet_2023mar.onnx)");
    }
    // Built lazily on the first frame: FaceDetectorYN wants the input size up
    // front, and we do not know it here.
    yunet_size_ = cv::Size(0, 0);
  }
  else
  {
    throw std::runtime_error("FaceFeature: unknown backend '" + config_.backend + "' (expected 'haar' or 'yunet')");
  }
}

FaceFeature::~FaceFeature() = default;

std::vector<cv::Rect> FaceFeature::detect(const cv::Mat& image) const
{
  std::vector<cv::Rect> faces;

  if (config_.backend == "haar")
  {
    cv::Mat gray;
    if (image.channels() == 3)
    {
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
      gray = image;
    }
    cv::equalizeHist(gray, gray);
    cascade_.detectMultiScale(gray, faces, config_.scale_factor, config_.min_neighbors, 0,
                              cv::Size(config_.min_size, config_.min_size));
    return faces;
  }

  // YuNet needs three channels and a fixed input size; rebuild only when the
  // frame size changes, which for a stream is once.
  cv::Mat bgr;
  if (image.channels() == 1)
  {
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
  }
  else
  {
    bgr = image;
  }
  if (!yunet_ || yunet_size_ != bgr.size())
  {
    yunet_ = cv::FaceDetectorYN::create(config_.model_path, "", bgr.size(), config_.score_threshold,
                                        config_.nms_threshold);
    if (!yunet_)
    {
      throw std::runtime_error("FaceFeature: failed to create YuNet from '" + config_.model_path + "'");
    }
    yunet_size_ = bgr.size();
  }

  cv::Mat detections;
  yunet_->detect(bgr, detections);
  for (int i = 0; i < detections.rows; ++i)
  {
    const float x = detections.at<float>(i, 0);
    const float y = detections.at<float>(i, 1);
    const float w = detections.at<float>(i, 2);
    const float h = detections.at<float>(i, 3);
    if (w >= static_cast<float>(config_.min_size) && h >= static_cast<float>(config_.min_size))
    {
      faces.emplace_back(cvRound(x), cvRound(y), cvRound(w), cvRound(h));
    }
  }
  return faces;
}

core::FeatureMap FaceFeature::extract(const core::Frame& frame) const
{
  DebugContext dummy;
  return extract(frame, dummy);
}

core::FeatureMap FaceFeature::extract(const core::Frame& frame, DebugContext& debug) const
{
  if (frame.empty())
  {
    throw std::runtime_error("FaceFeature: cannot extract from empty frame");
  }

  cv::Mat map = cv::Mat::zeros(frame.size(), CV_32F);
  const std::vector<cv::Rect> faces = detect(frame.image);

  for (const auto& face : faces)
  {
    const float sigma = std::max(1.0f, config_.sigma_frac * static_cast<float>(face.width));
    const cv::Point2f centre(face.x + face.width * 0.5f, face.y + face.height * 0.5f);
    // Paint the blob over a bounded neighbourhood rather than the whole frame:
    // beyond 3 sigma the Gaussian is below 2% and the cost is quadratic in the
    // radius.
    const int radius = cvRound(3.0f * sigma);
    const cv::Rect roi = cv::Rect(cvRound(centre.x) - radius, cvRound(centre.y) - radius, 2 * radius + 1,
                                  2 * radius + 1) &
                         cv::Rect(0, 0, map.cols, map.rows);
    for (int y = roi.y; y < roi.y + roi.height; ++y)
    {
      float* row = map.ptr<float>(y);
      for (int x = roi.x; x < roi.x + roi.width; ++x)
      {
        const float dx = x - centre.x;
        const float dy = y - centre.y;
        row[x] += std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
      }
    }
  }

  // Normalize only when something was found. An empty map must stay all-zero
  // rather than being stretched: "no faces" is a real answer, and normalizing
  // noise up to 1 would make a frame without faces look like a frame full of
  // them — the per-frame-normalization mistake the dossier's finding A is about.
  double hi = 0.0;
  cv::minMaxLoc(map, nullptr, &hi);
  if (hi > 1e-6)
  {
    map /= static_cast<float>(hi);
  }

  debug.add_annotation("faces", std::to_string(faces.size()));
  debug.add_annotation("backend", config_.backend);
  return core::FeatureMap(name(), map);
}

} // namespace features
} // namespace attention
