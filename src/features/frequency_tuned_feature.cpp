#include "attention/features/frequency_tuned_feature.h"
#include <stdexcept>

namespace attention
{
namespace features
{

FrequencyTunedFeature::FrequencyTunedFeature(const Config& config) : config_(config) {}

core::FeatureMap FrequencyTunedFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("FrequencyTunedFeature: cannot extract from empty frame");
  }

  // Suppress fine texture/noise (the high-frequency band) with a small Gaussian.
  const int k = config_.blur_size | 1; // force odd
  cv::Mat blurred;
  cv::GaussianBlur(frame.image, blurred, cv::Size(k, k), 0);

  cv::Mat saliency;
  if (frame.channels() == 3)
  {
    // Distance in CIELab from the image's mean colour (removes the low-frequency
    // background band). Convert via float BGR so Lab has its true ranges.
    cv::Mat bgr_f, lab;
    blurred.convertTo(bgr_f, CV_32F, 1.0 / 255.0);
    cv::cvtColor(bgr_f, lab, cv::COLOR_BGR2Lab);

    const cv::Scalar mu = cv::mean(lab);
    std::vector<cv::Mat> ch;
    cv::split(lab, ch);

    cv::Mat dl = ch[0] - mu[0];
    cv::Mat da = ch[1] - mu[1];
    cv::Mat db = ch[2] - mu[2];

    cv::Mat term;
    cv::multiply(dl, dl, saliency);
    cv::multiply(da, da, term);
    saliency += term;
    cv::multiply(db, db, term);
    saliency += term;
    cv::sqrt(saliency, saliency);
  }
  else
  {
    cv::Mat gray_f;
    blurred.convertTo(gray_f, CV_32F);
    const cv::Scalar mu = cv::mean(gray_f);
    saliency = cv::abs(gray_f - mu[0]);
  }

  cv::normalize(saliency, saliency, 0.0, 1.0, cv::NORM_MINMAX);
  return core::FeatureMap("frequency-tuned", saliency, 1.0f);
}

} // namespace features
} // namespace attention
