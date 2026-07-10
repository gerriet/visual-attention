#include "attention/features/phase_spectrum_feature.h"
#include <stdexcept>
#include <vector>

namespace attention
{
namespace features
{

namespace
{

// Squared phase-only reconstruction of one CV_32F channel: forward DFT,
// amplitude normalized to 1 (keeping the phase), inverse DFT, magnitude².
cv::Mat phase_only_response(const cv::Mat& channel)
{
  cv::Mat planes[] = {channel.clone(), cv::Mat::zeros(channel.size(), CV_32F)};
  cv::Mat complex;
  cv::merge(planes, 2, complex);
  cv::dft(complex, complex);
  cv::split(complex, planes);

  cv::Mat amplitude;
  cv::magnitude(planes[0], planes[1], amplitude);
  cv::max(amplitude, 1e-8, amplitude); // guard empty bins
  cv::divide(planes[0], amplitude, planes[0]);
  cv::divide(planes[1], amplitude, planes[1]);

  cv::merge(planes, 2, complex);
  cv::idft(complex, complex);
  cv::split(complex, planes);

  cv::Mat response;
  cv::magnitude(planes[0], planes[1], response);
  cv::multiply(response, response, response);
  return response;
}

} // namespace

PhaseSpectrumFeature::PhaseSpectrumFeature(const Config& config) : config_(config) {}

core::FeatureMap PhaseSpectrumFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("PhaseSpectrumFeature: cannot extract from empty frame");
  }

  const cv::Size orig = frame.image.size();
  const int longer = std::max(orig.width, orig.height);
  const double scale = static_cast<double>(config_.input_size) / std::max(1, longer);
  const cv::Size work(std::max(8, cvRound(orig.width * scale)), std::max(8, cvRound(orig.height * scale)));
  const int interp = (work.area() < orig.area()) ? cv::INTER_AREA : cv::INTER_LINEAR;

  cv::Mat small;
  cv::resize(frame.image, small, work, 0, 0, interp);

  // Channels per Guo et al.: intensity + colour opponency (RG, BY) for colour
  // input, intensity only for grayscale.
  std::vector<cv::Mat> channels;
  cv::Mat intensity;
  if (small.channels() == 3)
  {
    cv::Mat f;
    small.convertTo(f, CV_32F, 1.0 / 255.0);
    cv::Mat bgr[3];
    cv::split(f, bgr);
    const cv::Mat &b = bgr[0], &g = bgr[1], &r = bgr[2];

    intensity = (r + g + b) / 3.0f;
    cv::Mat R = r - (g + b) / 2.0f;
    cv::Mat G = g - (r + b) / 2.0f;
    cv::Mat B = b - (r + g) / 2.0f;
    cv::Mat Y = (r + g) / 2.0f - cv::abs(r - g) / 2.0f - b;
    channels.push_back(intensity);
    channels.push_back(R - G);
    channels.push_back(B - Y);
  }
  else
  {
    small.convertTo(intensity, CV_32F, 1.0 / 255.0);
    channels.push_back(intensity);
  }

  // Motion channel on temporal streams: |I_t − I_{t−1}| at working resolution.
  // Both sides use BGR2GRAY luma (previous_gray's weighting) so a static scene
  // is exactly zero — the PQFT intensity channel weights channels differently.
  if (config_.use_motion && !frame.previous_gray.empty())
  {
    cv::Mat current_gray;
    if (small.channels() == 3)
    {
      cv::cvtColor(small, current_gray, cv::COLOR_BGR2GRAY);
      current_gray.convertTo(current_gray, CV_32F, 1.0 / 255.0);
    }
    else
    {
      current_gray = intensity;
    }
    cv::Mat prev;
    cv::resize(frame.previous_gray, prev, work, 0, 0, interp);
    prev.convertTo(prev, CV_32F, 1.0 / 255.0);
    channels.push_back(cv::abs(current_gray - prev));
  }

  cv::Mat saliency = cv::Mat::zeros(work, CV_32F);
  for (const auto& channel : channels)
  {
    saliency += phase_only_response(channel);
  }

  // Smooth, upscale to the original size, normalize to [0, 1].
  cv::GaussianBlur(saliency, saliency, cv::Size(0, 0), config_.gaussian_sigma);
  cv::Mat result;
  cv::resize(saliency, result, orig, 0, 0, cv::INTER_LINEAR);
  cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);

  return core::FeatureMap("phase-spectrum", result, 1.0f);
}

} // namespace features
} // namespace attention
