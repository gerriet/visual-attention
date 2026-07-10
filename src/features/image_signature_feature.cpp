#include "attention/features/image_signature_feature.h"
#include <stdexcept>
#include <vector>

namespace attention
{
namespace features
{

namespace
{

// sign(x) in {-1, 0, +1} as CV_32F.
cv::Mat sign_of(const cv::Mat& m)
{
  cv::Mat pos, neg;
  cv::compare(m, 0.0, pos, cv::CMP_GT);
  cv::compare(m, 0.0, neg, cv::CMP_LT);
  pos.convertTo(pos, CV_32F, 1.0 / 255.0);
  neg.convertTo(neg, CV_32F, 1.0 / 255.0);
  return pos - neg;
}

// Squared reconstruction from the DCT sign signature of one CV_32F channel.
cv::Mat signature_response(const cv::Mat& channel)
{
  cv::Mat coeffs, recon;
  cv::dct(channel, coeffs);
  cv::dct(sign_of(coeffs), recon, cv::DCT_INVERSE);
  cv::multiply(recon, recon, recon);
  return recon;
}

} // namespace

ImageSignatureFeature::ImageSignatureFeature(const Config& config) : config_(config) {}

core::FeatureMap ImageSignatureFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("ImageSignatureFeature: cannot extract from empty frame");
  }

  // Working resolution: small, aspect-preserving, EVEN dimensions (cv::dct
  // supports even-sized arrays only).
  const cv::Size orig = frame.image.size();
  const int longer = std::max(orig.width, orig.height);
  const double scale = static_cast<double>(config_.input_size) / std::max(1, longer);
  auto even = [](double v) { return std::max(8, 2 * cvRound(v / 2.0)); };
  const cv::Size work(even(orig.width * scale), even(orig.height * scale));
  const int interp = (work.area() < orig.area()) ? cv::INTER_AREA : cv::INTER_LINEAR;

  cv::Mat small;
  cv::resize(frame.image, small, work, 0, 0, interp);

  // Per-channel signature response, summed. Colour input runs in CIE Lab (as
  // in the paper's colour experiments); grayscale uses the single channel.
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
  for (const auto& channel : channels)
  {
    saliency += signature_response(channel);
  }

  // Smooth, upscale to the original size, normalize to [0, 1].
  cv::GaussianBlur(saliency, saliency, cv::Size(0, 0), config_.gaussian_sigma);
  cv::Mat result;
  cv::resize(saliency, result, orig, 0, 0, cv::INTER_LINEAR);
  cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);

  return core::FeatureMap("image-signature", result, 1.0f);
}

} // namespace features
} // namespace attention
