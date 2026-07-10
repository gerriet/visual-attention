#include "attention/features/spectral_residual_feature.h"
#include <stdexcept>

namespace attention
{
namespace features
{

SpectralResidualFeature::SpectralResidualFeature(const Config& config) : config_(config) {}

core::FeatureMap SpectralResidualFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("SpectralResidualFeature: cannot extract from empty frame");
  }

  // 1. Grayscale copy at a small, aspect-preserving working resolution. The
  //    model is defined at low resolution (Hou & Zhang use ~64 px).
  cv::Mat gray;
  if (frame.channels() == 3)
  {
    cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    gray = frame.image;
  }

  const cv::Size orig = gray.size();
  const int longer = std::max(orig.width, orig.height);
  const double scale = static_cast<double>(config_.input_size) / std::max(1, longer);
  const cv::Size work(std::max(8, cvRound(orig.width * scale)), std::max(8, cvRound(orig.height * scale)));

  cv::Mat small;
  // INTER_AREA only downsamples well; a small input is upsampled to the working
  // size, where AREA degrades to blocky nearest-neighbor — use LINEAR there.
  const int interp = (work.area() < orig.area()) ? cv::INTER_AREA : cv::INTER_LINEAR;
  cv::resize(gray, small, work, 0, 0, interp);
  small.convertTo(small, CV_32F);

  // 2. Forward DFT into a two-channel complex spectrum.
  cv::Mat planes[] = {small, cv::Mat::zeros(work, CV_32F)};
  cv::Mat complex;
  cv::merge(planes, 2, complex);
  cv::dft(complex, complex);
  cv::split(complex, planes);

  // 3. Log-amplitude spectrum and phase.
  cv::Mat amplitude, phase;
  cv::magnitude(planes[0], planes[1], amplitude);
  cv::phase(planes[0], planes[1], phase);
  cv::max(amplitude, 1e-8, amplitude); // guard log(0) at empty bins
  cv::Mat log_amplitude;
  cv::log(amplitude, log_amplitude);

  // 4. Spectral residual: log spectrum minus its local average (h_n * A). The
  //    box filter uses OpenCV's default BORDER_REFLECT_101 (the conventional SR
  //    choice); the numpy twin in eval/.../spectral_residual.py zero-pads, so
  //    the two are the same model but not bit-identical at the spectrum edges.
  cv::Mat averaged;
  const int k = std::max(1, config_.avg_filter_size);
  cv::boxFilter(log_amplitude, averaged, -1, cv::Size(k, k));
  cv::Mat residual = log_amplitude - averaged;

  // 5. Reconstruct with residual amplitude and original phase, inverse DFT.
  cv::Mat residual_amplitude;
  cv::exp(residual, residual_amplitude);
  cv::polarToCart(residual_amplitude, phase, planes[0], planes[1]);
  cv::merge(planes, 2, complex);
  cv::idft(complex, complex);
  cv::split(complex, planes);

  cv::Mat saliency;
  cv::magnitude(planes[0], planes[1], saliency);
  cv::multiply(saliency, saliency, saliency); // squared response (Hou & Zhang)

  // 6. Smooth, upscale to the original size, normalize to [0, 1].
  cv::GaussianBlur(saliency, saliency, cv::Size(0, 0), config_.gaussian_sigma);
  cv::Mat result;
  cv::resize(saliency, result, orig, 0, 0, cv::INTER_LINEAR);
  cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);

  return core::FeatureMap("spectral-residual", result, 1.0f);
}

} // namespace features
} // namespace attention
