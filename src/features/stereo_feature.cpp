#include "attention/features/stereo_feature.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace attention
{
namespace features
{

StereoFeature::StereoFeature(const Config& config) : config_(config)
{
  if (config_.window_size < 3 || config_.window_size % 2 == 0)
  {
    throw std::runtime_error("StereoFeature: window_size must be odd and >= 3");
  }
  if (config_.min_disparity > config_.max_disparity)
  {
    throw std::runtime_error("StereoFeature: min_disparity must be <= max_disparity");
  }
}

void StereoFeature::orientation_scheme(std::vector<double>& thetas, std::vector<double>& weights) const
{
  // Near-vertical orientations only (vertical carries horizontal disparity;
  // §5.4.2). theta is the edge-normal angle; 0 == vertical edges. Weights and
  // angle sets mirror the original StereoFeature presets.
  const double d30 = 30.0 * M_PI / 180.0;
  const double d15 = 15.0 * M_PI / 180.0;
  switch (config_.num_orientations)
  {
    case 1:
      thetas = {0.0};
      weights = {1.0};
      break;
    case 5:
      thetas = {0.0, d15, -d15, d30, -d30};
      weights = {0.4, 0.2, 0.2, 0.1, 0.1};
      break;
    case 3:
    default:
      thetas = {0.0, d30, -d30};
      weights = {0.5, 0.25, 0.25};
      break;
  }
}

cv::Mat StereoFeature::gabor_magnitude(const cv::Mat& gray, double theta) const
{
  // Phase-invariant edge energy = magnitude of the quadrature Gabor pair.
  int ksize = static_cast<int>(std::ceil(config_.gabor_wavelength * 1.5));
  if (ksize % 2 == 0)
  {
    ksize++;
  }
  const double sigma = config_.gabor_wavelength * config_.gabor_bandwidth / M_PI;
  const double gamma = 1.0;
  cv::Mat kernel_even =
      cv::getGaborKernel(cv::Size(ksize, ksize), sigma, theta, config_.gabor_wavelength, gamma, 0.0, CV_32F);
  cv::Mat kernel_odd = cv::getGaborKernel(cv::Size(ksize, ksize), sigma, theta, config_.gabor_wavelength, gamma,
                                          M_PI * 0.5, CV_32F);
  cv::Mat resp_even, resp_odd;
  cv::filter2D(gray, resp_even, CV_32F, kernel_even);
  cv::filter2D(gray, resp_odd, CV_32F, kernel_odd);

  cv::Mat magnitude;
  cv::magnitude(resp_even, resp_odd, magnitude);
  return magnitude;
}

core::FeatureMap StereoFeature::extract(const core::Frame& frame) const
{
  DebugContext dummy_debug;
  return extract(frame, dummy_debug);
}

core::FeatureMap StereoFeature::extract(const core::Frame& frame, DebugContext& /*debug*/) const
{
  if (frame.empty())
  {
    throw std::runtime_error("StereoFeature: cannot extract from empty frame");
  }
  if (frame.stereo_right.empty())
  {
    throw std::runtime_error("StereoFeature: no right stereo image on the frame");
  }

  // Left/right to grayscale float, right resized to the left's geometry.
  cv::Mat left_gray, right_gray;
  if (frame.image.channels() == 3)
  {
    cv::cvtColor(frame.image, left_gray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    left_gray = frame.image;
  }
  if (frame.stereo_right.channels() == 3)
  {
    cv::cvtColor(frame.stereo_right, right_gray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    right_gray = frame.stereo_right;
  }
  if (right_gray.size() != left_gray.size())
  {
    cv::resize(right_gray, right_gray, left_gray.size(), 0, 0, cv::INTER_AREA);
  }

  // Downscale to the working resolution (correlation cost is linear in width
  // and in the disparity count, which itself scales with width — §5.4.2).
  const cv::Size full_size = left_gray.size();
  double scale = 1.0;
  const int longer = std::max(full_size.width, full_size.height);
  if (longer > config_.max_working_size)
  {
    scale = static_cast<double>(config_.max_working_size) / longer;
  }
  cv::Mat left_w, right_w;
  if (scale < 1.0)
  {
    cv::resize(left_gray, left_w, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::resize(right_gray, right_w, cv::Size(), scale, scale, cv::INTER_AREA);
  }
  else
  {
    left_w = left_gray;
    right_w = right_gray;
  }
  left_w.convertTo(left_w, CV_32F);
  right_w.convertTo(right_w, CV_32F);

  const int W = left_w.cols;
  const int H = left_w.rows;
  const int dl = config_.window_size;
  const int hdl = (dl - 1) / 2;
  const int min_d = config_.min_disparity;
  const int max_d = config_.max_disparity;
  const int num_d = max_d - min_d + 1;

  if (W <= dl)
  {
    // Too narrow at the working resolution to run the correlation window — no
    // disparity can be recovered, so return a flat (no-depth) map rather than
    // throw. applicable() only screens for a right image, not for size, and a
    // thrown exception here would still be caught by the pipeline, but a small
    // valid pair should simply produce zero depth saliency.
    return core::FeatureMap(name(), cv::Mat::zeros(full_size, CV_32F));
  }

  std::vector<double> thetas, weights;
  orientation_scheme(thetas, weights);
  const int num_o = static_cast<int>(thetas.size());

  // Gabor magnitude responses per orientation, for both images.
  std::vector<cv::Mat> magL(num_o), magR(num_o);
  for (int b = 0; b < num_o; ++b)
  {
    magL[b] = gabor_magnitude(left_w, thetas[b]);
    magR[b] = gabor_magnitude(right_w, thetas[b]);
  }

  // Confidence volume: one CV_32F plane per disparity, accumulated over
  // orientations. Each row is written by a single thread (see the parallel
  // loop), so the slices need no locking.
  std::vector<cv::Mat> conf(num_d);
  for (int d = 0; d < num_d; ++d)
  {
    conf[d] = cv::Mat::zeros(H, W, CV_32F);
  }

  // Gaussian window weights along the correlation window (envelope of the
  // Gabor). Unnormalized — the factor cancels in the normalized correlation.
  std::vector<float> win(dl);
  const double win_sigma2 = 2.0 * 7.06; // matches the original's window spread
  for (int k = 0; k < dl; ++k)
  {
    const double dpos = k - hdl;
    win[k] = static_cast<float>(std::exp(-dpos * dpos / win_sigma2));
  }

  const float var_thresh = static_cast<float>(config_.variance_threshold);
  const int last_x = W - dl; // last valid window start

#pragma omp parallel for schedule(dynamic)
  for (int y = 0; y < H; ++y)
  {
    // Per-orientation windowed statistics for this row.
    std::vector<float> con_l(last_x + 1), con_r(last_x + 1);
    std::vector<float> sig_l(last_x + 1), sig_r(last_x + 1);

    for (int b = 0; b < num_o; ++b)
    {
      const float* lrow = magL[b].ptr<float>(y);
      const float* rrow = magR[b].ptr<float>(y);

      // Windowed energy (weighted) and std-dev (unweighted, for gating).
      for (int x = 0; x <= last_x; ++x)
      {
        float el = 0.0f, er = 0.0f, ml = 0.0f, mr = 0.0f;
        for (int k = 0; k < dl; ++k)
        {
          const float lv = lrow[x + k];
          const float rv = rrow[x + k];
          el += win[k] * lv * lv;
          er += win[k] * rv * rv;
          ml += lv;
          mr += rv;
        }
        con_l[x] = el;
        con_r[x] = er;
        ml /= dl;
        mr /= dl;
        float vl = 0.0f, vr = 0.0f;
        for (int k = 0; k < dl; ++k)
        {
          const float dvl = lrow[x + k] - ml;
          const float dvr = rrow[x + k] - mr;
          vl += dvl * dvl;
          vr += dvr * dvr;
        }
        sig_l[x] = std::sqrt(vl / dl);
        sig_r[x] = std::sqrt(vr / dl);
      }

      // Windowed normalized cross-correlation over the disparity range (eq. 5.13).
      for (int x = 0; x <= last_x; ++x)
      {
        if (sig_l[x] < var_thresh)
        {
          continue; // too little structure on the left — no reliable match
        }
        const int lo = std::max(min_d, -x);
        const int hi = std::min(max_d, last_x - x);
        for (int disp = lo; disp <= hi; ++disp)
        {
          if (sig_r[x + disp] < var_thresh)
          {
            continue;
          }
          float num = 0.0f;
          for (int k = 0; k < dl; ++k)
          {
            num += win[k] * lrow[x + k] * rrow[x + k + disp];
          }
          const float denom = std::sqrt(con_l[x]) * std::sqrt(con_r[x + disp]);
          if (denom <= 0.0f)
          {
            continue;
          }
          const float rho = num / denom;
          if (rho > 0.0f)
          {
            // Accumulate at the window centre, weighted by orientation.
            conf[disp - min_d].at<float>(y, x + hdl) += static_cast<float>(weights[b]) * rho;
          }
        }
      }
    }
  }

  // Spatial integration of the confidence (the Gaussian neighborhood wσ of
  // eq. 5.14): blur each disparity slice.
  if (config_.confidence_blur >= 3 && config_.confidence_blur % 2 == 1)
  {
    for (int d = 0; d < num_d; ++d)
    {
      cv::GaussianBlur(conf[d], conf[d], cv::Size(config_.confidence_blur, config_.confidence_blur), 0);
    }
  }

  // Winner-take-all per pixel (eq. 5.15) → normalized disparity magnitude
  // saliency (eq. 5.16). Near surfaces (large |d|) are salient; pixels with no
  // reliable correspondence (winning confidence ~0) stay dark.
  const float disp_span = static_cast<float>(std::max(1, std::max(std::abs(min_d), std::abs(max_d))));
  cv::Mat depth = cv::Mat::zeros(H, W, CV_32F);
  for (int y = 0; y < H; ++y)
  {
    float* out = depth.ptr<float>(y);
    for (int x = 0; x < W; ++x)
    {
      float best = 0.0f;
      int best_d = 0;
      for (int d = 0; d < num_d; ++d)
      {
        const float v = conf[d].at<float>(y, x);
        if (v > best)
        {
          best = v;
          best_d = d;
        }
      }
      if (best > 1e-6f)
      {
        const int disparity = min_d + best_d;
        out[x] = std::abs(disparity) / disp_span;
      }
    }
  }

  // Back to full resolution.
  cv::Mat result;
  if (depth.size() != full_size)
  {
    cv::resize(depth, result, full_size, 0, 0, cv::INTER_LINEAR);
  }
  else
  {
    result = depth;
  }

  return core::FeatureMap(name(), result);
}

} // namespace features
} // namespace attention
