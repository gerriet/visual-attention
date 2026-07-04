#include "attention/core/frame.h"
#include "attention/core/constants.h"

namespace attention
{
namespace core
{

void Frame::compute_pyramids(int levels)
{
  if (pyramids_computed)
    return; // Already computed

  if (image.empty())
    return;

  // Compute RGB pyramid (if color image)
  if (channels() == 3)
  {
    cv::Mat rgb_float;
    image.convertTo(rgb_float, CV_32F, 1.0 / 255.0);
    rgb_pyramid.clear();
    rgb_pyramid.push_back(rgb_float.clone());

    cv::Mat current = rgb_float;
    for (int i = 1; i < levels; ++i)
    {
      cv::Mat downsampled;
      cv::pyrDown(current, downsampled);
      rgb_pyramid.push_back(downsampled);
      current = downsampled;
    }
  }

  // Compute grayscale pyramid (always)
  cv::Mat gray;
  if (channels() == 1)
  {
    image.convertTo(gray, CV_32F, 1.0 / 255.0);
  }
  else
  {
    cv::Mat gray_temp;
    cv::cvtColor(image, gray_temp, cv::COLOR_BGR2GRAY);
    gray_temp.convertTo(gray, CV_32F, 1.0 / 255.0);
  }

  gray_pyramid.clear();
  gray_pyramid.push_back(gray.clone());

  cv::Mat current = gray;
  for (int i = 1; i < levels; ++i)
  {
    cv::Mat downsampled;
    cv::pyrDown(current, downsampled);
    gray_pyramid.push_back(downsampled);
    current = downsampled;
  }

  pyramids_computed = true;
}

void Frame::compute_gabor_pyramids(int levels, int num_orientations, double wavelength, double bandwidth)
{
  if (image.empty() || !pyramids_computed)
  {
    // Need grayscale pyramid first
    if (!pyramids_computed)
      compute_pyramids(levels);
  }

  // Check if already computed with same parameters (orientations AND levels)
  int actual_levels = std::min(levels, static_cast<int>(gray_pyramid.size()));

  if (gabor_pyramids_computed &&
      num_gabor_orientations >= num_orientations &&
      static_cast<int>(gabor_pyramids.size()) >= actual_levels)
  {
    return; // Already have enough
  }

  // Note: no incremental "extend with more orientations" path. Orientation
  // angles are spaced 180°/num_orientations, so adding orientations changes
  // the spacing of ALL entries — extending in place would mix two spacings
  // in one bank. A larger request always triggers a full recompute.

  // Full recompute needed
  num_gabor_orientations = num_orientations;
  gabor_pyramids.clear();
  gabor_pyramids.resize(actual_levels);

  // For each pyramid level
  for (int level = 0; level < actual_levels; ++level)
  {
    const cv::Mat& source = gray_pyramid[level];
    gabor_pyramids[level].resize(num_orientations);

    // For each orientation
    for (int orient = 0; orient < num_orientations; ++orient)
    {
      double theta = M_PI * orient / num_orientations; // Orientation angle

      // Create Gabor kernel
      cv::Mat gabor_kernel = create_gabor_kernel(wavelength, theta, bandwidth);

      // Apply Gabor filter
      cv::Mat gabor_response;
      cv::filter2D(source, gabor_response, CV_32F, gabor_kernel);

      // Store magnitude
      gabor_response = cv::abs(gabor_response);
      gabor_pyramids[level][orient] = gabor_response;
    }
  }

  gabor_pyramids_computed = true;
}

cv::Mat Frame::create_gabor_kernel(double wavelength, double theta, double bandwidth) const
{
  // Kernel size based on wavelength
  int kernel_size = static_cast<int>(std::ceil(wavelength * constants::GABOR_KERNEL_SIZE_FACTOR));
  if (kernel_size % 2 == 0)
    kernel_size++; // Ensure odd size

  // Sigma (spatial extent) from bandwidth
  double sigma = wavelength * bandwidth / M_PI;

  // Standard OpenCV Gabor parameters
  double psi = constants::GABOR_PHASE_OFFSET;  // Phase offset (M_PI * 0.5)
  double gamma = constants::GABOR_ASPECT_RATIO; // Spatial aspect ratio

  // Create Gabor kernel using OpenCV
  cv::Mat kernel =
      cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, wavelength, gamma, psi, CV_32F);

  return kernel;
}

} // namespace core
} // namespace attention
