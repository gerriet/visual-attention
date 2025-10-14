#include "attention/core/frame.h"

namespace attention
{
namespace core
{

void Frame::compute_gabor_pyramids(int levels, int num_orientations, double wavelength, double bandwidth)
{
  if (gabor_pyramids_computed && num_gabor_orientations == num_orientations)
    return; // Already computed with same parameters

  if (image.empty() || !pyramids_computed)
  {
    // Need grayscale pyramid first
    if (!pyramids_computed)
      compute_pyramids(levels);
  }

  num_gabor_orientations = num_orientations;
  gabor_pyramids.clear();
  gabor_pyramids.resize(levels);

  // For each pyramid level
  for (int level = 0; level < levels && level < static_cast<int>(gray_pyramid.size()); ++level)
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
  int kernel_size = static_cast<int>(std::ceil(wavelength * 2.5));
  if (kernel_size % 2 == 0)
    kernel_size++; // Ensure odd size

  // Sigma (spatial extent) from bandwidth
  double sigma = wavelength * bandwidth / M_PI;

  // Standard OpenCV Gabor parameters
  double psi = M_PI * 0.5; // Phase offset
  double gamma = 0.5;      // Spatial aspect ratio

  // Create Gabor kernel using OpenCV
  cv::Mat kernel =
      cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, wavelength, gamma, psi, CV_32F);

  return kernel;
}

} // namespace core
} // namespace attention
