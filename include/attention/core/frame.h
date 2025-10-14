#pragma once

#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace attention
{
namespace core
{

/**
 * Frame represents a single image with optional metadata and cached pyramids.
 * Supports move semantics for efficient transfer of image data.
 * Caches multi-scale pyramids to avoid redundant computation across features.
 */
struct Frame
{
  // Image data
  cv::Mat image;

  // Cached pyramids (computed once, shared across features)
  std::vector<cv::Mat> rgb_pyramid;                       // RGB color pyramid
  std::vector<cv::Mat> gray_pyramid;                      // Grayscale intensity pyramid
  std::vector<std::vector<cv::Mat>> gabor_pyramids;       // Gabor filter pyramids [level][orientation]
  bool pyramids_computed = false;                         // Flag to track if pyramids are cached
  bool gabor_pyramids_computed = false;                   // Flag for Gabor pyramids
  int num_gabor_orientations = 12;                        // Number of Gabor orientations (default 12)

  // Optional metadata
  std::string source_path;
  int frame_number = -1;
  std::chrono::system_clock::time_point timestamp;

  // Default constructor
  Frame() = default;

  // Constructor from cv::Mat
  explicit Frame(const cv::Mat& img) : image(img), timestamp(std::chrono::system_clock::now()) {}

  // Constructor from cv::Mat with source path
  Frame(const cv::Mat& img, const std::string& path)
    : image(img), source_path(path), timestamp(std::chrono::system_clock::now())
  {
  }

  // Move constructor
  Frame(Frame&& other) noexcept = default;

  // Move assignment
  Frame& operator=(Frame&& other) noexcept = default;

  // Copy constructor (explicit to avoid accidental copies)
  Frame(const Frame& other) = default;

  // Copy assignment
  Frame& operator=(const Frame& other) = default;

  // Query methods
  bool empty() const { return image.empty(); }
  int width() const { return image.cols; }
  int height() const { return image.rows; }
  int channels() const { return image.channels(); }
  cv::Size size() const { return image.size(); }

  /**
   * Compute and cache multi-scale pyramids.
   * @param levels Number of pyramid levels to compute
   */
  void compute_pyramids(int levels)
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

  /**
   * Compute and cache Gabor filter pyramids.
   * Creates Gabor responses for multiple orientations at each pyramid level.
   * @param levels Number of pyramid levels
   * @param num_orientations Number of orientations (default 12, typical values: 4, 8, 12)
   * @param wavelength Wavelength of the Gabor filter (default 4.0)
   * @param bandwidth Bandwidth parameter (default 1.0)
   */
  void compute_gabor_pyramids(int levels, int num_orientations = 12, double wavelength = 4.0, double bandwidth = 1.0);

private:
  /**
   * Create a Gabor filter kernel.
   * @param wavelength Wavelength of the sinusoidal factor
   * @param theta Orientation of the normal to the parallel stripes
   * @param bandwidth Bandwidth parameter (sigma relative to wavelength)
   * @return Gabor kernel
   */
  cv::Mat create_gabor_kernel(double wavelength, double theta, double bandwidth) const;

public:
};

} // namespace core
} // namespace attention
