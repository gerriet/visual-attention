#pragma once

#include <chrono>
#include <cmath>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

namespace attention
{
namespace core
{

/**
 * Key identifying one Gabor filter bank: banks with different parameters are
 * distinct (angular spacing is 180°/orientations, so they cannot share data).
 */
struct GaborBankKey
{
  int orientations;
  double wavelength;
  double bandwidth;

  bool operator<(const GaborBankKey& other) const
  {
    return std::tie(orientations, wavelength, bandwidth) <
           std::tie(other.orientations, other.wavelength, other.bandwidth);
  }
};

/**
 * Frame represents a single image with optional metadata and cached pyramids.
 * Supports move semantics for efficient transfer of image data.
 * Caches multi-scale pyramids to avoid redundant computation across features.
 */
struct Frame
{
  // Gabor bank: responses indexed [level][orientation]
  using GaborBank = std::vector<std::vector<cv::Mat>>;

  // Image data
  cv::Mat image;

  // Optional companion inputs for multi-image / temporal features (M5). Both
  // are empty for a plain monocular still; a feature that needs one declares
  // itself inapplicable (FeatureExtractor::applicable) when its input is absent.
  //   stereo_right  — the right image of a stereo pair (StereoFeature)
  //   previous_gray — the previous frame's grayscale (OnsetFeature); the
  //                   pipeline injects it from RunState before extraction
  cv::Mat stereo_right;
  cv::Mat previous_gray;

  // Cached pyramids (computed once, shared across features)
  std::vector<cv::Mat> rgb_pyramid;              // RGB color pyramid
  std::vector<cv::Mat> gray_pyramid;             // Grayscale intensity pyramid
  std::map<GaborBankKey, GaborBank> gabor_banks; // Gabor banks by parameter set
  bool pyramids_computed = false;                // Flag to track if pyramids are cached

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
  void compute_pyramids(int levels);

  /**
   * Compute and cache a Gabor filter bank for the given parameter set.
   * No-op when a bank with these parameters and at least as many levels
   * already exists. NOT thread-safe: the pipeline precomputes all banks the
   * enabled features require before parallel extraction (features declare
   * their needs via FeatureExtractor::gabor_requirement()).
   * @param levels Number of pyramid levels
   * @param num_orientations Number of orientations (angular spacing 180°/N)
   * @param wavelength Wavelength of the Gabor filter
   * @param bandwidth Bandwidth parameter
   */
  void compute_gabor_bank(int levels, int num_orientations, double wavelength, double bandwidth);

  /**
   * Access a previously computed Gabor bank.
   * @throws std::runtime_error if no bank with these parameters exists —
   *         it must be precomputed (the pipeline does this)
   */
  const GaborBank& gabor_bank(int num_orientations, double wavelength, double bandwidth) const;

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
