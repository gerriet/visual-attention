#pragma once

namespace attention
{
namespace constants
{

// Gabor filter parameters
constexpr double GABOR_KERNEL_SIZE_FACTOR = 2.5;   // Multiplier for kernel size based on wavelength
constexpr double GABOR_PHASE_OFFSET = 1.5707963267948966;  // M_PI * 0.5
constexpr double GABOR_ASPECT_RATIO = 0.5;         // Spatial aspect ratio (gamma)

// Peak detection parameters
constexpr int PEAK_DETECTION_DOWNSAMPLE_THRESHOLD = 640;  // Image dimension threshold for downsampling

// Symmetry feature weights
constexpr float BILATERAL_SYMMETRY_WEIGHT = 0.4f;  // Weight for bilateral symmetry
constexpr float RADIAL_SYMMETRY_WEIGHT = 0.6f;     // Weight for radial symmetry

// Visualization parameters
constexpr float HEATMAP_ALPHA = 0.6f;              // Alpha blend for saliency heatmap overlay
constexpr float ORIGINAL_ALPHA = 0.4f;             // Alpha blend for original image
constexpr float IOR_SIGMA_FACTOR = 2.5f;           // Divisor for IOR Gaussian sigma calculation

} // namespace constants
} // namespace attention
