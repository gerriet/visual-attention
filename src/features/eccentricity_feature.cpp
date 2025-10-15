#include "attention/features/eccentricity_feature.h"
#include <iostream>
#include <map>

namespace attention
{
namespace features
{

EccentricityFeature::EccentricityFeature() : config_() {}

EccentricityFeature::EccentricityFeature(const Config& config) : config_(config) {}

core::FeatureMap EccentricityFeature::extract(const core::Frame& frame) const
{
  if (frame.empty())
  {
    throw std::runtime_error("EccentricityFeature: Cannot extract from empty frame");
  }

  // C-2 FIX: Validate pyramid state and bounds check scale index
  if (!frame.pyramids_computed || frame.gray_pyramid.empty())
  {
    throw std::runtime_error("EccentricityFeature: Grayscale pyramid not computed");
  }

  int scale_index = std::min(config_.compute_at_scale, static_cast<int>(frame.gray_pyramid.size()) - 1);
  if (scale_index < 0)
  {
    throw std::runtime_error("EccentricityFeature: Invalid pyramid configuration");
  }

  // Select the appropriate pyramid level
  const cv::Mat& gray = frame.gray_pyramid[scale_index];

  // Compute edges using Sobel gradient magnitude
  cv::Mat grad_x, grad_y;
  cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

  // Compute gradient magnitude
  cv::Mat edges;
  cv::magnitude(grad_x, grad_y, edges);
  cv::normalize(edges, edges, 0.0, 1.0, cv::NORM_MINMAX);

  // Segment the image
  cv::Mat labels = segment_image(gray, edges);

  // Filter segments by area and compute moments
  int image_area = gray.rows * gray.cols;
  std::map<int, cv::Moments> valid_segments = filter_segments(labels, image_area);

  // Create eccentricity map
  cv::Mat eccentricity_map = cv::Mat::zeros(gray.size(), CV_32F);

  // For each valid segment, compute eccentricity and fill the region
  for (const auto& pair : valid_segments)
  {
    int label = pair.first;
    const cv::Moments& m = pair.second;

    float ecc = compute_eccentricity(m);

    // Fill all pixels belonging to this segment with eccentricity value
    for (int y = 0; y < labels.rows; ++y)
    {
      for (int x = 0; x < labels.cols; ++x)
      {
        if (labels.at<int>(y, x) == label)
        {
          eccentricity_map.at<float>(y, x) = ecc;
        }
      }
    }
  }

  // Resize to original frame size if needed
  cv::Mat result;
  if (scale_index > 0)
  {
    cv::resize(eccentricity_map, result, frame.size(), 0, 0, cv::INTER_LINEAR);
  }
  else
  {
    result = eccentricity_map;
  }

  // Normalize to [0, 1]
  cv::normalize(result, result, 0.0, 1.0, cv::NORM_MINMAX);

  // Create feature map
  core::FeatureMap feature;
  feature.name = "eccentricity";
  feature.data = result;
  feature.confidence = 1.0f;

  return feature;
}

cv::Mat EccentricityFeature::segment_image(const cv::Mat& gray, const cv::Mat& edges) const
{
  // Create binary edge map based on threshold
  cv::Mat edge_binary;
  cv::threshold(edges, edge_binary, config_.edge_threshold, 1.0, cv::THRESH_BINARY);
  edge_binary.convertTo(edge_binary, CV_8U, 255);

  // Use distance transform and watershed for segmentation
  cv::Mat dist;
  cv::distanceTransform(~edge_binary, dist, cv::DIST_L2, 3);

  // Find local maxima as seeds
  cv::Mat dist_8u;
  cv::normalize(dist, dist_8u, 0, 255, cv::NORM_MINMAX, CV_8U);

  // Threshold distance transform to get markers
  cv::Mat markers_8u;
  cv::threshold(dist_8u, markers_8u, 0.3 * 255, 255, cv::THRESH_BINARY);

  // Find connected components as initial markers
  cv::Mat markers;
  int num_labels = cv::connectedComponents(markers_8u, markers, 8, CV_32S);

  // Apply watershed - requires CV_8UC3 input
  cv::Mat gray_8u, gray_bgr;
  if (gray.type() != CV_8U)
  {
    gray.convertTo(gray_8u, CV_8U);
  }
  else
  {
    gray_8u = gray;
  }
  cv::cvtColor(gray_8u, gray_bgr, cv::COLOR_GRAY2BGR);
  cv::watershed(gray_bgr, markers);

  // Watershed sets boundaries to -1, we need to reassign those to nearest segment
  for (int y = 0; y < markers.rows; ++y)
  {
    for (int x = 0; x < markers.cols; ++x)
    {
      if (markers.at<int>(y, x) == -1)
      {
        // Find nearest non-boundary pixel (simple 3x3 search)
        bool found = false;
        for (int dy = -1; dy <= 1 && !found; ++dy)
        {
          for (int dx = -1; dx <= 1 && !found; ++dx)
          {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < markers.cols && ny >= 0 && ny < markers.rows)
            {
              int neighbor_label = markers.at<int>(ny, nx);
              if (neighbor_label > 0)
              {
                markers.at<int>(y, x) = neighbor_label;
                found = true;
              }
            }
          }
        }
        if (!found)
        {
          markers.at<int>(y, x) = 1; // Default to label 1 if no neighbor found
        }
      }
    }
  }

  return markers;
}

float EccentricityFeature::compute_eccentricity(const cv::Moments& m) const
{
  // C-3 FIX: Guard against zero or near-zero area
  if (m.m00 < 1e-10)
  {
    return 0.0f;
  }

  // Compute central moments
  double mu20 = m.mu20 / m.m00;
  double mu02 = m.mu02 / m.m00;
  double mu11 = m.mu11 / m.m00;

  // Compute eigenvalues of the covariance matrix
  // The covariance matrix is:
  // [ mu20  mu11 ]
  // [ mu11  mu02 ]
  //
  // Eigenvalues: lambda = (mu20 + mu02 +/- sqrt((mu20-mu02)^2 + 4*mu11^2)) / 2

  double diff = mu20 - mu02;
  double disc = std::sqrt(diff * diff + 4 * mu11 * mu11);

  double lambda1 = (mu20 + mu02 + disc) / 2.0;
  double lambda2 = (mu20 + mu02 - disc) / 2.0;

  // Eccentricity from eigenvalues: ecc = sqrt(1 - lambda_min/lambda_max)
  if (lambda1 <= 0)
  {
    return 0.0f; // Degenerate case
  }

  double ecc_squared = 1.0 - std::abs(lambda2) / std::abs(lambda1);
  if (ecc_squared < 0)
  {
    ecc_squared = 0;
  }

  return static_cast<float>(std::sqrt(ecc_squared));
}

std::map<int, cv::Moments> EccentricityFeature::filter_segments(const cv::Mat& labels, int image_area) const
{
  std::map<int, cv::Moments> valid_segments;

  // Find unique labels and compute moments
  std::map<int, int> label_counts;

  // Count pixels per label
  for (int y = 0; y < labels.rows; ++y)
  {
    for (int x = 0; x < labels.cols; ++x)
    {
      int label = labels.at<int>(y, x);
      if (label > 0)
      {
        label_counts[label]++;
      }
    }
  }

  // Compute area thresholds
  int min_pixels = static_cast<int>(config_.min_area * image_area / 100.0);
  int max_pixels = static_cast<int>(config_.max_area * image_area / 100.0);

  // For each label, check area constraints and compute moments
  for (const auto& pair : label_counts)
  {
    int label = pair.first;
    int count = pair.second;

    if (count >= min_pixels && count <= max_pixels)
    {
      // Create binary mask for this segment
      cv::Mat mask = (labels == label);

      // Compute moments
      cv::Moments moments = cv::moments(mask, true);

      if (moments.m00 > 0)
      {
        valid_segments[label] = moments;
      }
    }
  }

  return valid_segments;
}

} // namespace features
} // namespace attention
