#include "attention/features/eccentricity_feature.h"
#include <chrono>
#include <iostream>
#include <map>
#include <set>

namespace attention
{
namespace features
{

EccentricityFeature::EccentricityFeature() : config_() {}

EccentricityFeature::EccentricityFeature(const Config& config) : config_(config) {}

core::FeatureMap EccentricityFeature::extract(const core::Frame& frame) const
{
  DebugContext dummy_debug;
  return extract(frame, dummy_debug);
}

core::FeatureMap EccentricityFeature::extract(const core::Frame& frame, DebugContext& debug) const
{
  // Timing (only if debugging)
  auto t_start = std::chrono::high_resolution_clock::now();

  // Validation
  if (frame.empty())
  {
    throw std::runtime_error("EccentricityFeature: Cannot extract from empty frame");
  }

  if (!frame.pyramids_computed || frame.gray_pyramid.empty())
  {
    throw std::runtime_error("EccentricityFeature: Grayscale pyramid not computed");
  }

  // Auto scale (-1): quarter resolution for large images, full otherwise
  // (moved verbatim from the v1 pipeline's size heuristic)
  int requested_scale = config_.compute_at_scale;
  if (requested_scale < 0)
  {
    requested_scale = (frame.width() > 640 || frame.height() > 640) ? 2 : 0;
  }

  int scale_index = std::min(requested_scale, static_cast<int>(frame.gray_pyramid.size()) - 1);
  if (scale_index < 0)
  {
    throw std::runtime_error("EccentricityFeature: Invalid pyramid configuration");
  }

  const cv::Mat& gray = frame.gray_pyramid[scale_index];

  // Step 1: Compute edges using Sobel gradient magnitude
  auto t_edge_start = std::chrono::high_resolution_clock::now();
  cv::Mat grad_x, grad_y;
  cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

  cv::Mat edges;
  cv::magnitude(grad_x, grad_y, edges);
  cv::normalize(edges, edges, 0.0f, 1.0f, cv::NORM_MINMAX);
  auto t_edge_end = std::chrono::high_resolution_clock::now();

  // Step 2: Segment the image using watershed
  auto t_segment_start = std::chrono::high_resolution_clock::now();
  cv::Mat labels = segment_image(gray, edges);
  auto t_segment_end = std::chrono::high_resolution_clock::now();

  // Step 3: Compute eccentricity for each segment
  auto t_ecc_start = std::chrono::high_resolution_clock::now();
  int image_area = gray.rows * gray.cols;
  std::map<int, cv::Moments> valid_segments = filter_segments(labels, image_area);

  // Pre-compute eccentricity for all valid segments (O(N+M) optimization)
  std::map<int, float> label_to_ecc;
  for (const auto& pair : valid_segments)
  {
    label_to_ecc[pair.first] = compute_eccentricity(pair.second);
  }

  // Create eccentricity map - single pass over pixels
  cv::Mat eccentricity_map = cv::Mat::zeros(gray.size(), CV_32F);
  for (int y = 0; y < labels.rows; ++y)
  {
    for (int x = 0; x < labels.cols; ++x)
    {
      int label = labels.at<int>(y, x);
      auto it = label_to_ecc.find(label);
      if (it != label_to_ecc.end())
      {
        eccentricity_map.at<float>(y, x) = it->second;
      }
    }
  }
  auto t_ecc_end = std::chrono::high_resolution_clock::now();

  // Step 4: Resize to original frame size and normalize
  auto t_resize_start = std::chrono::high_resolution_clock::now();
  cv::Mat result;
  if (scale_index > 0)
  {
    cv::resize(eccentricity_map, result, frame.size(), 0, 0, cv::INTER_LINEAR);
  }
  else
  {
    result = eccentricity_map.clone();
  }

  cv::normalize(result, result, 0.0f, 1.0f, cv::NORM_MINMAX);
  auto t_resize_end = std::chrono::high_resolution_clock::now();

  // Capture debug data if requested (keeps algorithm code clean above)
  if (debug.enabled)
  {
    double total_ms = std::chrono::duration<double, std::milli>(t_resize_end - t_start).count();
    double edge_ms = std::chrono::duration<double, std::milli>(t_edge_end - t_edge_start).count();
    double segment_ms = std::chrono::duration<double, std::milli>(t_segment_end - t_segment_start).count();
    double ecc_ms = std::chrono::duration<double, std::milli>(t_ecc_end - t_ecc_start).count();
    double resize_ms = std::chrono::duration<double, std::milli>(t_resize_end - t_resize_start).count();

    capture_debug_data(debug, frame, gray, edges, labels, eccentricity_map, result, total_ms, edge_ms, segment_ms,
                       ecc_ms, resize_ms, static_cast<int>(valid_segments.size()));
  }

  return core::FeatureMap("eccentricity", result, 1.0f);
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

  // Collect valid labels first, then reassign boundaries
  std::set<int> valid_labels;
  for (int y = 0; y < markers.rows; ++y)
  {
    for (int x = 0; x < markers.cols; ++x)
    {
      int label = markers.at<int>(y, x);
      if (label > 0)
      {
        valid_labels.insert(label);
      }
    }
  }

  // If no valid labels, return empty markers (fallback)
  if (valid_labels.empty())
  {
    return cv::Mat::zeros(markers.size(), CV_32S);
  }

  // Use first valid label as fallback
  int fallback_label = *valid_labels.begin();

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
          markers.at<int>(y, x) = fallback_label; // Use valid label as fallback
        }
      }
    }
  }

  return markers;
}

float EccentricityFeature::compute_eccentricity(const cv::Moments& m) const
{
  // Guard against zero or near-zero area
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

void EccentricityFeature::capture_debug_data(DebugContext& debug, const core::Frame& frame, const cv::Mat& gray,
                                             const cv::Mat& edges, const cv::Mat& labels,
                                             const cv::Mat& eccentricity_map, const cv::Mat& result, double total_ms,
                                             double edge_computation_ms, double segmentation_ms,
                                             double eccentricity_computation_ms, double resize_ms,
                                             int num_segments) const
{
  // Annotations
  debug.add_annotation("num_segments", std::to_string(num_segments));
  debug.add_annotation("compute_scale", std::to_string(config_.compute_at_scale));
  debug.add_annotation("edge_threshold", std::to_string(config_.edge_threshold));
  debug.add_annotation("output_size", std::to_string(result.cols) + "x" + std::to_string(result.rows));

  // Timings
  debug.add_timing("edge_computation", edge_computation_ms);
  debug.add_timing("segmentation", segmentation_ms);
  debug.add_timing("eccentricity_computation", eccentricity_computation_ms);
  debug.add_timing("resize_and_normalize", resize_ms);
  debug.add_timing("total_time", total_ms);

  // Basic level: Key intermediate results
  if (debug.is_level(DebugContext::Level::Basic))
  {
    debug.add_image("edges", edges);
    debug.add_image("eccentricity_map_before_resize", eccentricity_map);
  }

  // Detailed level: Add segmentation labels visualization
  if (debug.is_level(DebugContext::Level::Detailed))
  {
    // Visualize segments by converting labels to colors
    cv::Mat labels_viz;
    cv::normalize(labels, labels_viz, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(labels_viz, labels_viz, cv::COLORMAP_JET);
    debug.add_image("segment_labels", labels_viz);

    // Also save the raw grayscale used for computation
    debug.add_image("input_grayscale", gray);
  }
}

} // namespace features
} // namespace attention
