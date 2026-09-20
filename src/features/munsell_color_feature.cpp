#include "attention/features/munsell_color_feature.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>

namespace attention
{
namespace features
{

namespace
{

constexpr int kHueClasses = 12;
constexpr double kRadToDeg = 57.29578;

double mtm_distance(const double a[3], const double b[3])
{
  const double d0 = a[0] - b[0];
  const double d1 = a[1] - b[1];
  const double d2 = a[2] - b[2];
  return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
}

// Standard deviation of the MTM colour in the 3x3 window around each pixel
// (mean squared Euclidean distance to the window mean), counting only the
// pixels inside the image — the "variable threshold after Yokoya" of the
// original win_sigma(), computed for the whole image with box sums.
cv::Mat local_colour_sigma(const cv::Mat& mtm)
{
  const cv::Size window(3, 3);
  cv::Mat count;
  cv::boxFilter(cv::Mat::ones(mtm.size(), CV_32F), count, CV_32F, window, cv::Point(-1, -1), false,
                cv::BORDER_CONSTANT);

  std::vector<cv::Mat> bands;
  cv::split(mtm, bands);
  cv::Mat variance = cv::Mat::zeros(mtm.size(), CV_32F);
  for (const cv::Mat& band : bands)
  {
    cv::Mat sum, sum_sq;
    cv::boxFilter(band, sum, CV_32F, window, cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
    cv::boxFilter(band.mul(band), sum_sq, CV_32F, window, cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
    cv::Mat mean = sum / count;
    variance += sum_sq / count - mean.mul(mean);
  }
  cv::max(variance, 0.0f, variance); // rounding can leave a flat window slightly negative
  cv::Mat sigma;
  cv::sqrt(variance, sigma);
  return sigma;
}

// Hue class of a segment mean: 12 classes of 30°, class 0 centred on 0°
// (345°..15°), exactly as the original bins them.
int hue_class_of(double s1, double s2)
{
  int hue = static_cast<int>(kRadToDeg * std::atan2(s2, s1) + 0.5);
  if (hue < 0)
  {
    hue += 360;
  }
  return (hue >= 15 && hue < 345) ? (hue + 15) / 30 : 0;
}

} // namespace

MunsellColorFeature::MunsellColorFeature(const Config& config) : config_(config)
{
  if (config_.max_contrast <= 0.0f)
  {
    throw std::invalid_argument("MunsellColorFeature: max_contrast must be positive");
  }
}

cv::Mat MunsellColorFeature::bgr_to_mtm(const cv::Mat& bgr)
{
  if (bgr.channels() != 3)
  {
    throw std::runtime_error("MunsellColorFeature: needs a 3-channel image");
  }
  // The original works on 0..255 values; the white-point divisors below
  // (250.410, 255, 301.665 = the matrix rows applied to white) presuppose it.
  cv::Mat source;
  if (bgr.depth() == CV_8U)
  {
    bgr.convertTo(source, CV_32F);
  }
  else
  {
    bgr.convertTo(source, CV_32F, 255.0);
  }

  cv::Mat mtm(source.size(), CV_32FC3);
  for (int y = 0; y < source.rows; ++y)
  {
    const cv::Vec3f* in = source.ptr<cv::Vec3f>(y);
    cv::Vec3f* out = mtm.ptr<cv::Vec3f>(y);
    for (int x = 0; x < source.cols; ++x)
    {
      const float b = in[x][0], g = in[x][1], r = in[x][2];
      // FCC RGB -> CIE XYZ (eq. 5.7), scaled to a 0..100 tristimulus
      const float X = std::cbrt(100.0f * (0.607f * r + 0.174f * g + 0.201f * b) / 250.410f);
      const float Y = std::cbrt(100.0f * (0.299f * r + 0.587f * g + 0.114f * b) / 255.000f);
      const float Z = std::cbrt(100.0f * (0.066f * g + 1.117f * b) / 301.665f);
      // Adams chromatic value space (eq. 5.8); V(x) = 11.6 x^(1/3) - 1.6
      const float m1 = 11.6f * (X - Y);
      const float m2 = 0.4f * 11.6f * (Z - Y);
      const float lightness = 11.6f * Y - 1.6f;
      // MTM (eq. 5.9): hue-dependent stretch of the two chromatic axes
      const float phi = std::atan2(m2, m1);
      out[x][0] = lightness;
      out[x][1] = (8.880f + 0.966f * std::cos(phi)) * m1;
      out[x][2] = (8.025f + 2.558f * std::sin(phi)) * m2;
    }
  }
  return mtm;
}

MunsellColorFeature::Result MunsellColorFeature::evaluate(const cv::Mat& mtm) const
{
  if (mtm.type() != CV_32FC3 || mtm.empty())
  {
    throw std::runtime_error("MunsellColorFeature::evaluate: needs a non-empty CV_32FC3 MTM image");
  }
  const int w = mtm.cols;
  const int h = mtm.rows;
  const cv::Mat sigma = local_colour_sigma(mtm);

  Result result;
  result.labels = cv::Mat(h, w, CV_32S, cv::Scalar(-1));
  std::vector<Segment>& segments = result.segments;
  // Shared border lengths, per segment: neighbour index -> pixel adjacencies
  std::vector<std::map<int, int>> borders;

  auto make_neighbours = [&borders](int a, int b)
  {
    if (a < 0 || b < 0 || a == b)
    {
      return;
    }
    ++borders[a][b];
    ++borders[b][a];
  };

  // --- Region growing: one raster pass, direction alternating per row so the
  // segmentation depends less on the processing order.
  for (int row = 0; row < h; ++row)
  {
    const int direction = (row % 2 == 0) ? 1 : -1;
    for (int step = 0; step < w; ++step)
    {
      const int col = direction == 1 ? step : w - 1 - step;
      const cv::Vec3f& pixel = mtm.at<cv::Vec3f>(row, col);
      const double value[3] = {pixel[0], pixel[1], pixel[2]};

      // Already-visited neighbours: previous in scan order, above, and the two
      // diagonals above. Only the first two count as *borders* (4-connected).
      const int previous_col = col - direction;
      std::array<int, 4> neighbour = {-1, -1, -1, -1};
      if (previous_col >= 0 && previous_col < w)
      {
        neighbour[0] = result.labels.at<int>(row, previous_col);
      }
      if (row > 0)
      {
        neighbour[1] = result.labels.at<int>(row - 1, col);
        if (col > 0)
        {
          neighbour[2] = result.labels.at<int>(row - 1, col - 1);
        }
        if (col < w - 1)
        {
          neighbour[3] = result.labels.at<int>(row - 1, col + 1);
        }
      }

      const double limit = config_.threshold + config_.threshold_sigma * sigma.at<float>(row, col);
      int best = -1;
      double best_distance = 0.0;
      for (int j = 0; j < 4; ++j)
      {
        const int label = neighbour[j];
        if (label < 0 || std::find(neighbour.begin(), neighbour.begin() + j, label) != neighbour.begin() + j)
        {
          continue; // absent, or the same segment seen through another neighbour
        }
        const double distance = mtm_distance(value, segments[label].mean);
        if (distance < limit && (best < 0 || distance < best_distance))
        {
          best = label;
          best_distance = distance;
        }
      }

      if (best < 0)
      {
        best = static_cast<int>(segments.size());
        segments.emplace_back();
        borders.emplace_back();
      }
      Segment& segment = segments[best];
      const double n = segment.pixels;
      for (int k = 0; k < 3; ++k)
      {
        segment.mean[k] = (n * segment.mean[k] + value[k]) / (n + 1.0); // centroid linkage
      }
      ++segment.pixels;
      result.labels.at<int>(row, col) = best;
      make_neighbours(best, neighbour[0]);
      make_neighbours(best, neighbour[1]);
    }
  }

  // --- Border-weighted mean contrast to the neighbours (eq. 5.11), using the
  // final segment means.
  const int count = static_cast<int>(segments.size());
  for (int i = 0; i < count; ++i)
  {
    double weighted = 0.0;
    for (const auto& border : borders[i])
    {
      segments[i].perimeter += border.second;
      weighted += mtm_distance(segments[i].mean, segments[border.first].mean) * border.second;
    }
    segments[i].contrast = segments[i].perimeter > 0 ? weighted / segments[i].perimeter : 0.0;
    segments[i].hue_class = hue_class_of(segments[i].mean[1], segments[i].mean[2]);
  }

  // --- Exclusivity (§5.5.3): how many conspicuous segments share each hue class
  std::array<int, kHueClasses> quantity{};
  for (const Segment& segment : segments)
  {
    if (segment.contrast >= config_.attribute_threshold)
    {
      ++quantity[segment.hue_class];
    }
  }

  const int min_pixels = static_cast<int>(config_.min_segment * w * h);
  const int max_pixels = static_cast<int>(config_.max_segment * w * h);
  for (Segment& segment : segments)
  {
    if (segment.pixels < min_pixels || segment.pixels > max_pixels)
    {
      continue; // too small to matter for attention, or background-sized
    }
    const double chroma = std::hypot(segment.mean[1], segment.mean[2]);
    // An achromatic segment has no meaningful hue, hence no hue class to share
    const int sharing = chroma >= config_.attribute_threshold ? quantity[segment.hue_class] : 0;
    const double contrast = segment.contrast / config_.exclusivity.divisor(sharing);
    // Sigmoid (eq. 5.12): high contrasts are rare — suppress small, stress large
    segment.saliency = static_cast<float>(
        1.0 / (1.0 + std::exp(-config_.sigmoid_beta * (2.0 * contrast / config_.max_contrast - 1.0))));
  }

  result.saliency = cv::Mat(h, w, CV_32F);
  for (int y = 0; y < h; ++y)
  {
    const int* label = result.labels.ptr<int>(y);
    float* out = result.saliency.ptr<float>(y);
    for (int x = 0; x < w; ++x)
    {
      out[x] = segments[label[x]].saliency;
    }
  }
  return result;
}

core::FeatureMap MunsellColorFeature::extract(const core::Frame& frame) const
{
  DebugContext no_debug;
  return extract(frame, no_debug);
}

core::FeatureMap MunsellColorFeature::extract(const core::Frame& frame, DebugContext& debug) const
{
  const auto start = std::chrono::high_resolution_clock::now();
  if (frame.empty())
  {
    throw std::runtime_error("MunsellColorFeature: Cannot extract from empty frame");
  }
  if (frame.channels() != 3)
  {
    throw std::runtime_error("MunsellColorFeature: needs a colour image");
  }

  cv::Mat working = frame.image;
  const int long_side = std::max(working.cols, working.rows);
  if (config_.max_working_size > 0 && long_side > config_.max_working_size)
  {
    const double scale = static_cast<double>(config_.max_working_size) / long_side;
    cv::resize(frame.image, working, cv::Size(), scale, scale, cv::INTER_AREA);
  }

  const cv::Mat mtm = bgr_to_mtm(working);
  const Result result = evaluate(mtm);

  cv::Mat saliency;
  if (result.saliency.size() != frame.size())
  {
    cv::resize(result.saliency, saliency, frame.size(), 0, 0, cv::INTER_LINEAR);
  }
  else
  {
    saliency = result.saliency;
  }

  if (debug.enabled)
  {
    const auto end = std::chrono::high_resolution_clock::now();
    debug.add_timing("total_time", std::chrono::duration<double, std::milli>(end - start).count());
    debug.add_annotation("num_segments", std::to_string(result.segments.size()));
    debug.add_annotation("working_size", std::to_string(working.cols) + "x" + std::to_string(working.rows));
    if (debug.is_level(DebugContext::Level::Basic))
    {
      cv::Mat labels_viz;
      cv::normalize(result.labels, labels_viz, 0, 255, cv::NORM_MINMAX, CV_8U);
      cv::applyColorMap(labels_viz, labels_viz, cv::COLORMAP_JET);
      debug.add_image("segment_labels", labels_viz);
      debug.add_image("saliency_before_resize", result.saliency);
    }
  }

  return core::FeatureMap(name(), saliency, 1.0f);
}

} // namespace features
} // namespace attention
