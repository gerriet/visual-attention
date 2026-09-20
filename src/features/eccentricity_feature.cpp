#include "attention/features/eccentricity_feature.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>

namespace attention
{
namespace features
{

namespace
{

using Segment = EccentricityFeature::Segment;

constexpr int kOffsetX[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
constexpr int kOffsetY[8] = {0, 0, -1, 1, -1, 1, -1, 1};
// The original caps the growth threshold: a Sobel response this strong is an
// edge however uniform the rest of the image is.
constexpr int kMaxGrowthThreshold = 250;

void add_pixel(Segment& segment, int x, int y, int gray)
{
  ++segment.pixels;
  segment.sum_gray += gray;
  segment.sum_gray_sq += static_cast<double>(gray) * gray;
  segment.sum_x += x;
  segment.sum_y += y;
  segment.sum_xx += static_cast<double>(x) * x;
  segment.sum_yy += static_cast<double>(y) * y;
  segment.sum_xy += static_cast<double>(x) * y;
}

void absorb(Segment& into, Segment& from)
{
  into.pixels += from.pixels;
  into.sum_gray += from.sum_gray;
  into.sum_gray_sq += from.sum_gray_sq;
  into.sum_x += from.sum_x;
  into.sum_y += from.sum_y;
  into.sum_xx += from.sum_xx;
  into.sum_yy += from.sum_yy;
  into.sum_xy += from.sum_xy;
  from = Segment{};
}

// Segments merge during the dilation passes; the label image keeps the labels
// it was painted with and is resolved through this forest.
class LabelForest
{
 public:
  int add()
  {
    parent_.push_back(static_cast<int>(parent_.size()));
    return parent_.back();
  }
  int find(int label)
  {
    while (parent_[label] != label)
    {
      parent_[label] = parent_[parent_[label]];
      label = parent_[label];
    }
    return label;
  }
  void join(int child, int root) { parent_[child] = root; }

 private:
  std::vector<int> parent_;
};

} // namespace

EccentricityFeature::EccentricityFeature(const Config& config) : config_(config)
{
  if (config_.connectivity != 4 && config_.connectivity != 8)
  {
    throw std::invalid_argument("EccentricityFeature: connectivity must be 4 or 8");
  }
  if (config_.edge_threshold <= 0.0f || config_.edge_threshold >= 1.0f)
  {
    throw std::invalid_argument("EccentricityFeature: edge_threshold is a share of pixels, in (0, 1)");
  }
  if (config_.saliency_offset < 0.0f || config_.saliency_offset >= 1.0f)
  {
    throw std::invalid_argument("EccentricityFeature: saliency_offset must be in [0, 1)");
  }
}

float EccentricityFeature::eccentricity(double mu20, double mu02, double mu11)
{
  const double sum = mu20 + mu02;
  if (sum <= 0.0)
  {
    return 0.0f; // a single pixel (or an empty segment) has no shape
  }
  const double diff = mu20 - mu02;
  return static_cast<float>((diff * diff + 4.0 * mu11 * mu11) / (sum * sum));
}

EccentricityFeature::Result EccentricityFeature::evaluate(const cv::Mat& gray_8u) const
{
  if (gray_8u.type() != CV_8UC1 || gray_8u.empty())
  {
    throw std::runtime_error("EccentricityFeature::evaluate: needs a non-empty 8-bit grey image");
  }
  const int w = gray_8u.cols;
  const int h = gray_8u.rows;

  cv::Mat gray = gray_8u;
  if (config_.equalize)
  {
    cv::equalizeHist(gray_8u, gray);
  }

  Result result;

  // --- 1. Homogeneity: max(|gx|, |gy|), zero on the one-pixel image border
  cv::Mat grad_x, grad_y;
  cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
  const cv::Mat abs_x = cv::abs(grad_x);
  const cv::Mat abs_y = cv::abs(grad_y);
  cv::max(abs_x, abs_y, result.sobel);
  result.sobel.row(0).setTo(0.0f);
  result.sobel.row(h - 1).setTo(0.0f);
  result.sobel.col(0).setTo(0.0f);
  result.sobel.col(w - 1).setTo(0.0f);

  // Growth threshold from the histogram: the smallest value below which the
  // configured share of all pixels lies.
  std::array<int, 256> histogram{};
  for (int y = 0; y < h; ++y)
  {
    const float* row = result.sobel.ptr<float>(y);
    for (int x = 0; x < w; ++x)
    {
      ++histogram[std::min(255, static_cast<int>(row[x]))];
    }
  }
  const long target = static_cast<long>(config_.edge_threshold * w * h);
  int threshold = 0;
  for (long sum = 0; sum < target && threshold < kMaxGrowthThreshold; ++threshold)
  {
    sum += histogram[threshold];
  }
  result.growth_threshold = threshold;
  auto is_area = [&](int x, int y) { return result.sobel.at<float>(y, x) < threshold; };

  // --- 2. Region growing. Label 0 = boundary / unassigned.
  result.labels = cv::Mat::zeros(h, w, CV_32S);
  std::vector<Segment>& segments = result.segments;
  LabelForest forest;
  segments.emplace_back(); // label 0
  forest.add();

  cv::Mat visited = cv::Mat::zeros(h, w, CV_8U);
  std::queue<std::pair<int, int>> frontier;
  for (int sy = 0; sy < h; ++sy)
  {
    for (int sx = 0; sx < w; ++sx)
    {
      if (visited.at<uchar>(sy, sx) || !is_area(sx, sy))
      {
        continue;
      }
      const int label = forest.add();
      segments.emplace_back();
      visited.at<uchar>(sy, sx) = 1;
      frontier.emplace(sx, sy);
      while (!frontier.empty())
      {
        const auto [cx, cy] = frontier.front();
        frontier.pop();
        if (!is_area(cx, cy))
        {
          continue; // reached, but a boundary pixel: growth stops here
        }
        result.labels.at<int>(cy, cx) = label;
        add_pixel(segments[label], cx, cy, gray.at<uchar>(cy, cx));
        for (int i = 0; i < config_.connectivity; ++i)
        {
          const int nx = cx + kOffsetX[i];
          const int ny = cy + kOffsetY[i];
          if (nx >= 0 && nx < w && ny >= 0 && ny < h && !visited.at<uchar>(ny, nx))
          {
            visited.at<uchar>(ny, nx) = 1;
            frontier.emplace(nx, ny);
          }
        }
      }
    }
  }
  // A boundary pixel reached during a growth stays `visited` but unlabelled;
  // that is intended — it is what the dilation below works on.
  result.initial_segments = static_cast<int>(segments.size()) - 1;

  // --- 3. Merging + dilation
  auto mergeable = [&](const Segment& a, const Segment& b)
  {
    if (a.pixels == 0 || b.pixels == 0)
    {
      return false;
    }
    if (std::abs(a.mean_gray() - b.mean_gray()) > config_.merge_mean_difference)
    {
      return false;
    }
    // +1: two flat segments (variance 0) are alike, not 0/0 — see the header
    const double high = std::max(a.variance(), b.variance()) + 1.0;
    const double low = std::min(a.variance(), b.variance()) + 1.0;
    return high / low < config_.variance_threshold;
  };

  for (int pass = 0; pass < config_.merge_iterations; ++pass)
  {
    // Dilation is applied after the pass (every pixel sees the same state);
    // merges take effect at once, as in the original.
    std::vector<std::pair<cv::Point, int>> dilated;
    for (int y = 1; y < h - 1; ++y)
    {
      for (int x = 1; x < w - 1; ++x)
      {
        if (result.labels.at<int>(y, x) != 0)
        {
          continue;
        }
        // Distinct neighbouring segments and how many neighbours each holds
        std::array<std::pair<int, int>, 8> around{};
        int distinct = 0;
        for (int i = 0; i < config_.connectivity; ++i)
        {
          const int raw = result.labels.at<int>(y + kOffsetY[i], x + kOffsetX[i]);
          if (raw == 0)
          {
            continue;
          }
          const int label = forest.find(raw);
          auto end = around.begin() + distinct;
          auto hit = std::find_if(around.begin(), end, [label](const auto& e) { return e.first == label; });
          if (hit == end)
          {
            around[distinct++] = {label, 1};
          }
          else
          {
            ++hit->second;
          }
        }

        if (distinct == 2)
        {
          int into = around[0].first;
          int from = around[1].first;
          if (mergeable(segments[into], segments[from]))
          {
            if (segments[into].pixels < segments[from].pixels)
            {
              std::swap(into, from);
            }
            absorb(segments[into], segments[from]);
            forest.join(from, into);
            result.labels.at<int>(y, x) = into;
            add_pixel(segments[into], x, y, gray.at<uchar>(y, x));
          }
          // Two unlike segments: the pixel stays a boundary between them.
        }
        else if (distinct > 0)
        {
          const auto dominant = std::max_element(around.begin(), around.begin() + distinct,
                                                 [](const auto& a, const auto& b) { return a.second < b.second; });
          dilated.emplace_back(cv::Point(x, y), dominant->first);
        }
      }
    }
    for (const auto& [point, label] : dilated)
    {
      const int root = forest.find(label); // it may have been merged later in the pass
      result.labels.at<int>(point) = root;
      add_pixel(segments[root], point.x, point.y, gray.at<uchar>(point));
    }
  }

  // Resolve merged labels
  for (int y = 0; y < h; ++y)
  {
    int* row = result.labels.ptr<int>(y);
    for (int x = 0; x < w; ++x)
    {
      row[x] = forest.find(row[x]);
    }
  }

  // --- 4./5. Size filter, moments, orientation classes, saliency
  const int min_pixels = static_cast<int>(config_.min_area / 100.0f * w * h);
  const int max_pixels = static_cast<int>(config_.max_area / 100.0f * w * h);
  std::array<int, kOrientationClasses + 1> per_class{};
  for (std::size_t label = 1; label < segments.size(); ++label)
  {
    Segment& segment = segments[label];
    if (segment.pixels == 0)
    {
      continue; // merged away
    }
    if (segment.pixels < min_pixels || segment.pixels > max_pixels)
    {
      segment = Segment{}; // too small to matter for attention, or background-sized
      continue;
    }
    const double n = segment.pixels;
    const double mu20 = segment.sum_xx - segment.sum_x * segment.sum_x / n;
    const double mu02 = segment.sum_yy - segment.sum_y * segment.sum_y / n;
    const double mu11 = segment.sum_xy - segment.sum_x * segment.sum_y / n;
    segment.eccentricity = eccentricity(mu20, mu02, mu11);
    if (segment.eccentricity >= config_.min_oriented)
    {
      double angle = 0.5 * std::atan2(2.0 * mu11, mu20 - mu02); // eq. 5.5
      if (angle < 0.0)
      {
        angle += CV_PI;
      }
      segment.angle = static_cast<float>(angle);
      segment.orientation_class =
          std::min(kOrientationClasses - 1, static_cast<int>(kOrientationClasses * angle / CV_PI));
    }
    ++per_class[segment.orientation_class];
  }

  for (Segment& segment : segments)
  {
    if (segment.pixels == 0)
    {
      continue;
    }
    const float above = std::max(0.0f, segment.eccentricity - config_.saliency_offset);
    segment.saliency =
        above / (1.0f - config_.saliency_offset) / config_.exclusivity.divisor(per_class[segment.orientation_class]);
  }

  result.saliency = cv::Mat::zeros(h, w, CV_32F);
  for (int y = 0; y < h; ++y)
  {
    int* label = result.labels.ptr<int>(y);
    float* out = result.saliency.ptr<float>(y);
    for (int x = 0; x < w; ++x)
    {
      if (segments[label[x]].pixels == 0)
      {
        label[x] = 0; // dropped by the size filter
      }
      out[x] = segments[label[x]].saliency;
    }
  }
  return result;
}

core::FeatureMap EccentricityFeature::extract(const core::Frame& frame) const
{
  DebugContext no_debug;
  return extract(frame, no_debug);
}

core::FeatureMap EccentricityFeature::extract(const core::Frame& frame, DebugContext& debug) const
{
  const auto start = std::chrono::high_resolution_clock::now();

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
  const int scale_index = std::min(requested_scale, static_cast<int>(frame.gray_pyramid.size()) - 1);

  // The pyramid holds float grey values in [0, 1]; the algorithm's thresholds
  // (Sobel histogram, max_mu = 20) are defined on 0..255.
  cv::Mat gray_8u;
  frame.gray_pyramid[scale_index].convertTo(gray_8u, CV_8U, 255.0);

  const Result result = evaluate(gray_8u);

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
    const int kept = static_cast<int>(
        std::count_if(result.segments.begin(), result.segments.end(), [](const Segment& s) { return s.pixels > 0; }));
    debug.add_annotation("initial_segments", std::to_string(result.initial_segments));
    debug.add_annotation("num_segments", std::to_string(kept));
    debug.add_annotation("growth_threshold", std::to_string(result.growth_threshold));
    debug.add_annotation("compute_scale", std::to_string(scale_index));
    debug.add_timing("total_time", std::chrono::duration<double, std::milli>(end - start).count());
    if (debug.is_level(DebugContext::Level::Basic))
    {
      cv::Mat sobel_viz;
      cv::normalize(result.sobel, sobel_viz, 0.0f, 1.0f, cv::NORM_MINMAX);
      debug.add_image("edges", sobel_viz);
      debug.add_image("eccentricity_map_before_resize", result.saliency);
    }
    if (debug.is_level(DebugContext::Level::Detailed))
    {
      cv::Mat labels_viz;
      cv::normalize(result.labels, labels_viz, 0, 255, cv::NORM_MINMAX, CV_8U);
      cv::applyColorMap(labels_viz, labels_viz, cv::COLORMAP_JET);
      debug.add_image("segment_labels", labels_viz);
    }
  }

  return core::FeatureMap("eccentricity", saliency, 1.0f);
}

} // namespace features
} // namespace attention
