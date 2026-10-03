#include "attention/fusion/top_down_weights.h"
#include <cmath>
#include <stdexcept>

namespace attention
{
namespace fusion
{

namespace
{

// Scale a map to [0, 1]. A constant map becomes all-zero rather than all-one:
// a channel that says the same thing everywhere carries no information, and
// normalizing it to 1 would let it dominate the sum it is mixed into.
cv::Mat unit_normalize(const cv::Mat& map)
{
  double lo = 0.0;
  double hi = 0.0;
  cv::minMaxLoc(map, &lo, &hi);
  cv::Mat out;
  if (hi - lo > 1e-12)
  {
    map.convertTo(out, CV_32F, 1.0 / (hi - lo), -lo / (hi - lo));
  }
  else
  {
    out = cv::Mat::zeros(map.size(), CV_32F);
  }
  return out;
}

} // namespace

TopDownWeights::TopDownWeights(const TopDownWeightConfig& config) : config_(config) {}

cv::Mat TopDownWeights::top_down_map(const cv::Mat& bottom_up, const std::vector<core::FeatureMap>& features) const
{
  cv::Mat excitation = cv::Mat::zeros(bottom_up.size(), CV_32F);
  cv::Mat inhibition = cv::Mat::zeros(bottom_up.size(), CV_32F);
  bool any = false;

  for (const auto& feature : features)
  {
    const auto it = config_.weights.find(feature.name);
    if (it == config_.weights.end() || feature.data.empty())
    {
      continue;
    }
    const float w = it->second;
    // w == 1 means the channel does not separate target from background; it
    // belongs to neither sum. The guard band keeps a weight of 1.0001 from
    // entering the excitation with a near-zero contribution and a 1/w twin.
    if (std::abs(w - 1.0f) < 1e-3f || w <= 0.0f)
    {
      continue;
    }

    cv::Mat data = feature.data;
    if (data.size() != bottom_up.size())
    {
      cv::resize(data, data, bottom_up.size(), 0, 0, cv::INTER_LINEAR);
    }
    if (data.type() != CV_32F)
    {
      data.convertTo(data, CV_32F);
    }

    if (w > 1.0f)
    {
      excitation += w * data;
    }
    else
    {
      inhibition += (1.0f / w) * data;
    }
    any = true;
  }

  if (!any)
  {
    return cv::Mat();
  }
  return unit_normalize(excitation - inhibition);
}

cv::Mat TopDownWeights::apply(const cv::Mat& bottom_up, const std::vector<core::FeatureMap>& features) const
{
  if (!config_.active() || bottom_up.empty())
  {
    return bottom_up;
  }
  const cv::Mat top_down = top_down_map(bottom_up, features);
  if (top_down.empty())
  {
    return bottom_up;
  }
  const float t = std::min(1.0f, std::max(0.0f, config_.factor));
  cv::Mat mixed = (1.0f - t) * bottom_up + t * top_down;
  return unit_normalize(mixed);
}

std::map<std::string, float> learn_top_down_weights(const std::vector<core::FeatureMap>& features,
                                                    const cv::Mat& target_mask, float max_weight)
{
  if (target_mask.empty())
  {
    throw std::runtime_error("learn_top_down_weights: empty target mask");
  }
  const int inside_count = cv::countNonZero(target_mask);
  const int total = target_mask.rows * target_mask.cols;
  if (inside_count == 0 || inside_count == total)
  {
    throw std::runtime_error(
        "learn_top_down_weights: the mask must contain both target and background "
        "(got " +
        std::to_string(inside_count) + " of " + std::to_string(total) + " pixels)");
  }

  cv::Mat outside_mask;
  cv::bitwise_not(target_mask, outside_mask);

  std::map<std::string, float> weights;
  for (const auto& feature : features)
  {
    if (feature.data.empty())
    {
      continue;
    }
    cv::Mat data = feature.data;
    if (data.size() != target_mask.size())
    {
      cv::resize(data, data, target_mask.size(), 0, 0, cv::INTER_LINEAR);
    }
    const double inside = cv::mean(data, target_mask)[0];
    const double outside = cv::mean(data, outside_mask)[0];

    // A channel that is silent on the target separates nothing in the
    // direction the rule cares about; 1/max_weight is the symmetric partner of
    // the max_weight clamp below, so excitation and inhibition are bounded
    // alike.
    float w;
    if (outside <= 1e-9)
    {
      w = inside <= 1e-9 ? 1.0f : max_weight;
    }
    else
    {
      w = static_cast<float>(inside / outside);
    }
    w = std::min(max_weight, std::max(1.0f / max_weight, w));
    weights[feature.name] = w;
  }
  return weights;
}

std::map<std::string, float> combine_top_down_weights(const std::vector<std::map<std::string, float>>& examples)
{
  std::map<std::string, double> log_sum;
  std::map<std::string, int> counts;
  for (const auto& example : examples)
  {
    for (const auto& entry : example)
    {
      if (entry.second <= 0.0f)
      {
        continue;
      }
      log_sum[entry.first] += std::log(static_cast<double>(entry.second));
      counts[entry.first] += 1;
    }
  }

  std::map<std::string, float> combined;
  for (const auto& entry : log_sum)
  {
    const int n = counts[entry.first];
    if (n > 0)
    {
      combined[entry.first] = static_cast<float>(std::exp(entry.second / n));
    }
  }
  return combined;
}

cv::Mat salient_region_in_box(const cv::Mat& saliency, const cv::Rect& box, float rel_threshold)
{
  cv::Mat mask = cv::Mat::zeros(saliency.size(), CV_8U);
  const cv::Rect clipped = box & cv::Rect(0, 0, saliency.cols, saliency.rows);
  if (clipped.width <= 0 || clipped.height <= 0)
  {
    return mask;
  }

  const cv::Mat roi = saliency(clipped);
  double lo = 0.0;
  double hi = 0.0;
  cv::Point peak;
  cv::minMaxLoc(roi, &lo, &hi, nullptr, &peak);

  // A flat box has no "most salient region" to find: fall back to the whole
  // box rather than to an empty mask, so the caller degrades to the naive
  // rectangle instead of losing the example.
  if (hi - lo <= 1e-9)
  {
    mask(clipped).setTo(255);
    return mask;
  }

  cv::Mat above;
  cv::threshold(roi, above, lo + rel_threshold * (hi - lo), 255, cv::THRESH_BINARY);
  above.convertTo(above, CV_8U);

  // Keep only the component the peak belongs to: a box can contain two bright
  // things, and the target is the one at the maximum.
  cv::Mat labels;
  const int components = cv::connectedComponents(above, labels, 8, CV_32S);
  if (components > 1)
  {
    const int peak_label = labels.at<int>(peak);
    if (peak_label != 0)
    {
      above = (labels == peak_label);
    }
  }
  above.copyTo(mask(clipped));
  return mask;
}

} // namespace fusion
} // namespace attention
