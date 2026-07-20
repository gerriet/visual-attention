#include "attention/fusion/priority_map.h"
#include <algorithm>
#include <map>
#include <stdexcept>

namespace attention
{
namespace fusion
{

namespace
{
// Min-max normalize to [0, 1]; a flat map stays flat (zeros).
cv::Mat normalized(const cv::Mat& map)
{
  double min_val = 0.0, max_val = 0.0;
  cv::minMaxLoc(map, &min_val, &max_val);
  cv::Mat out;
  if (max_val - min_val > 1e-9)
  {
    map.convertTo(out, CV_32F, 1.0 / (max_val - min_val), -min_val / (max_val - min_val));
  }
  else
  {
    out = cv::Mat::zeros(map.size(), CV_32F);
  }
  return out;
}

void splat_gaussian(cv::Mat& map, const cv::Point& at, float radius, float strength)
{
  const int r = static_cast<int>(radius * 3.0f);
  const cv::Rect patch_rect = cv::Rect(at.x - r, at.y - r, 2 * r + 1, 2 * r + 1) & cv::Rect(0, 0, map.cols, map.rows);
  if (patch_rect.area() <= 0)
  {
    return;
  }
  for (int y = patch_rect.y; y < patch_rect.y + patch_rect.height; ++y)
  {
    float* row = map.ptr<float>(y);
    const float dy = static_cast<float>(y - at.y);
    for (int x = patch_rect.x; x < patch_rect.x + patch_rect.width; ++x)
    {
      const float dx = static_cast<float>(x - at.x);
      row[x] += strength * std::exp(-(dx * dx + dy * dy) / (2.0f * radius * radius));
    }
  }
}
} // namespace

cv::Vec3b parse_colour(const std::string& spec)
{
  static const std::map<std::string, cv::Vec3b> named = {
      {"red", {0, 0, 220}},       {"green", {0, 200, 0}},    {"blue", {220, 60, 0}},   {"yellow", {0, 210, 230}},
      {"orange", {0, 140, 255}},  {"purple", {200, 0, 160}}, {"cyan", {200, 200, 0}},  {"magenta", {200, 0, 200}},
      {"white", {245, 245, 245}}, {"black", {10, 10, 10}},   {"gray", {128, 128, 128}}};

  auto it = named.find(spec);
  if (it != named.end())
  {
    return it->second;
  }
  if (spec.size() == 7 && spec[0] == '#')
  {
    try
    {
      const int r = std::stoi(spec.substr(1, 2), nullptr, 16);
      const int g = std::stoi(spec.substr(3, 2), nullptr, 16);
      const int b = std::stoi(spec.substr(5, 2), nullptr, 16);
      return cv::Vec3b(static_cast<uchar>(b), static_cast<uchar>(g), static_cast<uchar>(r));
    }
    catch (const std::exception&)
    {
      // Fall through to the friendly error rather than surfacing a raw stoi throw.
    }
  }
  throw std::runtime_error("Unparseable colour '" + spec + "' (use #rrggbb or a named colour)");
}

TopDownChannel::TopDownChannel(const PriorityConfig& config) : config_(config)
{
  if (!config_.top_down_active())
  {
    return; // channel off (weight 0): don't touch a leftover map path or colour
  }
  if (!config_.top_down_map_path.empty())
  {
    const cv::Mat loaded = cv::imread(config_.top_down_map_path, cv::IMREAD_GRAYSCALE);
    if (loaded.empty())
    {
      throw std::runtime_error("PriorityMap: cannot read top-down map: " + config_.top_down_map_path);
    }
    loaded.convertTo(file_map_, CV_32F, 1.0 / 255.0);
  }
  if (!config_.target_color.empty())
  {
    const cv::Vec3b bgr = parse_colour(config_.target_color);
    cv::Mat px(1, 1, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2]));
    cv::cvtColor(px, px, cv::COLOR_BGR2Lab);
    const cv::Vec3b lab = px.at<cv::Vec3b>(0, 0);
    target_lab_ = cv::Vec3f(lab[0], lab[1], lab[2]);
    have_target_colour_ = true;
  }
}

cv::Mat TopDownChannel::colour_similarity(const cv::Mat& frame_bgr) const
{
  cv::Mat lab;
  cv::cvtColor(frame_bgr, lab, cv::COLOR_BGR2Lab);
  cv::Mat lab_f;
  lab.convertTo(lab_f, CV_32FC3);

  cv::Mat diff;
  cv::subtract(lab_f, cv::Scalar(target_lab_[0], target_lab_[1], target_lab_[2]), diff);
  std::vector<cv::Mat> ch;
  cv::split(diff, ch);
  // Chroma-weighted distance: colour identity lives in a/b; L carries
  // illumination, which should not disqualify a shaded target.
  cv::Mat d2 = 0.25f * ch[0].mul(ch[0]) + ch[1].mul(ch[1]) + ch[2].mul(ch[2]);
  cv::Mat sim;
  cv::exp(d2 * (-1.0 / (2.0 * config_.target_color_sigma * config_.target_color_sigma)), sim);
  return sim;
}

cv::Mat TopDownChannel::relevance(const cv::Size& size, const cv::Mat& frame_bgr) const
{
  cv::Mat channel;
  if (have_target_colour_ && !frame_bgr.empty())
  {
    channel = colour_similarity(frame_bgr);
  }
  if (!file_map_.empty())
  {
    cv::Mat resized;
    cv::resize(file_map_, resized, size, 0, 0, cv::INTER_LINEAR);
    channel = channel.empty() ? resized : cv::Mat(normalized(channel + resized));
  }
  if (!channel.empty() && channel.size() != size)
  {
    cv::resize(channel, channel, size, 0, 0, cv::INTER_LINEAR);
  }
  return channel;
}

cv::Mat TopDownChannel::apply(const cv::Mat& bottom_up, const cv::Mat& frame_bgr) const
{
  if (!config_.top_down_active())
  {
    return bottom_up; // default path: bit-identical to the thesis map
  }
  const cv::Mat channel = relevance(bottom_up.size(), frame_bgr);
  if (channel.empty())
  {
    return bottom_up;
  }
  return normalized(bottom_up + config_.top_down_weight * channel);
}

HistoryChannels::HistoryChannels(const PriorityConfig& config) : config_(config) {}

void HistoryChannels::reset()
{
  location_.release();
}

void HistoryChannels::decay_and_record(const cv::Point& focus, const cv::Size& frame_size, bool has_focus)
{
  if (!config_.history_active())
  {
    return;
  }
  if (location_.empty() || location_.size() != frame_size)
  {
    location_ = cv::Mat::zeros(frame_size, CV_32F);
  }
  location_ *= config_.location_history_decay;
  if (has_focus)
  {
    splat_gaussian(location_, focus, config_.location_history_radius, 1.0f);
  }
}

cv::Mat HistoryChannels::apply(const cv::Mat& saliency, const std::vector<system::ObjectFile>& objects) const
{
  if (!config_.history_active())
  {
    return saliency; // default path: untouched
  }

  cv::Mat priority = saliency.clone();
  if (config_.object_value_weight > 0.0f)
  {
    cv::Mat value_map = cv::Mat::zeros(saliency.size(), CV_32F);
    for (const auto& object : objects)
    {
      if (object.value > 0.0f)
      {
        // Value rides on the object file: the boost appears wherever the
        // object *is now*, sized to its extent.
        const float radius = std::max(8.0f, 0.5f * std::max(object.bbox.width, object.bbox.height));
        splat_gaussian(value_map, object.centroid, radius, object.value);
      }
    }
    priority += config_.object_value_weight * value_map;
  }
  // Size-guard the location term: apply() runs at the top of the frame, before
  // decay_and_record() rebuilds location_ for the current frame — so on a
  // stream whose resolution changes, last frame's map may not match.
  if (config_.location_history_weight > 0.0f && location_.size() == saliency.size())
  {
    priority += config_.location_history_weight * location_;
  }
  return normalized(priority);
}

} // namespace fusion
} // namespace attention
