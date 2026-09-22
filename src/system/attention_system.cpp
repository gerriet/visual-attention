#include "attention/system/attention_system.h"
#include "attention/config/yaml_reader.h"
#include "attention/selection/neural_field_selection.h"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace attention
{
namespace system
{

namespace
{
// A config section may carry only these keys; anything else is a typo, which
// must not pass silently (it would quietly keep the default).
void reject_unknown_keys(const YAML::Node& node, const std::set<std::string>& known, const std::string& where)
{
  if (!node.IsMap())
  {
    throw std::runtime_error(where + ": expected a mapping");
  }
  for (const auto& entry : node)
  {
    const std::string key = entry.first.as<std::string>();
    if (known.count(key) == 0)
    {
      std::string list;
      for (const auto& k : known)
      {
        list += (list.empty() ? "" : ", ") + k;
      }
      throw std::runtime_error(where + ": unknown key '" + key + "' (known: " + list + ")");
    }
  }
}
} // namespace

void AttentionSystem::apply_config_yaml(const std::string& yaml, Config& cfg)
{
  using attention::config::read_param;
  if (yaml.empty())
  {
    return;
  }
  const YAML::Node node = YAML::Load(yaml);
  if (node.IsNull())
  {
    return;
  }
  reject_unknown_keys(
      node,
      {"cluster_source", "camera_compensation", "camera_max_shift", "segment_fraction", "segment_min", "min_cluster_size", "segment_close", "max_cluster_fraction",
       "proto_objects", "proto_min_contrast", "proto_tolerance", "proto_window", "object_files"},
      "attention_system");
  if (node["cluster_source"])
  {
    const std::string source = node["cluster_source"].as<std::string>();
    if (source != "saliency" && source != "field")
    {
      throw std::runtime_error("attention_system.cluster_source: '" + source + "' (known: saliency, field)");
    }
    cfg.cluster_source = source == "field" ? Config::ClusterSource::Field : Config::ClusterSource::Saliency;
  }
  read_param(node, "camera_compensation", cfg.camera_compensation);
  read_param(node, "camera_max_shift", cfg.camera_max_shift);
  read_param(node, "segment_fraction", cfg.segment_fraction);
  read_param(node, "segment_min", cfg.segment_min);
  read_param(node, "min_cluster_size", cfg.min_cluster_size);
  read_param(node, "segment_close", cfg.segment_close);
  read_param(node, "max_cluster_fraction", cfg.max_cluster_fraction);
  read_param(node, "proto_objects", cfg.proto_objects);
  read_param(node, "proto_min_contrast", cfg.proto_min_contrast);
  read_param(node, "proto_tolerance", cfg.proto_tolerance);
  read_param(node, "proto_window", cfg.proto_window);

  const YAML::Node files = node["object_files"];
  if (!files)
  {
    return;
  }
  reject_unknown_keys(
      files,
      {"correspondence", "feature_tolerance", "feature_gate", "merge_resolve_frames", "correspondence_radius",
       "max_inactive_age", "motion_prediction", "appearance_matching", "appearance_weight", "persistent_identity",
       "reid_colour_gate", "reid_colour_veto", "gate_growth"},
      "attention_system.object_files");
  ObjectFileStore::Config& store = cfg.object_store;
  if (files["correspondence"])
  {
    const std::string rule = files["correspondence"].as<std::string>();
    if (rule != "position" && rule != "thesis")
    {
      throw std::runtime_error("attention_system.object_files.correspondence: '" + rule +
                               "' (known: position, thesis)");
    }
    store.rule = rule == "thesis" ? ObjectFileStore::Rule::Thesis : ObjectFileStore::Rule::Position;
  }
  read_param(files, "feature_tolerance", store.feature_tolerance);
  read_param(files, "feature_gate", store.feature_gate);
  read_param(files, "merge_resolve_frames", store.merge_resolve_frames);
  read_param(files, "correspondence_radius", store.correspondence_radius);
  read_param(files, "max_inactive_age", store.max_inactive_age);
  read_param(files, "motion_prediction", store.motion_prediction);
  read_param(files, "appearance_matching", store.appearance_matching);
  read_param(files, "appearance_weight", store.appearance_weight);
  read_param(files, "persistent_identity", store.persistent_identity);
  read_param(files, "reid_colour_gate", store.reid_colour_gate);
  read_param(files, "reid_colour_veto", store.reid_colour_veto);
  read_param(files, "gate_growth", store.gate_growth);
}

AttentionSystem::AttentionSystem(const Config& config)
  : config_(config),
    pipeline_(config.pipeline),
    object_store_(config.object_store),
    behavior_(create_behavior(config.behavior, config.ior_params, config.identification_params)),
    history_(config.pipeline.priority)
{
  for (const auto& name : config_.processors)
  {
    processors_.push_back(create_processor(name));
  }
  if (config_.cluster_source == Config::ClusterSource::Field)
  {
    const pipeline::PipelineConfig& p = config_.pipeline;
    selection::SelectionParams shared;
    shared.min_distance = p.peak_min_distance;
    shared.threshold = p.peak_threshold;
    shared.max_count = p.peak_max_count;
    const bool own_params = !p.selection_params_yaml.empty();
    field_ = selection::create_selection_strategy("neural-field", shared,
                                                  own_params ? YAML::Load(p.selection_params_yaml) : YAML::Node());
  }
}

namespace
{
// Per-channel median colour of an 8-bit image: the frame's figure-ground
// estimate (objects rarely cover half the view).
cv::Vec3f median_colour(const cv::Mat& image)
{
  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  cv::Vec3f median(0.0f, 0.0f, 0.0f);
  for (int c = 0; c < std::min(3, static_cast<int>(channels.size())); ++c)
  {
    int hist[256] = {0};
    for (int y = 0; y < channels[c].rows; ++y)
    {
      const uchar* row = channels[c].ptr<uchar>(y);
      for (int x = 0; x < channels[c].cols; ++x)
      {
        ++hist[row[x]];
      }
    }
    const int half = static_cast<int>(channels[c].total() / 2);
    int seen = 0;
    int v = 0;
    while (v < 255 && seen + hist[v] <= half)
    {
      seen += hist[v++];
    }
    median[c] = static_cast<float>(v);
  }
  return median;
}

// Per-pixel colour distance (L2, 0-255 scale) of `image` to `colour`.
cv::Mat colour_distance(const cv::Mat& image, const cv::Vec3f& colour)
{
  std::vector<cv::Mat> channels;
  cv::split(image, channels);
  cv::Mat sum = cv::Mat::zeros(image.size(), CV_32F);
  for (int c = 0; c < std::min(3, static_cast<int>(channels.size())); ++c)
  {
    cv::Mat diff;
    channels[c].convertTo(diff, CV_32F, 1.0, -colour[c]);
    sum += diff.mul(diff);
  }
  cv::sqrt(sum, sum);
  return sum;
}

cv::Vec3f mean_colour(const cv::Mat& image, const cv::Mat& mask)
{
  const cv::Scalar m = cv::mean(image, mask);
  return cv::Vec3f(static_cast<float>(m[0]), static_cast<float>(m[1]), static_cast<float>(m[2]));
}
} // namespace

std::vector<Cluster> AttentionSystem::proto_objects(const cv::Mat& region, const cv::Mat& image, const Cluster& cluster,
                                                    const cv::Vec3f& ground) const
{
  // A fill covering more than this share of its window has flooded the
  // background rather than traced an object.
  constexpr double kLeakFraction = 0.6;
  constexpr int kMaxObjectsPerCluster = 4;

  const int mx = static_cast<int>(config_.proto_window * cluster.bbox.width);
  const int my = static_cast<int>(config_.proto_window * cluster.bbox.height);
  const cv::Rect window =
      cv::Rect(cluster.bbox.x - mx, cluster.bbox.y - my, cluster.bbox.width + 2 * mx, cluster.bbox.height + 2 * my) &
      cv::Rect(0, 0, image.cols, image.rows);
  const cv::Mat distance = colour_distance(image, ground);
  cv::Mat remaining = region.clone();
  std::vector<Cluster> objects;
  bool untextured = false; // a figure seed that wouldn't grow (texture): keep the salient cluster
  for (int i = 0; i < kMaxObjectsPerCluster && cv::countNonZero(remaining) >= config_.min_cluster_size; ++i)
  {
    double contrast = 0.0;
    cv::Point seed;
    cv::minMaxLoc(distance, nullptr, &contrast, nullptr, &seed, remaining);
    if (contrast < config_.proto_min_contrast)
    {
      break; // nothing left that stands out from the ground
    }
    cv::Mat roi = image(window).clone(); // floodFill wants a mutable image even with MASK_ONLY
    cv::Mat fill = cv::Mat::zeros(window.height + 2, window.width + 2, CV_8U);
    const double t = config_.proto_tolerance;
    cv::floodFill(roi, fill, seed - window.tl(), cv::Scalar(), nullptr, cv::Scalar(t, t, t), cv::Scalar(t, t, t),
                  8 | cv::FLOODFILL_MASK_ONLY | cv::FLOODFILL_FIXED_RANGE | (255 << 8));
    const cv::Mat grown = fill(cv::Rect(1, 1, window.width, window.height));
    cv::Mat grown_full = cv::Mat::zeros(region.size(), CV_8U);
    grown.copyTo(grown_full(window));
    remaining.setTo(0, grown_full);
    remaining.at<uchar>(seed) = 0; // progress even if the fill were empty
    const int area = cv::countNonZero(grown);
    if (area > kLeakFraction * window.area())
    {
      continue; // flooded the window: background
    }
    if (area < config_.min_cluster_size)
    {
      untextured = true;
      continue;
    }
    Cluster object = cluster;
    const cv::Moments m = cv::moments(grown, true);
    object.size = area;
    object.bbox = cv::boundingRect(grown) + window.tl();
    object.centroid =
        cv::Point(static_cast<int>(m.m10 / m.m00 + 0.5) + window.x, static_cast<int>(m.m01 / m.m00 + 0.5) + window.y);
    object.appearance = mean_colour(image, grown_full);
    objects.push_back(object);
  }
  if (objects.empty() && untextured)
  {
    objects.push_back(cluster);
  }
  return objects;
}

std::vector<float> AttentionSystem::feature_means(const cv::Mat& region) const
{
  std::vector<float> means;
  for (const auto& feature : pipeline_.get_features())
  {
    if (feature.data.empty())
    {
      means.push_back(0.0f);
      continue;
    }
    cv::Mat mask = region;
    if (mask.size() != feature.data.size())
    {
      cv::resize(region, mask, feature.data.size(), 0, 0, cv::INTER_NEAREST);
    }
    means.push_back(static_cast<float>(cv::mean(feature.data, mask)[0]));
  }
  return means;
}

std::vector<Cluster> AttentionSystem::field_clusters(const cv::Mat& saliency)
{
  std::vector<Cluster> clusters;
  const auto* field = dynamic_cast<const selection::NeuralFieldSelection*>(field_.get());
  if (field == nullptr || saliency.empty())
  {
    return clusters;
  }
  const cv::Mat& image = pipeline_.get_frame().image;
  const bool have_image = !image.empty() && image.size() == saliency.size();
  for (const auto& active : field->track(saliency, field_activity_))
  {
    Cluster cluster;
    cluster.centroid =
        cv::Point(static_cast<int>(active.centroid.x + 0.5f), static_cast<int>(active.centroid.y + 0.5f));
    cluster.centroid_exact = active.centroid;
    cluster.bbox = active.bbox;
    cluster.size = active.size;
    cluster.mean_saliency = static_cast<float>(cv::mean(saliency, active.mask)[0]);
    if (have_image)
    {
      cluster.appearance = mean_colour(image, active.mask);
    }
    cluster.features = feature_means(active.mask);
    clusters.push_back(cluster);
  }
  return clusters;
}

std::vector<Cluster> AttentionSystem::segment(const cv::Mat& saliency) const
{
  return segment(saliency, pipeline_.get_frame().image);
}

std::vector<Cluster> AttentionSystem::segment(const cv::Mat& saliency, const cv::Mat& image) const
{
  std::vector<Cluster> clusters;
  if (saliency.empty())
  {
    return clusters;
  }

  double max_val = 0.0;
  cv::minMaxLoc(saliency, nullptr, &max_val);
  if (max_val <= 0.0)
  {
    return clusters; // nothing salient this frame
  }

  const double thresh = std::max(static_cast<double>(config_.segment_min), config_.segment_fraction * max_val);
  cv::Mat mask = saliency > thresh; // CV_8U
  if (config_.segment_close > 0)
  {
    // Bridge an object's fragments (e.g. a moving disk's leading and trailing
    // onset crescents) into one cluster, hence one object file.
    const int k = 2 * config_.segment_close + 1;
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
  }
  const int max_area = config_.max_cluster_fraction > 0.0f
                           ? static_cast<int>(config_.max_cluster_fraction * static_cast<float>(mask.total()))
                           : std::numeric_limits<int>::max();

  // The native frame, at saliency resolution, gives each cluster an appearance
  // descriptor (mean colour) for identity-stable correspondence (M12) — computed
  // only over the already-selected regions, at no extra segmentation cost.
  const bool have_image = !image.empty() && image.size() == saliency.size();
  const bool proto = config_.proto_objects && have_image && image.depth() == CV_8U;
  const cv::Vec3f ground = proto ? median_colour(image) : cv::Vec3f();

  cv::Mat labels, stats, centroids;
  const int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
  for (int label = 1; label < num_labels; ++label) // 0 == background
  {
    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
    if (area < config_.min_cluster_size || area > max_area)
    {
      continue; // too small to track, or too large to be an object
    }
    Cluster cluster;
    cluster.size = area;
    cluster.bbox = cv::Rect(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                            stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));
    cluster.centroid = cv::Point(static_cast<int>(centroids.at<double>(label, 0) + 0.5),
                                 static_cast<int>(centroids.at<double>(label, 1) + 0.5));
    const cv::Mat region = (labels == label);
    cluster.mean_saliency = static_cast<float>(cv::mean(saliency, region)[0]);
    if (have_image)
    {
      cluster.appearance = mean_colour(image, region);
    }
    cluster.features = feature_means(region);
    if (proto)
    {
      for (const auto& object : proto_objects(region, image, cluster, ground))
      {
        clusters.push_back(object);
      }
    }
    else
    {
      clusters.push_back(cluster);
    }
  }
  if (proto)
  {
    // Fragments of one object grow into the same proto-object: keep the most
    // salient of any heavily overlapping pair, or of a pair where one sits
    // mostly inside the other and looks the same (a partial growth). A
    // different-looking object inside another's box (a phone in a hand) stays.
    std::stable_sort(clusters.begin(), clusters.end(),
                     [](const Cluster& a, const Cluster& b) { return a.mean_saliency > b.mean_saliency; });
    std::vector<Cluster> kept;
    for (const auto& c : clusters)
    {
      const bool duplicate = std::any_of(
          kept.begin(), kept.end(),
          [&](const Cluster& k)
          {
            const double inter = (c.bbox & k.bbox).area();
            const double uni = c.bbox.area() + k.bbox.area() - inter;
            const double smaller = std::min(c.bbox.area(), k.bbox.area());
            const bool alike = cv::norm(c.appearance - k.appearance) < config_.object_store.reid_colour_gate;
            return (uni > 0.0 && inter / uni > 0.5) || (smaller > 0.0 && inter >= 0.8 * smaller && alike);
          });
      if (!duplicate)
      {
        kept.push_back(c);
      }
    }
    clusters = std::move(kept);
  }
  return clusters;
}

void AttentionSystem::process_second_stage()
{
  has_focus_ = false;
  if (config_.action_mode == ActionMode::Feature)
  {
    return; // saliency only
  }

  // M17: the map the second stage sees is the priority map — the pipeline's
  // (already top-down-adjusted) saliency plus the history/value channels.
  // With inactive channels this is the pipeline map untouched.
  compensate_camera();
  const cv::Mat priority = history_.apply(pipeline_.get_saliency_map().map, object_store_.active_files());
  std::vector<Cluster> clusters =
      config_.cluster_source == Config::ClusterSource::Field ? field_clusters(priority) : segment(priority);
  object_store_.update(clusters, frame_index_);
  object_store_.decay_values(config_.pipeline.priority.object_value_decay);

  const ObjectFile* focus = behavior_->select_focus(object_store_, frame_index_);
  if (focus != nullptr)
  {
    current_focus_.frame = frame_index_;
    current_focus_.label = focus->label;
    current_focus_.location = focus->centroid;
    current_focus_.bbox = focus->bbox;
    current_focus_.saliency = focus->saliency;
    has_focus_ = true;
    scanpath_.push_back(current_focus_);
    // M17 facilitation: being selected accrues value on the object file
    // (rewards from a task arrive via ObjectFileStore::add_value).
    object_store_.add_value(focus->label, config_.pipeline.priority.object_value_per_selection);
  }
  history_.decay_and_record(has_focus_ ? current_focus_.location : cv::Point(), priority.size(), has_focus_);

  run_processors();
}

void AttentionSystem::compensate_camera()
{
  last_camera_shift_ = cv::Point2f(0, 0);
  if (!config_.camera_compensation)
  {
    return;
  }
  const cv::Mat& image = pipeline_.get_frame().image;
  if (image.empty())
  {
    return;
  }
  // Phase correlation at a reduced size gives the frame's global translation
  cv::Mat gray;
  if (image.channels() == 3)
  {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    gray = image;
  }
  const double factor = std::min(1.0, 256.0 / std::max(gray.cols, gray.rows));
  cv::Mat small;
  cv::resize(gray, small, cv::Size(), factor, factor, cv::INTER_AREA);
  small.convertTo(small, CV_32F);
  if (!previous_gray_.empty() && previous_gray_.size() == small.size())
  {
    cv::Mat window;
    cv::createHanningWindow(window, small.size(), CV_32F);
    // A scene point at p in the previous frame is at p + shift in this one
    const cv::Point2d shift = cv::phaseCorrelate(previous_gray_, small, window);
    const cv::Point2f in_image(static_cast<float>(shift.x / factor), static_cast<float>(shift.y / factor));
    const float limit = config_.camera_max_shift * std::max(image.cols, image.rows);
    if (std::abs(in_image.x) < limit && std::abs(in_image.y) < limit)
    {
      last_camera_shift_ = in_image;
      object_store_.displace(in_image);
      behavior_->displace(in_image);
      history_.displace(in_image);
      if (!field_activity_.empty())
      {
        // The field is at its own resolution (the thesis's displacefield())
        const double field_factor =
            static_cast<double>(field_activity_.cols) / pipeline_.get_saliency_map().map.cols;
        const cv::Mat warp =
            (cv::Mat_<double>(2, 3) << 1, 0, in_image.x * field_factor, 0, 1, in_image.y * field_factor);
        cv::Mat moved;
        const cv::Scalar resting(field_activity_.at<float>(0, 0));
        cv::warpAffine(field_activity_, moved, warp, field_activity_.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                       resting);
        field_activity_ = moved;
      }
    }
  }
  previous_gray_ = small;
}

void AttentionSystem::run_processors()
{
  if (processors_.empty())
  {
    return;
  }

  const cv::Mat& image = pipeline_.get_frame().image;
  if (image.empty())
  {
    return;
  }

  // The FullFrame baseline: every processor over the entire frame, every
  // frame — what recognition costs without attention gating (H2's comparison
  // arm; annotations carry object_label -1).
  if (config_.processor_cadence == ProcessorCadence::FullFrame)
  {
    ObjectFile whole_frame;
    whole_frame.label = -1;
    whole_frame.bbox = cv::Rect(0, 0, image.cols, image.rows);
    for (const auto& processor : processors_)
    {
      record_annotation(run_processor(*processor, whole_frame, image, cv::Point(0, 0), frame_index_));
    }
    return;
  }

  // Gated: processors see only the focus ROI. PerDwell fires when the focus
  // settles on an object (a new visit) and re-fires every process_repeat_frames
  // while the focus stays put — a continuously held focus is a sequence of
  // attentive computations, not one; EveryFrame re-fires every focused frame.
  if (!has_focus_)
  {
    last_processed_label_ = -1; // losing the focus ends the dwell
    return;
  }
  if (config_.processor_cadence == ProcessorCadence::PerDwell && current_focus_.label == last_processed_label_)
  {
    ++frames_since_processed_;
    if (frames_since_processed_ < std::max(1, config_.process_repeat_frames))
    {
      return;
    }
  }
  last_processed_label_ = current_focus_.label;
  frames_since_processed_ = 0;

  // Expand the tight object bbox by a margin — detectors want context around
  // the object (HOG's person window includes background around the person).
  const ObjectFile* focus = object_store_.find_active(current_focus_.label);
  if (focus == nullptr)
  {
    return;
  }
  cv::Rect box = focus->bbox;
  const int mx = static_cast<int>(box.width * config_.roi_margin);
  const int my = static_cast<int>(box.height * config_.roi_margin);
  box += cv::Point(-mx, -my);
  box += cv::Size(2 * mx, 2 * my);
  box &= cv::Rect(0, 0, image.cols, image.rows);
  const cv::Mat roi = box.area() > 0 ? image(box) : cv::Mat();

  for (const auto& processor : processors_)
  {
    record_annotation(run_processor(*processor, *focus, roi, box.tl(), frame_index_));
  }
}

void AttentionSystem::record_annotation(const Annotation& annotation)
{
  if (annotation.object_label >= 0)
  {
    object_store_.record_inspection(annotation.object_label);
    object_store_.add_label_vote(annotation.object_label, annotation.class_label, annotation.confidence);
  }
  // Keep the log bounded: a long-running live session records annotations
  // every frame and would otherwise grow without limit. Dropping the oldest
  // half keeps recent history; label memory and stats are unaffected.
  constexpr size_t kMaxAnnotationLog = 100000;
  if (annotations_.size() >= kMaxAnnotationLog)
  {
    annotations_.erase(annotations_.begin(), annotations_.begin() + kMaxAnnotationLog / 2);
  }
  annotations_.push_back(annotation);

  ProcessorStats& stats = processor_stats_[annotation.processor];
  stats.calls += 1;
  stats.pixels += annotation.pixels;
  stats.ms += annotation.ms;
}

void AttentionSystem::reset_stage2()
{
  object_store_.reset();
  behavior_->reset();
  scanpath_.clear();
  has_focus_ = false;
  frame_index_ = 0;
  annotations_.clear();
  processor_stats_.clear();
  last_processed_label_ = -1;
  frames_since_processed_ = 0;
  history_.reset();
  field_activity_ = cv::Mat();
  previous_gray_ = cv::Mat();
  last_camera_shift_ = cv::Point2f(0, 0);
}

void AttentionSystem::reset()
{
  pipeline_.reset_state();
  reset_stage2();
}

void AttentionSystem::process_frame(const cv::Mat& image, const std::string& source_name)
{
  pipeline_.load_image(image, source_name);
  pipeline_.process();
  process_second_stage();
  ++frame_index_;
}

void AttentionSystem::process_stream(pipeline::FrameSource& source, const FocusCallback& on_frame)
{
  // pipeline_.process_stream resets the pipeline RunState itself; reset only
  // the second-stage state here.
  reset_stage2();

  pipeline_.process_stream(source,
                           [&](pipeline::AttentionPipeline& /*p*/)
                           {
                             process_second_stage();
                             if (on_frame)
                             {
                               on_frame(*this);
                             }
                             ++frame_index_;
                           });
}

} // namespace system
} // namespace attention
