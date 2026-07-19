#include "attention/system/attention_system.h"
#include <algorithm>

namespace attention
{
namespace system
{

AttentionSystem::AttentionSystem(const Config& config)
  : config_(config),
    pipeline_(config.pipeline),
    object_store_(config.object_store),
    behavior_(create_behavior(config.behavior, config.ior_params, config.identification_params))
{
  for (const auto& name : config_.processors)
  {
    processors_.push_back(create_processor(name));
  }
}

std::vector<Cluster> AttentionSystem::segment(const cv::Mat& saliency) const
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

  // The native frame, at saliency resolution, gives each cluster an appearance
  // descriptor (mean colour) for identity-stable correspondence (M12) — computed
  // only over the already-selected regions, at no extra segmentation cost.
  const cv::Mat& image = pipeline_.get_frame().image;
  const bool have_image = !image.empty() && image.size() == saliency.size();

  cv::Mat labels, stats, centroids;
  const int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
  for (int label = 1; label < num_labels; ++label) // 0 == background
  {
    const int area = stats.at<int>(label, cv::CC_STAT_AREA);
    if (area < config_.min_cluster_size)
    {
      continue;
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
      const cv::Scalar mean_colour = cv::mean(image, region);
      cluster.appearance = cv::Vec3f(static_cast<float>(mean_colour[0]), static_cast<float>(mean_colour[1]),
                                     static_cast<float>(mean_colour[2]));
    }
    clusters.push_back(cluster);
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

  const cv::Mat& saliency = pipeline_.get_saliency_map().map;
  std::vector<Cluster> clusters = segment(saliency);
  object_store_.update(clusters, frame_index_);

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
  }

  run_processors();
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
