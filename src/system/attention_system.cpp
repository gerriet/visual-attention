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
    behavior_(create_behavior(config.behavior, config.ior_params))
{
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
    cluster.mean_saliency = static_cast<float>(cv::mean(saliency, labels == label)[0]);
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
}

void AttentionSystem::reset_stage2()
{
  object_store_.reset();
  behavior_->reset();
  scanpath_.clear();
  has_focus_ = false;
  frame_index_ = 0;
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
