#include "attention/selection/random_selection.h"

namespace attention
{
namespace selection
{

RandomSelection::RandomSelection(const SelectionParams& shared, const Params& params)
  : shared_(shared), params_(params), rng_(params.seed)
{
}

std::vector<core::Peak> RandomSelection::select(const cv::Mat& saliency, core::RunState& /*state*/) const
{
  std::vector<core::Peak> peaks;
  if (saliency.empty() || shared_.max_count <= 0)
  {
    return peaks;
  }

  std::uniform_int_distribution<int> x_dist(0, saliency.cols - 1);
  std::uniform_int_distribution<int> y_dist(0, saliency.rows - 1);
  const double min_sq = static_cast<double>(shared_.min_distance) * shared_.min_distance;

  peaks.reserve(static_cast<size_t>(shared_.max_count));
  for (int i = 0; i < shared_.max_count; ++i)
  {
    bool placed = false;
    for (int attempt = 0; attempt < params_.max_attempts && !placed; ++attempt)
    {
      const cv::Point candidate(x_dist(rng_), y_dist(rng_));
      const float value = saliency.at<float>(candidate);
      if (value < params_.min_value)
      {
        continue;
      }
      // Same spacing rule as the other strategies, so this arm spends a
      // comparable budget on a comparably spread set of locations; only
      // *where* differs.
      bool too_close = false;
      for (const auto& existing : peaks)
      {
        const double dx = existing.location.x - candidate.x;
        const double dy = existing.location.y - candidate.y;
        if (dx * dx + dy * dy < min_sq)
        {
          too_close = true;
          break;
        }
      }
      if (too_close)
      {
        continue;
      }
      peaks.emplace_back(candidate, value);
      placed = true;
    }
    if (!placed)
    {
      // The frame could not take another well-spaced point. Returning fewer is
      // right: silently relaxing the spacing would make this arm denser than
      // the one it is a control for, which is exactly the comparison it exists
      // to keep fair.
      break;
    }
  }
  return peaks;
}

} // namespace selection
} // namespace attention
