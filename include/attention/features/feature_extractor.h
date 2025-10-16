#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/features/debug_context.h"
#include <memory>

namespace attention
{
namespace features
{

/**
 * Base interface for all feature extractors.
 *
 * This interface allows polymorphic feature extraction,
 * enabling parallel processing and dynamic feature selection.
 */
class FeatureExtractor
{
 public:
  virtual ~FeatureExtractor() = default;

  /**
   * Extract feature from frame.
   * @param frame Input frame
   * @return Feature map with saliency values [0, 1]
   */
  virtual core::FeatureMap extract(const core::Frame& frame) const = 0;

  /**
   * Extract feature from frame with debug context.
   * @param frame Input frame
   * @param debug Debug context for capturing intermediate results
   * @return Feature map with saliency values [0, 1]
   */
  virtual core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const
  {
    // Default implementation ignores debug context
    return extract(frame);
  }

  /**
   * Get the name of this feature extractor.
   * @return Feature name (e.g., "color", "intensity", "symmetry")
   */
  virtual std::string name() const = 0;
};

} // namespace features
} // namespace attention
