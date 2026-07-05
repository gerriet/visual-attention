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

  /**
   * Whether this feature applies to the given frame. Inapplicable features
   * are skipped silently by the pipeline (e.g., color on grayscale input).
   */
  virtual bool applicable(const core::Frame& frame) const { return true; }

  /**
   * The Gabor bank this feature reads (orientations == 0: none). The
   * pipeline precomputes every required bank BEFORE parallel extraction —
   * features must never mutate shared Frame state from their extraction
   * threads; they access their bank via Frame::gabor_bank().
   */
  struct GaborRequirement
  {
    int orientations = 0;
    double wavelength = 4.0;
    double bandwidth = 1.0;
  };

  virtual GaborRequirement gabor_requirement() const { return {}; }

  /**
   * Whether this feature's map doubles as a per-pixel depth cue. When true,
   * the pipeline captures the extracted map into RunState::depth_map so the
   * 3D neural-field selection (M5) can lift the fused 2D saliency into a
   * depth volume. Only the stereo/disparity feature sets this.
   */
  virtual bool produces_depth() const { return false; }
};

} // namespace features
} // namespace attention
