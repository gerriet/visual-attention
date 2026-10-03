#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/features/feature_extractor.h"
#include <opencv2/opencv.hpp>
#include <string>

namespace attention
{
namespace features
{

/**
 * FaceFeature: a stage-1 channel that responds to faces.
 *
 * **Not** a thesis feature, and deliberately not in any default profile — the
 * replication profiles must stay what the dissertation described. This is a
 * modern-track channel (docs/adr/0005) that exists because H4's own verdict
 * named it:
 *
 *   "What would change these numbers, and is cheap: a centre prior on the
 *    priority map, and the face channel — MIT1003 is full of faces and text."
 *    (docs/SCANPATH_VS_HUMAN.md)
 *
 * The finding it is meant to probe is that the model's scanpaths are above
 * random but **not above a centre baseline** on MIT1003. Human free viewing of
 * photographs is dominated by faces; a bottom-up model with no face channel is
 * being asked to predict gaze driven by something it cannot see. Adding the
 * channel does not make the thesis model better — it tests whether the gap is
 * *about faces*, which is a different and more useful statement than "the model
 * is weak".
 *
 * Two backends, both from OpenCV, so nothing new is vendored:
 *
 * - `haar` (default): the classic cascade. Ships with OpenCV, no download.
 *   Weak on profile and small faces, which is worth knowing when reading an
 *   ablation that uses it.
 * - `yunet`: OpenCV's `FaceDetectorYN` CNN, which is markedly better but needs
 *   an ONNX file (`model_path`). Falls back to nothing if the file is absent —
 *   loudly, at construction, not silently at run time.
 *
 * The map is a sum of Gaussian blobs, one per detection, each scaled by the
 * detector's confidence where it has one. A blob rather than a filled box
 * because everything downstream — peak selection, the neural field — expects a
 * smooth map with a maximum to find, and a flat rectangle has no peak.
 */
class FaceFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    std::string backend = "haar"; ///< "haar" or "yunet"
    /// Cascade XML (haar) or ONNX (yunet). Empty = the cascade CMake found at
    /// configure time; for yunet it is required.
    std::string model_path;
    double scale_factor = 1.1; ///< haar: pyramid step
    int min_neighbors = 4;     ///< haar: higher = fewer false positives
    int min_size = 24;         ///< px; faces smaller than this are ignored
    /// Gaussian sigma as a fraction of the detected face's width. 0.35 puts
    /// most of the blob inside the face box.
    float sigma_frac = 0.35f;
    float score_threshold = 0.7f; ///< yunet: detection confidence cut
    float nms_threshold = 0.3f;   ///< yunet
  };

  FaceFeature() : FaceFeature(Config{}) {}
  explicit FaceFeature(const Config& config);
  ~FaceFeature() override;

  core::FeatureMap extract(const core::Frame& frame) const override;
  core::FeatureMap extract(const core::Frame& frame, DebugContext& debug) const override;

  std::string name() const override { return "face"; }

  /// Needs pixels; works on colour or grayscale (Haar converts, YuNet needs BGR).
  bool applicable(const core::Frame& frame) const override { return !frame.empty(); }

  /// Where the default Haar cascade was found, or "" if CMake found none.
  static std::string default_cascade_path();

 private:
  std::vector<cv::Rect> detect(const cv::Mat& image) const;

  Config config_;
  // Detectors are stateful and not cheap to build, so they are constructed
  // once. `mutable` because extract() is const by the interface's contract and
  // OpenCV's detect() is not.
  mutable cv::CascadeClassifier cascade_;
  mutable cv::Ptr<cv::FaceDetectorYN> yunet_;
  mutable cv::Size yunet_size_;
};

} // namespace features
} // namespace attention
