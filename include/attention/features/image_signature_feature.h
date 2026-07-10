#pragma once

#include "attention/core/feature_map.h"
#include "attention/core/frame.h"
#include "attention/features/feature_extractor.h"
#include <opencv2/opencv.hpp>

namespace attention
{
namespace features
{

/**
 * ImageSignatureFeature — DCT sign-signature saliency (not part of the thesis).
 *
 * The "image signature" is sign(DCT(image)): if an image is a spatially sparse
 * foreground on a spectrally sparse background, reconstructing from the sign of
 * the DCT coefficients alone concentrates energy on the foreground support.
 * Saliency is the smoothed, squared reconstruction, computed per channel (CIE
 * Lab for colour input) at a small working resolution and summed.
 *
 * This is an ALTERNATIVE saliency operator, opt-in via config; the thesis
 * feature set remains the default (see docs/ALTERNATIVE_FEATURES.md).
 *
 * Reference:
 *   X. Hou, J. Harel, C. Koch, "Image Signature: Highlighting Sparse Salient
 *   Regions", IEEE TPAMI 34(1), 2012.
 */
class ImageSignatureFeature : public FeatureExtractor
{
 public:
  struct Config
  {
    // Longer side of the working image (the paper operates at ~64 px). Working
    // dimensions are rounded to even values (cv::dct requirement).
    int input_size = 64;
    // Gaussian sigma (at working resolution) smoothing the squared
    // reconstruction; the paper suggests ~0.05 of the image width.
    double gaussian_sigma = 3.0;
  };

  ImageSignatureFeature() : ImageSignatureFeature(Config{}) {}
  explicit ImageSignatureFeature(const Config& config);

  core::FeatureMap extract(const core::Frame& frame) const override;

  std::string name() const override { return "image-signature"; }

 private:
  Config config_;
};

} // namespace features
} // namespace attention
