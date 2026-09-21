// The visualizers return an image; they open a window only when asked to.

#include "attention/core/feature_map.h"
#include "attention/visualization/visualizer.h"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/opencv.hpp>

using namespace attention;

TEST_CASE("visualize_feature_map without a window name opens no window", "[visualization]")
{
  // Regression: the window name defaulted to the feature's name, so every call
  // ran cv::imshow — also under --no-display. On macOS that turns the process
  // into a Dock application which takes the keyboard focus; a study launching
  // the binary hundreds of times made the machine unusable while it ran.
  const core::FeatureMap feature("no-window-please", cv::Mat(32, 32, CV_32F, cv::Scalar(0.5f)), 1.0f);
  const cv::Mat image = visualization::visualize_feature_map(feature);
  REQUIRE(image.size() == cv::Size(32, 32));

  double visible = -1.0;
  try
  {
    visible = cv::getWindowProperty(feature.name, cv::WND_PROP_VISIBLE);
  }
  catch (const cv::Exception&)
  {
    // No such window (or a build without a GUI backend): nothing was opened
  }
  CHECK(visible <= 0.0);
}
