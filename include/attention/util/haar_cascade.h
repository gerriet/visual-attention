#pragma once

#include <string>

namespace attention
{
namespace util
{

/**
 * Locate a Haar cascade XML that ships with OpenCV.
 *
 * OpenCV installs these but their directory varies by platform and package, so
 * the search order is: `$ATTENTION_HAAR_DIR` (the escape hatch), then the
 * directory CMake found next to the linked OpenCV at configure time, then the
 * usual system locations.
 *
 * Shared by the `haar-face` recognition processor (M13) and the opt-in `face`
 * stage-1 feature (H4), so the two cannot disagree about where cascades live.
 *
 * @param file bare filename, e.g. "haarcascade_frontalface_default.xml"
 * @return the full path, or "" when nothing matched — callers decide whether
 *         that is fatal, since one of them is optional and the other is not.
 */
std::string find_haar_cascade(const std::string& file);

} // namespace util
} // namespace attention
