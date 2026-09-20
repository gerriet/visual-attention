#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace attention
{
namespace features
{

/**
 * Exclusivity weighting (thesis §5.5.3, "Bewertung der Exklusivität") — the
 * model's odd-man-popout: a segment's saliency is divided by a factor that
 * grows with the number n of segments sharing its category (orientation class
 * for eccentricity, hue class for colour contrast), so a property that is
 * frequent in the image is suppressed and a unique one stands out. It is an
 * evaluation *within* a feature — the result is a modified feature saliency,
 * combinable with any fusion — not a fusion step.
 *
 * Two forms exist, and they disagree:
 *  - Exponential — the thesis text: divide by c^n, c >= 1, "1.1 in all
 *    experiments"; c = 1 switches the weighting off.
 *  - Power — the surviving original sources (feature/color.C,
 *    feature/eccentricity.C): divide by n^p for n > 1; p = 0 switches it off
 *    (and was the compiled-in default).
 * The thesis form is the default here; the code form is kept selectable so
 * either can be replicated.
 *
 * The thesis's side note applies to both: the weighting also amplifies noise —
 * a spurious segment that is alone in its category is boosted relative to the
 * rest.
 */
struct Exclusivity
{
  enum class Mode
  {
    Exponential, // thesis: c^n
    Power        // original sources: n^p
  };

  Mode mode = Mode::Exponential;
  // c (Exponential, >= 1) or p (Power, >= 0). The defaults switch the
  // weighting off, so a feature behaves as plain saliency unless a config
  // (configs/thesis.yaml: 1.1) turns it on.
  float strength = 1.0f;

  bool enabled() const { return mode == Mode::Exponential ? strength > 1.0f : strength > 0.0f; }

  /// The divisor for a segment whose category holds `count` segments (itself
  /// included). Always >= 1; 1 when the weighting is off or count is 0.
  float divisor(int count) const
  {
    if (!enabled() || count <= 0)
    {
      return 1.0f;
    }
    if (mode == Mode::Exponential)
    {
      return std::pow(strength, static_cast<float>(count));
    }
    return count > 1 ? std::pow(static_cast<float>(count), strength) : 1.0f;
  }

  static Mode parse_mode(const std::string& name)
  {
    if (name == "exponential" || name == "thesis")
    {
      return Mode::Exponential;
    }
    if (name == "power")
    {
      return Mode::Power;
    }
    throw std::runtime_error("Unknown exclusivity_mode '" + name + "' (exponential | power)");
  }
};

} // namespace features
} // namespace attention
