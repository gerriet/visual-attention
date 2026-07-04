#pragma once

namespace attention
{
namespace core
{

/**
 * RunState carries per-run state across the frames of a stream.
 *
 * The v2 pipeline is stream-oriented: motion/onset features, neural-field
 * dynamics (roadmap M3), cross-frame inhibition of return, and ESAB2 object
 * tracking (M6) all need state that outlives a single frame. This struct is
 * their home; it grows as those milestones land.
 */
struct RunState
{
  int frame_index = 0;

  void reset() { *this = RunState(); }
};

} // namespace core
} // namespace attention
