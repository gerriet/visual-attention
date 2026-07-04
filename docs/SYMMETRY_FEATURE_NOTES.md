# Symmetry Feature Implementation Notes

## Overview

The symmetry feature was completely reimplemented (October 2025) to match the thesis specification. The original implementation used a statistical variance-based approach which was incorrect. The new implementation uses geometric radius-sampling as specified in the thesis.

## Thesis Specification

At each pyramid level, for each pixel:
1. Test multiple radii (e.g., 5-30 pixels with step=2)
2. For each radius and orientation:
   - Sum Gabor filter values within a specified width (e.g., 3 pixels) orthogonal to the radius direction
   - Add responses from opposite sides (bilateral symmetry)
3. Select maximum across all radii for each pixel
4. Combine results from multiple scales

### Key Parameters (from thesis)
- **Pyramid levels**: Originally 64×64, 128×128, 256×256
- **Radii**: Originally {6, 9, 12, 15} pixels
- **Width**: 3 pixels (orthogonal summation box)
- **Orientations**: 12 orientations

## Current Implementation

### Architecture
- Uses existing pyramid levels directly (no resizing to fixed sizes)
- Adaptive scale selection: finds first level where min(width, height) < 256px
- Processes 3 consecutive pyramid levels starting from that point
- Continuous radius sampling with configurable step (currently step=2)

### Configuration
```cpp
// Example for large image (1918×1370):
// start_level = 3 (where 1370/8 = 171 < 256)
ScaleConfig(3, 5, 20, 2, 3, 0.3f)   // Level 3: 240×172, radii 5-20, step=2, threshold=0.3
ScaleConfig(4, 8, 25, 2, 3, 0.5f)   // Level 4: 120×86, radii 8-25, step=2, threshold=0.5
ScaleConfig(5, 10, 30, 2, 3, 0.65f) // Level 5: 60×43, radii 10-30, step=2, threshold=0.65
```

### Key Improvements

#### 1. Blob-like Responses (vs. Edge Detection)
**Problem**: Discrete radii {6, 9, 12, 15} created ring-like responses at object boundaries
**Solution**: Continuous radius sampling (e.g., 5, 7, 9, 11, 13, 15, 17, 19) fills symmetric regions

#### 2. Localized Peaks (vs. Over-saturation)
**Problem**: Too much of the image appeared symmetric
**Solution**:
- **Threshold**: Only count symmetry values ≥ threshold (normalized to [0,1])
- **Radius consistency**: Require ≥2 radii above threshold
- **Consistency weighting**: More consistent radii = higher score
  ```cpp
  if (num_above_threshold >= 2) {
      float consistency_weight = min(1.0f, num_above_threshold / 4.0f);
      result = max_val * consistency_weight;
  }
  ```

#### 3. Scale-Dependent Thresholds
**Problem**: Coarse scales showed too many false positives
**Solution**: Higher thresholds at coarser scales
- Finest scale: 0.3
- Medium scale: 0.5
- Coarsest scale: 0.65

This suppresses weak symmetry at coarse scales while preserving strong symmetric features.

### Performance Optimizations

#### Optimization #1: Reduce Redundancy (~1.5-2x speedup)
Moved `box_points` calculation outside the pixel loop. Previously computed once per pixel, now computed once per orientation/radius combination.

```cpp
// Build box_points ONCE per orientation/radius
std::vector<cv::Point> box_points;
// ... build box_points ...

// Then iterate over all pixels
for (int y = ...) {
    for (int x = ...) {
        for (const auto& offset : box_points) {  // Reuse same box_points
            // ...
        }
    }
}
```

#### Optimization #2: Parallelization (~3-4x speedup)
Added OpenMP parallelization across orientations (12 orientations can run on separate cores).

```cpp
#pragma omp parallel for schedule(dynamic) if(num_orientations >= 4)
for (int orientation = 0; orientation < num_orientations; ++orientation) {
    // ... compute contribution ...

    #pragma omp critical
    {
        radius_bands[radius_idx] += contribution;
    }
}
```

**Dependencies**:
- Added OpenMP to CMakeLists.txt
- Uses `schedule(dynamic)` for load balancing
- Critical section needed for accumulation

#### Optimization #3: Adaptive Scales
Use existing pyramid levels instead of resizing to fixed 64/128/256 sizes.

#### Optimization #4: Radius Step
Use step=2 instead of step=1 (tests every other radius) for ~2x speedup with minimal quality loss.

### Total Performance Improvement
- **Original correct implementation**: 13 seconds
- **After scale/radius optimization**: 1.4 seconds
- **After code optimizations**: 0.25 seconds
- **Total speedup**: 52x faster

### Implementation Files
- `include/attention/features/symmetry_feature.h` - Interface and configuration
- `src/features/symmetry_feature.cpp` - Core implementation
- `src/pipeline/attention_pipeline.cpp` - Pipeline integration with adaptive scale selection

### Core Algorithm Functions

#### `compute_radial_symmetry_at_scale()`
- Creates storage for each radius band
- Parallelizes across orientations
- For each orientation/radius: calls `compute_orientation_radius_contribution()`
- Applies threshold and radius consistency check
- Returns final symmetry map for this scale

#### `compute_orientation_radius_contribution()`
- Implements `symmetry_intern()` from thesis code
- Builds orthogonal summation box for given radius and orientation
- Samples Gabor responses in opposite directions
- Returns contribution map (normalized)

### Testing Results

**Test Image 1: Face (1918×1370)**
- Symmetric facial features (eyes, face structure) clearly highlighted
- Processing time: ~247 ms
- Scales: 240×172, 120×86, 60×43

**Test Image 2: Ball (input3.png)**
- Ball clearly visible in finest scale
- Coarser scales clean with minimal false positives
- Processing time: ~85 ms
- Scales: 128×128, 64×64, 32×32

## Future Work

### Pending Optimization: Integral Images
Could provide another 2-3x speedup for very large radii by making box summation O(1) instead of O(box_size).

**Current approach**: Iterate over all points in box and sum
**Integral image approach**: Use 4 lookups to get sum of any rectangle

Not implemented yet because current performance (~70-250ms) is already practical.

### Potential Tuning Parameters
If results need adjustment:
- **Threshold values**: Currently 0.3/0.5/0.65 - increase to suppress more, decrease to detect weaker symmetry
- **Radius ranges**: Currently 5-20/8-25/10-30 - adjust based on expected object sizes
- **Radius step**: Currently 2 - use 1 for highest quality (slower) or 3 for speed (may miss details)
- **Min radii above threshold**: Currently 2 - increase for stricter consistency requirement

## Important Context for Future Development

### Why Not Use Fixed 64/128/256 Sizes?
The thesis used these for performance in old C++ code. Modern approach: leverage existing pyramid infrastructure, avoiding redundant resizing.

### Why Continuous Radius Sampling?
Testing only discrete radii {6, 9, 12, 15} creates responses **only at those specific scales**. An object might be symmetric at radius 8 but we'd miss it. Continuous sampling (5, 7, 9, 11, ...) ensures we detect symmetry at any scale within the range.

### Why Scale-Dependent Thresholds?
At coarse pyramid levels (e.g., 32×32), more things appear symmetric due to loss of detail. Without higher thresholds, coarse scales dominate with false positives. The solution: require stronger evidence at coarser scales.

### Why Normalize Per-Scale Before Thresholding?
Raw Gabor responses have different magnitudes at different scales. Normalizing to [0,1] per-scale allows a single threshold value to be meaningful across scales.

## Commit Information

**Commit**: 25bcf34
**Date**: October 2025
**Summary**: Complete rewrite from statistical approach to geometric radius-sampling with 52x speedup and improved quality

## Dependencies
- OpenCV 4+ (for cv::Mat, cv::minMaxLoc, etc.)
- OpenMP (for parallelization) - optional but recommended
- Existing Gabor pyramid infrastructure in `core::Frame`
