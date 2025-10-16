# Feature Extraction Debugging Guide

This document explains how to use the debugging capabilities in the attention framework to inspect intermediate computation steps during feature extraction.

## Overview

The debugging system allows you to capture and visualize intermediate results from feature extraction algorithms. This is useful for:

- **Understanding algorithm behavior**: See how each processing step contributes to the final result
- **Debugging extraction issues**: Identify where computations go wrong
- **Visualizing multi-scale processing**: Inspect pyramid levels and center-surround differences
- **Academic presentations**: Generate figures showing internal algorithm states
- **Performance analysis**: Measure timing of each computation step

## Debug Levels

The `DebugContext` supports four levels of verbosity:

### `Level::None` (0)
- No debug output
- Default mode for production use
- Zero performance overhead

### `Level::Basic` (1)
- Captures major intermediate results
- Examples for ColorFeature:
  - RGB pyramid
  - RG and BY opponent color pyramids
  - Combined result before final resize

### `Level::Detailed` (2)
- Captures all significant intermediate steps
- Examples for ColorFeature:
  - Everything from Basic level
  - RG and BY channels at full resolution (level 0)
  - Individual RG and BY center-surround results
  - Timing information for each stage

### `Level::Verbose` (3)
- Maximum detail (currently equivalent to Detailed)
- Reserved for future fine-grained debugging

## Usage

### 1. Programmatic Usage

```cpp
#include "attention/features/debug_context.h"
#include "attention/features/color_feature.h"

// Create debug context with desired level
attention::features::DebugContext debug(
    attention::features::DebugContext::Level::Detailed
);

// Create feature extractor
attention::features::ColorFeature color_extractor;

// Extract with debugging enabled
attention::core::FeatureMap result = color_extractor.extract(frame, debug);

// Print debug information to console
attention::features::DebugVisualizer::print_debug_info(debug, "ColorFeature");

// Save all debug images to directory
attention::features::DebugVisualizer::save_debug_images(
    debug,
    "debug_output",  // output directory
    "color"          // filename prefix
);

// Create combined visualization
cv::Mat viz = attention::features::DebugVisualizer::create_debug_visualization(debug);
cv::imwrite("debug_combined.png", viz);
```

### 2. Command-Line Usage

The main `attention` executable supports debug flags:

```bash
# Basic debugging - saves intermediate images
./attention input.jpg --debug

# Detailed debugging with specific output directory
./attention input.jpg --debug=detailed --debug-output=my_debug_dir

# Debug specific features only
./attention input.jpg --debug --debug-features=color,intensity
```

### 3. Building and Running Test Program

A simple test program is included to demonstrate debugging:

```bash
# Build
mkdir -p build && cd build
cmake ..
cmake --build .

# Run test
./test_debug ../data/test_images/lena.png
```

This will:
1. Extract color features with detailed debugging
2. Print timing and annotation information
3. Save individual debug images to `debug_output/`
4. Create a combined visualization

## Debug Output Structure

When you call `save_debug_images()`, the following structure is created:

```
debug_output/
├── color_0_rg_channel_level0.png          # RG opponent color at full res
├── color_1_by_channel_level0.png          # BY opponent color at full res
├── color_2_rg_center_surround.png         # RG center-surround result
├── color_3_by_center_surround.png         # BY center-surround result
├── color_4_combined_before_resize.png     # Combined before final resize
├── color_pyramid_rgb_pyramid.png          # RGB multi-scale pyramid
├── color_pyramid_rg_pyramid.png           # RG opponent color pyramid
├── color_pyramid_by_pyramid.png           # BY opponent color pyramid
└── color_debug_combined.png               # All images in one grid
```

## Captured Data Types

The `DebugContext` captures four types of data:

### Images
Single 2D images at various processing stages.

```cpp
debug.add_image("my_intermediate_result", cv::Mat);
```

### Pyramids
Multi-scale image pyramids.

```cpp
debug.add_pyramid("my_pyramid", std::vector<cv::Mat>);
```

### Annotations
Text metadata about the computation.

```cpp
debug.add_annotation("num_levels", "9");
debug.add_annotation("output_size", "512x512");
```

### Timings
Performance measurements in milliseconds.

```cpp
debug.add_timing("opponent_color_computation", 15.3);
debug.add_timing("center_surround_computation", 42.7);
```

## Example Output

When you run with `Level::Detailed` on the ColorFeature, you'll see console output like:

```
=== Debug Info: ColorFeature ===

Annotations:
  pyramid_levels: 9
  output_size: 512x512

Timings:
  opponent_color_computation        :  15.34 ms
  center_surround_computation       :  42.73 ms
  normalize_and_resize              :   3.21 ms
  total_time                        :  61.28 ms

Captured Data:
  Images: 5
  Pyramids: 3

  Image details:
    rg_channel_level0                 : 512x512 channels=1
    by_channel_level0                 : 512x512 channels=1
    rg_center_surround                : 32x32 channels=1
    by_center_surround                : 32x32 channels=1
    combined_before_resize            : 32x32 channels=1

  Pyramid details:
    rgb_pyramid                       : 9 levels
    rg_pyramid                        : 9 levels
    by_pyramid                        : 9 levels
===========================
```

## Feature-Specific Debug Output

### ColorFeature

**Basic Level:**
- RGB pyramid (from frame)
- RG opponent color pyramid
- BY opponent color pyramid
- Combined result before resize

**Detailed Level:**
- All Basic outputs
- RG channel at full resolution (level 0)
- BY channel at full resolution (level 0)
- Individual RG center-surround result
- Individual BY center-surround result
- Detailed timing for each stage

### IntensityFeature

(Similar structure to ColorFeature, capturing grayscale pyramid and center-surround differences)

### OrientationFeature

**Basic Level:**
- Gabor pyramid (all orientations)
- Per-orientation saliency maps

**Detailed Level:**
- Individual Gabor responses for each orientation
- Center-surround differences per orientation
- Combined orientation map

### EccentricityFeature

**Basic Level:**
- Edge detection result
- Watershed segmentation
- Eccentricity map

**Detailed Level:**
- Distance transform
- Watershed markers
- Individual segment eccentricities

### SymmetryFeature

**Basic Level:**
- Gabor responses at computed scale
- Bilateral symmetry map
- Radial symmetry map

**Detailed Level:**
- Individual orientation responses
- Symmetry components before combination

## Performance Considerations

Debug output has minimal performance impact when disabled:
- `Level::None`: Zero overhead (all checks are compile-time eliminated)
- `Level::Basic`: ~5-10% overhead (image cloning)
- `Level::Detailed`: ~10-20% overhead (more clones + timing)
- `Level::Verbose`: ~15-25% overhead (maximum data capture)

For production use, always use `Level::None` (the default).

## Advanced Usage

### Custom Debug Context

You can create custom debug visualizations:

```cpp
// Create your own debug context
attention::features::DebugContext my_debug(Level::Basic);

// Extract features
auto result = extractor.extract(frame, my_debug);

// Access captured data directly
for (const auto& [name, image] : my_debug.images)
{
  // Do custom processing
  cv::Mat processed = my_custom_processing(image);
  cv::imwrite("custom_" + name + ".png", processed);
}

// Access pyramids
for (const auto& [name, pyramid] : my_debug.pyramids)
{
  // Visualize specific pyramid levels
  for (size_t i = 0; i < pyramid.size(); ++i)
  {
    cv::imshow("Level " + std::to_string(i), pyramid[i]);
  }
}
```

### Conditional Debugging

```cpp
// Enable debugging only for specific conditions
attention::features::DebugContext debug;

if (image_is_problematic)
{
  debug.level = DebugContext::Level::Verbose;
  debug.enabled = true;
}

auto result = extractor.extract(frame, debug);

if (debug.enabled)
{
  DebugVisualizer::save_debug_images(debug, "problem_cases", filename);
}
```

## Integration with Pipeline

The `AttentionPipeline` can be configured to enable debugging for all features:

```cpp
pipeline.enable_feature_debugging(DebugContext::Level::Basic);
pipeline.process(image);

// Access debug contexts for each feature
auto debug_data = pipeline.get_debug_data();
for (const auto& [feature_name, debug_context] : debug_data)
{
  DebugVisualizer::print_debug_info(debug_context, feature_name);
}
```

## Troubleshooting

### No images saved
- Check that the output directory is writable
- Verify debug level is not `None`
- Ensure `debug.enabled == true`

### Missing pyramid visualizations
- Some features don't use pyramids (e.g., if computed at single scale)
- Check debug level is at least `Basic`

### Performance degradation
- Reduce debug level
- Disable debugging for features you don't need to inspect
- Save images less frequently

## See Also

- `include/attention/features/debug_context.h` - Full API documentation
- `src/features/debug_context.cpp` - Implementation details
- `test_debug.cpp` - Complete working example
- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system design

