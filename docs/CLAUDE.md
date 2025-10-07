# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a legacy C++ visual attention research codebase from a PhD dissertation (circa 2003-2005). The code implements various visual attention models using neural fields and feature extraction. The codebase has not been updated in over 20 years and uses outdated C++ standards and libraries.

## Key Dependencies

- **jtools (jt)**: Custom C++ class library (symlinked as `./jt -> ../jtools/`)
- **vigra**: Vision with Generic Algorithms library
- **Standard libraries**: libjpeg, libpng, pthread
- **Architecture**: Originally designed for Pentium 3 with x86 optimizations

## Build System

### Build Commands

```bash
# Build main applications (k_sample and a_sample)
make

# Build individual applications
make k_sample    # Koch et al. attention model sample
make a_sample    # General attention model sample
make nf_sample   # Neural field sample (not built by default)

# Update dependencies
make dep

# Generate documentation
make doc

# Clean build artifacts
rm -rf *.o src/*.o src/feature/*.o lib/libgtools.a
```

### Build Process

The Makefile automatically:
1. Compiles all `.C` files in `src/` directory into `lib/libgtools.a`
2. For `a_sample`, runs `./enumsamples` script before compilation to generate sample enumerations
3. Links against vigra, jtools, jpeg, png, and pthread libraries
4. Uses `-O2` optimization with Pentium 3-specific flags (`-march=pentium3`)

## Architecture

### Core Components

1. **Neural Fields** (`include/nf*.h`, `src/nf*.C`)
   - `NeuralField3D`: 3D neural field implementation for depth-based attention
   - `NeuralField2D`: Classic 2D neural field
   - `NeuralField2DSystem`: System of multiple 2D neural fields for feature integration

2. **Attention System** (`include/esab2.h`, `src/esab2.C`)
   - `ESAB2`: Main attention system class
   - Integrates multiple feature maps through neural field dynamics
   - Supports different input modes: camera (Xilcam, V4L), files, simulator
   - Action modes: `feature_mode`, `move_sensor_mode`, `scanpath_mode`

3. **Feature Extraction** (`include/feature/*.h`, `src/feature/*.C`)
   - `AttentionFeature`: Base class for all features
   - Feature types: color, stereo (depth), symmetry, eccentricity
   - Each feature can run in its own thread

4. **Image Handling**
   - `pic2d<T>`: 2D images (from jtools)
   - `pic3d<T>`: 3D volumes (local implementation)
   - Supports grayscale and color (3-plane) images

5. **Object Tracking** (`include/objectfile.h`, `src/objectfile.C`)
   - Maintains object files across frames
   - Tracks object identities and generates scanpaths
   - Computes similarities between frames

### Sample Applications

- **k_sample.C**: Koch et al. classic attention model implementation
- **a_sample.C**: Test harness that runs various samples from `samples/` directory
- **nf_sample.C**: Neural field testing with artificial trajectories
- **samples/**: Collection of feature-specific test programs

### Data Flow

1. Input acquisition (camera/files) → grayscale/color images
2. Feature computation (parallel threads) → feature maps
3. Feature integration → master saliency map
4. Neural field dynamics → attention allocation
5. Object file generation → tracking/scanpath
6. Optional: sensor movement based on selected focus

## Historical Context

- Written in early 2000s C++ (pre-C++11)
- Uses deprecated features: `strstream`, old-style casts
- Platform-specific: Linux/Unix (some references to Solaris)
- German comments throughout ("Merkmale", "Verhalten", etc.)
- File extension: `.C` (capital C) for implementation files
- Uses custom minmax operators (`>?`, `<?`) - GCC extensions removed in later versions
