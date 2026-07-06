# Reference Code - DO NOT COMPILE

⚠️ **This code is for reference only. It will NOT compile with modern tools.**

This directory contains selected files from the original 2003-2005 dissertation
implementation. Use them to understand algorithms and parameters, then
reimplement in modern C++.

## What's Here

### Core Neural Field Implementation
- `nf2d.h` - 2D neural field dynamics
- `nf3d.h` - 3D neural field (unique contribution)
- `nf.h` - Base neural field class

### Feature Extraction
- `feature/color.h` - Color saliency (opponent colors)
- `feature/stereo.h` - Stereo depth saliency
- `feature/symmetry.h` - Symmetry detection
- `feature/eccentricity.h` - Eccentricity feature

### System Architecture
- `attention.h` - Feature interface pattern
- `esab2.C` - Main system initialization (see parameter values)
- `readargs.h` - Default parameter values

### Utilities
- `nf_sample.h` - Helper functions for testing (draw_gauss, etc.)

## How to Use

1. **Read** algorithm in old file
2. **Check** corresponding equations in `../../docs/thesis/`
3. **Understand** the approach
4. **Reimplement** in modern C++ using OpenCV
5. **Test** against expected outputs in `../../data/expected_outputs/`

## DO NOT

- ❌ Copy-paste this code into the new project
- ❌ Try to compile these files
- ❌ Port the jtools dependencies
- ❌ Use deprecated C++ features

## DO

- ✅ Extract algorithms and convert to pseudocode
- ✅ Note parameter values that worked
- ✅ Understand the mathematical approach
- ✅ Use as specification for modern reimplementation

## Why This Code Won't Compile

- Uses `strstream` (removed in C++17)
- Uses GCC-specific operators `>?` and `<?`
- Depends on custom jtools library
- Uses deprecated threading APIs
- Platform-specific (XIL, old V4L)
- No longer maintained dependencies

## Reading Notes (Historical Context)

- Written in early-2000s C++ (pre-C++11) for Linux/Solaris on a Pentium 3;
  the original build used a Makefile linking jtools, vigra, libjpeg/png.
- Implementation files use the `.C` extension (capital C).
- Comments are largely German ("Merkmale" = features, "Verhalten" = behavior).
- Images are `pic2d<T>`/`pic3d<T>` from jtools (grayscale or 3-plane color).
- `esab2.C` is the original system level (ESAB = DFG project "Entwicklung von
  Systembausteinen der Aktiven Bildanalyse"); v2 reimplements it as
  `AttentionSystem`. Its input modes (Xilcam/V4L camera, files, simulator)
  and action modes (`feature_mode`, `move_sensor_mode`, `scanpath_mode`) map
  to the v2 CLI's `--sequence`/`--attend`/`--live` and the ActionMode enum.

## Example Workflow

```bash
# 1. Read old algorithm
cat reference/old_code/nf2d.h

# 2. Extract to pseudocode
# ... document in docs/algorithms.md

# 3. Check thesis equations
# ... verify in docs/thesis/thesis.pdf

# 4. Implement modern version
# ... write src/integrators/neural_field_2d.cpp using OpenCV

# 5. Test
# ... compare output to data/expected_outputs/
```

Think of this as a **design specification**, not a **code base**.
