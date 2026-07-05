# Visual Attention Framework

Modern C++ reimplementation of neural field-based visual attention system from doctoral dissertation (2003-2005).

## Current Status: v2 Phase

Development happens on the `v2` branch following `docs/V2_ROADMAP.md`
(milestones M0–M7: swappable architecture, neural-field selection, stereo,
motion, ESAB2 system level, modern learned-model comparison).

**Implemented Features:**

- ✅ Color attention (red-green, blue-yellow opponent channels)
- ✅ Intensity attention (center-surround contrast)
- ✅ Orientation (Gabor filter pyramids, Itti-Koch style)
- ✅ Symmetry (Gabor-based radial symmetry, per thesis specification)
- ✅ Eccentricity (region segmentation + moments)
- ✅ Stereo/disparity (depth saliency from a rectified pair, thesis §5.4)
- ✅ Onset/motion (rectified temporal change of edge energy, on the stream)
- ✅ Multi-scale processing with cached pyramids
- ✅ Parallel feature extraction
- ✅ Winner-take-all peak detection with inhibition of return
- ✅ Neural-field selection (2D Amari dynamics from the dissertation, with
  cluster/Objectfile readout and decaying spatial IOR)
- ✅ 3D neural-field selection (depth-aware, cross-depth inhibition, thesis §6.4)
- ✅ YAML configuration system
- ✅ Batch processing mode
- ✅ Golden regression tests (characterization + behavioral scanpath)
- ✅ Result interchange format (`docs/INTERCHANGE_FORMAT.md`)

**Performance:**

- 512×512 image: ~210ms
- 1436×2011 image: ~3000ms
- Optimized with shared pyramids and parallel extraction

## Quick Start

### Build

```bash
mkdir build && cd build
cmake ..
make
```

### Usage

```bash
# Process single image (display results)
./attention ../data/test_images/lena.png

# Process without display (save to results/)
./attention ../data/test_images/lena.png --no-display

# Batch process directory
./attention --batch ../data/test_images/ --output ../results

# Stereo pair (adds a disparity/depth channel; 3D field with the stereo config)
./attention --stereo ../data/test_images/stereo/left.png ../data/test_images/stereo/right.png \
    --config ../configs/stereo.yaml --no-display

# Temporal sequence (directory of frames or a video) — onset/motion
./attention --sequence ../data/test_images/motion_seq --output ../results/seq

# Use configuration file
./attention --config ../configs/default.yaml

# Run a profile on any image (thesis feature set + IOR, or the modern set)
./attention --config ../configs/thesis.yaml ../data/test_images/inputc.png --no-display
./attention --config ../configs/modern.yaml ../data/test_images/inputc.png --no-display

# Emit results in the interchange format (JSON + 16-bit saliency PNG)
./attention ../data/test_images/input.png --no-display --emit-json out/result.json
```

### Tests

```bash
ctest --test-dir build
```

Two layers, both against golden data in `tests/golden/`:

- **Characterization** (C++/Catch2): feature and saliency maps within tolerance —
  refactor tripwires, not ground truth.
- **Behavioral** (CLI + `eval/compare_scanpaths.py`): fixation sequence matches
  within loose position/order tolerance — the replication bar of the project.

After an *intentional* algorithm change, review and regenerate:

```bash
ATTENTION_UPDATE_GOLDEN=1 ./build/tests/characterization_tests
```

## Documentation

- `docs/MODERN_ARCHITECTURE.md` - System design and architecture
- `docs/PHASE1_ACTION_PLAN.md` - Week-by-week implementation plan
- `docs/REALISTIC_TIMELINE.md` - Timeline and effort estimates
- `docs/MODERN_ATTENTION_RESEARCH.md` - Survey of modern attention systems
- `docs/MIGRATION_GUIDE.md` - How to use old code as reference
- `docs/DEVELOPMENT_GUIDELINES.md` - **Development best practices and coding guidelines**
- `docs/CODE_STYLE.md` - Code style guide (Google-Allman hybrid)
- `docs/FORMATTING.md` - Code formatting with clang-format
- `docs/thesis/` - Original dissertation and extracted equations
- `reference/old_code/` - Original implementation (reference only, not compiled)

## Project Structure

```text
attention-framework/
├── docs/              # Documentation (see docs/V2_ROADMAP.md for direction)
├── reference/         # Original thesis code (reference only, not compiled)
├── data/              # Test images
├── configs/           # YAML configuration files
├── include/           # C++ headers
├── src/               # C++ implementation
├── tests/             # Golden regression tests (Catch2 + CTest)
├── eval/              # Python evaluation layer (comparators, metrics)
└── examples/          # Usage examples (built with the project)
```

## Architecture

**Core Pipeline:**

```
Input Image → Multi-scale Pyramids → Parallel Feature Extraction →
Feature Integration → Winner-Take-All → Attention Peaks
```

**Key Design Decisions:**

- Shared pyramid computation (eliminates redundancy)
- Parallel feature extraction (3 threads for color images)
- Cached data structures for efficiency
- YAML-based configuration for flexibility

## Development

### Current Progress

Phase 1 (2025, weeks 1–4): core infrastructure, five features, pipeline,
configuration, optimization — complete. See `docs/PHASE1_ACTION_PLAN.md`.

Phase 2 ("v2", 2026): see `docs/V2_ROADMAP.md` — guardrails (M1), swappable
architecture (M2), neural-field selection (M3: the dissertation's 2D Amari
field with two-stage cluster readout, plus per-feature Gabor banks), the
Python evaluation layer core (M4: metrics, scanpath comparison, report
generator — see `eval/README.md`), and stereo + motion/onset (M5: the
disparity feature, onset/motion on the stream pipeline, and the 3D
neural field) are done. Next up: the ESAB2 system level (M6).

### Code Formatting

This project uses clang-format with automatic formatting on commit:

```bash
# Format all code manually
cmake --build build --target format

# Check formatting (CI/CD)
cmake --build build --target format-check
```

See `docs/FORMATTING.md` and `docs/DEVELOPMENT_GUIDELINES.md` for coding standards.

## License

[Add your license here]

## Citation

If you use this work, please cite the original dissertation:

- Backer, G. 2004. Modellierung visueller Aufmerksamkeit im Computer-Sehen: Ein zweistufiges Selektionsmodell für ein Aktives Sehsystem. Ph.D. thesis, Universität Hamburg, Germany.
