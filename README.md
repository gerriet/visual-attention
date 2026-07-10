# Visual Attention Framework

Modern C++ reimplementation of neural field-based visual attention system from doctoral dissertation (2003-2005).

## Current Status: v2 Phase

Development happens on the `v2` branch following `docs/V2_ROADMAP.md`
(milestones M0–M8, all complete: swappable architecture, neural-field
selection, stereo, motion/onset, the AttentionSystem symbolic second stage,
the modern-model comparison, and a live demonstrator).

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
- ✅ AttentionSystem: symbolic second stage — object files tracked across
  frames, Exploration behavior with dwell + object-based IOR, scanpaths
- ✅ Modern-track benchmark: modern saliency models (spectral residual,
  center-bias, DeepGaze slot) compared head-to-head with the thesis model
  (`docs/thesis_vs_modern.md`)
- ✅ Live demonstrator: real-time attention on webcam/video with object-file
  plugins running only on attended regions (the attention premise made visible)
- ✅ Pluggable alternative (non-thesis) saliency features from the 2007–2015
  literature — spectral residual, phase spectrum (PQFT-style, motion-aware on
  streams), frequency-tuned, image signature, boolean-map, minimum barrier —
  opt-in via config; the thesis set stays the default
  (`docs/ALTERNATIVE_FEATURES.md`)
- ✅ YAML configuration system
- ✅ Batch processing mode
- ✅ Golden regression tests (characterization + behavioral scanpath)
- ✅ Result interchange format (`docs/INTERCHANGE_FORMAT.md`)

**Performance** (Apple Silicon, 512×512 input, July 2026; single-image runs
include one-time setup):

- Default profile (5 features, NMS selection): ~480 ms
- Thesis profile (3 features, neural-field selection): ~125 ms
- Live profile, streaming: ~30 ms/frame at processing size 120,
  ~165 ms/frame at 480 — see `docs/PERFORMANCE.md` for the breakdown and
  the open M8 tuning gap

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

# Full attention system: object files + Exploration behavior + scanpath
./attention --attend ../data/test_images/motion_seq \
    --output ../results/attend --emit-scanpath ../results/scanpath.json

# Live demonstrator: attention + object-file plugins on a webcam (ESC to quit)
./attention --live 0 --config ../configs/live.yaml --processors region-descriptor
# ...or headless on a video / frame directory, saving annotated frames
./attention --live ../data/test_images/motion_seq --no-display --frames 30 --output ../results/live

# Use configuration file
./attention --config ../configs/default.yaml

# Run a profile on any image (thesis feature set + IOR, or the modern set)
./attention --config ../configs/thesis.yaml ../data/test_images/inputc.png --no-display
./attention --config ../configs/modern.yaml ../data/test_images/inputc.png --no-display

# Alternative (non-thesis) saliency operators, opt-in via config
./attention --config ../configs/alternative.yaml ../data/samples/images/soccer.jpg --no-display

# Emit results in the interchange format (JSON + 16-bit saliency PNG)
./attention ../data/test_images/input.png --no-display --emit-json out/result.json
```

Every binary answers `--help` (as do the Python tools in `tools/` and `eval/`).

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

- `docs/V3_ROADMAP.md` - **Direction (current): the science phase — replication, dynamic-IOR study, model lab, recognition, scenarios (M10–M16)**
- `docs/V2_ROADMAP.md` - v2 history: goals, locked decisions, milestones M0–M9
- `docs/ALTERNATIVE_FEATURES.md` - Non-thesis saliency features and how to plug them in
- `docs/INTERCHANGE_FORMAT.md` - Result/scanpath JSON + saliency-map format all models emit
- `docs/PERFORMANCE.md` - Timing instrumentation, optimization history, current numbers
- `docs/thesis_vs_modern.md` - The M7 deliverable: thesis model vs. modern saliency models
- `docs/MODERN_ARCHITECTURE.md` - System design and architecture
- `docs/MODERN_ATTENTION_RESEARCH.md` - Survey of modern attention systems
- `docs/SYMMETRY_FEATURE_NOTES.md` - Thesis-spec symmetry feature notes
- `docs/MIGRATION_GUIDE.md` - How to use old code as reference
- `docs/DEVELOPMENT_GUIDELINES.md` - **Development best practices and coding guidelines**
- `docs/CODE_STYLE.md` - Code style guide (Google-Allman hybrid)
- `docs/FORMATTING.md` - Code formatting with clang-format
- `docs/PHASE1_ACTION_PLAN.md`, `docs/REALISTIC_TIMELINE.md` - Phase-1 history (superseded by the roadmap)
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
├── eval/              # Python evaluation layer (comparators, metrics, models)
├── tools/             # Synthetic test-data generators (stereo pair, motion sequence)
└── examples/          # Usage examples (built with the project)
```

## Architecture

**Core Pipeline** (each stage is a config-selected strategy):

```
Frame stream → Pyramids/Gabor banks → Feature Extraction (parallel) →
Fusion → Selection (WTA or 2D/3D neural field) → Peaks/Clusters →
AttentionSystem second stage (object files → behavior → focus/scanpath)
```

**Key Design Decisions:**

- Stream-oriented and stateful: a single image is a stream of length one;
  field activity, IOR, and object files persist across frames (`RunState`)
- Features, fusion, selection, and object-file processors are registries —
  composition is fully YAML-driven (`configs/*.yaml`), no code changes to swap
- Shared pyramid and parameter-keyed Gabor-bank caches per frame
- Every model (C++ or Python) emits the same interchange format, so the
  evaluation harness never special-cases a model

## Development

### Current Progress

Phase 1 (2025, weeks 1–4): core infrastructure, five features, pipeline,
configuration, optimization — complete. See `docs/PHASE1_ACTION_PLAN.md`.

Phase 2 ("v2", 2026): see `docs/V2_ROADMAP.md` — guardrails (M1), swappable
architecture (M2), neural-field selection (M3: the dissertation's 2D Amari
field with two-stage cluster readout, plus per-feature Gabor banks), the
Python evaluation layer core (M4: metrics, scanpath comparison, report
generator — see `eval/README.md`), stereo + motion/onset (M5: the disparity
feature, onset/motion on the stream pipeline, and the 3D neural field), and
the AttentionSystem (M6: the symbolic second stage — object files tracked
across frames, the Exploration behavior with dwell and object-based IOR, and
scanpaths over a stream), the modern track (M7: modern saliency models as
interchange peers, a benchmark harness, and the "thesis, 20 years later"
comparison report — see `docs/thesis_vs_modern.md`), and the live demonstrator
(M8: real-time attention on webcam/video with object-file plugins running only
on attended regions) are done. The full v2 roadmap (M0–M8) is complete.

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
