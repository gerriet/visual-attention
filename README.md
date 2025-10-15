# Visual Attention Framework

Modern C++ reimplementation of neural field-based visual attention system from doctoral dissertation (2003-2005).

## Current Status: Week 4 Complete ✓

**Implemented Features:**

- ✅ Color attention (red-green, blue-yellow opponent channels)
- ✅ Intensity attention (center-surround contrast)
- ✅ Radial symmetry (gradient voting, Reisfeld et al. 1995)
- ✅ Multi-scale processing with cached pyramids
- ✅ Parallel feature extraction
- ✅ Winner-take-all peak detection with inhibition of return
- ✅ YAML configuration system
- ✅ Batch processing mode

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

# Use configuration file
./attention --config ../configs/default.yaml
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
├── docs/              # Documentation
├── reference/         # Old code for reference
├── data/              # Test images and expected outputs
├── configs/           # YAML configuration files
├── include/           # C++ headers
├── src/               # C++ implementation
├── tests/             # Unit tests
├── examples/          # Usage examples
└── tools/             # Utilities
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

**Week 1-3:** Core infrastructure, features, pipeline ✓
**Week 4:** Configuration system, optimization, batch processing ✓
**Week 5:** Additional features (orientation, motion) - *pending*
**Week 6+:** Neural field dynamics, 3D stereo attention - *pending*

See `docs/PHASE1_ACTION_PLAN.md` for detailed development phases.

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
