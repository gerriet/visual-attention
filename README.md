# Visual Attention Framework

Modern reimplementation of neural field-based visual attention system.

Based on doctoral dissertation work (2003-2005), reimplemented with modern C++ and OpenCV.

## Features

- Multi-feature visual attention (color, intensity, stereo depth, symmetry)
- Neural field dynamics for feature integration
- 3D attention with stereo vision
- Configurable via YAML
- Python bindings for analysis

## Quick Start

### Build

```bash
mkdir build && cd build
cmake ..
make
./attention ../configs/phase1_simple.yaml
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

```
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

## Development

See `docs/PHASE1_ACTION_PLAN.md` for the phased development approach.

### Code Formatting

This project uses clang-format for automatic code formatting:

```bash
# Format all code
cmake --build build --target format

# Check formatting (CI/CD)
cmake --build build --target format-check
```

A pre-commit hook automatically formats code before each commit. See `docs/FORMATTING.md` for details.

**Phase 1 (3-4 weeks)**: Minimal working system
- Core data structures
- 2-3 basic features
- Simple integration
- Visualization

**Phase 2 (2-3 weeks)**: Video + more features
**Phase 3 (2-3 weeks)**: Neural field integration
**Phase 4 (2-3 weeks)**: Stereo/3D attention

## License

[Add your license here]

## Citation

If you use this work, please cite the original dissertation:
[Add citation information]
