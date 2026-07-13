# Performance Optimization

This document describes performance measurement, analysis, and optimization techniques applied to the attention framework.

> **Note:** the sections up to "Conclusion" document the Phase-1 (2025)
> optimization pass over the original five-feature pipeline. For measured
> numbers on the current v2 pipeline (profiles, neural-field selection, live
> streaming), see [Current performance (v2)](#current-performance-v2-2026-07)
> at the end.

## Performance Measurement

### Per-Feature Timing

The pipeline tracks detailed timing for each stage:

```cpp
const auto& timing = pipeline.get_timing();
timing.pyramid_ms        // Pyramid computation
timing.feature_ms        // Map of feature name -> time (ms)
timing.integration_ms    // Feature integration
timing.peak_detection_ms // Peak detection
timing.total_ms()        // Total time
```

### Batch Mode Timing Export

In batch processing mode, timing information is automatically saved to `timing.txt` in each result directory:

```text
Performance Timing (ms)
======================
Image size: 512x512

Pyramid computation: 0 ms
Feature 'color': 1 ms
Feature 'intensity': 0 ms
Feature 'symmetry': 5 ms
Integration: 0 ms
Peak detection: 12 ms

Total: 18 ms
```

Usage:

```bash
./attention --batch ../data/test_images/ --output ../results
cat ../results/input/timing.txt
```

## Performance Analysis

### Timing Breakdown (Before Optimization)

On a 512×512 test image:

```text
Pyramid computation:  4 ms   (1.8%)
Feature 'color':      2 ms   (0.9%)
Feature 'intensity':  1 ms   (0.5%)
Feature 'symmetry': 198 ms  (91.2%) ← BOTTLENECK
Integration:          0 ms   (0%)
Peak detection:      12 ms   (5.5%)
Total:              217 ms
```

On a 1436×2011 image:

```text
Pyramid computation:   10 ms   (0.3%)
Feature 'color':       26 ms   (0.9%)
Feature 'intensity':    3 ms   (0.1%)
Feature 'symmetry':  2718 ms  (94.8%) ← BOTTLENECK
Integration:            3 ms   (0.1%)
Peak detection:       132 ms   (4.6%)
Total:               2892 ms
```

**Key Insight:** Symmetry feature dominates processing time (90-95%)

### Symmetry Computational Complexity

The radial symmetry algorithm:

```text
For each pixel (x, y):
  For each direction (forward, backward):
    For distance d = 1 to max_radius:
      Accumulate vote
```

Complexity: **O(width × height × max_radius)**

For 512×512 with max_radius=128: ~33M operations
For 1436×2011 with max_radius=500: ~1.4B operations

## Optimization Strategies Implemented

### 1. Shared Pyramid Computation ✓

**Problem:** Each feature was computing its own Gaussian pyramid independently.

**Solution:** Compute pyramids once in the Frame structure, share across all features.

**Implementation:**

```cpp
// In AttentionPipeline::process()
frame_.compute_pyramids(pyramid_levels);  // Compute once

// In each feature
const auto& gray_pyramid = frame_.gray_pyramid;  // Reuse
```

**Impact:**

- Small images (512×512): 82ms → 57ms (30% faster)
- Large images (1436×2011): 479ms → 394ms (18% faster)

### 2. Parallel Feature Extraction ✓

**Problem:** Features were computed sequentially.

**Solution:** Extract features in parallel using std::thread (3 threads for color images).

**Implementation:**

```cpp
std::vector<std::thread> threads;
for (size_t i = 0; i < extractors.size(); ++i) {
  threads.emplace_back([this, i, &extractors]() {
    features_[i] = extractors[i]->extract(frame_);
  });
}
for (auto& thread : threads) {
  thread.join();
}
```

**Impact:**

- Modest speedup (~15%) due to OpenCV internal threading
- Enables per-feature timing measurement
- Better CPU utilization

### 3. Reduced Resolution for Expensive Features ✓

**Problem:** Symmetry computation at full resolution is extremely slow.

**Solution:** Compute symmetry at reduced resolution (pyramid scale 2 = quarter resolution).

**Rationale:**

- Symmetry is a coarse, global feature (faces, objects)
- Fine details not critical for attention
- Upsampling preserves spatial structure
- Reduces computation by 16× (4× width, 4× height)

**Implementation:**

```cpp
struct Config {
  int compute_at_scale; // 0=full, 1=half, 2=quarter, 3=eighth
};

// In SymmetryFeature::extract()
if (config_.compute_at_scale > 0) {
  int scale_index = std::min(config_.compute_at_scale,
                             static_cast<int>(frame.gray_pyramid.size()) - 1);
  const auto& level = frame.gray_pyramid[scale_index];
  cv::Mat sym = compute_radial_symmetry(level);
  result = normalize_and_resize(sym, frame.size());
}

// In AttentionPipeline::extract_features()
// Adaptive resolution: quarter res for large images (>640px)
if (frame_.width() > 640 || frame_.height() > 640) {
  sym_config.compute_at_scale = 2; // Quarter resolution
} else {
  sym_config.compute_at_scale = 0; // Full resolution
}
```

**Impact:**

The system uses **adaptive resolution** based on image size:

- Images ≤640px: Full resolution (best quality)
- Images >640px: Quarter resolution (best performance)

Small images (256×256, full res):

```text
Before: Symmetry  38ms, Total  43ms
After:  Symmetry  33ms, Total  37ms
Speedup: 1.2× (minimal, already fast)
```

Medium images (512×512, full res):

```text
Symmetry: 194ms, Total 210ms
Quality: Full detail preserved
Use case: Standard images, balanced approach
```

Large images (1436×2011, quarter res):

```text
Before: Symmetry 2718ms, Total 2892ms
After:  Symmetry   52ms, Total  202ms
Speedup: 52× for symmetry, 14× overall
Quality: Excellent (coarse features preserved)
```

**Quality Assessment:**

- ✓ Face detection preserved
- ✓ Symmetric regions still highlighted
- ✓ Less noise (better generalization)
- ✓ Suitable for attention (coarse features)

## Current Performance (Optimized)

### Timing Breakdown (After All Optimizations)

256×256 image (small, full res):

```text
Pyramid computation:  1 ms   (2.7%)
Feature 'color':      0 ms   (0%)
Feature 'intensity':  0 ms   (0%)
Feature 'symmetry':  33 ms  (89.2%)
Integration:          0 ms   (0%)
Peak detection:       3 ms   (8.1%)
Total:               37 ms
```

512×512 image (medium, full res):

```text
Pyramid computation:  2 ms   (1.0%)
Feature 'color':      2 ms   (1.0%)
Feature 'intensity':  0 ms   (0%)
Feature 'symmetry': 194 ms  (92.4%)
Integration:          0 ms   (0%)
Peak detection:      12 ms   (5.7%)
Total:              210 ms
```

1436×2011 image (large, quarter res):

```text
Pyramid computation:  12 ms   (5.9%)
Feature 'color':      27 ms  (13.4%)
Feature 'intensity':   4 ms   (2.0%)
Feature 'symmetry':   52 ms  (25.7%)
Integration:           3 ms   (1.5%)
Peak detection:      133 ms  (65.8%)
Total:               202 ms
```

**Note:** Adaptive resolution keeps symmetry from dominating on large images.

### Performance Summary Table

| Image Size | Resolution | Before | After | Speedup |
|------------|-----------|--------|-------|---------|
| 256×256    | Full      | 38ms   | 37ms  | 1.0×    |
| 512×512    | Full      | 217ms  | 210ms | 1.0×    |
| 895×1100   | Quarter   | 1949ms | ~100ms| 19.5×   |
| 1436×2011  | Quarter   | 2892ms | 202ms | 14.3×   |
| 1918×1370  | Quarter   | 2386ms | ~200ms| 11.9×   |

**Speedup for large images (>640px): ~15×**
**Note:** Small/medium images maintain full quality at original speed.

## Optimization Recommendations

### Completed ✓

1. ✓ **Shared pyramid computation** - Eliminated redundant computation
2. ✓ **Parallel feature extraction** - Better CPU utilization
3. ✓ **Per-feature timing** - Identified bottlenecks
4. ✓ **Reduced resolution for symmetry** - 40-50× speedup

### Future Opportunities

1. **Configurable resolution per feature**
   - Make `compute_at_scale` configurable via YAML
   - Allow different scales for different images/scenarios
   - Trade quality vs speed based on application

2. **GPU acceleration**
   - OpenCV CUDA backend for feature extraction
   - Most beneficial for large images
   - Requires CUDA-capable hardware

3. **Adaptive processing**
   - Detect static regions (video processing)
   - Only recompute changed areas
   - Region-of-interest processing

4. **Algorithm optimizations**
   - Early termination in symmetry voting
   - Spatial hashing for vote accumulation
   - SIMD vectorization for inner loops

5. **Multi-scale integration optimization**
   - Currently combines all scales
   - Could use single scale or fewer scales
   - Balance quality vs performance

## Best Practices

### For Development

1. **Profile before optimizing**: Use per-feature timing to identify bottlenecks
2. **Batch testing**: Use `--batch` mode to test across multiple images
3. **Quality assessment**: Always check visual results after optimization
4. **Document tradeoffs**: Record quality vs performance decisions

### For Deployment

1. **Adaptive resolution (default)**:
   - Automatically uses full res for ≤640px
   - Automatically uses quarter res for >640px
   - Balances quality and performance
   - No configuration needed

2. **Manual override** (if needed):

   ```cpp
   features::SymmetryFeature::Config sym_config;
   sym_config.compute_at_scale = 0; // Force full resolution
   // or
   sym_config.compute_at_scale = 2; // Force quarter resolution
   ```

3. **Monitor timing**:
   ```cpp
   const auto& timing = pipeline.get_timing();
   if (timing.total_ms() > threshold) {
     // Adjust quality settings
   }
   ```

4. **Batch processing**:
   - Use `--output` to organize results
   - Check `timing.txt` files for performance analysis
   - Identify problematic images

## Measurement Tools

### Command-line

```bash
# Single image with timing
./attention image.png --no-display

# Batch with timing files
./attention --batch images/ --output results/
grep "Total:" results/*/timing.txt

# Time multiple runs
for i in {1..10}; do
  ./attention image.png --no-display 2>&1 | grep "Processing time"
done | awk '{sum+=$3; count++} END {print "Average:", sum/count, "ms"}'
```

### Programmatic

```cpp
AttentionPipeline pipeline(config);
pipeline.load_image("test.png");
pipeline.process();

const auto& timing = pipeline.get_timing();
std::cout << "Pyramid: " << timing.pyramid_ms << "ms\n";
for (const auto& [name, ms] : timing.feature_ms) {
  std::cout << "Feature '" << name << "': " << ms << "ms\n";
}
std::cout << "Total: " << timing.total_ms() << "ms\n";
```

## Conclusion

Through systematic profiling and targeted optimization:

- **15× average speedup** on large images (>640px)
- **Full quality preserved** on small/medium images (≤640px)
- **Adaptive resolution**: Automatically balances quality vs performance
- **Real-time capable**: 512×512 in 210ms (~5 FPS), large images in ~200ms

The key insights:

1. Identified symmetry as the bottleneck (90-95% of time)
2. Applied reduced-resolution computation for large images only
3. Maintained full quality for typical image sizes
4. Achieved 15× speedup on large images with minimal quality loss

## Current performance (v2, 2026-07)

Measured on Apple Silicon (Release build), a 512×512 test image.
Single-image runs include one-time setup (Gabor-bank construction, pyramid
allocation); streaming numbers are steady-state wall clock per frame over a
30-frame run, including the annotated-PNG write in headless mode.

| Configuration | Time |
|---------------|------|
| Default profile (5 features, NMS selection), single image | ~480 ms |
| Thesis profile (color/eccentricity/symmetry, 2D neural field), single image | ~125 ms |
| Live profile, `--live` streaming, processing size 480 | ~165 ms/frame (~6 fps) |
| Live profile, `--live` streaming, processing size 240 | ~113 ms/frame (~9 fps) |
| Live profile, `--live` streaming, processing size 120 | ~30 ms/frame (~30 fps) |

### Known gap: untimed per-frame work (M8 tuning)

The per-stage timers (`pipeline.get_timing()`) account for only ~23 ms of the
~155 ms live frame at processing size 480. The remainder lives in stages the
timing struct does not cover:

- `Frame` construction and the per-frame rebuild of the parameter-keyed
  Gabor banks (cached per `Frame`, so a fresh frame rebuilds them)
- system-level work: saliency segmentation into clusters, object-file
  correspondence, behavior selection
- overlay drawing and (headless) PNG writes

Consequence: the M8 target of 25+ fps at VGA processing size does not hold
yet — it currently requires processing size ≈120. The first candidates are
caching Gabor banks across frames (they depend only on parameters, not frame
content) and extending the timing struct to cover the untimed stages so the
gap is visible per run.
