# Phase 1 Action Plan - Minimal Working System (3-4 weeks)

## Goal
Build a minimal system that:
1. Loads a single image
2. Computes 2-3 basic features
3. Integrates them with simple weighting
4. Shows a saliency map visualization
5. Runs from command line with simple config

**Success Metric**: `./attention config.yaml` produces meaningful saliency maps

---

## Week-by-Week Breakdown

### **Week 1: Foundation & Setup**

#### **Session 1 (3-4 hours): Project Setup**
**Tasks:**
- [ ] Create new project directory structure
- [ ] Generate CMakeLists.txt with AI (OpenCV, Eigen, yaml-cpp)
- [ ] Setup vcpkg/Docker for dependencies
- [ ] Create basic `main.cpp` that loads an image with OpenCV
- [ ] Verify build works: `cmake .. && make && ./attention test.jpg`

**AI Prompts to Use:**
```
"Generate a CMakeLists.txt for a C++17 project that uses:
- OpenCV 4.x
- Eigen3
- yaml-cpp
- Builds an executable called 'attention'
- Uses vcpkg for package management"

"Create a basic main.cpp that loads an image using OpenCV
and displays it in a window"
```

**Deliverable**: Image loads and displays ✅

---

#### **Session 2 (3-4 hours): Core Data Structures**
**Tasks:**
- [ ] Create `include/attention/core/` directory
- [ ] Implement `Frame` struct (holds cv::Mat + metadata)
- [ ] Implement `FeatureMap` struct (name, cv::Mat data, confidence)
- [ ] Implement `SaliencyMap` struct (cv::Mat, peaks vector)
- [ ] Add simple unit tests (optional but recommended)

**AI Prompts:**
```
"Create a C++17 struct called Frame that wraps cv::Mat
with optional metadata. Should support move semantics."

"Create FeatureMap and SaliencyMap structs following modern C++ practices"
```

**Deliverable**: Core data structures defined ✅

---

#### **Session 3 (3-4 hours): Basic Pipeline Structure**
**Tasks:**
- [ ] Create `AttentionPipeline` class skeleton
- [ ] Add `load_image()`, `process()`, `visualize()` methods
- [ ] Wire up basic flow: load → process → visualize
- [ ] Test with dummy data (just pass image through)

**Code Skeleton:**
```cpp
class AttentionPipeline {
public:
    void load_image(const std::string& path);
    void process();
    void visualize();

    cv::Mat get_saliency_map() const;

private:
    Frame frame_;
    std::vector<FeatureMap> features_;
    SaliencyMap saliency_;
};
```

**Deliverable**: Pipeline structure in place ✅

---

### **Week 2: Feature Extraction**

#### **Session 1 (3-4 hours): Color Feature**
**Tasks:**
- [ ] Create `ColorFeature` class
- [ ] Implement opponent color computation (R-G, B-Y)
- [ ] Use OpenCV to extract color channels
- [ ] Compute color saliency using center-surround difference
- [ ] Visualize feature map to verify it works

**AI Prompts:**
```
"Implement a color saliency feature using OpenCV that:
1. Converts RGB to opponent color space (RG, BY)
2. Computes center-surround differences at 2-3 scales
3. Returns a normalized saliency map

Use cv::pyrDown for scale space, cv::GaussianBlur for smoothing"
```

**Reference**: Your old `src/feature/color.C` has the algorithm
**Deliverable**: Color feature produces reasonable output ✅

---

#### **Session 2 (3-4 hours): Intensity/Edge Feature**
**Tasks:**
- [ ] Create `IntensityFeature` class
- [ ] Convert to grayscale
- [ ] Compute intensity saliency (center-surround on intensity)
- [ ] OR use edge detection (Sobel/Canny) as proxy for visual saliency
- [ ] Test on several images

**AI Prompts:**
```
"Implement intensity-based saliency using:
1. Gaussian pyramid (cv::buildPyramid)
2. Center-surround difference at 3 scales
3. Normalize and combine scales
Return cv::Mat with float saliency values [0,1]"
```

**Deliverable**: Second feature working ✅

---

#### **Session 3 (3-4 hours): Integration & Testing**
**Tasks:**
- [ ] Integrate features into pipeline
- [ ] Add feature computation to `process()` method
- [ ] Test with multiple images
- [ ] Debug any issues
- [ ] Visualize individual feature maps

**Code Structure:**
```cpp
void AttentionPipeline::process() {
    // Extract features
    features_.push_back(compute_color_feature(frame_));
    features_.push_back(compute_intensity_feature(frame_));

    // For now, just store them (integrate next week)
}
```

**Deliverable**: Multiple features computed per image ✅

---

### **Week 3: Integration & Visualization**

#### **Session 1 (3-4 hours): Simple Integration**
**Tasks:**
- [ ] Implement weighted sum integration
- [ ] Combine feature maps: `saliency = w1*feat1 + w2*feat2`
- [ ] Normalize result to [0, 1]
- [ ] Make weights configurable

**Code:**
```cpp
SaliencyMap integrate_features(
    const std::vector<FeatureMap>& features,
    const std::vector<double>& weights)
{
    cv::Mat combined = cv::Mat::zeros(
        features[0].data.size(), CV_32F);

    for (size_t i = 0; i < features.size(); ++i) {
        combined += weights[i] * features[i].data;
    }

    cv::normalize(combined, combined, 0, 1, cv::NORM_MINMAX);

    SaliencyMap result;
    result.map = combined;
    // TODO: Find peaks
    return result;
}
```

**Deliverable**: Combined saliency map ✅

---

#### **Session 2 (3-4 hours): Visualization**
**Tasks:**
- [ ] Create heatmap overlay on original image
- [ ] Use `cv::applyColorMap()` for nice visualization
- [ ] Mark peak(s) with circles
- [ ] Side-by-side view: original | features | saliency
- [ ] Save output images

**AI Prompts:**
```
"Create a visualization function that:
1. Takes original image and saliency map
2. Creates heatmap using cv::applyColorMap(COLORMAP_JET)
3. Overlays on original with alpha blending
4. Marks top 3 saliency peaks with circles
Return combined visualization as cv::Mat"
```

**Deliverable**: Nice visualizations ✅

---

#### **Session 3 (3-4 hours): Peak Detection**
**Tasks:**
- [ ] Implement local maxima detection
- [ ] Use non-maximum suppression
- [ ] Find top N peaks
- [ ] Add to `SaliencyMap` structure

**Use OpenCV:**
```cpp
// Simple approach
cv::Point find_global_max(const cv::Mat& saliency) {
    cv::Point max_loc;
    cv::minMaxLoc(saliency, nullptr, nullptr,
                  nullptr, &max_loc);
    return max_loc;
}

// Better: use cv::dilate for non-max suppression
std::vector<cv::Point> find_local_maxima(
    const cv::Mat& saliency,
    int min_distance = 20)
{
    // Dilate and compare
    cv::Mat dilated;
    cv::dilate(saliency, dilated,
               cv::getStructuringElement(
                   cv::MORPH_ELLIPSE,
                   cv::Size(min_distance, min_distance)));

    cv::Mat peaks = (saliency == dilated) &
                    (saliency > 0.5); // threshold

    // Find non-zero points
    std::vector<cv::Point> locations;
    cv::findNonZero(peaks, locations);
    return locations;
}
```

**Deliverable**: Peak detection working ✅

---

### **Week 4: Configuration & Polish**

#### **Session 1 (3-4 hours): YAML Configuration**
**Tasks:**
- [ ] Create config structure
- [ ] Load from YAML file
- [ ] Configure: input image, feature weights, output path
- [ ] Update main.cpp to use config

**Example config.yaml:**
```yaml
input:
  image: "test_images/beach.jpg"

features:
  - name: "color"
    weight: 1.2

  - name: "intensity"
    weight: 0.8

output:
  save_features: true
  save_saliency: true
  output_dir: "results/"

visualization:
  show_windows: true
  colormap: "jet"
  alpha: 0.6
```

**AI Prompts:**
```
"Create a C++ class ConfigLoader that:
1. Uses yaml-cpp to load configuration
2. Parses input, features, output sections
3. Returns a Config struct with all settings
Include error handling for missing fields"
```

**Deliverable**: Config-driven execution ✅

---

#### **Session 2 (3-4 hours): Testing & Debugging**
**Tasks:**
- [ ] Test on 5-10 different images
- [ ] Fix any bugs discovered
- [ ] Tune feature weights for reasonable results
- [ ] Document what works and what doesn't
- [ ] Create test image collection

**Good Test Images:**
- Faces (should attend to faces)
- Text on background (should attend to text)
- Isolated objects (should attend to object)
- Symmetric objects (test for symmetry later)
- Cluttered scenes (test integration)

**Deliverable**: Working on diverse images ✅

---

#### **Session 3 (2-3 hours): Documentation & Cleanup**
**Tasks:**
- [ ] Write README.md with build instructions
- [ ] Add usage examples
- [ ] Comment tricky code sections
- [ ] Create TODO.md for Phase 2 features
- [ ] Clean up debug prints

**README Template:**
```markdown
# Attention Framework - Phase 1

## Build
```bash
mkdir build && cd build
cmake ..
make
```

## Usage
```bash
./attention config.yaml
```

## Current Features
- Color saliency (opponent colors)
- Intensity saliency (center-surround)
- Weighted integration
- Heatmap visualization

## Example Results
[Include screenshot]

## Next Steps
See TODO.md for Phase 2 plans
```

**Deliverable**: Documented Phase 1 system ✅

---

## End of Week 4: Decision Point

### **Evaluate Success**

**Green Light (Continue to Phase 2) if:**
- ✅ Saliency maps look reasonable on test images
- ✅ System runs reliably
- ✅ You understand the code and can modify it
- ✅ You're still motivated/interested
- ✅ Build process is smooth

**Yellow Light (Iterate on Phase 1) if:**
- ⚠️ Results are poor but fixable
- ⚠️ Some bugs remain but not blocking
- ⚠️ Need to understand algorithms better

**Red Light (Reconsider Project) if:**
- ❌ Can't get results that make sense
- ❌ Too many technical issues
- ❌ Lost interest
- ❌ Old code is "good enough"

---

## Troubleshooting Guide

### **Common Issues Week 1**

**Problem**: CMake can't find OpenCV
```bash
# Solution: Use vcpkg
vcpkg install opencv4 eigen3 yaml-cpp
cmake .. -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
```

**Problem**: Image won't load
```cpp
// Debug:
cv::Mat img = cv::imread("test.jpg");
if (img.empty()) {
    std::cerr << "Failed to load: " << path << std::endl;
    // Check: file exists? correct path? OpenCV built with image support?
}
```

---

### **Common Issues Week 2-3**

**Problem**: Feature maps are all black/white
```cpp
// Debug: Check value ranges
double min_val, max_val;
cv::minMaxLoc(feature_map, &min_val, &max_val);
std::cout << "Range: [" << min_val << ", " << max_val << "]" << std::endl;

// Normalize if needed
cv::normalize(feature_map, feature_map, 0, 1, cv::NORM_MINMAX);
```

**Problem**: Saliency map doesn't highlight obvious features
```cpp
// Debug: Visualize individual features
for (const auto& feat : features_) {
    cv::imshow(feat.name, feat.data);
}
cv::waitKey(0);

// Try different weights
// Disable one feature at a time to see impact
```

**Problem**: Crashes or segfaults
```cpp
// Common causes:
// 1. Size mismatch between features
assert(feat1.data.size() == feat2.data.size());

// 2. Wrong type
assert(feat.data.type() == CV_32F);

// 3. Empty mats
assert(!feat.data.empty());
```

---

## AI Assistance Strategy

### **When to Use AI**

✅ **Always:**
- Generating boilerplate code
- CMake configuration
- YAML parsing
- Documentation

✅ **Frequently:**
- OpenCV API questions ("How do I...?")
- Debugging compile errors
- Explaining error messages
- Code review/improvement suggestions

✅ **Sometimes:**
- Algorithm implementation (verify against papers/old code)
- Performance optimization
- Design decisions

❌ **Rarely:**
- Numerical debugging (AI struggles with "why does this diverge?")
- Architecture decisions (understand trade-offs yourself)

### **Effective AI Prompts**

**Good:**
```
"Show me how to use cv::pyrDown to create a 3-level
Gaussian pyramid from a cv::Mat image. Include error checking."
```

**Better:**
```
"I'm implementing center-surround saliency. I have:
- Input: cv::Mat (grayscale, CV_8U)
- Need: 3-scale pyramid with Gaussian blur
- Then: subtract across scales

Show me the OpenCV code with proper type conversions."
```

**Best:**
```
"Here's my current code for center-surround [paste code].
I'm getting [error/unexpected behavior].
The input image is [describe], output should be [describe].
What's wrong and how do I fix it?"
```

---

## Success Checklist

After Week 4, you should have:

- [ ] ✅ Git repository with clean commit history
- [ ] ✅ CMakeLists.txt that builds on your system
- [ ] ✅ `./attention config.yaml` runs successfully
- [ ] ✅ 2-3 features implemented and tested
- [ ] ✅ Saliency maps that make intuitive sense
- [ ] ✅ Visualization that's useful for debugging
- [ ] ✅ README explaining how to use it
- [ ] ✅ 5-10 test images with known good results
- [ ] ✅ TODO list for Phase 2
- [ ] ✅ Confidence you can extend the system

**Most Important**:
- [ ] ✅ You understand the code you wrote
- [ ] ✅ You can explain how it works
- [ ] ✅ You want to continue to Phase 2

---

## Next Steps After Phase 1

If Phase 1 succeeds, you have options:

**Option A: Continue to Phase 2** (Video + More Features)
- Add video input
- Add 2-3 more features (symmetry, motion)
- Estimated: 2-3 weeks

**Option B: Jump to Neural Fields** (Your old research)
- Skip more features for now
- Implement neural field integration
- Compare to weighted sum
- Estimated: 2-3 weeks

**Option C: Add Stereo** (If you have stereo data)
- Implement stereo depth feature
- 3D attention
- Estimated: 2-3 weeks

**Option D: Pause and Experiment**
- Use Phase 1 system for experiments
- Try different images, weights, combinations
- Write up findings
- Continue when needed

**Recommendation**: Option A or D
- Option A builds momentum
- Option D lets you validate Phase 1 is useful

---

## Weekly Time Tracking Template

Use this to track reality vs. plan:

```markdown
## Week 1
- Session 1 (Target: 3h): ___ hours - ✅/❌
  - What worked:
  - What didn't:
  - Blockers:

- Session 2 (Target: 3h): ___ hours - ✅/❌
  - What worked:
  - What didn't:
  - Blockers:

- Session 3 (Target: 3h): ___ hours - ✅/❌
  - What worked:
  - What didn't:
  - Blockers:

Total Week 1: ___ hours (Target: 9-12h)
Status: On track / Slightly behind / Behind / Ahead
```

This helps you adjust expectations and catch issues early.

---

## Final Prep Before Starting

1. **Block calendar time**: Schedule your 3 sessions/week now
2. **Prepare workspace**: Clear desk, close distractions
3. **Gather resources**:
   - Old dissertation code for reference
   - 5-10 test images ready
   - OpenCV documentation bookmarked
   - AI tools logged in
4. **Set up dev environment**:
   - IDE configured
   - Git repository initialized
   - First commit: empty project structure
5. **Mental preparation**: This is for learning/fun, not work. No pressure.

**Start Date**: _______________
**Target Completion**: _______________ (3-4 weeks later)

**Let's build something cool! 🚀**
