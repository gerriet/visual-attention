# Migration Guide - What to Take to New Project

## TL;DR

**Essential:**
- ✅ All 4 markdown files I created
- ✅ Your thesis PDF (algorithms, equations, parameters)
- ✅ Selected old source code (reference only)
- ✅ Test images/videos if you have them

**DON'T migrate:**
- ❌ Don't copy-paste old code into new project
- ❌ Don't try to compile old code as-is
- ❌ Don't bring dependencies (jtools, old Makefiles)

**Strategy:** Reference old code for algorithms, rewrite in modern C++

---

## 1. Markdown Files (Essential)

### **Take ALL of these:**

```bash
new-project/
├── docs/
│   ├── CLAUDE.md                      # Architecture overview of old system
│   ├── CODE_ASSESSMENT.md             # What was good/bad in old code
│   ├── MODERN_ARCHITECTURE.md         # Design for new system
│   ├── REALISTIC_TIMELINE.md          # Timeline estimates
│   ├── PHASE1_ACTION_PLAN.md         # Week-by-week plan
│   └── MODERN_ATTENTION_RESEARCH.md  # Field overview, baselines
```

**Why**: These are your roadmap. Reference them when:
- Designing new architecture
- Understanding what old system did
- Comparing to modern approaches
- Planning work sessions

---

## 2. Thesis/Dissertation (Critical)

### **What to extract:**

✅ **The PDF** - Full thesis document
- Algorithms with equations
- Parameter values that worked
- Experimental results
- Figures showing expected outputs

✅ **Key sections to bookmark:**
1. **Neural field equations** (Chapter on NF dynamics)
   - Differential equations
   - Kernel definitions (DoG parameters)
   - Integration method
   - Convergence criteria

2. **Feature extraction details**
   - Color space conversions (RGB → opponent colors)
   - Symmetry computation (orientations, scales)
   - Stereo matching (disparity range, correlation method)
   - Eccentricity (exact parameters)

3. **Integration weights**
   - Feature weights that produced good results
   - Normalization methods
   - Temporal smoothing parameters

4. **Evaluation results**
   - Example images that worked well
   - Parameter sensitivities
   - Performance benchmarks

### **How to organize:**

```bash
new-project/
├── docs/
│   ├── thesis/
│   │   ├── thesis.pdf                    # Full PDF
│   │   ├── neural_field_equations.md     # Extracted equations
│   │   ├── feature_parameters.md         # Working parameters
│   │   └── test_results/                 # Screenshots from thesis
│   │       ├── figure_3_2_symmetry.png
│   │       ├── figure_4_5_stereo.png
│   │       └── ...
```

**Action item:** Extract key equations to markdown for easy reference:

```markdown
# Neural Field 2D - Equations from Thesis

## Activation Update (Equation 3.14)
```
τ * dA(x,y,t)/dt = -A(x,y,t) + ρ + h + I(x,y,t) + ∫∫ w(x',y') * σ(A(x',y',t)) dx'dy'
```

Where:
- τ = time constant (typically 10ms)
- ρ = resting level (-0.25)
- h = global inhibition (0.0)
- I = external input (feature map)
- w = lateral interaction kernel (DoG)
- σ = sigmoid function: σ(x) = 1/(1 + exp(-β*x)), β=30

## Kernel Definition (Equation 3.18)
```
w(x,y) = A_exc * exp(-(x²+y²)/(2*σ_exc²)) - A_inh * exp(-(x²+y²)/(2*σ_inh²))
```

Parameters that worked:
- σ_exc = 2.0
- σ_inh = 10.0
- A_exc = 1.5
- A_inh = 0.5
```

---

## 3. Old Source Code (Reference Only)

### **What to take:**

**Strategy:** Create a `reference/` directory with selected files, but DON'T integrate into new codebase.

```bash
new-project/
├── reference/
│   ├── old_code/
│   │   ├── README.md          # "This is reference only, not compiled"
│   │   ├── nf2d.h             # Neural field 2D header
│   │   ├── nf3d.h             # Neural field 3D header
│   │   ├── feature/
│   │   │   ├── color.h        # Color feature algorithms
│   │   │   ├── stereo.h       # Stereo feature
│   │   │   ├── symmetry.h     # Symmetry detection
│   │   │   └── eccentricity.h # Eccentricity
│   │   ├── attention.h        # Base feature interface
│   │   ├── esab2.h           # Main system (architecture reference)
│   │   └── nf_sample.h       # Test trajectories, helper functions
```

### **Specifically useful files:**

#### **From `include/`:**

1. **`nf2d.h`** - Neural field 2D implementation
   - Algorithm structure
   - Parameter definitions
   - Kernel setup methods

2. **`nf3d.h`** - Neural field 3D (your unique contribution)
   - 3D extension of dynamics
   - Depth layer handling

3. **`attention.h`** - Feature interface
   - Good abstraction to copy
   - Thread handling pattern

4. **`feature/color.h`** - Color saliency
   - RGB → Munsell conversion
   - Segmentation algorithm
   - Exclusivity computation

5. **`feature/stereo.h`** - Stereo depth
   - Gabor filter orientations
   - Correlation method
   - Disparity-to-saliency conversion

6. **`feature/symmetry.h`** - Symmetry detection
   - Scale space approach
   - Orientation handling

7. **`feature/eccentricity.h`** - Eccentricity
   - Region growing
   - Shape analysis

8. **`nf_sample.h`** - Useful utilities
   - `draw_gauss()` - Gaussian drawing (for synthetic tests)
   - `normal()` - Gaussian function
   - `object_trajectory` - For generating test data

#### **From `src/`:**

9. **`esab2.C`** (first ~200 lines)
   - Feature initialization with parameters
   - Integration setup
   - Good reference for configuration

10. **`pictureop.h/C`** - Image operations
    - May have useful utilities
    - Check for anything not in OpenCV

### **What to SKIP:**

❌ **Don't take:**
- `Makefile` (obsolete build system)
- `.depend` (outdated dependencies)
- `jt/` symlink (entire jtools library - use OpenCV instead)
- `*_cviuAnimationC` (obsolete visualization)
- `test.C`, `videorecorder.C` (one-off tools)
- Compiled binaries (`a_sample`, `k_sample`)
- `samples/` directory (old test programs)

---

## 4. Test Data

### **If you have it, take:**

✅ **Input images** that produced good results
```bash
new-project/
├── data/
│   ├── test_images/
│   │   ├── beach.jpg           # Color test
│   │   ├── face.jpg            # Should attend to face
│   │   ├── text_scene.jpg      # Text detection
│   │   └── symmetric_object.jpg
│   ├── stereo_pairs/
│   │   ├── left_001.jpg
│   │   ├── right_001.jpg
│   │   └── ...
│   └── videos/
│       └── moving_objects.mp4
```

✅ **Expected outputs** (from thesis figures)
```bash
├── data/
│   └── expected_outputs/
│       ├── beach_saliency.png        # What it should look like
│       ├── face_attention.png
│       └── ...
```

**Why:** Use as regression tests. If new implementation produces similar results, you're on track.

### **If you DON'T have test data:**

Use standard datasets:
- PASCAL VOC (objects)
- COCO (complex scenes)
- MIT1003 (eye-tracking, for validation)
- Middlebury Stereo (for stereo feature)

---

## 5. Configuration/Parameters

### **Extract working configurations:**

From your old code, find parameter sets that worked:

```yaml
# Create: new-project/configs/classic_parameters.yaml
# These are from the dissertation, verified to work

neural_field_2d:
  alpha: 0.5              # From thesis Chapter 3
  beta: 30.0
  rest_value: -0.25
  kernel:
    sigma_exc: 2.0
    sigma_inh: 10.0
    A_exc: 1.5
    A_inh: 0.5
  convergence:
    max_iterations: 50
    threshold: 0.01

features:
  color:
    weight: 1.2           # From readargs.h:89-92
    num_scales: 3
    threshold: 8.0        # From esab2.C:93

  intensity:
    weight: 0.35
    num_scales: 3

  symmetry:
    weight: 1.6
    num_orientations: 6
    scales: [1, 2, 4]

  stereo:
    weight: 0.7
    min_disparity: -25    # From your experiments
    max_disparity: 0
    block_size: 15
```

**Source files to check:**
- `readargs.h` lines 85-95 (default weights)
- `esab2.C` lines 60-150 (feature initialization)
- Thesis appendix (parameter tables)

---

## 6. What NOT to Take

### **Don't bring technical debt:**

❌ **jtools library**
- Custom `pic2d<T>` → Use `cv::Mat`
- Custom `matrix<T>` → Use `Eigen::Matrix`
- Custom threading → Use `std::thread`
- Custom file I/O → Use OpenCV `imread/imwrite`

❌ **Platform-specific code**
- Xilcam (Sun XIL library - obsolete)
- ORB client (obsolete simulator)
- V4L (old Video4Linux) → Use OpenCV `VideoCapture`

❌ **German comments**
- Translate if keeping algorithms
- But better: write new comments in English

❌ **Hardcoded paths**
- `/tmp/` everywhere
- Specific file locations

❌ **Build artifacts**
- `.o` files, executables
- `.depend` file

---

## 7. Recommended File Structure for New Project

```bash
attention-framework/
├── README.md
├── CMakeLists.txt
├── .gitignore
│
├── docs/                           # All documentation
│   ├── CLAUDE.md
│   ├── CODE_ASSESSMENT.md
│   ├── MODERN_ARCHITECTURE.md
│   ├── REALISTIC_TIMELINE.md
│   ├── PHASE1_ACTION_PLAN.md
│   ├── MODERN_ATTENTION_RESEARCH.md
│   ├── thesis/
│   │   ├── thesis.pdf
│   │   ├── neural_field_equations.md
│   │   └── feature_parameters.md
│   └── progress/                   # Your dev journal
│       └── weekly_log.md
│
├── reference/                      # Old code (not compiled)
│   ├── old_code/
│   │   ├── README.md              # "Reference only"
│   │   ├── nf2d.h
│   │   ├── nf3d.h
│   │   └── feature/
│   │       ├── color.h
│   │       ├── stereo.h
│   │       └── ...
│   └── algorithms.md              # Extracted algorithms in pseudocode
│
├── data/                           # Test data
│   ├── test_images/
│   ├── expected_outputs/
│   └── README.md                  # Data sources, licenses
│
├── configs/                        # Configuration files
│   ├── classic_parameters.yaml    # From dissertation
│   ├── phase1_simple.yaml
│   └── development.yaml
│
├── include/                        # New C++ headers
│   └── attention/
│       ├── core/
│       ├── features/
│       └── ...
│
├── src/                            # New C++ implementation
│   ├── core/
│   ├── features/
│   └── ...
│
├── tests/                          # Unit tests
│   └── ...
│
├── examples/                       # Example usage
│   └── simple_attention.cpp
│
└── tools/                          # Utilities
    └── extract_equations.py       # Script to extract from thesis
```

---

## 8. Extraction Script

### **Automated helper:**

Create a script to copy relevant files:

```bash
#!/bin/bash
# migrate_files.sh

OLD_DIR="$HOME/source/diss/att"
NEW_DIR="$HOME/projects/attention-framework"

echo "Migrating files from old project..."

# Create structure
mkdir -p "$NEW_DIR"/{docs,reference/old_code,data,configs}

# Copy documentation
cp "$OLD_DIR"/*.md "$NEW_DIR/docs/"

# Copy selected headers (reference only)
cp "$OLD_DIR/include/nf2d.h" "$NEW_DIR/reference/old_code/"
cp "$OLD_DIR/include/nf3d.h" "$NEW_DIR/reference/old_code/"
cp "$OLD_DIR/include/attention.h" "$NEW_DIR/reference/old_code/"

mkdir -p "$NEW_DIR/reference/old_code/feature"
cp "$OLD_DIR/include/feature/color.h" "$NEW_DIR/reference/old_code/feature/"
cp "$OLD_DIR/include/feature/stereo.h" "$NEW_DIR/reference/old_code/feature/"
cp "$OLD_DIR/include/feature/symmetry.h" "$NEW_DIR/reference/old_code/feature/"
cp "$OLD_DIR/include/feature/eccentricity.h" "$NEW_DIR/reference/old_code/feature/"

cp "$OLD_DIR/include/nf_sample.h" "$NEW_DIR/reference/old_code/"

# Copy test images if they exist
if [ -d "$OLD_DIR/test_images" ]; then
    cp -r "$OLD_DIR/test_images" "$NEW_DIR/data/"
fi

# Create README in reference
cat > "$NEW_DIR/reference/old_code/README.md" << 'EOF'
# Old Code Reference

**DO NOT COMPILE THESE FILES**

This directory contains selected files from the 2003-2005 dissertation code
for reference only. Use them to understand algorithms and parameters, but
reimplement in modern C++.

## Key Files

- `nf2d.h`, `nf3d.h` - Neural field implementations
- `feature/*.h` - Feature extraction algorithms
- `attention.h` - Feature interface pattern
- `nf_sample.h` - Utility functions for testing

## How to Use

1. Read algorithm in old file
2. Check corresponding equations in ../docs/thesis/
3. Implement in modern C++ with OpenCV
4. Test against expected outputs in ../../data/

Do NOT copy-paste this code. It uses:
- Deprecated C++ (strstream, old casts)
- Custom jtools library (use OpenCV instead)
- GCC extensions that no longer compile
EOF

echo "Migration complete!"
echo "Review $NEW_DIR and remove anything you don't need."
```

**Run it:**
```bash
chmod +x migrate_files.sh
./migrate_files.sh
```

---

## 9. How to Use Old Code as Reference

### **Workflow for each algorithm:**

#### **Example: Implementing Neural Field 2D**

**Step 1: Read old code**
```bash
# Open reference file
code reference/old_code/nf2d.h
```

**Step 2: Extract algorithm to pseudocode**
```markdown
## Neural Field 2D Update (from nf2d.h:32-40)

```
for iteration in 1..N:
    1. Compute sigmoid of activity
       sigmoid = 1 / (1 + exp(-beta * activity))

    2. Convolve sigmoid with kernel
       lateral = convolve2d(sigmoid, kernel)

    3. Update activity
       delta = -alpha * activity + global * lateral + input
       activity += delta + rest_value

    4. Check convergence
       if norm(delta) < threshold: break
```

Parameters:
- alpha = 0.5 (line 17)
- beta = 30.0 (line 64)
- rest = -0.25
```

**Step 3: Cross-reference with thesis**
- Find equation 3.14 in thesis
- Verify parameters match
- Check any special cases

**Step 4: Implement in modern C++**
```cpp
// src/integrators/neural_field_2d.cpp
// Based on nf2d.h but using OpenCV

void NeuralField2D::update(const cv::Mat& input, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // 1. Sigmoid
        cv::Mat sigmoid;
        compute_sigmoid(activity_, sigmoid);

        // 2. Convolution
        cv::Mat lateral;
        cv::filter2D(sigmoid, lateral, CV_32F, kernel_);

        // 3. Update
        cv::Mat delta = -alpha_ * activity_
                      + global_mult_ * lateral
                      + input;
        activity_ += delta;
        cv::add(activity_, cv::Scalar(rest_value_), activity_);

        // 4. Convergence
        if (cv::norm(delta, cv::NORM_L1) < convergence_threshold_)
            break;
    }
}
```

**Step 5: Test against expected output**
```cpp
// Compare to thesis Figure 3.7
cv::Mat result = nf.get_saliency();
cv::imshow("Should match Figure 3.7", result);
```

---

## 10. Gradual Migration Strategy

### **Don't migrate everything at once!**

**Phase 1 (Week 1-4):** New core only
- Implement from scratch
- Reference old code for algorithms
- Don't copy any old code

**Phase 2 (Week 5-8):** Add features
- Check old feature code for parameters
- Reimplement with OpenCV
- Test: Do results match thesis figures?

**Phase 3 (Week 9+):** Neural fields
- This is the tricky part
- Old code is invaluable here
- Line-by-line comparison for debugging

**Phase 4 (Later):** Advanced features
- 3D neural field (unique to your work)
- Active vision
- Reference old code heavily

---

## 11. Quick Start Checklist

Before starting Phase 1:

- [ ] Create new project directory
- [ ] Copy all 6 markdown files to `docs/`
- [ ] Copy thesis PDF to `docs/thesis/`
- [ ] Run migration script (or manually copy reference files)
- [ ] Create `reference/old_code/README.md` warning not to compile
- [ ] Extract key equations from thesis to markdown
- [ ] Create `configs/classic_parameters.yaml` with working params
- [ ] Collect 5-10 test images
- [ ] Find or screenshot thesis figures for expected outputs
- [ ] Set up git repository
- [ ] First commit: "Initial structure with documentation"

**Then start Phase 1 (new code from scratch)**

---

## 12. Final Recommendations

### **DO:**
✅ Keep old code as **reference**
✅ Extract **algorithms** and **parameters**
✅ Use thesis for **equations** and **expected results**
✅ Reimplement in **modern C++** with **OpenCV**
✅ Test against **thesis figures**

### **DON'T:**
❌ Copy-paste old code
❌ Try to compile old code
❌ Bring jtools or other dependencies
❌ Port line-by-line (understand first, then rewrite)

### **Golden Rule:**
**"Reference, don't migrate"**

The old code is a **design document**, not a **starting point**.

Think of it like rebuilding a house:
- Keep the blueprints (algorithms, equations)
- Keep photos (expected results, thesis figures)
- Demolish the structure (don't port old code)
- Build new foundation (modern C++, OpenCV)
- Follow the design (same algorithms, new implementation)

---

## 13. One-Time Setup Script

```bash
#!/bin/bash
# setup_new_project.sh

set -e

PROJECT_NAME="attention-framework"
OLD_DIR="$HOME/source/diss/att"
NEW_DIR="$HOME/projects/$PROJECT_NAME"

echo "Setting up new project: $PROJECT_NAME"

# Create structure
mkdir -p "$NEW_DIR"/{docs/{thesis,progress},reference/old_code/feature,data/{test_images,expected_outputs},configs,include/attention,src,tests,examples,tools}

# Copy docs
echo "Copying documentation..."
cp "$OLD_DIR"/{CLAUDE,CODE_ASSESSMENT,MODERN_ARCHITECTURE,REALISTIC_TIMELINE,PHASE1_ACTION_PLAN,MODERN_ATTENTION_RESEARCH}.md "$NEW_DIR/docs/" 2>/dev/null || true

# Copy selected reference files
echo "Copying reference code..."
for file in nf2d.h nf3d.h attention.h nf_sample.h; do
    [ -f "$OLD_DIR/include/$file" ] && cp "$OLD_DIR/include/$file" "$NEW_DIR/reference/old_code/"
done

for file in color.h stereo.h symmetry.h eccentricity.h; do
    [ -f "$OLD_DIR/include/feature/$file" ] && cp "$OLD_DIR/include/feature/$file" "$NEW_DIR/reference/old_code/feature/"
done

# Create README files
cat > "$NEW_DIR/README.md" << 'EOF'
# Visual Attention Framework

Modern reimplementation of neural field-based visual attention system.

## Quick Start

See `docs/PHASE1_ACTION_PLAN.md` for development plan.

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Documentation

- `docs/MODERN_ARCHITECTURE.md` - System design
- `docs/PHASE1_ACTION_PLAN.md` - Implementation plan
- `docs/thesis/` - Original research
- `reference/old_code/` - Reference implementation (not compiled)
EOF

cat > "$NEW_DIR/reference/old_code/README.md" << 'EOF'
# Reference Code - DO NOT COMPILE

Original 2003-2005 implementation for reference only.

Use to understand algorithms, then reimplement in modern C++.
EOF

cat > "$NEW_DIR/.gitignore" << 'EOF'
build/
*.o
*.a
*.so
.vscode/
.idea/
__pycache__/
*.pyc
EOF

# Initialize git
cd "$NEW_DIR"
git init
git add .
git commit -m "Initial project structure"

echo "✅ Project setup complete: $NEW_DIR"
echo ""
echo "Next steps:"
echo "1. Copy your thesis PDF to docs/thesis/"
echo "2. Add test images to data/test_images/"
echo "3. Review docs/PHASE1_ACTION_PLAN.md"
echo "4. Start coding!"
```

**Run it once, you're ready to go:**
```bash
chmod +x setup_new_project.sh
./setup_new_project.sh
```

---

## Bottom Line

**Take with you:**
1. ✅ All markdown files (your roadmap)
2. ✅ Thesis PDF (algorithms, equations, results)
3. ✅ Selected old headers (reference, not compiled)
4. ✅ Test data (if available)
5. ✅ Working parameters (extracted to YAML)

**Leave behind:**
- Everything else (dependencies, build system, executables)

**Mindset:**
- Old code is a **specification**, not a starting point
- Understand algorithm → Reimplement modern
- Use for **verification**, not **migration**

You're building a **new system inspired by** the old one, not **porting** the old one.

Good luck! 🚀
