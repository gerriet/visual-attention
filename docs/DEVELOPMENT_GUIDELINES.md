# Development Guidelines for Attention Framework

## Core Principles

Readability, testability, maintainability, and changeability of code are (alongside actual functionality, correctness, and efficiency) central aspects of code quality.

## Structure

### Keep Units Small
- Limit lines per method and per class
- Limit number of methods per class
- Limit number of parameters per method
- Limit line length (max 120 characters)
- **Goal**: High cohesion, low coupling

**Recommended limits:**
- Functions: < 50 lines (ideally < 30)
- Classes: < 300 lines (ideally < 200)
- Parameters: < 5 per function (ideally < 3)
- Class methods: < 20 public methods

### Platform Independence
- Prefer language standards over specialized solutions
- Ideally test with multiple compilers/OS
- Use standard C++17 features consistently

### Scope and Visibility
- Minimize scope of local variables (declare close to use)
- Minimize visibility of class members (prefer `private`)
- Reduce global state as much as possible
- Document (unexpected) side effects

### Modern C++ Practices
- Use `const` whenever possible
- Consider inheritance vs. aggregation carefully
- Prefer references over pointers for parameters (unless nullable)
- Use complete includes
  - Order: local/specific first, general last (string, iostream)
- Use smart pointers instead of raw pointers
- Avoid pointer arithmetic
- Use STL containers

### "Adult Classes" Pattern
Classes should contain data **AND** offer methods that operate on that data:

❌ **Avoid**:
- Pure data classes with only getters/setters (use `struct` instead)
- Classes without data (use `namespace` instead)

✅ **Good**:
- Classes that encapsulate both state and behavior
- Initialize completely in constructor (consider error handling)

### Domain-Specific Types
Create domain-specific types instead of using built-in types everywhere:

❌ Bad: `double angle`
✅ Good: `Angle angle` (custom type)

This prevents bugs and makes intent clear.

### Documentation
- Document usage in headers (Doxygen format)
- Namespaces mirror directory structure

## What to Avoid

### Don't Use
- ❌ `using namespace` in header files
- ❌ Trick programming / clever hacks
- ❌ Excessive casts (use sparingly)
- ❌ Non-trivial literals without comments
- ❌ Code duplication (use local variables for repeated constructs)
- ❌ Uninitialized variables
- ❌ Macros (use `const`, `constexpr`, templates instead)
- ❌ Raw pointers (use smart pointers or STL containers)
- ❌ Pointer arithmetic
- ❌ Manual memory management (`new`/`delete`)

### Reduce Side Effects
Strive for functional programming where possible:
- Pure functions that don't modify state
- Const-correctness
- Immutability where appropriate

## Error Handling

**Principle**: Use **exceptions** for handling unexpected situations.

❌ **Don't**:
```cpp
bool load_image(const std::string& path) {
    // Returns false on error - forces caller to check
    if (!file_exists(path)) return false;
    // ...
}
```

✅ **Do**:
```cpp
void load_image(const std::string& path) {
    // Throws exception - errors propagate naturally
    if (!file_exists(path)) {
        throw std::runtime_error("Image not found: " + path);
    }
    // ...
}
```

**When to use exceptions:**
- File not found
- Invalid input
- Resource allocation failure
- Precondition violations

**When NOT to use exceptions:**
- Expected conditions (e.g., end of file)
- Performance-critical inner loops
- Cross-API boundaries (C libraries)

## Naming Conventions

### Language
- **English only** for all identifiers and comments

### Philosophy
- Good names reduce need for comments
- Focus on **what** (meaning/purpose), not **how** (implementation)
- No type prefixes (`p`, `b`) or member prefixes (`m_`)

### Style

Following Google C++ Style Guide with Allman braces:

| Element | Convention | Example |
|---------|-----------|---------|
| Classes/Structs | PascalCase | `FeatureMap`, `SaliencyMap` |
| Functions | snake_case | `compute_saliency()`, `find_peaks()` |
| Variables | snake_case | `image_path`, `peak_count` |
| Constants | kPascalCase | `kMaxIterations`, `kDefaultWeight` |
| Namespaces | snake_case | `attention::core` |
| Member variables | trailing `_` | `member_data_`, `frame_number_` |
| Macros | UPPER_CASE | `MAX_FEATURES` (avoid if possible) |

**Rationale for trailing underscore:**
- Distinguishes members from local variables
- No visual clutter at the start
- Google C++ style convention

**Examples:**
```cpp
namespace attention {
namespace core {

const int kDefaultBufferSize = 1024;

class FeatureExtractor
{
public:
  FeatureExtractor(int feature_count);

  FeatureMap extract_features(const Frame& input_frame);

private:
  int feature_count_;
  std::vector<float> weights_;
  bool is_initialized_;
};

} // namespace core
} // namespace attention
```

### Special Cases

**Abbreviations:**
Use underscore to separate abbreviations:
- `compute_rgb_histogram()` (not `computeRGBHistogram()`)
- `load_xml_config()` (not `loadXMLConfig()`)

**Boolean variables:**
Use question-like names:
- `is_valid`, `has_data`, `can_process`

## Comments

### Focus
- Purpose and usage (the **why**)
- Constraints and preconditions
- Special design decisions
- Edge cases and gotchas

### Don't Comment
- What is directly obvious from code
- Commented-out code (use version control instead)
- Redundant information

### Header Documentation

Use Doxygen format for public APIs:

```cpp
/**
 * Compute saliency map from feature maps.
 *
 * @param features Vector of input feature maps (must be non-empty)
 * @param weights Per-feature weights (must match features size)
 * @return Integrated saliency map normalized to [0, 1]
 * @throws std::invalid_argument if features is empty or sizes mismatch
 *
 * The integration uses weighted summation followed by normalization.
 * Features are expected to be pre-normalized to [0, 1].
 */
SaliencyMap integrate_features(
    const std::vector<FeatureMap>& features,
    const std::vector<double>& weights);
```

## File Organization

### File Names
- **snake_case** (lowercase with underscores)
- English
- Extensions: `.cpp` and `.h`
- Example: `feature_extractor.h`, `feature_extractor.cpp`

### Include Order

From specific to general:

1. **Corresponding header** (in .cpp files)
2. **Project headers** (`"attention/..."`)
3. **Third-party libraries** (`<opencv2/...>`)
4. **C++ standard library** (`<iostream>`, `<vector>`)
5. **C library headers** (`<cmath>`, `<cstdlib>`)

**Example:**
```cpp
// In feature_extractor.cpp
#include "attention/features/feature_extractor.h"  // Own header first

#include "attention/core/frame.h"                  // Project headers
#include "attention/core/feature_map.h"

#include <opencv2/opencv.hpp>                      // Third-party

#include <algorithm>                               // C++ stdlib
#include <vector>

#include <cmath>                                   // C stdlib
```

### Directory Structure
Use directories for logical organization:
```
include/attention/
├── core/           # Core data structures
├── features/       # Feature extraction
├── visualization/  # Visualization utilities
└── integration/    # Feature integration
```

Namespaces mirror directory structure.

## Best Practices

### Use More Of
- ✅ Unit tests (Google Test)
- ✅ Refactoring (continuous improvement)
- ✅ Code reviews
- ✅ Static analysis (cppcheck)
- ✅ Documentation (Doxygen)

### Class Organization

**Preferred order:**
```cpp
class MyClass
{
public:
  // Public interface first (most important for users)
  MyClass();
  void public_method();

protected:
  // Protected interface (for derived classes)
  void protected_method();

private:
  // Private implementation (least important for users)
  void private_method();
  int member_data_;
};
```

**Rationale:** Users care about public interface first.

### Memory Management

**Modern C++ hierarchy (prefer top):**
1. ✅ **Stack allocation** - `FeatureMap feature;`
2. ✅ **STL containers** - `std::vector<FeatureMap> features;`
3. ✅ **Smart pointers** - `std::unique_ptr<FeatureMap>`
4. ⚠️ **Raw pointers** - Only for non-owning references
5. ❌ **`new`/`delete`** - Avoid in modern C++

**Example:**
```cpp
// Bad: manual memory management
FeatureMap* create_feature() {
    return new FeatureMap(...);  // Who deletes this?
}

// Good: smart pointer
std::unique_ptr<FeatureMap> create_feature() {
    return std::make_unique<FeatureMap>(...);
}

// Better: return by value (move semantics)
FeatureMap create_feature() {
    return FeatureMap(...);  // Efficient with move
}
```

## Code Style

See [CODE_STYLE.md](CODE_STYLE.md) for complete formatting rules.

**Summary:**
- Based on [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Brace style: [Allman](https://en.wikipedia.org/wiki/Indentation_style#Allman_style)
- No brace-less blocks (always use braces)
- Max line length: 120 characters
- Indentation: 2 spaces
- Spaces around operators and after commas
- Pointer/reference aligned to type: `int* ptr`, `const Frame& frame`
- See `.clang-format` for complete configuration

**Automatic formatting:**
```bash
cmake --build build --target format
```

## Recommended Tools

- **Build system**: CMake
- **Package manager**: vcpkg (or Conan)
- **Testing**: Google Test
- **Documentation**: Doxygen
- **Static analysis**: cppcheck
- **Libraries**: OpenCV, Eigen3, yaml-cpp
- **Formatting**: clang-format
- **Version control**: Git

## References

Essential reading for writing quality C++ code:

- [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [Modern C++ best practices](https://github.com/cpp-best-practices/cppbestpractices)

## Quick Checklist

Before committing code, verify:

- [ ] ✅ Runs through `clang-format`
- [ ] ✅ No compiler warnings
- [ ] ✅ Doxygen comments on public APIs
- [ ] ✅ Unit tests pass
- [ ] ✅ Const-correctness enforced
- [ ] ✅ No raw pointers (unless non-owning)
- [ ] ✅ No `using namespace` in headers
- [ ] ✅ Includes are complete and ordered
- [ ] ✅ Functions < 50 lines
- [ ] ✅ Classes < 300 lines
- [ ] ✅ Good names that explain intent

---

**Remember**: Code is read far more often than it is written. Optimize for readability!
