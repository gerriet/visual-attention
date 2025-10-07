# Code Style Guide

## Overview

This project uses a **hybrid Google-Allman style** for C++ code formatting, enforced via `.clang-format`.

For comprehensive development guidelines including best practices, naming conventions, and code organization, see [DEVELOPMENT_GUIDELINES.md](DEVELOPMENT_GUIDELINES.md). This document focuses specifically on formatting rules.

## Style Philosophy

We combine:
- **Google C++ Style Guide** as the base (spacing, naming, organization)
- **Allman brace style** for improved readability (braces on separate lines)

This hybrid provides Google's modern C++ conventions with Allman's visual clarity for control structures.

## Key Style Rules

### Brace Placement (Allman)

```cpp
// Functions
void my_function()
{
    // body
}

// Control structures
if (condition)
{
    // body
}
else
{
    // body
}

// Classes
class MyClass
{
public:
    void method()
    {
        // body
    }
};

// Namespaces
namespace attention
{
namespace core
{
    // declarations
}
}
```

### Indentation (Google)

- **2 spaces** for indentation (no tabs)
- **2 spaces** for constructor initializer lists
- **No** indent for case labels in switch statements

```cpp
MyClass::MyClass(int a, int b)
  : member_a_(a)
  , member_b_(b)
{
    switch (type)
    {
    case TYPE_A:
        handle_a();
        break;
    case TYPE_B:
        handle_b();
        break;
    }
}
```

### Line Length

- **120 characters** maximum per line
- Allows for modern wide displays while maintaining readability
- Splits long function signatures, expressions, and comments appropriately

### Short Forms

```cpp
// Inline functions allowed on single line
int get_value() const { return value_; }

// Short if/loops NOT allowed on single line - must use braces
if (x > 0)  // ERROR - not allowed
    do_something();

if (x > 0)  // CORRECT
{
    do_something();
}

// Empty blocks can be on single line
void empty_override() override {}
```

### Comments

```cpp
// Single space before trailing comments
int value = 42; // This is a trailing comment

// Block comments for complex sections
/*
 * Multi-line explanation of complex algorithm
 * continues here...
 */
```

### Pointers and References

- Pointer/reference alignment: **left** (attached to type)
- Configuration: `PointerAlignment: Left`, `ReferenceAlignment: Left`

```cpp
// Correct: pointer/reference attached to type
int* ptr;
const Frame& frame;
std::unique_ptr<FeatureMap> feature;

// Incorrect: attached to name
int *ptr;     // Wrong
const Frame &frame;  // Wrong
```

**Rationale:** Attaches to type because `int*` is conceptually "pointer to int", not "int ptr".

### Template Declarations

```cpp
// Template declarations can stay on same line if short
template <typename T> class MyClass { };

// Longer templates should break
template <typename T, typename U>
class MyComplexClass
{
    // ...
};
```

### Include Order

Headers are grouped and sorted by priority:

1. **stdafx.h** (if used) - precompiled headers
2. **Project headers** `"attention/..."` - local includes
3. **System C++ headers** `<iostream>`, `<vector>` - standard library
4. **C headers** `<stdlib.h>` - C compatibility
5. **Third-party headers** `<opencv2/...>` - external libraries

```cpp
#include "attention/core/frame.h"     // Priority 2: Project

#include <opencv2/opencv.hpp>         // Priority 3: Third-party
#include <iostream>                   // Priority 3: C++ stdlib
#include <vector>                     // Priority 3: C++ stdlib
```

Note: Current `IncludeBlocks: Preserve` means clang-format won't reorder your includes automatically - you maintain control.

## Applying Format

### Format entire project

```bash
find include src -name "*.h" -o -name "*.cpp" | xargs clang-format -i
```

### Format specific file

```bash
clang-format -i src/main.cpp
```

### Check formatting (dry run)

```bash
clang-format --dry-run --Werror src/main.cpp
```

### Git pre-commit hook (optional)

```bash
#!/bin/bash
# .git/hooks/pre-commit
files=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|h)$')
if [ -n "$files" ]; then
    clang-format -i $files
    git add $files
fi
```

## Editor Integration

### VS Code

```json
{
  "editor.formatOnSave": true,
  "C_Cpp.clang_format_style": "file"
}
```

### CLion / IntelliJ

- Settings → Editor → Code Style → C/C++
- Set Scheme to "Project" (reads `.clang-format` automatically)
- Enable "Reformat code" in commit dialog

### Vim

```vim
" .vimrc
autocmd FileType cpp,h setlocal formatprg=clang-format
```

## Rationale for Hybrid Style

### Why Google base?
- Modern C++ conventions (smart pointers, nullptr, etc.)
- Compact indentation (2 spaces) saves vertical space
- Consistent with many open-source projects
- Good defaults for includes, comments, spacing

### Why Allman braces?
- Visual clarity: braces aligned make block boundaries obvious
- Matches original dissertation code style (circa 2003-2005)
- Preferred by developers with scientific computing background
- Reduces "where does this block end?" questions

### Tradeoffs accepted
- Slightly more verbose than pure Google (extra lines for braces)
- Not a "pure" style guide (requires explaining to contributors)
- Mixing styles can confuse automated tools expecting pure Google

## When to Deviate

Generally, **don't deviate** - consistency matters more than personal preference.

However, reasonable exceptions:
- **Generated code** - don't format auto-generated headers
- **Third-party code** - preserve original formatting in `reference/` directory
- **ASCII art comments** - disable clang-format with `// clang-format off/on`
- **Alignment for readability** - matrices, lookup tables, bit flags

```cpp
// clang-format off
const float matrix[3][3] = {
    { 1.0f,  0.0f,  0.0f },
    { 0.0f,  1.0f,  0.0f },
    { 0.0f,  0.0f,  1.0f }
};
// clang-format on
```

## Additional Style Notes

Beyond clang-format automation:

### Naming Conventions (Google)

See [DEVELOPMENT_GUIDELINES.md](DEVELOPMENT_GUIDELINES.md) for complete naming rules.

**Summary:**
- `ClassName` - PascalCase for types (classes, structs, enums)
- `function_name()` - snake_case for functions
- `variable_name` - snake_case for variables
- `member_variable_` - trailing underscore for class members
- `kConstantName` - k prefix + PascalCase for constants
- `MACRO_NAME` - UPPER_CASE for preprocessor macros (avoid when possible)
- `namespace_name` - snake_case for namespaces

### File Names

- `my_class.h` / `my_class.cpp` - snake_case matching class name
- Headers in `include/attention/` subdirectories
- Implementation in `src/` subdirectories

### Documentation

- Use `/** Doxygen-style */` comments for public APIs
- Use `//` single-line comments for implementation notes
- Document **why**, not **what** (code explains what)

## References

- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [Allman Style (Wikipedia)](https://en.wikipedia.org/wiki/Indentation_style#Allman_style)
- [ClangFormat Documentation](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)

---

*This style guide should be followed for all code contributions to maintain consistency across the project.*
