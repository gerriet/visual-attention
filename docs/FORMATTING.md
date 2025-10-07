# Code Formatting Guide

## Overview

This project uses **clang-format** for automatic C++ code formatting with a Google-Allman hybrid style.

See [CODE_STYLE.md](CODE_STYLE.md) for detailed style guidelines.

## Quick Start

### Format all code

```bash
# From build directory (works with any generator)
cmake --build . --target format

# Or directly with ninja (if using Ninja generator)
ninja format

# Or with make (if using Unix Makefiles generator)
make format
```

### Check formatting (without modifying files)

```bash
# From build directory (works with any generator)
cmake --build . --target format-check

# Or directly with ninja
ninja format-check

# Or with make
make format-check
```

This is useful for CI/CD pipelines to verify code is properly formatted.

### Format specific files

```bash
clang-format -i src/main.cpp
clang-format -i include/attention/core/*.h
```

### Format entire project manually

```bash
find include src -name "*.h" -o -name "*.cpp" | xargs clang-format -i
```

## Pre-commit Hook

A pre-commit hook is installed at `.git/hooks/pre-commit` that automatically formats staged C++ files before each commit.

### How it works

1. When you run `git commit`, the hook runs automatically
2. All staged `.cpp` and `.h` files are formatted with clang-format
3. Formatted files are re-staged
4. Commit proceeds with formatted code

### Disable for a single commit

```bash
git commit --no-verify
```

Use this sparingly - only when you have a good reason to skip formatting (e.g., committing generated code).

### Reinstall hook

If the hook gets deleted or needs updating:

```bash
cp docs/templates/pre-commit.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## CMake Integration

The build system automatically detects clang-format and adds formatting targets:

### Available targets

- **format** - Format all C++ source files in-place
- **format-check** - Check if formatting is correct (exit with error if not)

### Configuration output

When running `cmake ..`, you'll see:

```
-- clang-format found: /usr/bin/clang-format
--   Run 'make format' or 'cmake --build . --target format' to format code
--   Run 'make format-check' to check formatting (CI mode)
```

If clang-format is not found:

```
-- clang-format not found. Code formatting targets unavailable.
```

Install clang-format to enable formatting features.

## Installation

### macOS

```bash
# Homebrew
brew install clang-format

# Or via LLVM
brew install llvm
```

### Ubuntu/Debian

```bash
sudo apt install clang-format
```

### Windows

Download from [LLVM releases](https://releases.llvm.org/) or use chocolatey:

```bash
choco install llvm
```

### Verify installation

```bash
clang-format --version
```

## Editor Integration

### VS Code

Install the **C/C++** extension, then add to `settings.json`:

```json
{
  "editor.formatOnSave": true,
  "C_Cpp.clang_format_style": "file",
  "C_Cpp.clang_format_path": "/usr/bin/clang-format"
}
```

### CLion/IntelliJ

1. Go to **Settings → Editor → Code Style → C/C++**
2. Click **Set from... → Predefined Style → Create from .clang-format**
3. Enable **Format on save**

### Vim/Neovim

```vim
" .vimrc or init.vim
autocmd FileType cpp,h setlocal formatprg=clang-format

" Format with gq
vnoremap gq :!clang-format<CR>
```

### Emacs

```elisp
;; .emacs or init.el
(require 'clang-format)
(global-set-key [C-M-tab] 'clang-format-region)
(add-hook 'c++-mode-hook
  (lambda () (add-hook 'before-save-hook 'clang-format-buffer nil 'local)))
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Format Check

on: [pull_request]

jobs:
  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install clang-format
        run: sudo apt install clang-format
      - name: Check formatting
        run: |
          cmake -B build
          cmake --build build --target format-check
```

### Pre-push Hook (optional)

For stricter enforcement, add a pre-push hook:

```bash
#!/bin/bash
# .git/hooks/pre-push

echo "Checking code formatting before push..."
cmake --build build --target format-check

if [ $? -ne 0 ]; then
    echo "Error: Code is not properly formatted"
    echo "Run 'make format' to fix formatting"
    exit 1
fi
```

## Troubleshooting

### "clang-format not found"

Install clang-format (see Installation section above).

### Formatting conflicts in merge

If you get merge conflicts due to formatting differences:

1. Resolve conflicts normally
2. Run `make format` on the merged file
3. Add and commit

### Wrong version of clang-format

Different versions may format slightly differently. This project uses:

```bash
clang-format --version
# Expected: clang-format version 14.0 or later
```

Install a compatible version or update the `.clang-format` file.

### Disable formatting for code blocks

```cpp
// clang-format off
const int lookup_table[] = {
    1,  2,  3,  4,
    5,  6,  7,  8
};
// clang-format on
```

Use sparingly - only for special cases like aligned tables or ASCII art.

## Style Configuration

The formatting style is defined in `.clang-format` at the project root:

- **Base style**: Google
- **Brace style**: Allman (braces on new lines)
- **Indent**: 2 spaces
- **Line length**: 120 characters

See [CODE_STYLE.md](CODE_STYLE.md) for complete style documentation.

## Benefits

✅ **Consistency** - All code follows the same style automatically
✅ **No bikeshedding** - No arguments about formatting in code reviews
✅ **Less mental load** - Focus on logic, not spacing
✅ **Easier onboarding** - New contributors don't need to learn style manually
✅ **Cleaner diffs** - Only semantic changes show up in git diffs

## FAQ

**Q: Can I use a different style?**
A: Modify `.clang-format` and run `make format`. Commit the config change.

**Q: Does this slow down commits?**
A: Negligibly - formatting is very fast (< 1 second for typical changes).

**Q: What if I disagree with a formatting choice?**
A: Discuss in a team meeting and update `.clang-format` if consensus is reached. Don't make local changes.

**Q: Can I skip formatting for generated code?**
A: Yes, add generated files to a `.clang-format-ignore` file (if your version supports it), or exclude them from the CMake `GLOB_RECURSE`.

---

*Consistent formatting makes code reviews faster and reduces merge conflicts!*
