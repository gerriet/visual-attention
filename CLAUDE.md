# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

C++17/OpenCV reimplementation of the two-stage visual attention model from
the 2004 dissertation (Backer), plus a Python evaluation layer. Current
direction, open hypotheses (H1–H6), and milestones: `docs/V3_ROADMAP.md`
(the science phase); the v2 instrument-building history (M0–M9) it builds on:
`docs/V2_ROADMAP.md`.

Development happens on `main`: each milestone/module lands on its own
`module/<name>` branch and is merged when done. The `v1` tag marks the
pre-v2 (phase-1) state.

The original 2003–2005 implementation lives in `reference/old_code/` —
reference only, never compiled (see its README for how to read it).

## Build and test

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build
```

- Binaries: `build/attention` (main CLI — run `--help` for all modes),
  `build/pipeline_test`, `build/debug_color` (examples).
- Test layers: C++ characterization goldens (Catch2) plus behavioral scanpath
  goldens via the CLI + `eval/compare_scanpaths.py`. The replication bar is
  *loose behavioral equivalence* to the thesis, not pixel-exactness.
- After an intentional algorithm change, review the diffs, then regenerate
  goldens: `ATTENTION_UPDATE_GOLDEN=1 ./build/tests/characterization_tests`.
- The Python eval tests need a venv:
  `python3 -m venv eval/.venv && eval/.venv/bin/pip install -r eval/requirements.txt`
  (CTest picks it up automatically).

## Architecture orientation

The pipeline is stream-oriented and stateful: a single image is a stream of
length one; field activity, IOR, and object files persist across frames in
`RunState`. Features, fusion strategies, selection strategies, and object-file
processors are registries — composition is fully driven by `configs/*.yaml`.
Every model (C++ pipeline or Python model) emits the same interchange format
(`docs/INTERCHANGE_FORMAT.md`), which is the only thing the evaluation harness
consumes. The `AttentionSystem` second stage tracks object files across frames
and selects the focus through a behavior. Recognition/analysis processors run
on attended native-resolution ROIs only — headless under `--attend`
(attention-gated recognition, M13) and in the live demonstrator (`--live`).

## Conventions

- Style: `docs/CODE_STYLE.md` (Google-Allman hybrid), enforced with
  clang-format — `cmake --build build --target format` / `format-check`.
- Every binary and Python script answers `--help`; keep it that way for new
  entry points, and add a `help_<binary>` test in `tests/CMakeLists.txt`.
- Current measured performance and known gaps: `docs/PERFORMANCE.md`
  ("Current performance (v2)").
