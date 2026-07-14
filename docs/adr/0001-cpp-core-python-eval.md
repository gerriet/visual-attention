# ADR-0001 — C++ core, Python evaluation layer

**Status:** accepted · **Date:** 2026-07

## Context

The system has two very different kinds of work: a per-pixel image-processing
pipeline that must run per video frame, and an offline evaluation layer
(metrics against human fixations, dataset adapters, plots, learned-model
inference). The original dissertation code was C++. A single language for both
would force a compromise.

## Decision

Keep the **pipeline in C++17 / OpenCV** and put the **evaluation layer in
Python** (numpy / pillow, optional torch for learned models). The two never
link; they meet at a file-based interchange format ([ADR-0003](0003-file-interchange-not-ffi.md)).

## Consequences

- The hot path stays fast and matches the original implementation's language,
  so the reimplementation is a fair comparison to the thesis.
- Metrics, dataset adapters and modern learned baselines (DeepGaze-class) live
  where that ecosystem actually is — Python — without dragging it into the
  build.
- Cost: a result must be written to disk to cross the boundary. Acceptable — the
  evaluation layer is offline, and the artifact doubles as the reproducible
  record of a run.
