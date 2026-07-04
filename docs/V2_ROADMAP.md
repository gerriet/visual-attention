# V2 Roadmap — Bolder Modernization Phase

*Agreed 2026-07-04. Supersedes PHASE1_ACTION_PLAN.md / REALISTIC_TIMELINE.md for direction; those remain as history of the first phase.*

## Goals

1. **Full-scope thesis replication** — feature pipeline, two-stage neural-field
   selection, stereo, motion/onset, and the ESAB2 system level (object files,
   scanpaths over time, action modes). Fidelity bar: *mainly equivalent* to the
   original model, not exact reproduction.
2. **Classic vs. modern as a config switch** — swap features, fusion, and selection
   strategies (including learned saliency models) without code changes.
3. **Quantitative comparison** — the thesis model evaluated head-to-head against
   modern approaches on standard metrics and datasets.

## Decisions (locked)

- **Replication standard: loose behavioral equivalence.** A configuration
  replicates the thesis when it attends to essentially the same locations in a
  similar order on the thesis test images — not pixel-identical maps, and no
  strict tolerance regime. Spot-checks against thesis figures are enough;
  precision is explicitly *not* a goal of this phase.
- **Ecosystem: C++17 core, Python evaluation layer.** The pipeline stays
  C++/OpenCV. Metrics, datasets, plots, and learned-model inference live in
  Python. The two sides meet at a file-based interchange format (below), not FFI.
- **Modern track includes learned models** (DeepGaze-class), run Python-side as
  peers of the C++ pipeline in the same evaluation harness.

## Two architectural consequences to bake in early

1. **The pipeline is stateful and stream-oriented.** Motion/onset, neural-field
   dynamics, IOR, and ESAB2 object tracking all carry state across frames. V2's
   pipeline processes a frame *sequence* (single image = sequence of length 1),
   with explicit per-run state (field activations, IOR map, object files).
   Retrofitting this later would mean refactoring twice.
2. **A common result interchange format.** Every model — C++ classic, C++
   modern-classical, Python learned — emits the same artifact: a saliency map
   (PNG/EXR) plus a JSON result (ordered fixations with position, value,
   timestamp/iteration; per-feature timing; config hash). The Python harness only
   ever consumes this format, so adding a model never touches the harness.

## Milestones

### M0 — Housekeeping (small) ✓ 2026-07-04
- Triage untracked files: commit `SYMMETRY_FEATURE_NOTES.md`, `thesis.txt`, the
  `reference/old_code/feature/*.C` sources; delete or relocate `test_debug.cpp`,
  `debug_output/`, `results_batch/` (gitignore generated output).
- Fix README drift (`tests/`, `tools/`, `expected_outputs/` don't exist).
- Branch `v2` off `main`.

### M1 — Guardrails (before any refactor) ✓ 2026-07-04
- Catch2 + CTest. Two test layers:
  - **Characterization tests**: snapshot current feature/saliency maps on the
    test images with tolerances — pure refactor tripwires, not truth.
  - **Behavioral tests**: `--emit-json` on the CLI; fixation sequences compared
    by a minimal Python comparator (position tolerance + order).
- Define and document the interchange format (JSON schema + map files).
- Wire `examples/` into CMake so nothing rots silently.

### M2 — Architecture for swappability ✓ 2026-07-04
- Registries for `FeatureExtractor`, `FusionStrategy`, `SelectionStrategy`;
  construction fully config-driven.
- Complete the YAML loader: all feature weights (orientation/eccentricity are
  currently silently dropped), enable/disable per feature, variant + parameter
  selection (Gabor bank, symmetry schedule, output paths).
- Stream-oriented `Pipeline::process(FrameSource&)` with explicit `RunState`.
- Canonical profiles: `configs/thesis.yaml` (faithful parameters),
  `configs/modern.yaml`.
- Unify parallelism (currently std::thread-per-feature + OpenMP inside symmetry).

### M3 — Neural-field selection (the thesis core) ✓ 2026-07-04
- ✓ 2D neural-field dynamics (`nf2d.h`) ported as the `neural-field`
  `SelectionStrategy`: Amari relaxation update, Backer lateral kernel,
  logistic sigmoid, border suppression, convergence per thesis (≤55 cycles,
  mean |du| threshold).
- ✓ Two-stage readout in minimal form (thesis ch. 6/7): field activation
  clusters → Objectfile-lite (centroid, size, mean saliency) → fixations in
  priority order. Space-based IOR as decaying inhibition map across stream
  frames (thesis §8.3, −20 %/frame). The *full* symbolic stage 2 (behavior
  model, dwell time, object tracking) is M6; the *3D* field (`nf3d.h`) has
  depth/disparity as its third axis and therefore lands with stereo in M5.
- ✓ Gabor quirks resolved: `Frame` now caches parameter-keyed Gabor banks;
  each feature declares its requirement and gets exactly the bank it
  configured. Symmetry runs on wavelength 8.0 as intended, orientation on
  true 0°/45°/90°/135°. Goldens deliberately regenerated; scanpath drift was
  order-shuffling among the same regions (checked before regeneration).
- `thesis.yaml` = dissertation feature set + neural-field selection with the
  original code's parameters; locked by a behavioral golden test.

### M4 — Python evaluation layer
- `eval/` package: saliency metrics (AUC-Judd, NSS, CC, SIM, KL) and scanpath
  metrics (Levenshtein on gridded fixations, MultiMatch/ScanMatch).
- Dataset adapters (MIT1003, CAT2000, Toronto) with download/cache scripts.
- Report generator: given N result directories, produce comparison tables/plots.

### M5 — Stereo + motion/onset
- Port `stereo.C` / `stereomulti` as feature extractors; acquire stereo test
  data (Middlebury pairs + a few own captures).
- Port `onset.h` on the stream pipeline from M2; add video/frame-dir input.

### M6 — ESAB2 system level
- Object files, tracking across frames, scanpath maintenance, action modes
  (camera movement simulated on video sequences).

### M7 — Modern track
- Python-side learned models (DeepGaze IIE or successor) emitting the
  interchange format.
- C++ modern-classical strategies where cheap (spectral residual is in OpenCV;
  optionally GBVS/BMS).
- Deliverable: an evaluation report — thesis model vs. modern models on the
  thesis images *and* a public fixation dataset. "The thesis, 20 years later."

## Working agreement for this phase

- Bigger steps: one milestone = one brief, planned up front, executed largely
  autonomously, reviewed at the end (`/code-review`).
- The M1 guardrails are what make bold steps safe — they land before M2 starts.
- Milestones M4 and M5/M6 are independent of each other and can proceed in
  parallel (separate worktrees) once M2 is done.
