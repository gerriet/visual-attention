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
4. **Live demonstrator** — the pipeline running live on video, object files
   displayed as tracked annotations at their image positions, with a plugin
   mechanism that applies downstream processing (classification, measurement,
   …) to attended regions only — attention as a front-end for selective
   processing.

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

### M4 — Python evaluation layer ✓ core 2026-07-05
- ✓ `eval/attention_eval` package (numpy/pillow venv, wired into CTest):
  saliency metrics (AUC-Judd, NSS, CC, SIM, KL), scanpath metrics (greedy
  match stats, gridded Levenshtein), interchange-format loader.
- ✓ Report generator (`python -m attention_eval.report`): markdown comparison
  of N results against a reference. First classic-vs-modern numbers (inputc):
  thesis profile vs modern default — CC 0.69, SIM 0.82, NSS@ref 2.31,
  70 % scanpath agreement.
- ✓ MIT1003 adapter (fixation maps/points; download steps in the module).
- Pending: dataset downloads + benchmark runs (with M7), MultiMatch/ScanMatch,
  per-observer scanpaths (needs scipy), plots, CAT2000/Toronto adapters.

### M5 — Stereo + motion/onset ✓ 2026-07-05
- ✓ Stereo disparity feature (`stereo`, thesis §5.4): dedicated near-vertical
  Gabor magnitude responses, windowed normalized cross-correlation over a
  disparity range (eq. 5.13) with variance gating, confidence accumulation +
  spatial blur (eq. 5.14), per-pixel WTA (eq. 5.15), normalized-disparity
  depth saliency (eq. 5.16). Single-scale core; the multi-scale
  `StereoMultiFeature` and multi-hypothesis/exclusivity machinery are not
  ported (loose-equivalence bar). Runs only when the frame carries a right
  image; publishes its map as a depth cue for the 3D field.
- ✓ Onset/motion feature (`onset`): rectified positive temporal change in edge
  energy (structure appearing since the previous frame), on the stream
  pipeline. A principled reconstruction — the original `onset.C` was not
  preserved, only an empty header stub. Runs only inside a temporal stream.
- ✓ 3D neural field (`nf3d`, thesis §6.4): `NeuralField3D` core (stack of 2D
  Amari fields with cross-depth inhibition) + `neural-field-3d` selection that
  lifts the fused 2D saliency into a depth volume via the disparity cue and
  reads out clusters with a dominant depth. Field re-initializes per frame
  (no cross-frame volume persistence yet); uses the 2D space-based IOR map.
- ✓ Input plumbing: `Frame::stereo_right` / `previous_gray`, `RunState`
  `previous_gray` / `depth_map`; CLI `--stereo <l> <r>` (single pair) and
  `--sequence <dir|video>` (temporal stream, no per-frame reset);
  `StereoImageSource` + `VideoFrameSource`.
- ✓ Test data + eval: deterministic synthetic stereo pair
  (`tools/make_synthetic_stereo.py` → `data/test_images/stereo/`), stereo
  behavioral golden, C++ tests for stereo/onset/3D-field, and a Middlebury
  adapter (`eval/datasets/middlebury.py`, download steps in the module).
- Deferred: multi-scale stereo (`StereoMultiFeature`), cross-frame 3D-field
  volume persistence, real stereo-video / own captures (fold into M7 runs).

### M6 — AttentionSystem (system level) ✓ 2026-07-05
*(the original code called this class ESAB2, after the DFG project "Entwicklung
von Systembausteinen der Aktiven Bildanalyse II"; renamed to AttentionSystem as
the name carried no meaning outside the project.)*
- ✓ Object files (thesis §7.2): `ObjectFile` (label, centroid, bbox, size,
  current + leaky-averaged saliency, created/last-seen/last-selected frames,
  active flag, trajectory) + `ObjectFileStore` — correspondence by centroid
  proximity with a global minimal-distance assignment, new files beyond 2× the
  radius, an inactive stack that later clusters can revive, and aging out.
- ✓ Behavior model (thesis §8.5): `Behavior` interface + `Exploration` —
  priority classes (never-selected first, then longest-unselected), within a
  class by mean saliency, with a 3-frame dwell and object-based inhibition of
  return via the last-selected ordering. Visual search / MOT / search-and-track
  slot in behind the same interface.
- ✓ `AttentionSystem` orchestrator: per stream frame, segment the fused
  saliency into candidate clusters, correspond to object files, run the
  behavior to pick the focus, maintain the scanpath. Action modes: Feature
  (saliency only) and Scanpath (full second stage). move_sensor (overt gaze
  shift + field displacement) needs a controllable source and is deferred.
- ✓ CLI `--attend <dir|video>` with per-frame object annotations (focus/never/
  previously-selected coloring, as in thesis Fig. 8.2) and a scanpath JSON
  (`attention-scanpath/v1`); behavioral golden on the motion sequence.
- Deferred: the further behaviors (visual search, MOT, search-and-track),
  overt sensor control / field displacement, and the seeded-region-growing
  object segmentation refinement (thesis §7.3.2).

### M7 — Modern track ✓ 2026-07-06
- ✓ Modern saliency models as interchange-format peers
  (`eval/attention_eval/models/`): spectral-residual (Hou & Zhang, numpy) and
  a Gaussian center-bias baseline, runnable in CI; a torch-gated DeepGaze IIE
  adapter as the learned-model slot (documented, not run in CI). Each emits the
  same JSON + 16-bit PNG the C++ pipeline does, so all models are peers.
- ✓ Benchmark harness (`attention_eval.benchmark`): run model specs (Python
  names + `cpp:<binary>:<config>`) over an image set; aggregate cross-model
  (reference model) or against ground-truth fixations (`--dataset mit1003`).
  Markdown report + saliency montage / metric-bar plots (`attention_eval.plots`,
  matplotlib optional).
- ✓ Deliverable report `docs/thesis_vs_modern.md` (+ montage): the 2004 model
  vs spectral-residual vs center-bias on the thesis images. Finding: the modern
  baselines diverge markedly (CC 0.11 SR / 0.34 center-bias vs the thesis map,
  ~10 % scanpath overlap); the center bias tracks the thesis model more closely
  than spectral residual, so the thesis model carries a moderate central
  tendency. Ground-truth benchmarking runs against MIT1003 once downloaded.
- The C++ modern-classical strategies (spectral residual / GBVS in-pipeline)
  were kept Python-side instead (peers via the interchange format) — cleaner
  and sufficient for the comparison; a C++ port stays optional.

### M8 — Live demonstrator: annotated video + object-file plugins ✓ 2026-07-06
*(depends on M5 video input and M6 object files; independent of M7)*

- ✓ `VideoSource : FrameSource` (`cv::VideoCapture`) opens a webcam device
  index or a video file; a frame directory streams via `ImageListSource`.
  `LiveDemonstrator` **decouples processing from display resolution**: capture
  native, process at ≤ `process_max_side` (default 480), scale object-file
  coordinates back up for ROIs and the overlay. CLI `--live <cam|video|dir>`
  (interactive `imshow`, ESC quits) or headless (`--no-display --frames N
  --output dir`, saves annotated frames — the CI smoke test).
- ✓ `configs/live.yaml`: a single coarse symmetry scale and a **fixed number of
  field cycles per frame** (`cycles_per_frame`) instead of per-frame
  re-convergence — the field state in `RunState` carries over, so attention
  shifts emerge from the continuous dynamics at a deterministic per-frame cost.
- ✓ Annotation overlay: one marker per object file (stable M6 ID, box, colour by
  focus / previously / never selected) plus a scanpath trail.
- ✓ **Object-file plugin processes**: a `Processor` interface (object file + its
  native ROI → annotation), registry + config-driven like features/fusion/
  selection, run only on attended regions. Built-ins: `roi-probe` (ROI dims +
  timing — proves the mechanism) and `region-descriptor` (a real, dependency-
  free analysis: colour / brightness / edge density / aspect ratio).
- Deferred: a learned classifier/OCR plugin (external deps), and the on-device
  fps tuning (the headless path and the fixed per-frame budget are in place).

## Working agreement for this phase

- Bigger steps: one milestone = one brief, planned up front, executed largely
  autonomously, reviewed at the end (`/code-review`).
- The M1 guardrails are what make bold steps safe — they land before M2 starts.
- Milestones M4 and M5/M6 are independent of each other and can proceed in
  parallel (separate worktrees) once M2 is done.
