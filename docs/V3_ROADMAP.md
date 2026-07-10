# V3 Roadmap — the science phase

*Drafted 2026-07-10. v2 (`docs/V2_ROADMAP.md`, M0–M9) built the instrument: the
thesis model reimplemented (loose behavioral equivalence), a fully pluggable
pipeline (features / fusion / selection / behaviors / processors as
registries), six alternative saliency operators, a Python evaluation layer,
and a live demonstrator. v3 uses the instrument: replicate the thesis
findings, run the comparisons the thesis could only sketch, and prove the
model's central claims on today's data.*

A pleasing symmetry: thesis chapter 9 (Evaluation) proposed simulation-based
evaluation of active attention systems (§9.5, the "Orbital 3D" idea) as future
work. This phase is that chapter, built.

## The program in one paragraph

Replicate the dissertation's own findings first (they anchor everything).
Then attack the headline scientific claim — that **object-based,
multi-tracking inhibition of return beats spatial IOR in dynamic
environments** — with controlled synthetic sweeps and ground-truth-scored real
video. Around it: scanpath comparison against human fixations on stills,
recognition processes that run only on attended regions (the attention
premise turned into a measurable compute argument), a model lab for cooking up
and honestly scoring feature/fusion/selection combinations, and stereo-video +
virtual active vision scenarios that show the ideas earning their keep.

## Hypotheses (make the science explicit)

- **H1 — Dynamic IOR.** In multi-object dynamic scenes, object-based IOR
  (inhibition tags *tracked object files*) yields higher object coverage and
  lower novel-object detection latency than space-based IOR, and degrades
  gracefully with object speed — while space-based IOR collapses (moving
  objects escape their inhibited region → perseveration; empty inhibited
  regions starve revisits). *This is the thesis's raison d'être for stage 2.*
- **H2 — Gated recognition.** Recognition restricted to attended ROIs reaches
  near-full-frame accuracy at a fraction of the compute; the gap narrows as
  scene clutter grows.
- **H3 — Depth priority.** With the stereo/disparity channel and 3D field,
  near/approaching objects are attended earlier than in the 2D model —
  measurable as time-to-attend for approaching objects.
- **H4 — Scanpath plausibility.** On still images, the two-stage model's
  scanpaths land measurably above random/center baselines toward the
  inter-observer ceiling — and the *ordering* benefit of stage 2 is separable
  from the saliency map itself (same map, WTA readout vs object-file readout).

## Milestones

### M10 — Replication dossier (thesis findings, reproduced)

The thesis contains concrete, reproducible experiments — mostly ch. 5/6:

- Feature-variation curves: eccentricity/symmetry variation → monotone feature
  response (Abb. 5.13), colour-contrast variation (5.20), distance variation
  for stereo (5.28).
- Noise-robustness curves per feature (5.14, 5.21, 5.30).
- Parameter-sensitivity: segmentation thresholds (5.15/5.16), colour
  thresholds (5.22), stereo variance threshold (5.32).
- Feature superposition & exclusivity effects (5.33, 5.34); 2D vs 3D
  integration (5.35).
- Neural-field behavior: tracking two maxima, repulsion effects (6.10);
  per-feature-weighted field system (6.11); multiple salient objects (6.12);
  3D field with local inhibition (6.13); **multi-object tracking through
  temporary occlusion across field variants (6.14)** — the 2004 ancestor of
  the H1 study.
- Qualitative ch. 9 claims worth a demo each: flanker-compatibility effect
  (§9.3.2), early vs late selection (§9.3.3).

Work: mine `docs/thesis/thesis.txt` into a definitive findings list (ref →
claim → stimulus → expected curve/behavior); build each as a scripted
experiment under `experiments/replication/`; synthetic stimulus generators
where the thesis used lab imagery. Deliverable:
`docs/replication/REPLICATION_DOSSIER.md` — per finding: thesis figure ref,
v2 plot, verdict **replicated / partially / diverged** with explanation.
Divergences are findings, not failures (document, don't chase pixel parity).

### M10b — Selection backends: robust multi-blob tracking without the field's tuning

The Amari field works but is hard to parametrize (13 coupled, input-scale-
dependent knobs) — a documented DNF obstacle, not a local quirk. Add alternative
`SelectionStrategy` backends that do the same job (track multiple high-activity
blobs in 2D and 3D) by **decoupling** detect / compete / persist / inhibit /
repel, each with 1–2 semantic, monotone knobs. Full analysis and the menu:
`docs/SELECTION_BACKENDS.md`.

Build order (all opt-in; thesis field stays default):
- **B — blob detection + Kalman MOT** (SORT-style): the robustness/efficiency
  win; occlusion handling for free; 3D by adding disparity to the Kalman state.
  *Do this first — **agreed 2026-07-10**, ships as a `kalman-mot` selection
  strategy after the M9 commit.*
- **D — normalization-model competition** (Reynolds & Heeger): keep a
  field-like soft-WTA but make it input-scale invariant (kills the #1 tuning
  pain), one pass, ~2 knobs.
- **E — self-tuning field** (intrinsic plasticity + input normalization): the
  minimal-risk upgrade of the *existing* field — auto-tunes resting & slope,
  keeps thesis behavior.
- Optional extras: A (mean-shift/CAMShift), C (attention particles),
  F (GM-PHD, the principled MOT baseline for H1).

Deliverable: the backends behind config keys + `docs/SELECTION_BACKENDS.md`
verdicts. This **pairs with M12**: the payoff experiment is same saliency
stream, vary only the backend → tracking accuracy **and parameter sensitivity**
(how wide the working range is) and runtime, turning "the field was hard to
tune" into a quantified result and giving H1 rigorous baselines.

### M11 — Scanpaths vs human gaze (stills)

- Finalize the MIT1003 adapter (fixation *sequences*, not just maps);
  optional: CAT2000.
- Scanpath metrics in `eval/attention_eval`: MultiMatch (5 dims) and
  ScanMatch (Needleman-Wunsch), alongside the existing AUC/NSS/CC/SIM/KL.
- A generic WTA+IOR scanpath readout over any saliency map, so every model —
  the six alternatives, spectral/center baselines, DeepGaze — becomes a
  scanpath model and a fair peer (this also isolates H4's stage-2 ordering
  effect: same map, two readouts).
- Baselines that keep us honest: inter-observer consistency (leave-one-out
  ceiling), random + center-bias floors.

Deliverable: `docs/SCANPATH_VS_HUMAN.md` — metric tables + montages; the
thesis model placed on the floor↔ceiling axis. Done when the comparison runs
end-to-end from one command.

### M12 — The dynamic-IOR proving ground (H1 — the centerpiece)

Three arms, identical everywhere except inhibition:
1. **no-IOR** (baseline; saliency argmax each frame),
2. **space-IOR** (first stage only: decaying spatial inhibition),
3. **object-IOR** (full two-stage: inhibition rides on tracked object files).

Work:
- Synthetic dynamic-scene generator (Python → frame dirs + ground-truth JSON;
  the CLI already consumes frame dirs): N objects, controllable speed, size,
  trajectories, occluders, timed onsets; seeded.
- Scoring harness reading `attention-scanpath/v1` + ground truth: object
  **coverage@T** (unique objects attended), **novel-object latency**,
  **revisit waste**, **identity persistence**.
- Sweeps: speed × object count × occlusion, ≥20 seeds/cell, bootstrap CIs.
  The money plot: coverage vs object speed, three curves.
- Reproduce thesis Abb. 6.14's occlusion setup as one scenario (bridges M10).
- Real-video validation: DAVIS-2017 (per-frame object masks → exact
  "which object attended"); `vtest.avi` for the qualitative figure.

Deliverable: `docs/DYNAMIC_IOR_STUDY.md` — H1 confirmed/refuted, with the
regime map (where object-IOR pays, where it doesn't). Honesty requirement:
publish the failure regimes too (e.g. very slow scenes where space-IOR
suffices).

### M13 — Recognition processors (attention-gated perception, H2)

- Tier 1 (zero new dependencies): `hog-person` and `haar-face` processors
  (OpenCV built-ins) on attended ROIs.
- Tier 2: `dnn-classify` processor via `cv::dnn` — loads any ONNX classifier
  from config (model path + labels file); no new link dependency.
- **Label memory on object files**: accumulated majority-vote label +
  confidence per object file → stable semantic identities ("person #3").
- **Identification behavior** (new stage-2 behavior): unrecognized objects
  attract attention until confidently labeled; recognized ones drop priority —
  recognition-triggered semantic IOR, i.e. curiosity.
- H2 experiment: gated recognition vs full-frame detection at matched compute
  → accuracy-vs-compute curves on vtest/DAVIS.

Deliverable: processors + behavior + `docs/GATED_RECOGNITION.md`.

### M14 — The model lab (simulate, compare, combine)

- `configs/models/` presets: `thesis-2004`, `alternatives-suite`, per-operator
  singles, and named hybrids (e.g. thesis + minimum-barrier, symmetry ×
  phase-spectrum).
- `eval/lab.py`: matrix runner — (model preset × dataset) → results table
  (CSV/JSON) → auto-generated report with metric tables and montages.
- Combination search: weight sweeps / greedy forward selection over the
  feature registry, scored on a train split of MIT1003, reported on a held-out
  test split (no self-congratulation on training data).

Deliverable: one command reproduces the full comparison; the lab report and
the best honest hybrid checked in. This industrializes "cook up interesting
combinations."

### M15 — Stereo video + virtual active vision (H3)

- Stereo video input: KITTI raw adapter (rectified stereo streams) and/or a
  two-webcam capture path; stereo feature + 3D field running on streams
  (v2 deferred exactly this).
- H3 experiment: approaching-object time-to-attend, depth-prioritized vs 2D.
- **The virtual fovea** — active vision without the robot head: high-res
  input, the model drives a digital pan-tilt-zoom crop (the "fovea");
  recognition processors run only inside it. This is the thesis's active-vision
  premise (and §9.5's simulation idea) made runnable on any 4K video.

Deliverable: `--fovea` demonstrator mode + latency plot + doc section.

### M16 — Scenario showcase (the ideas earning their keep)

Packaged, one-command scenarios with captured demo clips:

- **Watchman**: static camera (vtest/webcam); guarantee: every person
  attended and identified within T seconds; alarm on new entrant
  (onset → capture → recognition). Metrics: coverage, entrant latency.
- **Left luggage**: PETS2006-style; a new object appears, persists, and its
  object file never moves while its owner leaves → alarm. Onset + object
  files + label memory, working together.
- **Find-my-object**: top-down feature weighting (a Search behavior sets
  feature weights from a target description) → search-time vs bottom-up.
- Stretch: foveated streaming (attention-driven encoding quality; bandwidth
  saved) — only if cheap after M15.

## Datasets

| Dataset | For | Status |
|---|---|---|
| `data/samples/` + synthetic generators | replication, IOR sweeps, demos | in repo |
| MIT1003 | still fixations/scanpaths (M11, M14) | adapter exists; download documented |
| DAVIS 2017 | video object masks → exact IOR scoring (M12, M13) | adapter to write |
| DIEM (or DHF1K) | video gaze (optional M14 extension) | choose when needed; DIEM easiest |
| KITTI raw (stereo) | stereo video (M15) | adapter to write |
| PETS2006 | left-luggage scenario (M16) | pointer only |

Corpora stay pointed-to, never redistributed (v2 convention).

## Conventions for the phase

- `experiments/<area>/<name>/` = config + runner + README; results land
  outside git (`results/`), reports and key plots land in `docs/`.
- Every quantitative claim: ≥20 seeds where stochastic, bootstrap CIs, and
  train/test splits for anything tuned.
- Every experiment reproducible from one command; smoke-test versions wired
  into CTest where they're fast enough.
- The thesis model remains the default everywhere; everything new is opt-in
  by config (v2's M9 principle).

## Recommended order & rationale

**M10 → M10b → M12 → M13 → M11 → M14 → M15 → M16.**
Replication first (anchors credibility, and its stimulus generators feed M12).
Then the selection backends (M10b), because the headline H1 study wants them as
baselines and the Kalman backend hands M12 its occlusion handling. Then the H1
study while the momentum is on stage 2. M13 next because
the recognition processors unlock the best scenarios. M11/M14 are
Python-heavy and independent — swap earlier if the human-comparison story is
more urgent. M15/M16 package the story. M10 is independent of everything and
can start immediately.

## Working agreement (unchanged)

One milestone = one brief, planned up front, executed largely autonomously,
adversarially reviewed at the end, locked behind tests. Direction lives here;
v2 history in `docs/V2_ROADMAP.md`.
