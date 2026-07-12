# V3 Roadmap — the science phase

*Drafted 2026-07-10; sharpened the same day against `docs/RESEARCH_POSITIONING.md`
(priority-map framing, scanpath-variability scoring, and the VLM front-end —
milestones M17/M18 and hypotheses H5/H6). v2 (`docs/V2_ROADMAP.md`, M0–M9) built the instrument: the
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
virtual active vision scenarios that show the ideas earning their keep. Two
threads added by the 2026 positioning pass (below) turn the master saliency map
into a full **priority map** (M17) and reframe the whole instrument as an
attention front-end for large vision models (M18).

## Positioning: sharpen, don't redirect (2026)

`docs/RESEARCH_POSITIONING.md` places this work against 2025–26 attention
research. Verdict: the plan is well-aimed, but three sharpenings move it from
"reimplementing a 2004 thesis" to a current contribution. They are folded into
the milestones below.

- **Reframe the master saliency map as a *priority map*.** The modern unifying
  construct (Wolfe's Guided Search 6.0; Awh, Belopolsky & Theeuwes) fuses three
  sources: bottom-up salience, top-down task relevance, and selection history /
  value. The thesis map has only the first. Adding the other two (**M17**) is
  the single highest-leverage upgrade and a clean ablation axis.
- **Own the dynamic, object-based scanpath niche (H1 + M11/M12)** — exactly
  where learned free-viewing models are weakest and a real mechanism exists
  here. Score scanpaths with MultiMatch/ScanMatch and add a *generative
  variability* baseline (a ScanDiff-style probabilistic peer), so the
  deterministic scanpath is evaluated as one sample of a human distribution,
  not compared unfairly.
- **Build the VLM front-end demo (**M18**)** — highest timeliness. The same
  instrument, reframed as an interpretable, model-free attention front-end that
  saves large-vision-model compute (foveated token pruning, à la GazeVLM), with
  a token/FLOP-vs-accuracy curve. The M8 processor architecture is already the
  right shape.

**Non-goals.** Do not chase DeepGaze on free-viewing salience — that benchmark
is saturated at the inter-observer ceiling. Do not drift into "another VLM."
The edge is the interpretable, stateful, object-based *dynamic controller* that
large learned models lack and increasingly need.

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
  The deterministic path is scored as **one sample of a human scanpath
  distribution** (MultiMatch/ScanMatch against a generative baseline), never as
  "the" path.
- **H5 — Priority map beats salience-only for task-driven search.** Adding a
  top-down/task channel (best sourced from a language prompt) and a
  selection-history/value term to the master map improves target-finding
  efficiency (search time, fixations-to-target) over bottom-up-only, and each
  term contributes measurably (ablation). *Turns a 2004 bottom-up model into a
  current priority-map model.*
- **H6 — Attention as a VLM token budget.** Feeding a vision-language model only
  the attended ROIs (fovea) plus a low-res global view preserves task accuracy
  at a large fraction of the visual tokens/FLOPs of the full-resolution image,
  and the saving grows with input resolution.

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
- Research-current variant to note: **rhythmic / theta (~4–8 Hz) sampling** —
  attention as periodic re-selection rather than steady field relaxation
  (genuinely new since the thesis, per RESEARCH_POSITIONING.md). A cheap
  "sample the priority map every N frames" readout is a biologically-topical
  alternative temporal story worth an experiment, not a core dependency.

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
- **Variability baseline + probabilistic scoring** (positioning sharpening):
  human scanpaths are a *distribution*, not one path. Add a generative peer
  (a ScanDiff-style diffusion sampler, or at minimum a stochastic WTA+IOR that
  samples proportional to saliency) and score our deterministic path as one
  draw — report best-of-N and distribution-level agreement, not just a single
  MultiMatch number. Keeps H4 honest against 2025 diffusion scanpath models.

Deliverable: `docs/SCANPATH_VS_HUMAN.md` — metric tables + montages; the
thesis model placed on the floor↔ceiling axis, scored as a sample of a
distribution. Done when the comparison runs end-to-end from one command.

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
- Tier 3 (positioning sharpening): a `vlm-caption` processor that hands the
  attended ROI to a vision-language model and accumulates the returned label on
  the object file. This makes the **symbolic second stage a queryable scene
  graph / working memory** — object files with labels, trajectories, and
  saliency history that an LLM can read and update (the neuro-symbolic framing
  of RESEARCH_POSITIONING.md, mode 4). Shares plumbing with M18.

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
  feature weights from a target description, ideally via the M17 priority-map
  top-down channel) → search-time vs bottom-up.
- Stretch: foveated streaming (attention-driven encoding quality; bandwidth
  saved) — only if cheap after M15.

### M17 — Priority map: the two missing terms (top-down + selection history)

The highest-leverage sharpening from the positioning pass (H5). Generalize the
fused "master saliency map" into a **priority map** = bottom-up salience +
top-down task relevance + selection history / value, each an opt-in channel
with its own weight (the thesis map = this map with the last two terms zeroed,
so nothing changes by default).

- **Top-down / task channel.** A language prompt ("find the exit sign") →
  target embedding → either per-feature weight vector (Guided Search) or a
  dense semantic-relevance map (open-vocabulary detector / CLIP-style
  similarity) fused into the master map. The LLM is the natural source
  (RESEARCH_POSITIONING.md mode 2); keep it behind the interchange boundary so
  it stays optional and swappable.
- **Selection-history / value channel.** A decaying map of recently/rewarded
  locations and objects (object files already carry the history) added as a
  third priority term — distinct from IOR (which only suppresses).
- Ablation: bottom-up only → +top-down → +history, on the M16 find-my-object
  and a target-present visual-search set (COCO-Search18). Each term must earn
  its weight.

Deliverable: a `PriorityMap` fusion stage (channels behind the fusion
registry) + `docs/PRIORITY_MAP.md`. Reframes the model in current terms.

### M18 — VLM front-end: attention as a token-budget allocator (flagship demo)

The timeliest repositioning (H6): the whole instrument as an interpretable,
model-free attention front-end that saves large-vision-model compute.

- A `--vlm-front-end` mode (built on the M15 virtual fovea + M8 processors):
  the priority map / object files pick K attended ROIs; feed the VLM only those
  fovea crops plus one low-res global view instead of the full-resolution image
  (the GazeVLM / "gaze tells you where to compute" recipe).
- Measure the **token/FLOP-vs-accuracy curve** against full-resolution and
  uniform-downsample baselines on a high-resolution VQA / small-object task
  (e.g. V*Bench-style), where downsampling makes VLMs blind to small objects.
- Bonus baseline (mode 3): compare our fast bottom-up/object-based crop
  sequence to RL-learned agentic visual-search loops (ZoomEye / DeepEyes-style)
  — do learned search policies resemble object-based human scanpaths?
- Keep the VLM behind the interchange/processor boundary (no hard dependency in
  the core; the adapter lives Python-side like the modern models).

Deliverable: the front-end mode + `docs/VLM_FRONT_END.md` with the savings
curve. This is the single result that reframes the project from "a 2004 thesis
reimplementation" to "a stateful, object-based attention front-end for large
vision models." Guard against the non-goal: stay the *controller*, don't become
another VLM.

## Datasets

| Dataset | For | Status |
|---|---|---|
| `data/samples/` + synthetic generators | replication, IOR sweeps, demos | in repo |
| MIT1003 | still fixations/scanpaths (M11, M14) | adapter exists; download documented |
| DAVIS 2017 | video object masks → exact IOR scoring (M12, M13) | adapter to write |
| DIEM (or DHF1K) | video gaze (optional M14 extension) | choose when needed; DIEM easiest |
| KITTI raw (stereo) | stereo video (M15) | adapter to write |
| PETS2006 | left-luggage scenario (M16) | pointer only |
| COCO-Search18 | target-present visual search (M17 top-down ablation) | adapter to write |
| V*Bench / hi-res VQA | VLM token-vs-accuracy curve (M18) | pointer only |

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

**M10 → M10b → M12 → M13 → M17 → M11 → M14 → M15 → M18 → M16.**
Replication first (anchors credibility, and its stimulus generators feed M12).
Then the selection backends (M10b), because the headline H1 study wants them as
baselines and the Kalman backend hands M12 its occlusion handling. Then the H1
study while the momentum is on stage 2. M13 next because the recognition
processors unlock the best scenarios *and* supply the labels/embeddings the
priority map's top-down term needs — so M17 (priority map) follows immediately;
it is the highest-leverage positioning upgrade, pull it earlier if the
task-driven story is more urgent than replication. M11/M14 are Python-heavy and
independent — swap earlier if the human-comparison story leads. M15 (virtual
fovea) sets up **M18, the flagship VLM-front-end demo** — though a minimal M18
can ride the M8 processors before the full fovea lands, and if timeliness
dominates it is the single most repositioning result to front-load. M16
packages the story. M10 is independent and can start immediately.

The two positioning milestones (M17, M18) are the parts a reviewer will read as
"current"; everything before them makes them credible. Don't skip the science
to reach them, but don't defer them to the end either.

## Working agreement (unchanged)

One milestone = one brief, planned up front, executed largely autonomously,
adversarially reviewed at the end, locked behind tests. Direction lives here;
v2 history in `docs/V2_ROADMAP.md`.
