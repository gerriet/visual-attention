# V3 Roadmap — the science phase

*Drafted 2026-07-10; sharpened the same day against `docs/RESEARCH_POSITIONING.md`
(priority-map framing, scanpath-variability scoring, and the VLM front-end —
milestones M17/M18 and hypotheses H5/H6; M19 and H7 were added 2026-09-15).
v2 (`docs/V2_ROADMAP.md`, M0–M9) built the instrument: the thesis model reimplemented (loose behavioral equivalence), a fully pluggable
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
- **H7 — Object files as a video token cache (H1 × H6).** At a matched
  visual-token budget per video, an object-file front-end (object-based IOR +
  persistent object files) answers object-centric questions better than
  budget-matched uniform frame sampling and better than a spatial-IOR
  front-end; the gap grows with object count and speed, and shrinks with
  tracking errors (every identity switch is a re-send). *Where the second
  stage earns its keep: M12 showed object-IOR only ties space-IOR on
  exploration; H7 tests the predicted win — persistent, identity-keyed memory.*

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

Work: mine the dissertation text (published open access) into a definitive findings list (ref →
claim → stimulus → expected curve/behavior); build each as a scripted
experiment under `experiments/replication/`; synthetic stimulus generators
where the thesis used lab imagery. Deliverable:
`docs/replication/REPLICATION_DOSSIER.md` — per finding: thesis figure ref,
v2 plot, verdict **replicated / partially / diverged** with explanation.
Divergences are findings, not failures (document, don't chase pixel parity).

**Status (2026-09-20): first version done** — `docs/replication/REPLICATION_DOSSIER.md`,
one command (`eval/replication.py --all`, CPU only). Ten findings of ch. 5 (and
Abb. 6.14 via the H1 occlusion regime): eccentricity and colour contrast — the
two features ported from the original sources — **replicate every finding
tested**, eccentricity on the thesis's eq. 5.6 to within 0.02; exclusivity
replicates; the growth-threshold range partially. **Symmetry diverges**: it does
not peak on symmetric objects but in rings around them — the original's
absolute scale and clip offset were replaced by per-band normalization and
relative thresholds over three scales, so one-sided responses survive. That is
the next replication-track fix, after which the dossier, H1 and M11 re-run. Not
attempted yet: the stereo experiments, the field dynamics of Abb. 6.10–6.13
(need a harness that drives the field directly), the qualitative ch. 9 demos.

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

**Status (2026-07-12): B and D done; the payoff experiment is open.**
`kalman-mot` (B) and `normalization` (D) ship as opt-in selection strategies
(`configs/kalman.yaml`, `configs/normalization.yaml`, Catch2 + CTest coverage).
E, F, A and C are not built. The A/B this milestone exists for — same saliency
stream, vary only the backend → tracking accuracy, parameter sensitivity,
runtime — has not been run, and neither backend has been entered as an arm in
M12 or M19; `docs/SELECTION_BACKENDS.md` therefore carries no verdicts yet.

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

**Status (2026-09-21): done — real run on all 1003 MIT1003 stimuli, predictions
written down beforehand.** H4: above random, weakly (ScanMatch +0.015 [+0.012,
+0.017]); not above the constant-centre path (−0.102); the ceiling is +0.107
away; **no ordering benefit of stage 2 on stills** (object-file readout −0.022
against a plain WTA readout of the same map). Methodological result: on MIT1003
ScanMatch cannot tell "stays in the middle" (0.754) from human-vs-human
agreement (0.759); MultiMatch's direction dimension can. The harness needed
fixing when it met the real archive (the model's own field scanpath as an arm, a
working object-file readout, pairing, resume). `docs/SCANPATH_VS_HUMAN.md`.

*Earlier status (2026-07-21): instrument built + verified end-to-end on synthetic
data; real MIT1003 run gated on the dataset + DATA archive + scipy.* In-repo
MultiMatch (4 spatial dims, DTW-aligned) and ScanMatch (Needleman-Wunsch,
signed substitution matrix) live in `eval/attention_eval/scanpath.py`; a generic
WTA+IOR readout + a seeded stochastic sampler
(`eval/attention_eval/readout.py`) turn any saliency map into a scanpath — and
give H4 its ablation for free: the same thesis map read out by WTA
(`--emit-json`) vs the object-file second stage (`--attend`), no new C++ code.
The MIT1003 adapter now recovers per-observer ordered sequences from the raw
DATA/ archive via an I-DT fixation filter (scipy, unit-tested against a
synthetic .mat); a CAT2000 adapter adds a second dataset — but note CAT2000's
public release ships pooled fixation *locations*, not ordered per-observer
scanpaths, so it's the saliency/coverage cross-check while MIT1003 carries the
ordered comparison. `eval/scanpath_vs_human.py` runs the full floor↔ceiling
axis (inter-observer LOO ceiling; thesis-objfile / thesis-wta / per-operator
WTA / stochastic-best; center + random floors) in one command; its synthetic
`--demo --check` validates the whole scoring stack (ceiling and map-driven
readout clear the floors) and is a CTest gate. No dataset/scipy in this env at
build, so the real MIT1003 table lands on a keyed download. Full story:
`docs/SCANPATH_VS_HUMAN.md`.

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

**Status (2026-07-12): first cut done** — the three arms are behaviors
(`greedy` / `spatial-ior` / `object-ior`, `attention --attend --behavior`),
identical except in what they inhibit; `tools/make_dynamic_scene.py` +
`eval/dynamic_ior.py` generate scenes and score coverage/latency/waste/
perseveration. Finding: **IOR ≫ no-IOR (H1's first half strongly supported);** but pushing into
fast-motion + occlusion (with a configurable `--ior-radius` and opt-in
**motion-predicted correspondence** in the object-file store, `--motion-prediction`)
shows, across seeds, that **space-based IOR is at least as good as — usually
better than — object-based IOR**. The thesis's headline object-IOR advantage does
*not* robustly materialize: object-IOR is only as good as the tracker, and every
label-switch under fast/dense motion costs it a wasted re-fixation; motion
prediction narrows but does not close the gap. Sharper H1: object-IOR beats
space-IOR only to the extent identities stay stable. So we built better tracking
**from the features we already compute** — opt-in motion-predicted + **appearance
(mean-colour) correspondence** in the object-file store (DeepSORT-style, using
the already-selected regions). It cuts object-IOR's high-speed waste **3×**
(0.413 → 0.126), confirming label-switches were the bottleneck and taking
object-IOR from clearly-worse to *nearly tied* with space-IOR — but not past it
on exploration metrics. Remaining (the likely object-IOR win): identity-centric
metrics + *persistent* object memory vs necessarily-decaying spatial memory;
seeds+CIs; DAVIS. See `docs/DYNAMIC_IOR_STUDY.md`.

**Update (2026-09-15, with M19):** opt-in *persistent identity* in the
object-file store takes object-IOR from clearly worse to level with space-IOR,
and ahead on latency at high speed — in a six-seed pilot without intervals: a
direction, not yet a result. **Still open from this milestone's own list:** ≥ 20
seeds per cell with intervals (and a seed loop in `eval/dynamic_ior.py` — the
tables so far were averaged by hand), the speed × count × occlusion sweep, the
Abb. 6.14 scenario, DAVIS scoring of H1, and a strengthened spatial baseline
(motion-compensated spatial IOR). Plan: `docs/HYPOTHESIS_CLOSURE_PLAN.md`.

**Re-run with the ported symmetry (2026-09-21): H1 supported, fully.** After
the replication dossier found and fixed the symmetry feature, the same study on
fresh seeds 2000–2029: off-object fixations 0.33 → 0.00; object-based IOR beats
space-based IOR on latency (−0.5 / −4.4 / −2.6 frames) *and* staleness in every
regime, with the thesis's own correspondence and against the motion-compensated
tag; three of the four (pessimistic) pre-registered predictions refuted.

**First confirmatory run (2026-09-20), superseded: supported for latency only.**
Thesis profile, 30 fresh scenes per regime, predictions written down first,
one command (`eval/dynamic_ior.py --regime all --seeds 30 --seed0 1000`).
Object-based IOR reaches new objects sooner than space-based IOR in every
regime (−1.2 / −2.2 / −0.8 frames; also against the motion-compensated tag) and
degrades gracefully with speed when identity is held (latency 1.79 → 1.85,
space-based 2.94 → 4.00); at high speed the thesis's own correspondence is not
enough. Coverage is 1.0 for all IOR arms. On sustained coverage (staleness) it
does not win — a third of its fixations go to object files that are not
objects: the open problem has moved from tracking to segmentation. Still open:
the speed × count sweep, the Abb. 6.14 scenario, DAVIS.

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

**Status (2026-07-19): done except Tier 3 (deferred to M18 by agreement).**
Tier 1 (`hog-person`, `haar-face`) and Tier 2 (`dnn-classify`, generic ONNX via
`cv::dnn`, weights via `tools/fetch_models.py`) ship as registry processors
runnable headless in `--attend` (`--processors`, `--process-cadence
dwell|frame|full-frame` — dwell = once per focus visit, the thesis's ~3-frame
attentive computation; full-frame = the ungated baseline arm). Label memory
(vote histogram + inspections) lives on object files; the `identification`
behavior implements recognition-triggered semantic IOR with a give-up rule.
Interchange grew additive `annotations`/`labels`/`processing` fields (still
v1). H2 result: **on cluttered multi-object video (vtest) gated recognition
recovers 51% of all full-frame detections within ±15 frames at 5.8% of the
recognition pixels** (honestly counted: what the detector actually scanned,
upscaling included) — H2 strongly supported where it matters. On DAVIS
single-actor sequences the gap decomposes cleanly into detector-limited
(parkour: HOG hits 17% of attended crops; every-frame re-inspection lifts
recall 0.23→0.71) vs allocation-limited (judo: 100% hit rate on only 3 looks —
salience never prioritizes the actors; a stage-1 problem, M17's top-down
channel is the fix). DAVIS-2017 adapter + curated person ids pay down the M12
leftover. Full story: `docs/GATED_RECOGNITION.md`.

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

**Status (2026-07-20): done.** `PriorityConfig` adds the two missing terms as
opt-in weighted channels — top-down task relevance (a native chroma-weighted
target-colour channel, plus an external `top_down_map` file that is the
interchange-boundary slot an M18 CLIP/LLM adapter fills) folded in at fusion,
and stage-2 history/value (object-value facilitation that rides on object
files, distinct from IOR, with a reward hook; plus a decaying location map).
All weights default 0 → the map stays bit-identical to the thesis map
(goldens unchanged). H5 result: on synthetic search the **dense value-specific
top-down channel is decisive (time-to-target 33.4 → 0.1 frames)** while
feature-*dimension* weighting is an honest near-null (28.9, CIs overlap) and
the history term buys target-hold not acquisition (0.10 → 0.16); on
**COCO-Search18 a content-blind category prior in the channel cuts mean
fixations-to-target 9.03 → 7.23 and lifts found@10 0.27 → 0.52** (human 2.58 /
0.92) — a lower bound on what a semantic source in the same slot (M18) buys.
COCO-Search18 adapter added. Full story: `docs/PRIORITY_MAP.md`.

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

**Status (2026-07-20): instrument built + verified end-to-end on a mock; real
V\*Bench run gated on a keyed VLM (ungated 2026-09: a local Ollama backend is
now the default).** The first cut rides the existing
`--emit-json` interchange (no new C++ mode; the full C++ virtual fovea stays
M15's): `eval/vlm_frontend.py` crops K native-res fovea windows around the top
saliency fixations plus one low-res global view, and scores three arms
(`full-res` / `uniform`-at-matched-budget / `fovea`) with a pluggable
`VLMBackend` (`mock` + `claude`, anthropic SDK, opus-5, base64 blocks, real
`count_tokens`; since 2026-09 an `ollama` default, local `qwen3.8:27b`, real
token counts from `prompt_eval_count`). Token cost is reported both as a provider-independent
patch estimate (CI-safe) and the backend's real token count when keyed; only the
fraction vs full-res is compared. The **mock answers correctly iff the target is
delivered at usable resolution**, so the synthetic `--demo --check` smoke is a
genuine end-to-end test — and it already shows the H6 shape: **fovea holds 100%
accuracy at 27% of full-res tokens where the token-matched uniform downsample
drops to 0%** (a crop lands on the small salient target the downsample loses).
Crops are bottom-up now with the M17 `top_down_map` slot wired for
question-conditioned (H5×H6) crops later. V\*Bench adapter added
(`eval/datasets/vstar.py`). **First real numbers (2026-09-15, local Qwen,
pilot scale, with an oracle-crop arm and the question-conditioned `fovea-td`
arm — VLM grounding on the global view → M17 `top_down_map` → crops; V\*Bench +
HR-Bench 8K):** oracle crops beat full resolution on single-target questions at
a third of the tokens (1.00 vs 0.85), so the front-end works when attention
lands; bottom-up crops rarely land (10% of targets covered); grounding lifts
coverage to 65% and accuracy 0.45 → 0.70 but doesn't yet beat the same-budget
uniform arm. (The HR-Bench pilot rows were first scored with the correct
option always "A", which inflated the blind uniform arm; fixed and rerun
2026-09-19 — and the two-way
relative-position questions sit at chance at this n, so "relations favour the
whole view" is not yet shown.) Open: the full V\*Bench run and a budget sweep.
Full story: `docs/VLM_FRONT_END.md`.

### M19 — Object files as a video token cache (H7 = H1 × H6)

Why: M18 works on stills, where the second stage is idle — a single image is a
stream of length one, so object files, IOR and identity never engage, and the
still-image gains come from *what* to look at (top-down), not from attention
dynamics. The system's distinctive part — tracked object files, object-based
IOR, persistent memory — shows its advantage only on dynamic scenes. M12's
honest result says where to look: object-IOR only ties space-IOR on
*exploration*; the predicted win is persistent identity-keyed memory vs
necessarily-decaying spatial memory. A video VLM front-end is exactly the task
that needs it: a video VLM pays tokens per frame, and deciding which pixels to
(re)send requires knowing *what* has been seen, not *where* — spatial memory
cannot tell "the same object moved" from "a new object arrived", so it either
re-sends moved objects (wasted tokens) or suppresses newcomers that appear
where an old crop was (misses). Paper A leads with this; the still-image study
is its per-frame component.

Arms (same VLM, same question, matched budget except the reference):
`frames-full` (T native frames — the naive reference, ~10× the budget),
`frames-uniform` (the same T frames downsampled to the budget — the standard
video-VLM input), `space-ior` (overview + K crops at spatial-IOR fixations,
deduplicated by *location*), `object-ior` (overview + K crops, one per *object
file*), `oracle` (one crop per ground-truth object). Later: question-
conditioned crops (M18's `fovea-td`), re-send-on-change, object files as text
memory.

Stages:
1. **Synthetic (v1):** `tools/make_dynamic_scene.py --tags --late` — disks
   carrying a small code legible only at native resolution, some arriving
   mid-video; `eval/vlm_video.py` asks per object "what code is on the
   ⟨colour⟩ disk?". Sweeps: speed × object count × K, ≥ 10 seeds.
2. **Real video:** DAVIS-2017 with templated questions from the masks.
3. **Working memory:** object files (labels, trajectories) handed to the VLM
   as text next to the crops — closes M13 Tier 3; re-send on appearance change.

Metrics: accuracy vs visual tokens; objects delivered legibly; crop
efficiency (crops spent on not-yet-seen objects); M12's scanpath coverage /
latency / waste for context. Honesty: publish the regimes where space-IOR ties
or wins (slow scenes, few objects) and the cost of label switches.
Deliverable: `docs/VLM_VIDEO.md`.

**Status (2026-09-18): v1 done and merged into main.**
The thesis object files lost identity to segmentation — saliency on moving
disks is hollow onset rings plus symmetry responses between objects — so
opt-in **proto-object segmentation** (seed by figure-ground colour contrast,
grow by colour, drop background clusters) and **persistent identity** were
added. At a code size calibrated to the VLM (8 px: readable from a native
crop, at chance from the downsampled views), **with a real VLM (local Qwen,
10 scenes, same token budget): identity-keyed crops 0.95, location-keyed
crops 0.80, budget-matched frames 0.27 (chance 0.25)** — H7's effect;
proto-objects carry it (thesis segmentation: 0.45 vs 0.52). DAVIS-2017 (30
sequences, categories hand-labelled): every arm 0.96–0.98 at 480p — no
separation. The pre-registered harder test (the same sequences at full
resolution, run 2026-09-18) **refuted the prediction**: every budget arm still
answers alike (0.957–0.978), because "which of these can be seen?" survives
downsampling — the task, not the resolution, is the limit, so the H7 evidence
stays synthetic until a real-video task needs fine detail. Identity on real,
textured video remains the open problem. Full
story: `docs/VLM_VIDEO.md`; where this stands for a publication (and what a
reviewer would object to): `docs/PAPER_READINESS.md`.

## Datasets

| Dataset | For | Status |
|---|---|---|
| `data/samples/` + synthetic generators | replication, IOR sweeps, demos | in repo |
| MIT1003 | still fixations/scanpaths (M11, M14) | adapter in repo (maps + per-observer sequences via DATA/ + scipy) |
| CAT2000 | second stills dataset (M11 cross-check) | adapter in repo (`eval/datasets/cat2000.py`; pooled fixation locations + maps) |
| DAVIS 2017 | video object masks → exact IOR scoring (M12, M13) | adapter in repo (`eval/datasets/davis2017.py`) |
| DIEM (or DHF1K) | video gaze (optional M14 extension) | choose when needed; DIEM easiest |
| KITTI raw (stereo) | stereo video (M15) | adapter to write |
| PETS2006 | left-luggage scenario (M16) | pointer only |
| COCO-Search18 | target-present visual search (M17 top-down ablation) | adapter in repo (`eval/datasets/cocosearch18.py`) |
| V*Bench / hi-res VQA | VLM token-vs-accuracy curve (M18) | adapter in repo (`eval/datasets/vstar.py`) |
| HR-Bench 4K/8K | the same at higher resolution (M18) | adapter in repo (`eval/datasets/hrbench.py`) |
| Synthetic tagged dynamic scenes | video token cache (M19) | generator in repo (`tools/make_dynamic_scene.py --tags`) |

Corpora stay pointed-to, never redistributed (v2 convention).

## Conventions for the phase

- Study runners live in `eval/` (one script per study, each with a mock-backed
  CTest smoke) and stimulus generators in `tools/`; results land outside git
  (`results/`), reports and key plots land in `docs/`. (The originally planned
  `experiments/<area>/<name>/` layout was never adopted.)
- Every quantitative claim: ≥20 seeds where stochastic, bootstrap CIs, and
  train/test splits for anything tuned. *As of 2026-09 only the H5 synthetic
  study meets this bar; closing the gap is `docs/HYPOTHESIS_CLOSURE_PLAN.md`.*
- Every experiment reproducible from one command; smoke-test versions wired
  into CTest where they're fast enough.
- The thesis model remains the default everywhere; everything new is opt-in
  by config (v2's M9 principle).

## Recommended order & rationale

**M10 → M10b → M12 → M13 → M17 → M11 → M14 → M15 → M18 → M19 → M16.**
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
