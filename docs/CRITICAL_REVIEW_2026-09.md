# Critical review — goals, progress, method, and the way forward (2026-09-19)

*A deliberately adversarial stock-take of the v3 science phase, requested
2026-09-19. Inputs: `docs/V3_ROADMAP.md`, the five study documents, ADR-0004,
`docs/PAPER_READINESS.md`, `docs/RESEARCH_POSITIONING.md`, the git state of
all branches and worktrees, a read-through of the evaluation code with
recomputation from stored results, a build/test/hygiene audit, and a
literature scan (2024–26). Everything marked **[verified]** was checked
against code, data or git during this review; literature items come from a
web scan and are flagged where they were not opened and read.*

*This document complements `docs/PAPER_READINESS.md` (which asks "how far
from a paper?") — it asks the wider question: are we working on the right
things, the right way, and is what we wrote down true?*

---

## 1. Summary

**Verdict.** The project is in better scientific shape than most side projects
of its kind — and in worse shape than its own documents suggest. The
instrument is excellent: a protected thesis default, registries, one
interchange format, mock backends that make every study CI-testable, and
decomposition arms (oracle, perfect tracker) that say *why* an arm loses. The
honesty culture is real: negatives are published, one prediction was
pre-registered and its refutation reported. But **breadth has outrun
closure**: seven hypotheses are open, none is closed at the roadmap's own
statistical bar, the two cheapest credibility anchors (M10 replication, M11
human scanpaths) have not been run, and the one positive headline (H7) was
developed and reported on the same ten seeds against baselines that a
reviewer would call weak.

**The five findings that matter most**

1. **HR-Bench scoring is broken [verified].** The adapter keeps only
   rotation 0 of CircularEval, in which the correct answer is option **A for
   all 400 items** (4K and 8K), and the harness never shuffles options. A
   model that says "A" when it cannot see the target scores 100%. The
   HR-Bench rows in `docs/VLM_FRONT_END.md` — including the "strong uniform
   arm" they were used to argue for — are uninterpretable until fixed (§4.1).
2. **No headline number meets the roadmap's own bar** (≥ 20 seeds, bootstrap
   CIs, train/test split for anything tuned) except the H5 synthetic search.
   H1's regime map — the basis of the README claim that object-IOR "moves
   ahead at high speed" — is 6 seeds, no CIs, hand-aggregated (§4.2).
3. **The H7 headline was tuned on its test seeds**, and its comparison arms
   are internal. Proto-objects, persistent identity, closing, size cap and
   the 8-px code size were all developed on seeds 0–9; the Qwen result is
   reported on seeds 0–9. The location-keyed baseline has a permanent,
   never-decaying, never-swept dedup rule, and the obvious trivial baseline —
   *one crop per new colour*, no object files at all — was never run, although
   on these scenes identity **is** colour (§4.3). The good news: a correct
   paired scene-level reanalysis of the stored results makes the
   object-vs-location gap *stronger* (+0.15, 95% CI [0.07, 0.23]; 9 vs 0
   discordant questions), so the effect will probably survive fresh seeds.
4. **The plan's first milestone was never started.** M10 (replication
   dossier) has an empty branch and worktree; M11 (human scanpaths) has been
   parked, instrument-complete and unmerged, since 2026-07-21 for want of a
   dataset download. Those are exactly the two items `PAPER_READINESS.md`
   lists as "missing" for the *closer* paper (§3, §6).
5. **Documents contradict each other and git** on the state of H1 and M19
   (§5). None is dishonest; all are drift — status lives in five places and
   is updated in one.

**The five recommendations that matter most**

1. **Stop opening hypotheses; close some.** A two-week "closure sprint" with
   no new mechanisms: fix HR-Bench, run H1 and H7 confirmatory on fresh
   seeds at n ≥ 20 with paired statistics, run the full V\*Bench, download
   MIT1003 and run M11 (§8, step 1).
2. **Make pre-registration the rule, not the exception.** The 2026-09-18
   DAVIS prediction-then-refutation is the best piece of method in the
   repository. Every confirmatory run gets a frozen config, fresh seeds, and
   a written prediction first (§7).
3. **Give every comparison a baseline we did not build** — and a *strong*
   version of the arm we expect to lose (motion-compensated spatial IOR;
   decaying location cache; colour-keyed dedup; random / grid / frame-
   difference crops; ViCrop-style VLM-attention crops) (§4.3, §9).
4. **Lead with paper B, and reframe it around the psychophysics.** The human
   literature says object-based IOR in dynamic scenes is fragile and depends
   on a stable object representation — which is exactly what H1 found. "A
   2004 model's central claim, re-tested: it fails the way human data says it
   should, and here is the identity condition under which it holds" is a
   distinctive, finishable paper (§8, §9.1).
5. **For paper A, change the task, not the model.** The crux is a real-video
   task whose answer needs fine detail. Candidates exist (HRVideoBench,
   EgoTextVQA, SoccerNet-GSR); authoring DAVIS questions by hand is the
   slower road (§9.2).

---

## 2. Goals — what we set out to do, and how they moved

| Layer | Stated goal | Where it stands |
|---|---|---|
| v2 (M0–M9) | Reimplement the thesis model at *loose behavioral equivalence*; pluggable pipeline; eval layer; live demo | **Done**, and done well. The protected-default discipline has held through every later milestone. |
| v3 original (2026-07-10) | Replicate thesis findings → prove H1 (the centerpiece) → human scanpaths → gated recognition → model lab → stereo/virtual fovea → scenarios | H1 studied (honest negative, then a partial rescue); H2 done; **M10, M11 (real data), M14, M15, M16 not done; H3, H4 untested** |
| Positioning pass (same day) | Priority map (M17), VLM front-end (M18) as the "current" contributions | Both built; H5 supported on constructed stimuli + a weak real-data effect; H6 currently *loses* on stills |
| Papers phase (2026-07-26) | Two arXiv papers, A (VLM front-end) then B (self-replication) | Re-assessed 2026-09-16: B is closer; A blocked on a real-video win |
| M19 (2026-09-15) | H7 = H1 × H6, "where the second stage earns its keep" | Synthetic win with a real VLM; pre-registered real-video test refuted |

**Observation: the goal has drifted from "prove the thesis's claim" to "find
a task where the second stage wins".** The sequence is: H1 negative →
identity-centric reframing → M19 video token cache → synthetic win → DAVIS
tie → "needs a different task". Each step is defensible and honestly
reported. But taken together it is a search over *tasks and mechanisms* for a
positive result, and it should be named as such — both because a reviewer
will name it, and because it changes what counts as evidence (§7.1).

**Observation: the recommended order was abandoned without a decision.** The
roadmap says *M10 → M10b → M12 → M13 → M17 → M11 → M14 → M15 → M18 → M19 →
M16* and argues replication comes first because it "anchors credibility".
What happened: M10b (half) → M12 → M13 → M17 → M18 → M19. The exciting,
timely milestones were pulled forward — the roadmap explicitly permits that —
but the anchoring ones were silently dropped rather than re-scheduled.

---

## 3. Scorecard

### 3.1 Hypotheses

Evidence grades: **A** = meets the roadmap bar (n ≥ 20, CIs, held-out);
**B** = real data or real VLM, but pilot-scale or tuned-on-test;
**C** = directional only (few seeds, no CIs, or constructed stimulus);
**–** = not tested.

| H | Claim (short) | Current reading | Grade | What the grade hides |
|---|---|---|---|---|
| H1a | IOR ≫ no-IOR in dynamic scenes | Supported, large effect | C | 1–3 seeds; but the effect is so large (waste 0.85 vs 0.1) that it will survive |
| H1b | Object-IOR > space-IOR | **Not supported** by default; "ahead on latency at speed" with persistent identity | C | 6 seeds, no CIs, radius chosen per regime; waste metric saturates (§4.2) |
| H2 | Gated recognition ≈ full-frame at a fraction of compute | 51% recovery at 5.8% pixels on vtest | C | One video; self-referential ground truth incl. false positives; ±15-frame window inflates recall via autocorrelation; **no random-ROI or motion-ROI control**; "strongly supported" overstates — 51% is not "near-full-frame" |
| H3 | Depth priority | — | – | M15 not started |
| H4 | Scanpath plausibility vs humans | — | – | Instrument built on an unmerged branch; zero real-data numbers |
| H5 | Priority map > salience for search | Supported | A (synthetic) / B (COCO) | Synthetic result is near-tautological (a red detector finds red). COCO: first-150 trials, not a random sample; **no category-agnostic centre-prior control**, so the gain may be centre bias, not category knowledge |
| H6 | Fovea crops preserve VLM accuracy at lower budget | **Currently refuted for bottom-up crops** on V\*Bench; oracle shows headroom | B | n = 20/category; "oracle beats full-res" is 3 vs 0 discordant (McNemar p ≈ 0.25); relative-position items are 2-way (chance 0.5, scores 0.60–0.70 are at chance); HR-Bench rows invalid |
| H7 | Object files as a video token cache | Supported on synthetic scenes (0.95 / 0.80 / 0.27); **refuted as pre-registered on DAVIS** | B (synthetic) | Tuned on test seeds; identity ≡ colour on these scenes; location baseline never tuned; stimulus calibrated to the VLM |

### 3.2 Milestones [verified against git, 2026-09-19]

| M | Deliverable | State |
|---|---|---|
| M10 Replication dossier | `docs/replication/REPLICATION_DOSSIER.md`, `experiments/replication/` | **Not started.** `module/replication` has zero own commits; neither directory exists on any branch. |
| M10b Selection backends | B, D, E (+ A, C, F optional) + verdicts + the A/B payoff experiment | B (`kalman-mot`) and D (`normalization`) on main since 2026-07-12. **E not built; the payoff experiment (same saliency stream, vary backend → accuracy + parameter sensitivity + runtime) never run; no verdicts.** `docs/SELECTION_BACKENDS.md` still says "to be built". Neither backend was used as an arm in M12 or M19, which is what they were for. |
| M11 Scanpaths vs human | `docs/SCANPATH_VS_HUMAN.md` with real numbers | Instrument + tests on `module/scanpath-human` (2 commits, 27 behind main, README conflict). Only a synthetic demo table. Blocked on downloading MIT1003 — scipy is already installed. |
| M12 Dynamic IOR | H1 confirmed/refuted with regime map, ≥ 20 seeds/cell, CIs, Abb. 6.14 scenario, DAVIS | First cut + two follow-ups. **None of: ≥ 20 seeds, CIs, the speed × count × occlusion sweep, the "money plot", the Abb. 6.14 scenario, DAVIS scoring of H1.** `eval/dynamic_ior.py` has no seed loop — every multi-seed table was aggregated by hand. |
| M13 Gated recognition | processors, label memory, behavior, H2 study | Done (Tier 3 deferred). |
| M14 Model lab | `eval/lab.py`, `configs/models/` | Not started. |
| M15 Stereo video + virtual fovea | `--fovea`, KITTI | Not started. |
| M16 Scenarios | Watchman, left luggage, find-my-object | Not started. |
| M17 Priority map | channels + H5 study | Done. |
| M18 VLM front-end | mode + savings *curve* | Instrument done; **pilot only** (20/category of 191; no budget sweep — single operating point; the full run the doc header promises "when it completes" was never launched). |
| M19 Video token cache | harness + synthetic + DAVIS | Done and **merged** (the roadmap says unmerged). Sweeps (speed × count × K) not run. |

---

## 4. Methodological critique

### 4.1 HR-Bench: the answer is always "A" [verified]

`eval/datasets/hrbench.py` keeps only `cycle_category == 0`. In HR-Bench's
CircularEval layout rotation 0 places the correct option first:
`extracted_4k/items.jsonl` and `extracted_8k/items.jsonl` are **200/200 "A"**
each. `eval/vlm_frontend.py` does not shuffle options (no `shuffle`/`random`
anywhere in it or `vlm_backends.py`), and the answer parser takes the first
isolated valid letter — so a reply beginning with the article "A …" also
parses as option A.

Consequences: (a) stated chance (0.25) is not the floor — a position-biased
or blind model can score up to 1.0; (b) the conclusion "Qwen reads small
detail from downsampled images better than the legibility floor assumes" is
partly supported by numbers that may be position bias; (c) the
full-res-vs-uniform-vs-fovea ordering on HR-Bench cannot be read at all.

**Fix (small):** shuffle options per item with a seeded RNG shared across
arms (as `vlm_video.py` already does), or score all four rotations
(CircularEval proper). Persist the raw reply and parsed letter per row so
position bias can be audited after the fact. Then rerun and replace the two
HR-Bench rows. Also state chance per V\*Bench category: relative-position
items are 2-way, so 0.60–0.70 at n = 20 is *at chance*, which changes the
"relations favour the whole view" reading from a finding to a non-result.

### 4.2 H1: the centerpiece has the thinnest statistics

- **n.** 1, 3, 4 and 6 seeds; no CIs anywhere; the study doc itself says
  "a direction, not yet a result" — but the README and `PAPER_READINESS.md`
  promote it to "moves ahead on latency at high speed" and "from clearly
  worse to ahead".
- **Not reproducible from one command.** `eval/dynamic_ior.py` has no seed
  loop; tables were averaged by hand. This violates the roadmap convention
  and means nobody (including us) can regenerate the regime map.
- **Revisit-waste saturates.** Waste only accrues while an unseen object is
  visible. Both IOR arms reach full coverage in 2–8 frames, so the metric is
  decided by a handful of fixations per scene and is silent for the remaining
  ~85% of the video. It is a low-power metric for exactly the comparison we
  care about. Off-object fixations are dropped from the denominators, so an
  arm that often fixates background (object-IOR: 20–37% under thesis
  segmentation) is not penalised for it.
- **The arms are not quite "identical except for the inhibition domain".**
  Spatial inhibition sums over spots without a cap and spreads through a
  60-px Gaussian on a 320×240 frame (it also inhibits *neighbours*); object
  inhibition is capped at 1.0. Probably second-order, but ADR-0004's claim of
  a single-variable ablation should say so.
- **Asymmetric engineering effort.** ADR-0004 commits to "do not tune
  object-IOR until it wins". Since then the object arm received motion
  prediction, appearance matching, persistent identity, duplicate folding,
  closing, a size cap and proto-objects; the spatial arm received one knob
  (`--ior-radius`). Every addition was reported honestly — but the fair
  comparison is *best object-IOR vs best space-IOR*. The missing strong
  spatial baseline is **motion-compensated spatial IOR**: shift the
  inhibition map by the local flow / predicted velocity. It needs no
  identities, keeps space-IOR's robustness, and removes its one theoretical
  weakness. If object-IOR cannot beat *that*, the H1 story is settled.

### 4.3 H7: a real effect, shown against the wrong opponents

What is solid: the paired scene-level reanalysis (object-ior − space-ior =
+0.15 [0.07, 0.23], 9 vs 0 discordant) is stronger than the unpaired
question-level CIs the doc reports; the perfect-tracker arms show the lever
is large; token budgets are genuinely matched in real tokens (530 / 533 /
535).

What a reviewer will say:

1. **Tuned on test.** Every mechanism the headline depends on was iterated
   against seeds 0–4 / 0–9 with the mock, then the Qwen number was taken on
   seeds 0–9. Fresh seeds cost nothing. *This is the cheapest serious defect
   in the project to fix.*
2. **Identity ≡ colour.** Objects have distinct flat colours; the question
   is keyed by colour; the tracker's appearance descriptor is mean colour;
   proto-objects seed by colour contrast against the median. A baseline that
   keeps "one crop per new colour cluster" — no object files, no tracking, no
   IOR — plausibly reaches the oracle. Until that is run, the result shows
   that *colour-keyed dedup beats location-keyed dedup*, not that *object
   files* are needed. The discriminating stimulus is **same-coloured objects**
   (identity carried only by spatiotemporal continuity) — the actual
   definition of an object file since Kahneman & Treisman, and the case where
   an appearance hash fails and tracking must work.
3. **The location baseline is a strawman.** `location_picks` dedups against
   every earlier crop forever, with a radius fixed at half the crop side,
   never swept. M12's own argument is that spatial memory *must decay* — yet
   the spatial arm here was given memory that never does. Sweep radius ×
   decay and report the best.
4. **The frames arm is one point.** 4 frames at scale 0.30. With the same
   budget one could send 1 frame at 0.6 or 16 at 0.15; and keyframe selectors
   (AKS-style) are the standard. The calibrated 8-px code makes any
   whole-frame arm blind by construction, which is fine for isolating the
   mechanism but means the 0.27 is a design parameter, not a finding.
5. **The mock and Qwen agreeing is not independent confirmation** — the
   mock's legibility floor was set (`--min-target-px 5`) to agree with Qwen.
6. **Small confound:** the prompt preamble differs by arm (`describe()`).

### 4.4 H6 (stills)

- **Addendum (same day):** a random-fixation baseline reproduces the coverage
  figures below exactly (top-3 0.069 vs 0.068, top-10 0.219 vs 0.220) — the
  bottom-up map carries no target information on V\*Bench; and a feature audit
  found that the static thesis features are partly not the thesis's algorithms
  and partly defective. See `docs/FEATURE_ASSESSMENT.md`. This supersedes the
  "clean negative on saliency as a crop selector" framing in §6 item 7: the
  negative is about *this* stage 1, not about bottom-up saliency.
- The most informative number in the project may be the model-free one:
  bottom-up fixations cover the V\*Bench target on **2% / 7% / 22%** of items
  (top-1 / 3 / 10), over all 191 items. That is a clean, citable negative —
  and, per the literature scan, apparently unpublished: nobody seems to have
  measured classic saliency as a VLM crop selector.
- "Oracle crops beat full resolution (1.00 vs 0.85)" is 3 discordant items
  of 20. Say "match or exceed", or run all 191.
- There is no **budget sweep**, only one operating point (≈ 0.36 of
  full-res), at which the uniform arm is still comfortable. H6 predicts the
  gap opens as the budget tightens and resolution grows; the design never
  goes there. The roadmap deliverable is a *curve*.
- The VLM runs locally and V\*Bench is 191 items. The full run has been
  "pending" for a month with no technical blocker.
- The one stage-1 lever never pulled: every VLM study ran the *thesis*
  feature set. The repository contains six alternative saliency operators
  (M9) and a documented finding that symmetry fires *between* objects and
  onset draws hollow rings. Crop-hit-rate per feature set / per operator is a
  model-free, one-afternoon experiment on data already on disk.

### 4.5 H2 and H5

- **H2** needs a control that separates *attention* from *any sparse ROI
  policy*: random ROIs and frame-difference ROIs at the same pixel budget.
  On a static-camera pedestrian video, motion gating is the incumbent
  engineering answer and may well beat saliency. The reference set
  (full-frame HOG detections, false positives included, ±15-frame window)
  lets one gated detection match up to 31 autocorrelated references; report
  per-track recovery instead.
- **H5** on COCO-Search18: add a *pooled* (category-agnostic) prior. If it
  recovers most of 9.03 → 7.23, the finding is "centre bias helps", not
  "category knowledge helps". Sample trials at random rather than the first
  150. The human CI treats 1375 subject×image trials as independent (too
  narrow). `hold-fraction` conflates holding with finding (never-found runs
  score 0).

### 4.6 Cross-cutting

- One unpaired percentile bootstrap (`eval/study_common.py`) is the only
  inferential tool. All arms run on identical scenes/items, so **paired**
  differences are free power — the H7 reanalysis shows how much.
- `--resume` reuses stored rows without checking that the config matches;
  all pilot runs used it. A config hash in the results file closes this.
- Raw VLM replies are not stored, so no result can be audited post hoc.
- The controller's compute is acknowledged everywhere and measured nowhere
  in the VLM studies. One table (ms/frame for attention vs. ms/token saved
  for the local VLM) would turn a caveat into an argument — or end it.

---

## 5. Inconsistencies and mistakes found

| # | Where | Problem | Status |
|---|---|---|---|
| 1 | `eval/datasets/hrbench.py`, `docs/VLM_FRONT_END.md` | Correct answer always "A"; no option shuffling; stated chance 0.25 is wrong as a floor | **Bug [verified]** |
| 2 | `docs/V3_ROADMAP.md` M19 status, memory index | "v1 done on `module/video-token-cache` (unmerged)" — it is merged (594210a); the status also omits the full-resolution DAVIS refutation | Stale |
| 3 | H1 across documents | Roadmap M12 status + H7 text: "only ties"; ADR-0004: "nearly tied — not a win"; README + `PAPER_READINESS.md`: "moves ahead on latency at high speed"; study doc: "six seeds without CIs: a direction, not yet a result". The strongest wording is in the most public file. | Contradiction |
| 4 | `docs/DYNAMIC_IOR_STUDY.md` | Body says "space-based IOR is at least as good … usually better" in *every* regime; the later table shows object-IOR ahead in the *standard* regime already with aids only (4.54/0.113 vs 5.54/0.145). Different seed sets, never reconciled. | Internal inconsistency |
| 5 | `docs/VLM_VIDEO.md` "Next" | Item 1 is "the crux run above, once a VLM is available" — it was run and refuted two sections earlier; list numbered 1, 2, 2, 3, 4; "How it works" says codes are ~14 px, the results use 8 px | Stale |
| 6 | `docs/VLM_FRONT_END.md` header | "headline numbers land when the full run completes" — no full run exists or is running | Stale promise |
| 7 | `docs/SELECTION_BACKENDS.md` | "To be built after the M9 commit lands" — B and D have been on main since 2026-07-12; no verdicts section | Stale |
| 8 | `docs/V3_ROADMAP.md` intro | Says hypotheses "H5/H6", CLAUDE.md says "H1–H6"; the roadmap defines H7 | Minor |
| 9 | H2 wording | Hypothesis: "near-full-frame accuracy"; result: 51% windowed recovery; verdict: "strongly supported" | Overclaim |
| 10 | Roadmap conventions | "`experiments/<area>/<name>/` = config + runner + README" — no `experiments/` directory exists; runners live in `eval/` and `tools/`. "≥ 20 seeds, bootstrap CIs, train/test splits" — met once (H5 synthetic) | Convention not followed |
| 11 | Papers plan vs `PAPER_READINESS.md` | Agreed venue: arXiv + website, no deadline pressure; the readiness doc plans toward "a main-conference submission" | Unreconciled |
| 12 | `module/replication` branch + worktree | Exists, empty — suggests work that does not exist. The `module/scanpath-human` worktree lives in a directory named `recognition` | Confusing |
| 13 | README caption | "driven by bottom-up saliency and **object-based** inhibition of return" for the default still-image run; `configs/default.yaml` is NMS selection on a single image, where (per `VLM_VIDEO.md`) "object files, IOR and identity never engage" | Check wording |
| 14 | `CLAUDE.md` (uncommitted) | "Hot loops get a **criterion** benchmark" — Criterion is the Rust benchmarking crate; this is a C++/Catch2 project (Catch2's `BENCHMARK` or Google Benchmark would be the equivalent) | Likely copy-paste |
| 15 | `docs/progress/weekly_log.md` | Empty week-1 template from the phase-1 workflow | Dead file |
| 16 | `docs/RESEARCH_POSITIONING.md` | Sources were not re-verified since July; the front-end's actual nearest neighbours (ViCrop, ZoomEye, FOCUS; TimeChat-Online, TrajViT for video) are absent | Out of date |

*(Build, test and repository-hygiene findings: Appendix A.)*

---

## 6. What is innovative and promising

Criticism above notwithstanding, several things here are genuinely good, and
a few are better than their current billing.

1. **The decomposition arms.** Oracle crops and perfect-tracker
   (`-gtid`) arms split every loss into *budget* vs *attention* vs *tracker*
   vs *recognizer*. M13's "detector-limited vs allocation-limited" and M19's
   "attention, not the token budget, is the binding constraint" are the kind
   of statements most VLM-efficiency papers cannot make. This is a reusable
   *methodological* contribution and `PAPER_READINESS.md` is right to call it
   the carrier of a workshop paper.
2. **The honest H1 negative, which matches human data.** The psychophysics
   literature (Müller & von Mühlenen 1996; Christ, McCrae & Abrams 2002;
   Krüger & Hunt 2013; review: Reppa, Schmidt & Leek 2012) reports that
   spatial IOR is the robust effect and object-based IOR under motion is
   small, fragile, and dependent on the strength of the object
   representation. We found computationally that object-IOR "is only as good
   as its tracker". That is not an embarrassing failure of the thesis — it is
   a *mechanistic explanation of a known human fragility*. This framing is
   not in any of our documents yet, and it is the most publishable idea in
   the repository. *(Citations from the literature scan; read the originals
   before relying on magnitudes.)*
3. **"What, not where" as the token-cache key.** Location/change-keyed
   token dropping is the state of the art for streaming video VLMs
   (TimeChat-Online's differential token drop); learned trajectory tokens
   (TrajViT) are the trained counterpart. A *training-free, identity-keyed*
   cache sits in an unoccupied spot between them. The idea is sound even
   though the current evidence is not yet.
4. **The mock-as-legibility-oracle.** A deterministic backend that answers
   correctly iff the target was delivered at usable resolution makes every
   VLM study a CI test, gives model-free "delivered" rates, and cleanly
   separates "attention missed" from "VLM misread". Few research repos can
   regression-test their experiments.
5. **The protected default.** Seven milestones of new mechanism, all opt-in,
   goldens untouched. This is why a replication paper is still possible after
   all the modern additions.
6. **Pre-registration in a side project.** Prediction, refutation criterion
   and "what to watch" written down before the 2026-09-18 run, and the
   refutation published within the day. Keep doing exactly this.
7. **The clean negative on bottom-up saliency as a crop selector** (2 / 7 /
   22% target coverage on V\*Bench). Apparently unmeasured in the literature;
   a useful data point for anyone tempted by "saliency-guided token pruning".
8. **Proto-object segmentation** — the diagnosis (onset rings, symmetry
   between objects) is a real insight into why thesis-style saliency
   clusters are not objects, and the fix connects to a living literature
   (Walther & Koch; Russell et al.'s border-ownership proto-objects).

---

## 7. Way of working — what to keep, what to change

**Keep:** milestone-sized briefs; adversarial review at the end of each
(it found 10 real defects in M11); opt-in-by-config; mock-first instruments;
"sharper H" restatements after each study; ADRs; the worktree layout.

### 7.1 Separate *exploration* from *confirmation*

The current loop is: build mechanism → look at result → diagnose → add
mechanism → look again, all on seeds 0–9, and the last look becomes the
headline. That is the right loop for *engineering* and the wrong one for
*claims*. Proposal:

- **Dev seeds** (0–9) for iteration; **test seeds** (e.g. 1000–1029) touched
  only by a confirmatory run with a frozen, committed config.
- Before each confirmatory run, a short **prediction block** in the study
  doc: expected ordering, the refutation criterion, the analysis (paired
  bootstrap over scenes). The DAVIS block is the template.
- A result is quoted in the README only after its confirmatory run. Until
  then the README says "pilot".

### 7.2 Add a definition of done for a *hypothesis*, not only for a milestone

Milestones currently close when the instrument exists and one table is in a
doc ("first cut done", "instrument built + mock-verified"). Hypotheses never
close. Suggested DoD: confirmatory run at the roadmap bar · paired
statistics · at least one external baseline and one strengthened version of
the losing arm · one-command reproduction · status updated in the single
status table (7.4). M12's own deliverable list is a good DoD — it just was
never checked off.

### 7.3 Limit work in progress

Seven hypotheses, four with partial evidence, three worktrees, one milestone
parked for two months behind a download. A simple rule — *no new H until one
is closed* — would have forced M11's real run and H1's 20-seed sweep before
M19. The September sessions (27 commits in three days) show how much
throughput is available; it went to a new hypothesis instead of closing old
ones.

### 7.4 One source of truth for status

Status currently lives in the roadmap (per-milestone prose), each study doc's
header, the README's "What it found", `PAPER_READINESS.md`, and ADR-0004.
Every contradiction in §5 is one of these lagging another. Proposal: a single
`docs/STATUS.md` table (hypothesis · claim · grade · n · last confirmatory
run · doc link) that the README links to instead of restating numbers; study
docs keep the detail; the roadmap keeps the plan, not the results. Add
"update STATUS.md" to the merge checklist.

### 7.5 Make experiments reproducible artifacts

- Every runner gets `--seeds N --seed0 S`, writes per-item rows with raw
  model replies, the git SHA, and a config hash; `--resume` refuses on a hash
  mismatch.
- One shared `paired_bootstrap(arm_a, arm_b, unit=…)` in
  `eval/study_common.py`; every doc table that compares arms reports the
  paired difference.
- Either create `experiments/` as the roadmap says or delete the convention.

### 7.6 Baselines before mechanisms

When an arm loses, the reflex so far has been to add a mechanism to it. Add a
counter-reflex: before building anything, run the dumbest baseline that could
explain the effect (random crops, colour dedup, centre prior, frame
differencing). Several of those would have redirected effort earlier.

### 7.7 Unblock data the day it blocks

M11 waited two months on a download that needs a human. Keep a short
"needs Gerriet" list (downloads, accounts, visibility flips, pushes) at the
top of the roadmap so that blocked work is visible instead of silently parked.

---

## 8. Recommended path forward

**Step 1 — Closure sprint (no new mechanisms; ~2 weeks of sessions).**

| # | Task | Cost | Why first |
|---|---|---|---|
| 1 | Fix HR-Bench option order; store raw replies; state per-category chance; rerun HR-Bench pilot | hours | A known-wrong number is in a public doc |
| 2 | Fix the stale/contradictory statements in §5 (#2–8); soften H1 wording in README to match the study doc | 1 h | Free credibility |
| 3 | `dynamic_ior.py`: seed loop, paired bootstrap, a time-averaged revisit metric that doesn't saturate; run the 3 regimes × 30 fresh seeds, frozen config, prediction written first | 1 day, no VLM | Turns the centerpiece from C to A — whichever way it falls |
| 4 | H7 confirmatory: 30 fresh seeds, frozen `attend_proto.yaml`, paired scene-level analysis; **add arms**: colour-keyed dedup, decaying-location dedup (radius × decay swept on dev seeds), random crops | 1–2 days incl. VLM time | The one positive headline becomes defensible — or is exposed cheaply |
| 5 | H7 discriminating stimulus: same-coloured disks (identity only by continuity); optionally colour *changes* mid-video | 1 day | Tests *object files* rather than a colour hash; this is the experiment that makes H7 about the thesis |
| 6 | Full V\*Bench (191) + budget sweep (3–4 operating points) | VLM time, unattended | Replaces pilot with the curve the roadmap promised |
| 7 | Download MIT1003; rebase + merge `module/scanpath-human`; run M11 (remember the face-channel ablation) | 1 day after download | H4 goes from – to B; half of paper B's missing evidence |
| 0 | Default `CMAKE_BUILD_TYPE` to Release (Appendix A1) — the local build is unoptimized **[verified]**; commit the staged `.gitignore`; push main | minutes | Every run below gets faster; timings become meaningful |
| 8 | Delete the empty `module/replication` branch/worktree *or* start M10 in it; rename the `recognition` worktree | minutes | Removes phantom state |

**Step 2 — Decide with evidence (one session).** After step 1 the grades in
§3.1 are real. Then choose:

- **Paper B first** *(recommended)*. Title direction: *"Object-based
  inhibition of return, re-tested: a 2004 attention model, its central claim,
  and the identity condition under which it holds."* Spine: replication
  dossier (M10, scoped to ~8 findings — feature-variation curves, noise
  robustness, Abb. 6.14 occlusion tracking — not all of ch. 5/6) → H1 at full
  statistics with the motion-compensated spatial baseline → the
  label-switch/identity analysis → the psychophysics parallel → M11 as
  behavioural plausibility → H7-synthetic (with same-colour scenes) as "where
  persistent identity pays". Venue: arXiv as agreed; ReScience C ("Ten Years
  Reproducibility Challenge" precedent for self-replication — ask the editors
  about scope), *Computational Brain & Behavior*, or *Attention, Perception,
  & Psychophysics* are plausible journals if wanted later.
- **Paper A** becomes a focused follow-up whose go/no-go is a single
  experiment (§9.2), not an open-ended search. If the real-video task fails
  again, publish the workshop version: the decomposition instrument + the
  saliency-as-crop-selector negative + the synthetic identity result.

**Step 3 — Prune the roadmap.** M14 (model lab), M15 (stereo/KITTI/H3) and
M16 (scenarios) have had no work in ten weeks and serve neither paper. Mark
them *deferred* explicitly; keep the virtual-fovea idea only as far as paper A
needs it. M10b: either run the promised A/B (it is the natural home for
"kalman-mot as the object-file tracker" — see §9.3) or close it with a
two-paragraph verdict.

---

## 9. Ideas not yet researched

### 9.1 For H1 / paper B

1. **Motion-compensated spatial IOR** as the strong spatial baseline (§4.2).
2. **Tracker quality as the independent variable.** We have the pieces for a
   dose–response curve: thesis correspondence → +motion → +appearance →
   persistent identity → kalman-mot → ground-truth identity. Plot object-IOR
   advantage against measured ID-switch rate (IDF1/HOTA-style). "Object-IOR
   pays once ID-switches fall below *x* per object per 100 frames" is a
   quantitative, falsifiable statement — and the modelling counterpart to
   Reppa et al.'s "depends on the strength of the object representation".
3. **Metrics that exercise the asymmetry.** Long videos (≥ 500 frames) with
   objects leaving and re-entering; *re-inspection cost* under a fixed
   inspection budget; *time-to-detect a change* on an already-inspected
   object; revisit scheduling (longest-unseen-first needs identity).
   Coverage/latency reward spreading, which spatial IOR does for free.
4. **Human comparison for dynamic scenes.** DIEM/DHF1K gaze on video is in
   the dataset table and untouched. Do human refixation patterns on moving
   objects look object-keyed or location-keyed? That would connect H1 to
   behaviour directly.
5. **Rhythmic / theta sampling** — flagged as "genuinely new since the
   thesis" in the positioning doc, never tried. Cheap: re-select every N
   frames; compare against continuous field relaxation on the M12 metrics.
6. **Use the backends we built.** `kalman-mot` and `normalization` exist
   precisely as H1 arms and were never entered.

### 9.2 For H6/H7 / paper A

1. **A real-video task that needs detail — use an existing benchmark before
   authoring one.** From the literature scan, in order of fit:
   **HRVideoBench** (200 MCQs on ≥ 1080p video about small regions: OCR,
   counting, state change; small and public); **EgoTextVQA** (scene-text QA,
   the paper itself reports a high-vs-low-resolution gap — i.e. a published
   downsampling ablation; caveat: most questions need one frame);
   **SoccerNet-GSR** (1080p, tracked players *with jersey numbers* — identity
   tracks plus a fine-detail attribute, so identity-keyed crops can be scored
   directly; questions must be templated from annotations). Run the
   *model-free* legibility check first (does uniform downsampling at budget
   destroy the evidence region?) — one afternoon, no VLM — and only then
   spend VLM time. This is the pre-registration that would have saved the
   DAVIS full-resolution run.
2. **External crop-selection baselines.** Minimum set: random / grid /
   frame-difference crops (trivial), **ViCrop**-style crops from the VLM's
   own attention (the cheap strong training-free baseline; needs attention
   access, so an HF-transformers backend next to Ollama), and one tracker
   baseline (ByteTrack or SAM2 masks → crops). Cite ZoomEye, FOCUS, V\*/SEAL,
   DeepEyes/Chain-of-Focus as the trained/agentic upper family; TimeChat-
   Online (location/change-keyed token drop) and TrajViT (trajectory tokens)
   as the video neighbours.
3. **Better stage 1 for crop selection, measured model-free.** Target
   coverage@K on V\*Bench per feature set, per M9 operator, and for DeepGaze
   (adapter exists). If *no* bottom-up source gets past ~30%, H6-bottom-up is
   closed for good and the paper says so; if one does, it is a free win.
4. **Two-pass grounding that pays for itself.** `fovea-td` loses partly
   because the grounding call is charged (0.53 vs 0.36). Ground on a much
   smaller view (e.g. 256 px), or let the VLM return *several* candidate
   boxes and let saliency + IOR order them — the controller's actual job.
5. **Object files as text memory** (M13 Tier 3 / M19 stage 3): labels,
   trajectories and "already sent" flags handed to the VLM as text. This is
   the neuro-symbolic angle (positioning mode 4) and the one place where the
   second stage offers something a crop selector cannot: *state*. Questions
   that need it: "how many distinct objects appeared?", "did the red one
   leave and come back?", "which object entered last?". Uniform frames and
   stateless croppers should fail these by construction — a fairer home for
   H7 than code-reading.
6. **Re-send on change.** A cache needs invalidation. Appearance-change
   detection on the object file (the descriptor is already there) → re-crop.
   Stimulus: codes that change mid-video. Location-keyed and change-keyed
   (TimeChat-style) baselines become interesting opponents here.
7. **A second VLM and a quantization check** — already listed in
   `PAPER_READINESS.md`; cheap with Ollama.
8. **Controller cost table** (§4.6).

### 9.3 For the instrument

1. **Kalman-MOT as the object-file tracker** (promote association into
   `ObjectFileStore`, as ADR-0004 and the M12 doc both suggest). It was the
   agreed fix in July; persistent identity was built instead.
2. **Texture-capable appearance descriptor** (colour histogram or the
   per-feature signature the pipeline already computes) — the named blocker
   for identity on DAVIS (13–16 labels per object).
3. **Learned-feature proto-objects**: DINO-style patch-feature clustering as
   a drop-in `proto_objects` source behind the interchange boundary — keeps
   the controller interpretable at the *mechanism* level while fixing
   "colour fill traces only the shirt".
4. **The WASM demo and README GIFs** (already on the wish list) matter more
   for an arXiv-plus-website release than M14–M16 do.

---

## Appendix A — build, test and repository hygiene

Audited 2026-09-19 on main (594210a); nothing was modified.

**Healthy.** 37/37 CTest tests pass (112 s; slowest: `priority_search_smoke`
51 s, `vlm_video_smoke` 19 s, `gated_recognition_smoke` 18 s);
`format-check` clean; 74 Catch2 cases + 70 Python test functions; every path
referenced from `README.md` and `CLAUDE.md` exists; CI
(`.github/workflows/ci.yml`, ubuntu, Release, full ctest) is in place.

**Findings**

| # | Finding | Why it matters |
|---|---|---|
| A1 | **The local build has no build type.** `build/CMakeCache.txt` has an empty `CMAKE_BUILD_TYPE`; `CMakeLists.txt` sets no default; the build command in `CLAUDE.md` omits `-DCMAKE_BUILD_TYPE=Release` (README and CI include it). | Every local experiment and any local timing runs **unoptimized**. `docs/PERFORMANCE.md` says "Release build" — check which build its numbers came from. It also makes every study slower than necessary. Fix: default to Release in `CMakeLists.txt` when unset. |
| A2 | No warning flags anywhere (`-Wall -Wextra` absent), so "zero warnings" carries no information. A clean build shows 4 deprecation warnings in `src/selection/kalman_mot_selection.cpp:35,176` (`cv::Mat_ <<` initializer). | Cheap static checking not used. CI has no warnings-as-errors and no `format-check` step, and no macOS job although development is on macOS. |
| A3 | 7 commits on main are unpushed (all 2026-09-18), including the DAVIS refutation. | The public/remote state lags the README's claims. |
| A4 | 11 local branches are fully merged and can be deleted, including `module/replication` (empty) and `module/video-token-cache`, both still holding worktrees. | Phantom state (§5 #12). |
| A5 | Tracked phase-1 leftovers: `config.yaml`, `test_ior.yaml` (referenced nowhere), `docs/progress/weekly_log.md` (empty template). `reference/` holds only a `.DS_Store`. `CODE_STYLE.md`, `FORMATTING.md` and `DEVELOPMENT_GUIDELINES.md` (976 lines together) likely overlap. | Noise in a repo that is about to be public. |
| A6 | `tmp/old_code` contains original thesis sources. It is ignored (the `.gitignore` change adding `tmp/` is staged but uncommitted). | Commit the ignore rule before anything else touches `git add`. |
| A7 | `eval/requirements.txt`: loose lower bounds, no lock file; `scipy`, `pyarrow` and `anthropic` are used by studies but not listed (documented as optional installs in prose only). `find_package(OpenCV 4)` pins the major version only; the Catch2 fallback tarball has no visible `URL_HASH`. | Reproducibility of reported numbers across machines/time. A `requirements-studies.txt` with pins recorded at run time would do. |
| A8 | Missing `help_*` tests for `dynamic_ior.py`, `compare_scanpaths.py`, `compare_scanpath_json.py`, `report_thesis_vs_modern.py`, and the three `tools/make_*` generators — against the `CLAUDE.md` convention. `study_common.py` (the only statistics code) has no unit test. | The statistics helper is the one place a silent bug would poison every table. |
| A9 | `src/main.cpp` is 1041 lines — the CLI has absorbed every mode (`--attend`, `--live`, `--emit-json`, batch, stereo, …). The three VLM harnesses (500–600 lines each) duplicate arm-building, token accounting and result I/O. | Maintainability; and the duplication is where the "resume without config check" defect lives three times. |
| A10 | `CONTRIBUTING.md`, the release/provenance note and `docs/METHODOLOGY.md` from the July "docs to write" list are still unwritten. | `METHODOLOGY.md` is the natural home for §7.1–7.5 of this review. |

## Appendix B — sources for the literature-dependent claims

Unverified = seen in search results only, not opened.

- ViCrop, "MLLMs Know Where to Look" (ICLR 2025): arxiv.org/abs/2502.17422
- ZoomEye (EMNLP 2025): arxiv.org/abs/2411.16044
- V\*/SEAL: arxiv.org/abs/2312.14135
- FOCUS (NeurIPS 2025): arxiv.org/abs/2506.21710
- Chain-of-Focus: arxiv.org/abs/2505.15436
- Divide, Conquer and Combine (HR-Bench): arxiv.org/abs/2408.15556
- GazeVLM: openreview.net/forum?id=RAUMsywTko
- TimeChat-Online (ACM MM 2025): arxiv.org/abs/2504.17343
- TrajViT (ICCV 2025): arxiv.org/abs/2505.23617
- AutoGaze / HLVid: arxiv.org/abs/2603.12254 (release status unverified)
- AKS: arxiv.org/abs/2502.21271 · LongVU: arxiv.org/abs/2410.17434 ·
  Q-Frame: arxiv.org/abs/2506.22139
- Token-compression list: github.com/cokeshao/Awesome-Multimodal-Token-Compression
- HRVideoBench: huggingface.co/datasets/TIGER-Lab/HRVideoBench
- EgoTextVQA: arxiv.org/abs/2502.07411
- SoccerNet-GSR: arxiv.org/abs/2404.11335
- Tipper, Driver & Weaver (1991): doi.org/10.1080/14640749108400971
- Tipper et al. (1994), JEP:HPP 20:478 (magnitudes unverified)
- Müller & von Mühlenen (1996): pubmed.ncbi.nlm.nih.gov/8838166
- Christ, McCrae & Abrams (2002): pubmed.ncbi.nlm.nih.gov/12026955
- Krüger & Hunt (2013): pubmed.ncbi.nlm.nih.gov/23046140
- Reppa, Schmidt & Leek (2012), AP&P 74:43
- ReScience C: rescience.github.io · MLRC: reproml.org/call_for_papers
