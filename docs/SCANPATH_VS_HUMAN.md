# Scanpaths vs human gaze on stills (M11, H4)

*Track: **replication** (docs/adr/0005) — H4 asks whether the thesis model's scanpaths are plausible; the study runs on the thesis profile (`configs/thesis/thesis.yaml`, `configs/thesis/attend.yaml`).*

*Status: instrument built and verified end-to-end on synthetic data, 2026-07.
The real MIT1003 numbers are gated on the dataset + its raw eye-tracking
archive + scipy (see "Running it for real"); the headline table lands when that
runs.*

**H4 — Scanpath plausibility.** *On still images the two-stage model's
scanpaths land measurably above random/center baselines toward the
inter-observer ceiling — and the ordering benefit of stage 2 is separable from
the saliency map itself (same map, WTA readout vs object-file readout). The
deterministic path is scored as one sample of a human scanpath distribution
(MultiMatch/ScanMatch against a generative baseline), never as "the" path.*

This is the human-comparison milestone. It scores the thesis model's fixation
*sequence* — not just its saliency map — against where people actually look, on
the floor↔ceiling axis, and it does so honestly: humans are a distribution, so
the deterministic path is one draw among a stochastic peer's samples.

## The pieces

| Piece | What it is |
|---|---|
| MultiMatch (4 spatial dims) | Dewhurst et al. (2012): shape, direction, length, position over DTW-aligned saccade-vector sequences. Duration is omitted — model scanpaths carry none (standard for saliency-model eval). Each dim ∈ [0, 1]. |
| ScanMatch | Cristino et al. (2010): grid-quantize, then Needleman-Wunsch with a **signed** substitution matrix (+1 same cell → −1 opposite corner, so distant substitutions are penalized and a gap can win), normalized by the **longer** sequence into [0, 1] — which is where its length dependence comes from. Gap value 0.2 here, 0 in Cristino et al.'s own experiments; `--scanmatch-gap` switches (see "Determinism, not centrality"). |
| Generic WTA+IOR readout | Turns *any* saliency map into a scanpath (repeated argmax + Gaussian inhibition of return). Makes every saliency operator a fair scanpath peer. |
| Stochastic readout | Samples fixations ∝ saliency (with the same IOR), seeded — the generative variability peer; the deterministic path is scored as one draw (best-of-N). |
| MIT1003 sequences | Per-observer ordered fixations recovered from the raw DATA/ eye-tracking archive by an I-DT fixation filter (`datasets/mit1003.py`). |
| CAT2000 | Second stills dataset; the public release ships *pooled* fixation locations + maps (not ordered per-observer scanpaths), so it serves the saliency/coverage cross-check, not the ordered metrics (`datasets/cat2000.py`). |

Everything is hand-rolled (math/numpy only) to match the eval layer's
dependency-light style; scipy is the one new optional dep, for the .mat archives.

## The arms (floor ↔ ceiling)

| Arm | What it is |
|---|---|
| `inter-observer` | leave-one-out human-vs-human agreement — the **ceiling** |
| `thesis-objfile` | the thesis saliency map read out by the object-file second stage (`--attend`) |
| `thesis-wta` | the **same** map read out by generic WTA+IOR — the H4 ablation partner |
| `thesis-stoch-best` | best-of-N stochastic samples of the thesis map (the distribution framing) |
| `<operator>-wta` | each alternative saliency map (spectral-residual, center-bias, DeepGaze if present), WTA+IOR readout — fair scanpath peers |
| `center` / `random` | the **floors** |

**H4's specific claim — the stage-2 ordering benefit is separable from the
map — is read directly off `thesis-objfile` vs `thesis-wta`:** same saliency
map, two orderings. Any gap between them is the object-based second stage's
contribution, isolated from the map that both share (they run from one pipeline
config so the map is identical).

## Verified on synthetic data

The whole scoring stack — readout → MultiMatch/ScanMatch → floors → inter-observer
ceiling — is exercised by `--demo`: a saliency map with three salience-ranked
blobs and eight synthetic "human" scanpaths that follow them with jitter.

```
arm                      scanmatch   shape  direction  length  position
inter-observer               0.944   0.979      0.974   0.974     0.970
thesis-wta                   0.960   0.984      0.981   0.980     0.980
thesis-stoch-best            0.828   0.888      0.826   0.860     0.832
center                       0.740   0.824      0.617   0.650     0.778
random                       0.664   0.680      0.174   0.891     0.671
```

The map-driven readout tracks the salience-ordered targets (near the ceiling),
the stochastic best-of-N sits below the deterministic path (sampling noise),
and the floors fall away — most sharply on `direction`, where random gaze has
no directional agreement (0.174). The `--demo --check` smoke asserts the
robust ordering (ceiling and thesis-WTA both clear the floors) and is a CTest
gate; it deliberately does **not** assert model ≤ ceiling, because a
deterministic model legitimately exceeds a *noisy* inter-observer ceiling on
order-tolerant metrics — that's a finding, not a bug, and one the real run will
surface.

```bash
eval/scanpath_vs_human.py --demo --check     # synthetic; no dataset/scipy/binary
```

## Running it for real (MIT1003)

```bash
# stimuli + maps (~400 MB) and the raw eye-tracking archive for sequences
curl -O https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip
curl -O https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip
curl -O https://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip
unzip 'ALLSTIMULI.zip' 'ALLFIXATIONMAPS.zip' 'DATA.zip' -d data/MIT1003/
eval/.venv/bin/pip install scipy
# the study (one command)
eval/scanpath_vs_human.py --mit1003 --limit 200
```

The prediction (H4): **`thesis-wta` and `thesis-objfile` land well above the
`random`/`center` floors and approach the `inter-observer` ceiling; the gap
between `thesis-objfile` and `thesis-wta` measures the stage-2 ordering benefit
on the same map; and the deterministic path, scored as one draw against the
stochastic peer's best-of-N, is a plausible sample of the human distribution
rather than an outlier.** This section carries the measured table + montages
once that run completes.

*Why the numbers aren't here yet:* this environment has neither MIT1003 nor its
DATA archive, and scipy is a fresh install. The instrument is complete and
unit-tested (metrics, readouts, the I-DT extractor against a synthetic .mat
fixture, the full scoring stack via `--demo`); the measurement is one download
away — the same gating pattern as M13's weights and M18's key.

## The real run (MIT1003, 2026-09-20)

**Changes to the instrument before it** (found when the harness first met the
real archive):

- `thesis-field` is new and is *the thesis model*: the fixations the thesis
  profile itself emits (neural-field selection + IOR). `thesis-wta` — a generic
  Python readout of the same map — had been standing in for it.
- `thesis-objfile` fed the second stage a stream of length one, which yields a
  single focus. The still is now presented as a stream of identical frames (a
  fixed gaze on a static scene); consecutive frames on one object file are one
  fixation.
- A Python model that cannot be constructed (DeepGaze without torch) is skipped
  instead of stopping the study. This run has no learned model by decision.
- Rows are kept per stimulus, so arms are compared *paired* over images, and a
  long run resumes. `--limit` draws a seeded random sample (the archive's file
  order groups images by collection).
- `--profile thesis-extended=…` adds the reimplementation's old five-feature
  default as a comparison arm.

```bash
eval/scanpath_vs_human.py --mit1003 --profile thesis-extended=configs/thesis-extended.yaml \
    --out results/scanpath_vs_human
```

### Prediction, written down before the run

Six images were scored while debugging the real-data path; nothing else of
MIT1003 has been looked at. ScanMatch is the primary score, MultiMatch's five
dimensions are reported; a difference "holds" if its paired 95% interval over
stimuli excludes zero.

1. **Above random, below centre.** `thesis-field` scores above the random
   floor and **below the constant-centre path**: MIT1003 has a strong centre
   bias and a central starting fixation, and the thesis model has neither a
   centre prior nor a face or text channel. *(H4's "above random": refuted if
   `thesis-field − random` does not exclude zero on the positive side. "Above
   centre" is predicted to fail.)*
2. **The ceiling is far.** `inter-observer − thesis-field` is large and
   positive.
3. **The second stage's ordering is not separable on stills.** `thesis-objfile
   − thesis-wta` does not differ from zero: on a static image object files,
   dwell and object-based inhibition re-order the same few salient regions, and
   ScanMatch/MultiMatch are dominated by *where*, not by *in which order*.
   *(This is H4's second clause; predicted to fail.)*
4. **The thesis features beat the old default here too.** `thesis-wta −
   thesis-extended-wta` is positive, as on V\*Bench and COCO-Search18.
5. **A caveat about the instrument, to check rather than predict:** on the six
   debugging images the constant-centre path scored as high as human-vs-human
   agreement on ScanMatch (0.77 vs 0.77). If that holds on the full set,
   ScanMatch at this grid does not separate "looks where people look" from
   "stays in the middle", and the verdict has to rest on MultiMatch's direction
   and position dimensions.

### Outcome (all 1003 stimuli, 15 observers, 2026-09-21)

Run with the thesis profile as it stands after the 2026-09 feature ports
(colour contrast, eccentricity, symmetry — the symmetry port landed while this
run was being prepared; an interrupted earlier run on the defective feature was
discarded unread). CPU only; no learned model by decision. Ten fixations per
model path.

| Arm | ScanMatch | shape | direction | length | position |
|---|---|---|---|---|---|
| inter-observer (ceiling) | 0.759 | 0.952 | 0.630 | 0.927 | 0.891 |
| **thesis-field** (the model's own scanpath) | 0.651 | 0.862 | 0.594 | 0.784 | 0.803 |
| thesis-objfile (same map, second stage) | 0.652 | 0.867 | 0.589 | 0.790 | 0.791 |
| thesis-wta (same map, generic readout) | 0.675 | 0.867 | 0.598 | 0.792 | 0.813 |
| thesis-stoch-best (best of 10 samples) | 0.691 | 0.876 | 0.600 | 0.809 | 0.823 |
| thesis-extended-wta (old five-feature default) | 0.671 | 0.862 | 0.597 | 0.783 | 0.808 |
| spectral-residual-wta | 0.649 | 0.873 | 0.603 | 0.802 | 0.780 |
| center-bias-wta | 0.716 | 0.892 | 0.605 | 0.840 | 0.865 |
| center (constant path) | 0.754 | 0.953 | 0.501 | 0.907 | 0.862 |
| random | 0.636 | 0.846 | 0.586 | 0.752 | 0.766 |

Paired over the 1003 stimuli (95% CI):

| Comparison | ScanMatch | direction | position |
|---|---|---|---|
| thesis-field − random | **+0.015 [+0.012, +0.017]** | **+0.008** | **+0.037** |
| thesis-field − center | **−0.102 [−0.105, −0.100]** | **+0.093** | **−0.059** |
| inter-observer − thesis-field | **+0.107 [+0.105, +0.110]** | **+0.035** | **+0.087** |
| thesis-objfile − thesis-wta | **−0.022 [−0.025, −0.020]** | **−0.010** | **−0.021** |
| thesis-field − thesis-wta | **−0.023 [−0.025, −0.021]** | | |
| thesis-wta − thesis-extended-wta | **+0.004 [+0.002, +0.006]** | | |
| thesis-wta − spectral-residual-wta | **+0.026 [+0.022, +0.029]** | | |
| inter-observer − center | +0.005 | **+0.128** | **+0.029** |

**The predictions, scored.**

1. *Above random, below centre* — **holds.** The thesis model's scanpath is above
   the random floor on every measure, by a small margin (ScanMatch +0.015), and
   well below the constant-centre path (−0.102).
2. *The ceiling is far* — **holds**: +0.107 ScanMatch below human-vs-human
   agreement; the model closes about an eighth of the distance from random to
   the ceiling.
3. *The second stage's ordering is not separable on stills* — **refuted, in the
   wrong direction for H4.** The object-file readout differs from the generic
   readout of the same map — it is *worse* (−0.022), as is the model's own
   neural-field readout (−0.023). The two thesis readouts are
   indistinguishable from each other. On still images the second stage's
   ordering does not help agreement with human scanpaths; a plain
   winner-take-all over the same map is closer.
4. *The thesis features beat the old default* — **holds, barely**: +0.004. On
   human free-viewing data the feature ports matter far less than on the
   target-coverage benchmarks (V\*Bench, COCO-Search18).
5. *The instrument caveat* — **confirmed, and it is the main methodological
   result.** On ScanMatch the constant-centre path (0.754) is indistinguishable
   from the inter-observer ceiling (0.759): at this grid ScanMatch rewards
   staying in the middle as much as looking where other people look, because
   MIT1003 viewing starts at the centre and is strongly centre-biased. It cannot
   carry a plausibility claim on this dataset. MultiMatch's **direction**
   dimension does separate them — ceiling 0.630, centre 0.501 (the lowest of all
   arms), thesis model 0.594, random 0.586 — and there the thesis model is above
   random by +0.008 and nearer to the ceiling than centre is.

### Verdict on H4

H4: *"the two-stage model's scanpaths land measurably above random/centre
baselines toward the inter-observer ceiling — and the ordering benefit of stage 2
is separable from the saliency map itself."*

- **Above random: supported, weakly.** Measurable on every dimension with 1003
  stimuli; small in size.
- **Above centre: not supported** on ScanMatch, shape, length and position;
  supported on direction only. A model without a centre prior, a face channel or
  a text channel does not beat "stay in the middle" on MIT1003 — nor does any
  other bottom-up arm here (spectral residual is at the random floor).
- **An ordering benefit of stage 2: not supported.** The second stage changes
  the scanpath measurably, and for the worse on this measure. That is consistent
  with what the stage is for: H1 shows its benefit in *dynamic* scenes, where
  there is identity to keep; on a still there is none, and object files, dwell
  and object-based inhibition only re-order a handful of static regions.

The honest summary for the replication paper: the thesis never claimed to model
human free viewing, and it does not — its scanpaths are slightly better than
chance and far from human agreement. What it claimed was a mechanism for dynamic
scenes, and that holds (H1).

**What would change these numbers, and is cheap:** a centre prior on the
priority map (the `top_down_map` slot takes one as it is), and the face channel
(YuNet, weight 0 by default) — MIT1003 is full of faces and text. Both belong to
the *modern* track (docs/adr/0005), not to the replication.

### Update (2026-09-21): the dissertation's field parameters, and a length-matched control

The replication dossier found that the neural field had been running on the
default arguments of the original's setter functions rather than on what the
dissertation system configured (`docs/replication/REPLICATION_DOSSIER.md`,
finding B). The thesis profile now carries the system's parameters, and with
them the field does what the thesis says it does: it decides *how many*
fixations a scene deserves — **4.0 per image on average instead of 10**. Only
the `thesis-field` arm depends on the selection stage, so only it was re-scored
(`--resume --refresh-field`); every other row of the table above is unchanged.

Scored as before — the model's path against the observers' full paths — the arm
falls from 0.651 to **0.544 ScanMatch, below the random path (0.636)**, while
its MultiMatch position, shape and length agreement all *rise* (+0.037, +0.019,
+0.055). That combination says the score is measuring path length: a
four-fixation path is aligned against human paths of eight to ten, and ScanMatch
charges for every unmatched fixation. So the comparison was repeated at matched
length — a control added after seeing this, and labelled as such: every arm at
the model's own length *k*, against the first *k* fixations of each observer
(964 stimuli with k ≥ 2).

| Arm, at the model's path length k | ScanMatch | shape | direction | length | position |
|---|---|---|---|---|---|
| inter-observer@k (ceiling) | 0.866 | 0.948 | 0.639 | 0.928 | 0.913 |
| **thesis-field@k** | 0.749 | 0.857 | 0.515 | 0.804 | 0.800 |
| thesis-wta@k (generic readout, same map) | 0.746 | 0.860 | 0.525 | 0.807 | 0.795 |
| center@k | 0.850 | 0.953 | 0.485 | 0.908 | 0.898 |
| random@k | 0.672 | 0.816 | 0.523 | 0.719 | 0.724 |

Paired over stimuli: thesis-field@k − random@k **+0.076 [+0.072, +0.081]**;
− center@k **−0.101 [−0.105, −0.098]**; − thesis-wta@k +0.003 [+0.000, +0.006];
inter-observer@k − thesis-field@k **+0.118**.

What this changes in the verdict, and what it does not:

- **Above random: supported more clearly than before** — +0.076 at matched
  length against +0.015 in the unmatched comparison. The field's few fixations
  are better placed than the ten the port's parameters produced; on position
  agreement the gain over random is the same size (+0.076).
- **Not above centre** — unchanged (−0.10).
- **The field's own readout is no longer worse than a generic readout of the
  same map** (+0.003; it was −0.023). With the right parameters the thesis's
  selection stage is at least as good as winner-take-all on its own map.
- At matched length ScanMatch separates the human ceiling (0.866) from the
  constant-centre path (0.850) — barely. The instrument caveat stands.
- A methodological point for anyone scoring a model that chooses its own number
  of fixations: compare at matched length, or the score is a length penalty.

## Determinism, not centrality (2026-09-22)

*Prompted by a literature check: [Schwinn et al. 2022](https://arxiv.org/abs/2204.09093)
ran a centre baseline on MIT1003 under a string-edit scanpath score and found it
"did not perform much better than Random" — the opposite of the result above.
Their centre baseline samples each fixation i.i.d. from a central Gaussian; ours
returns the same path every time. Only the second is in this harness, so the two
results were not comparable.*

`center-sampled` was added (i.i.d. draws from a Gaussian at the image centre,
σ = 0.22 of each dimension) and both arms were run on the same 200-image seeded
sample, at this implementation's ScanMatch gap (0.2) and at the value
[Cristino et al.](https://doi.org/10.3758/BRM.42.3.692) use in their own
experiments (0, `--scanmatch-gap 0`):

| Arm | ScanMatch, gap 0.2 | − ceiling | ScanMatch, gap 0 | − ceiling |
|---|---|---|---|---|
| inter-observer (ceiling) | 0.757 | | 0.782 | |
| **constant centre** | **0.752** | −0.005 [−0.010, +0.001] | **0.773** | −0.009 [−0.014, −0.004] |
| **sampled centre** | **0.688** | −0.069 [−0.074, −0.064] | **0.718** | −0.065 [−0.069, −0.061] |
| centre-prior map, WTA | 0.715 | | 0.740 | |
| random | 0.638 | | 0.674 | |

Constant minus sampled: **+0.064 [+0.060, +0.068]** at gap 0.2, **+0.055
[+0.051, +0.060]** at gap 0. Both baselines look at the middle; only the
deterministic one approaches the ceiling.

**So the finding is about determinism, not about the centre.** A repeatable path
is rewarded whatever it repeats — the empirical form of the argument
[Kümmerer & Bethge (2021)](https://arxiv.org/abs/2102.12239) make on synthetic
data. It also reconciles this study with Schwinn et al.: their sampled baseline
*should* be near random, and ours *should* be near the ceiling. Two consequences
for anyone scoring scanpaths on a centre-biased dataset: the baseline to beat is
a trivial constant path, and a stochastic centre baseline understates how easy
the metric is to satisfy.

**The gap value.** This implementation uses 0.2 where Cristino et al. use 0.
Every conclusion above and in the sections before holds at both (the table
shows the pair that matters; the arm ordering is unchanged throughout).

**On the length finding, credit where it is due.** That a string-edit measure
confounds similarity with path length is not new:
[Jarodzka, Holmqvist & Nyström (ETRA 2010)](https://doi.org/10.1145/1743666.1743718)
state it — "a long scanpath is per default dissimilar to a short one, even though
the shorter is a substring of the longer" — and
[Mathôt et al. (2012)](https://doi.org/10.16910/jemr.5.1.4) call normalisation
for sequence length "inherently problematic". The mechanism is the
normalisation by the *longer* sequence, not the gap penalty, which this
document previously got wrong. What is this study's own is the case above: a
model that chooses its *own* number of fixations can be made to score below
random by *improving* its parameters, while its position, shape and length
agreement all rise.

**Still to read** before any novelty claim: Fahimi & Bruce (2021), *On metrics
for measuring scanpath similarity*, Behav. Res. Methods 53(2):609–628
([doi](https://doi.org/10.3758/s13428-020-01441-0)) — paywalled, not obtained;
and the axiomatic section of Anderson et al. (2015).

## Honest caveats (to report with the real numbers)

- **I-DT assumptions.** MIT1003's `eyeData` column order and 240 Hz sample rate
  are read from Judd's released archive; if a copy differs, `EYE_X_COL` /
  `EYE_Y_COL` / `SAMPLE_RATE_HZ` in the adapter are the one-line fix. The
  fixation extractor itself is dataset-independent and unit-tested.
- **Duration dimension omitted.** Model scanpaths have no fixation durations, so
  MultiMatch runs on the four spatial dims; the duration dim would only ever
  compare human-to-human.
- **CAT2000 is a cross-check, not an ordered-scanpath source.** Its public
  release gives pooled fixation locations, not per-observer sequences, so it
  feeds the map/coverage framing across categories, not MultiMatch/ScanMatch.
- **The metric can flatter a deterministic path.** Order-tolerant metrics
  (ScanMatch, DTW-aligned MultiMatch) can score a consistent model above a noisy
  human ceiling. Reporting the stochastic best-of-N and the ceiling alongside
  keeps the single number honest.

## Files

- `eval/attention_eval/scanpath.py` — MultiMatch + ScanMatch (plus the existing loose match)
- `eval/attention_eval/readout.py` — generic WTA+IOR readout + stochastic sampler
- `eval/datasets/mit1003.py` — per-observer sequences via I-DT over the DATA archive
- `eval/datasets/cat2000.py` — CAT2000 fixation locations + maps
- `eval/scanpath_vs_human.py` — the study harness (`--demo` / `--mit1003`)
- `tests/test_scanpath.py`, `tests/test_readout.py` (eval); CTest `scanpath_vs_human_smoke` + help
