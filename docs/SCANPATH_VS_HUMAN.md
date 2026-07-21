# Scanpaths vs human gaze on stills (M11, H4)

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
| ScanMatch | Cristino et al. (2010): grid-quantize, then Needleman-Wunsch with a **signed** substitution matrix (+1 same cell → −1 opposite corner, so distant substitutions are penalized and a gap can win). Normalized to [0, 1]. |
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
