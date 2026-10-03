# Learned top-down weights (M20, H8)

*Track: **modern** (docs/adr/0005) — a usefulness claim. Nothing here touches
the frozen replication profiles; with the channel inactive the fused map is
bit-identical to the thesis's, which is the default and is covered by a test.*

**H8 — Learned top-down weights beat hand-built channels.** *Feature weights
derived from examples of the target by VOCUS's rule — each channel weighted by
how far it* separates *target from background, not by how strongly it fires on
the target — guide search at least as well as M17's hand-written target-colour
channel and category prior, across more targets, with no detector in the loop.*

## Why this exists

M17 gave the priority map a top-down slot and then filled it twice, both times
by hand: a target colour that we name, and a category-level spatial prior built
from where targets of that category were found in training. Both work
(`docs/PRIORITY_MAP.md`). Both are also a channel *we* chose, with a weight *we*
tuned, and neither generalises to a target we have not thought about.

VOCUS fills the same slot with a rule (Frintrop, Backer & Rome, KI 2005, §2.2 —
verified from the paper, not from a summary). Given an example of the target and
a box around it, compute, for every feature map *i*:

```
w_i = mean of map i inside the target region / mean of map i outside it
```

The consequence worth stating out loud, because it is the whole idea: **a
channel counts to the degree that it separates the target from its background,
not to the degree that it is strong on the target.** A feature that fires hard
on the target and just as hard everywhere else gets w ≈ 1 and drops out, which
is correct and is not what a "how much does the target have of this" rule would
do.

In search mode the weights split into an excitation and an inhibition,

```
E = Σ_{w_i > 1}  w_i · X_i       I = Σ_{w_i < 1} (1/w_i) · X_i       S_td = E − I
```

and the map actually searched is `S = (1 − t)·S_bu + t·S_td` for one top-down
factor `t ∈ [0, 1]`. Weights from several examples combine by the **geometric**
mean, because they are ratios: a channel that is 2 in one example and 0.5 in
another averages to 1 and correctly disappears. (The arithmetic mean would make
it 1.25 and keep it excitatory — this is tested.)

## What was built

- `include/attention/fusion/top_down_weights.h`, `src/fusion/top_down_weights.cpp`
  — the rule, the geometric combination, the E − I application, and VOCUS's own
  "most salient region *inside* the box" step, because a bounding box contains
  background and averaging over it dilutes exactly the contrast being measured.
- The config block, beside M17's channels because both answer "what does the
  task want", but a separate struct because this one acts on the **feature
  maps** rather than on the fused map:

  ```yaml
  priority:
    top_down_factor: 0.5
    top_down_weights:
      color: 2.21
      eccentricity: 0.84
  ```

- `attention --learn-weights <image>:<x,y,w,h> [...] --out weights.yaml` — so the
  rule has exactly one implementation, in C++, and the study calls it rather
  than reimplementing it in Python.
- `eval/top_down_weights.py` — the COCO-Search18 study. It imports
  `coco_search.py`'s scoring rather than copying it, so the new arms and M17's
  arms are scored by the same code; a comparison between two harnesses would
  not mean anything.
- `tests/test_top_down_weights.cpp` — 12 cases. The ones that matter: the weight
  is the ratio and not the level; a feature quieter on the target becomes
  inhibitory; contradicting examples cancel under the geometric mean; an
  inactive config returns *the same `cv::Mat`*, not merely an equal one; and a
  mask with no background is an error rather than a silent neutral weight.

## The experiment

COCO-Search18, target-present trials, 18 categories. Weights are learned **per
category from the training split** and the images they were learned from are
held out of every score.

This makes the comparison with M17's prior arm fair rather than flattering:
both are learned from training data, both are blind to the test image's
content, and they differ in *what* they carry. The prior says **where** targets
of this category tend to be; the weights say **which channels** tell this
category apart. They can also be added, which is a third arm.

Metric: fixations-to-target (first fixation inside the target box, cap + 1 when
never found within the budget of 10) and found@10, with a **paired** bootstrap
over trials, because every arm sees the same images.

Following `docs/HYPOTHESIS_CLOSURE_PLAN.md`, the two phases do not share trials:

```bash
eval/top_down_weights.py --tune --examples 10 --limit 100       # pick t here
eval/top_down_weights.py --confirm --factor <t> --with-prior    # report here
```

`--tune` scores on training trials *held out from weight learning*; `--confirm`
scores on the validation split, once, with `t` already fixed.

### Qualitative check, before any number

The learned weights are legible, which is the first evidence that the rule picks
up something real rather than noise. From 10 training images per category:

- **stop sign** gets the highest colour weight of all eighteen categories — stop
  signs are red, and red is what separates them from a street.
- **clock** gets the *lowest* colour weight (inhibitory) — clocks are
  achromatic, so colour is evidence *against* a clock.
- **knife**, **keyboard**, **toilet**, **oven** load on eccentricity, which is
  the channel that responds to elongation and compactness.
- **car** gets a symmetry weight below 1, i.e. inhibitory: in these images
  symmetry fires on other things more than on the cars.

None of that was put in by hand. It is the ratio rule reading the dataset.

### Prediction, written down before the confirmatory run

*Committed 2026-10-03, before the validation split was scored.*

1. **The weights arm beats bottom-up** on mean fixations-to-target, with a
   paired interval excluding zero. This is the weakest form of H8 and the one I
   expect to hold; if it fails, the rule does not work on natural images at this
   stage-1's channel set and H8 is refuted outright.
2. **The weights arm does *not* beat the category prior.** I expect to lose this
   one. The prior encodes scene layout — that keyboards sit on desks and ovens
   at floor level — and on natural indoor images layout is worth more than five
   bottom-up channels can say. Predicting the loss now is the point: the
   interesting claim is not that VOCUS wins but that a *rule* gets within reach
   of a hand-built prior while generalising to any target with an example.
3. **Weights + prior beats either alone.** They carry different information
   (what vs where), so adding them should help if neither is merely a proxy for
   the other.
4. **The best `t` is interior, not 1.0.** A pure top-down map throws away the
   bottom-up salience that finds the target once attention is in the right
   region.

What would refute H8 as stated: prediction 1 failing, or the weights arm being
indistinguishable from bottom-up at every `t`.

## Result 1 — on the thesis's five fused channels: a null, and the reason for it

*Development phase, on training trials held out from weight learning. n = 100
unique (image, task) trials, 10 training images per category, cap 10.*

| arm | mean fixations-to-target | found@10 | paired Δ vs bottom-up |
|---|---|---|---|
| bottom-up | 8.74 | 0.33 | — |
| weights, t = 0.25 | 8.73 | 0.31 | −0.01 [−0.32, +0.29] |
| weights, t = 0.50 | 8.72 | 0.31 | −0.02 [−0.42, +0.39] |
| weights, t = 0.75 | 8.60 | 0.32 | −0.14 [−0.62, +0.34] |
| weights, t = 1.00 | 8.56 | **0.34** | −0.18 [−0.70, +0.33] |

Every interval spans zero. **Prediction 1 fails on this channel set**: the
weights arm does not beat bottom-up. Prediction 4 also fails — the best `t` is
1.0, not interior, which is itself a hint: a top-down map that is barely
different from the bottom-up one costs nothing to use at full strength.

### Why: the weights are nearly category-blind

The learned vectors looked legible — *stop sign* has the highest colour weight of
all eighteen categories, *clock* the lowest, *knife* and *sink* load on
eccentricity. That legibility is real but it is a small part of the signal. The
large part is shared:

| | colour | eccentricity | intensity | orientation | symmetry |
|---|---|---|---|---|---|
| geometric mean over the 18 categories | 1.16 | **2.36** | 1.64 | 1.18 | 1.33 |
| spread across categories (log SD) | 0.26 | 0.32 | 0.17 | 0.11 | 0.37 |

**Every single category** is excitatory on eccentricity (1.37–4.24) and on
intensity (1.20–2.10). What the rule mostly learned is *"objects are eccentric
and bright"* — which is category-independent, and therefore a re-weighted
bottom-up map rather than a top-down one.

### The control that settles it

Three weight vectors, same trials, same `t` = 1.0: the category's **matched**
vector, a category-**blind** vector (the geometric mean over all 18, so the
category-specific part is averaged out), and a **mismatched** one (each category
gets a different category's weights — a rotation, so none keeps its own).

| arm | mean ftt | found@10 |
|---|---|---|
| matched | **8.56** | **0.34** |
| bottom-up | 8.74 | 0.33 |
| blind | 8.94 | 0.29 |
| mismatched | 9.03 | 0.25 |

| paired difference | Δ | 95% CI |
|---|---|---|
| matched − mismatched | −0.47 | [−1.07, +0.13] |
| matched − blind | −0.38 | [−0.93, +0.14] |
| blind − mismatched | −0.09 | [−0.61, +0.41] |
| matched − bottom-up | −0.18 | [−0.70, +0.33] |

The **ordering is exactly what H8 predicts** — the right category's weights beat
the average category's, which beat the wrong category's — and **not one interval
excludes zero**. So: there is a category-specific component, it points the right
way, and at n = 100 it is too small to call. It is worth about 0.4 fixations out
of 8.7.

That is the honest state of H8 on the thesis's channel set, and it is a
negative. But it is a negative about *this channel set*, which leads to the part
worth having.

## Result 2 — the diagnosis, and what follows from it

A weight vector can only express what the channel set keeps apart. Weighting one
combined orientation map cannot say *"this target is horizontal"*; weighting one
combined colour map cannot say *"this target is red"*, because the sum of the
red–green and blue–yellow axes has already discarded which one responded. VOCUS
weights about thirteen maps — four oriented, four coloured, at several scales.
We were asking five contrast maps, each of which answers *"is something here"*
for one modality, to answer *"is the thing here a knife"*.

Declaring H8 refuted on a channel set that makes it inexpressible would be a
weak negative. So the channel set was split (`configs/split_channels.yaml`):
four orientations, the two colour-opponent axes, plus intensity, eccentricity
and symmetry — nine channels. The capability behind it is general and opt-in: a
config entry's key now names an *instance* and `type:` names the registry entry,
so one extractor can appear several times with different parameters. Every
existing config is unaffected, and the goldens are unchanged.

### The split channels, same trials, same controls

*Development phase still — training trials held out from weight learning,
n = 100, t = 1.0.*

| arm | mean ftt | found@10 | paired Δ vs bottom-up |
|---|---|---|---|
| matched | **8.08** | **0.37** | **−0.77 [−1.40, −0.16]** |
| blind | 8.56 | 0.32 | −0.29 [−0.61, +0.04] |
| bottom-up | 8.85 | 0.30 | — |
| mismatched | 9.31 | 0.27 | +0.46 [−0.05, +1.02] |

| paired difference | Δ | 95% CI | |
|---|---|---|---|
| matched − mismatched | **−1.23** | [−1.87, −0.64] | excludes zero |
| matched − bottom-up | **−0.77** | [−1.40, −0.16] | excludes zero |
| blind − mismatched | **−0.75** | [−1.29, −0.27] | excludes zero |
| matched − blind | −0.48 | [−1.04, +0.05] | just includes zero |

On channels the rule can say something with, **H8's prediction 1 holds**: the
learned weights beat bottom-up, and the interval excludes zero. The control is
what makes it a result rather than a coincidence — giving a category the *wrong*
category's weights is worse than giving it the *average* category's weights
(−0.75, excluding zero), which cannot happen if the weights are noise.

### A correction: the mechanism is not the one stated above

The obvious story — "split channels let the rule learn more category-specific
weights" — is **wrong, and the data say so**. The spread of the weights across
categories is not larger on the split set; it is marginally smaller (mean log-SD
0.23 vs 0.25). The rule did not become more discriminative in its *numbers*.

What changed is the **leverage** those numbers have. The five fused channels all
respond to much the same places — they are five ways of saying "something is
here" — so re-weighting them can only slightly reorder the same peaks. A
horizontal-edge map and a vertical-edge map respond at *different pixels*, and
the two colour-opponent axes at different pixels again. The same amount of
weight variation now moves the priority map somewhere else.

Put as a design lesson rather than a result: **a top-down weight vector is worth
as much as the spatial independence of the channels it weights**, not as much as
the variance of the weights themselves. That is the useful thing this milestone
found, and it was not what it set out to look for.

## Reproducing

```bash
cmake --build build -j
eval/top_down_weights.py --tune --examples 10 --limit 100
eval/top_down_weights.py --confirm --factor <chosen t> --with-prior --limit 150
```

The weight vectors land in `results/top_down_weights*/weights_<category>.yaml`
and in `summary.json` beside the scores.
