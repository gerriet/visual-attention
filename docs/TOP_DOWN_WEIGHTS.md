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

## Results

*The confirmatory run has not been made yet. This section is deliberately empty
until it has; the tuning numbers do not go here, because t is chosen on them.*

## Reproducing

```bash
cmake --build build -j
eval/top_down_weights.py --tune --examples 10 --limit 100
eval/top_down_weights.py --confirm --factor <chosen t> --with-prior --limit 150
```

The weight vectors land in `results/top_down_weights*/weights_<category>.yaml`
and in `summary.json` beside the scores.
