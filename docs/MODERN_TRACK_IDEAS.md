# Ideas for the modern track, from the VOCUS line

*2026-09-23. Track: **modern** (docs/adr/0005) — none of this touches the frozen
replication profiles. Written after reading Simone Frintrop's work, which is the
other branch that grew out of the same group and the same period, and which has
kept going for twenty years. Sources are cited; where a claim comes from an
abstract rather than a full text it says so.*

The modern track has three open problems, and this line of work has something
concrete to say about each:

| Our problem | Where it bites | What the VOCUS line offers |
|---|---|---|
| **What deserves an object file** | H1's residual weakness; 42–61% of fixations on DAVIS land on things the ground truth does not name (`docs/DYNAMIC_IOR_STUDY.md`) | class-agnostic open-world instance segmentation (SOS, 2024) |
| **Where the top-down signal comes from** | H5's hand-built target-colour channel and category prior; H6 needs a crop source that is cheaper than the model it feeds | VOCUS's learned feature weights — a two-line rule, no detector |
| **Small targets, and the cost of the controller** | V\*Bench top-3 target coverage 0.15; 2×2 tiling helped insignificantly; the front-end must cost less than it saves | scale-specific objectness (AttentionMask, 2018) |

---

## 1 · Learned top-down weights instead of hand-built channels

**What VOCUS does** (verified from
[Frintrop, Backer & Rome, KI 2005](https://doi.org/10.1007/11551263_28), §2.2).
Given a training image and a rectangle around the target, it computes its own
bottom-up saliency, takes the most salient region *inside* the rectangle — so it
decides for itself what in the box is the object — and then, for every feature
and conspicuity map *i*:

```
w_i = mean value of map i inside the target region
    / mean value of map i outside it
```

A feature counts to the degree that it *separates* the target from its
background, not merely to the degree that it is strong on the target. In search
mode the top-down map is an excitation minus an inhibition term,

```
E = Σ_{w_i > 1}  w_i · X_i        I = Σ_{w_i < 1} (1/w_i) · X_i     S_td = E − I
```

and the final map is `S = (1 − t)·S_bu + t·S_td` with a single top-down factor
`t ∈ [0, 1]`. Weights from several training images are combined by the
**geometric mean**, because they are ratios: a feature that is 2 in one image
and 0.5 in another averages to 1 and drops out. Their Tab. 2 is the clean
demonstration — learning "red bar" from four images in two orientations on two
backgrounds leaves colour as the only surviving weight.

**Why this is better than what we have.** `configs/attend_proto.yaml` and the
priority map (`docs/PRIORITY_MAP.md`) get their top-down signal from a
hand-written target-colour channel plus a category prior. That is one feature,
chosen by us, with a weight we tuned. The VOCUS rule is general (every channel
gets a weight), needs no training beyond one example, costs two means per map,
and has a principled way to generalise over examples.

**What to build.**

1. `fusion/top_down_weights.{h,cpp}`: given a mask (or box) on a frame, compute
   `w_i` per feature map, and a `TopDownChannel` that applies `E − I` with a
   configurable `t`. This is a sibling of the existing `top_down_map` slot, not
   a replacement.
2. Sweep `t` from 0 to 1 on COCO-Search18 — the existing H5 harness
   (`eval/coco_search.py`) already scores found@k. The thesis's own model has no
   such dial; this is the first time the mixture would be explicit.
3. Report **average hit number** alongside found@k: the rank of the first
   fixation that lands on the target, with the detection rate beside it. That is
   VOCUS's metric, it is the same quantity as our "target coverage at top-k"
   read the other way round, and reporting it makes the numbers comparable with
   twenty years of that literature.

**The interesting variant for H6.** A VLM front-end has no training image — it
has a *question*. Two cheap ways to get weights without running a detector:

- **Text → channel weights.** Parse colour and size words out of the question
  ("the red mug on the left") into weights over the existing channels. Crude,
  but it costs nothing and is a real arm against which an open-vocabulary
  detector has to justify its cost.
- **One-shot from a crop.** Where the task supplies an example image of the
  target, VOCUS's rule applies unchanged.

That gives the H6 experiment a proper cost axis: bottom-up only → text-weighted
channels → learned weights from an example → open-vocabulary detector, all on
the same token budget. Right now we only have the first and the last.

## 2 · A real object source: open-world instance segmentation

[SOS (ECCV 2024)](https://doi.org/10.1007/978-3-031-73383-3_10) segments
*arbitrary unknown* objects: it prompts SAM with object priors — the strongest
being the self-attention maps of a self-supervised ViT — to produce pseudo
annotations, then trains an ordinary instance segmenter on them, reporting
precision improvements up to 81.6% over the previous state of the art and
generalising across COCO, LVIS and ADE20k. *(From the abstract; I have not read
the full paper.)*

**Why it matters to us.** Our second stage builds object files from thresholded
saliency, or from the field's activity clusters. Both are blob detectors. The
DAVIS run showed what that costs: the fixations are not wrong, but they are not
on the annotated objects either, and no inhibition-domain comparison can survive
that. An open-world segmenter gives object hypotheses that are *object-shaped
and class-agnostic* — which is exactly what an object file wants and exactly
what an open-vocabulary detector like OWLv2 does not give, because the latter
needs to be told what to look for.

Worth saying plainly: a self-supervised transformer's attention map used as an
"object prior" is a learned proto-object detector. It is the same job the 2004
stage 1 gave to symmetry, eccentricity and colour contrast, done by a model that
was trained rather than designed.

**What to build.** A `object-proposals` cluster source next to `saliency` and
`field` (`attention_system.cluster_source`), fed by a segmenter behind the
existing processor interface. Then re-run H1 on DAVIS with it. The prediction
worth committing beforehand: the off-object share collapses, and *only then*
does the object- versus space-based comparison become decidable on real video.

## 3 · Scale-specific objectness for small targets

[AttentionMask (ACCV 2018)](https://doi.org/10.1007/978-3-030-20890-5_43)
generates object proposals using *scale-specific objectness attention maps* that
focus processing on promising parts of the image, reporting a 33% speed-up and
+53% average recall on small objects. *(From the abstract.)*

**Why it matters to us.** V\*Bench is a small-target benchmark and we are at 0.15
top-3 coverage. Our answer so far was 2×2 tiling, which raised top-3 from 0.13 to
0.18 — not significant, and crude: it processes every tile equally. Scale-specific
objectness is the principled version of the same instinct, and it *saves*
computation rather than multiplying it, which is the whole argument of an
attention front-end. If the modern track has one thing to prove to a reviewer, it
is that the controller costs less than it saves; this is the mechanism that makes
that provable.

## 4 · Two cheap things, no learned model needed

- **Saliency-guided seeding.** [Gao, Lauri, Zhang & Frintrop
  (IROS 2017)](https://doi.org/10.1109/IROS.2017.8206374) use saliency to seed a
  supervoxel segmentation instead of thresholding it. Our `AttentionSystem::segment`
  thresholds the fused map at 35% of its maximum; seeding a region growth from
  the saliency maxima would produce object-shaped clusters from the same map, at
  almost no cost, and is closer to what the thesis's own region-based features
  did anyway.
- **Exclusivity, the other functional form.** VOCUS divides each map by the
  square root of the number of local maxima above a threshold; the thesis divides
  by `c^n`. Same idea, independently arrived at, different shape. We have an
  exclusivity implementation with a selectable mode already
  (`include/attention/features/exclusivity.h`) — adding `sqrt-peaks` as a third
  mode and ablating the three is an afternoon, and it is a genuinely interesting
  comparison for either paper.

## 5 · One precedent worth knowing about

[VOCUS2 (CVPR 2015)](https://doi.org/10.1109/CVPR.2015.7298603) is titled
*"Traditional saliency reloaded: a good old model in new shape"* — a classic
attention architecture rebuilt to be fast and competitive with what came after.
That is the modern track's own thesis, made ten years earlier by the person best
placed to make it. It belongs in the related work of the modern-track paper both
as a baseline and as the precedent for the claim.

---

## Suggested order

1. **Top-down weights** (§1) — smallest, self-contained, improves an existing
   hypothesis (H5), and gives H6 the missing middle arms.
2. **Object proposals as a cluster source** (§2) — the one that could make the
   real-video test of H1 decidable, which is currently the paper's weakest point.
3. **Scale-specific objectness** (§3) — the small-target and cost story for H6.
4. §4's two cheap items whenever convenient; the exclusivity ablation is also
   interesting for the replication paper.
