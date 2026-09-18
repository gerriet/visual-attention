# Object files as a video token cache (M19, H7)

*Status (2026-09-15): harness built; model-free findings (the mock is a
legibility oracle) and real-VLM results (local Qwen) on synthetic scenes and
DAVIS-2017 below. With a real VLM the H7 effect holds on the synthetic scenes
(identity-keyed crops 0.95 vs location-keyed 0.80 vs budget-matched frames
0.27, chance 0.25); DAVIS at 480p does not separate the arms.*

**H7 — Object files as a video token cache (H1 × H6).** *At a matched
visual-token budget per video, an object-file front-end (object-based IOR +
persistent object files) answers object-centric questions better than
budget-matched uniform frame sampling and better than a spatial-IOR front-end;
the gap grows with object count and speed, and shrinks with tracking errors
(every identity switch is a re-send).*

## Why video

The still-image front-end (`docs/VLM_FRONT_END.md`, M18) leaves the second
stage idle: a single image is a stream of length one, so object files, IOR and
identity never engage. M12 (`docs/DYNAMIC_IOR_STUDY.md`) found that
object-based IOR only ties space-based IOR on *exploration*; the predicted win
was always persistent, identity-keyed memory. A video VLM front-end is the task
that needs exactly that: a video VLM pays tokens for every frame, and deciding
which pixels to (re)send requires knowing *what* has been seen, not *where*.
Spatial memory cannot tell "the same object moved" from "a new object arrived",
so it re-sends moved objects (wasted tokens) and skips newcomers that appear
where an earlier crop was (misses).

## How it works

- **Scenes:** `tools/make_dynamic_scene.py --tags --late` — coloured disks
  that move and bounce at 1280×960, some arriving mid-video, each carrying a
  small code ("K7", ~14 px tall) legible only at native resolution. The flags
  are additive; M12's scenes are unchanged without them.
- **Attention:** the system runs `--attend` on a 320-px copy — exactly M12's
  scene geometry (radius 16, speed 6 px/frame) — under the `spatial-ior` and
  `object-ior` behaviors; crops come from the native frames.
- **Questions:** per object, "what code is written on the ⟨colour⟩ disk?"
  (four options) — accuracy is the share of objects whose detail reached the
  VLM.

| Arm | What the VLM sees (per question) | Tokens |
|---|---|---|
| `frames-full` | 4 evenly spaced native frames | reference (~11×) |
| `frames-uniform` | the same 4 frames, downsampled to the crop budget | matched |
| `space-ior` | overview frame + K crops at spatial-IOR fixations, one per new *location* | matched |
| `object-ior` | overview frame + K crops, one per *object file* | matched |
| `oracle` | overview frame + one crop per ground-truth object | matched |

`--gt-identity` adds `space-ior-gtid` / `object-ior-gtid`: the behaviors'
fixations deduplicated by *ground-truth* identity — a perfect tracker. The gap
between `object-ior` and `object-ior-gtid` is what the object-file tracker
loses to label switches; the gap to `space-ior` is what identity memory is
worth. M12's scanpath scores (coverage, latency, waste) and two identity
statistics (distinct labels per attended object; share of fixations on no
object) ride along.

## First findings (mock — what reaches the VLM)

Five scenes (6 objects, 2 arriving late, 40 frames, K = 6), 30 questions per
arm, thesis object files with M12's tracking aids:

| Arm | delivered | tokens / question |
|---|---|---|
| frames-full | 1.00 | 6440 |
| frames-uniform | **0.00** | 616 |
| space-ior | 0.50 | 560 |
| object-ior | 0.43 | 560 |
| space-ior-gtid (perfect tracker) | 0.97 | 560 |
| object-ior-gtid (perfect tracker) | 0.97 | 560 |
| oracle | 1.00 | 560 |

- **Budget-matched frames are blind** to the codes; crops aren't.
- **Attention finds every object** (scanpath coverage 1.00 for both
  behaviors), and with a perfect tracker either behavior's crops deliver 97%
  of the codes. **Identity memory is worth ~+0.5 of all codes at the same
  budget** — H7's lever is real and large.
- **The object files lose all of it:** 4.4 distinct labels per attended object
  and ~a quarter of fixations on no object leave `object-ior` slightly *below*
  location-deduplicated crops. The same holds on a plain native-320 M12 scene,
  so neither the codes nor the downscaling cause it.

**Diagnosis.** The annotated object-file frames show two sources, both in
*segmentation*, not correspondence: (1) each moving disk falls apart into
fragments — a moving uniform disk is salient mostly at its leading and
trailing edges (the onset channel), so the two crescents become two clusters,
two object files, and the focus hops between them; (2) diffuse background
saliency is segmented as one giant "object" (the off-object fixations).
Correspondence adds a third: an object unobserved for a few frames drifts past
the fixed 30-px revival radius and comes back as a new file.

**Fixes (opt-in; thesis defaults unchanged, goldens untouched).**

- *Persistent identity* (`ObjectFileStore`, `attention_system.object_files.
  persistent_identity`): an unmatched cluster revives the inactive file with
  the lowest position + colour cost, inside a gate that widens with the time
  unseen, or anywhere if it is a clear colour look-alike; a clearly different
  colour is never the same object (so a newcomer where an old object vanished
  gets its own file); inactive files never age out. Shipped as
  `configs/attend_identity.yaml`.
- *Segmentation* (`attention_system.segment_close`, `max_cluster_fraction`):
  a morphological closing that bridges an object's fragments, and a size cap
  that drops regions too large to be an object.

Persistent identity alone barely moves the video result (object-ior 0.43 →
0.47; labels per object unchanged) — the churn is mostly fragmentation. The
segmentation sweep (3 scenes, persistent identity on, mock):

| Segmentation | object-ior (perfect tracker) | space-ior (perfect tracker) | labels / object (obj, space) | off-object (obj, space) |
|---|---|---|---|---|
| thesis | 0.39 (0.94) | 0.44 (0.94) | 4.4, 2.9 | 0.22, 0.28 |
| closing 4 px | 0.39 (0.94) | 0.39 (0.94) | 3.7, 2.7 | 0.24, 0.35 |
| closing 8 px | 0.50 (0.78) | 0.39 (0.78) | 1.7, 1.5 | 0.37, 0.30 |
| size cap 15% | 0.44 (1.00) | 0.44 (1.00) | 4.1, 2.6 | 0.22, 0.20 |
| closing 8 + cap 15% | **0.56** (0.78) | 0.39 (0.67) | 1.6, 1.5 | 0.34, 0.26 |

Closing fixes the fragmentation (4.4 → 1.6 labels per object) and, with it,
`object-ior` overtakes `space-ior` for the first time (0.56 vs 0.39) — a hint
of H7 on 18 questions per arm, not a result. But blind closing also merges
neighbouring objects (the perfect-tracker ceiling drops to 0.78), and a third
of fixations still land on no object. The size cap has no downside.

**H1 side effect.** On M12's own exploration study (6 seeds per regime,
`eval/dynamic_ior.py`), persistent identity moves object-based IOR from worse
to tied-or-better in the regimes it was built for — see
`docs/DYNAMIC_IOR_STUDY.md`, "Persistent identity (M19)".

## Proto-objects: segmentation that knows what an object is

The per-frame feature maps show why blind closing can't be the answer. On
these scenes the colour channel is a smooth gradient, eccentricity is blank,
**symmetry fires in the empty space *between* the disks** (a large branch-
shaped response — the off-object fixations), and **onset draws a hollow ring
around each moving disk** (its arcs are the fragments). Salient pixels alone
don't say where an object is.

`attention_system.proto_objects` (opt-in, `configs/attend_proto.yaml`) turns
each salient cluster into the object(s) under it, in the spirit of Walther &
Koch's proto-objects: seed at the cluster pixel whose colour differs most from
the frame's median colour (a figure-ground estimate) — a cluster where nothing
does is background (a motion ghost, the symmetry branch) and is dropped; grow
the object from the seed by colour (flood fill in a window around the cluster;
a fill that floods the window is background too); re-seed on what's left, so
a cluster spanning touching objects yields each; merge results that grew into
the same object. Under persistent identity, the store also folds a second
look-alike file on one object into the first (active duplicates with
overlapping boxes, and inactive ones last seen within the correspondence
radius).

Mock, 5 scenes, persistent identity on:

| Segmentation | object-ior (perfect tracker) | space-ior (perfect tracker) | labels / object | off-object | latency (obj, space) |
|---|---|---|---|---|---|
| thesis | 0.47 (0.97) | 0.50 (0.97) | 4.5, 3.2 | 0.20, 0.27 | 3.8, 5.3 |
| closing 8 + size cap | 0.57 (0.73) | 0.33 (0.67) | 1.6, 1.6 | 0.27, 0.24 | — |
| proto-objects | **0.67** (1.00) | 0.60 (1.00) | 2.6, 1.7 | **0.01**, 0.02 | 1.4, 2.5 |

Proto-objects all but remove off-object fixations, keep the perfect-tracker
ceiling at 1.00 (closing merged neighbours), and put `object-ior` ahead of
`space-ior` — with object-based attention reaching new objects sooner (1.4 vs
2.5 frames). Identity is better but not solved: ≈ 2.6 labels per object
remain, and the duplicate folding barely moved it — the rest is files
trading places between objects, not duplicates.

## Calibrated to the VLM: 8-px codes

The first real-VLM run (local Qwen) at the default 20-px code size hit the
ceiling: Qwen read the codes even from the budget-matched downsampled frames
(3 scenes: frames-uniform 0.94, every crop arm 1.00) — the mock's legibility
floor was far too strict, as on V\*Bench. A probe (one scene per size, six
questions per cell, chance 0.25) found the regime H7 needs:

| Code size | native crop | frame at the frames-uniform scale (0.30) | overview (0.40) |
|---|---|---|---|
| 8 px | **1.00** | 0.17 | 0.33 |
| 10 px | 1.00 | 0.33 | 0.67 |
| 12 px | 1.00 | 0.33 | 1.00 |
| 14 px | 1.00 | 0.50 | 1.00 |

At 8 px (a code box of 11 × 5 px) Qwen reads every code from a native crop and
is at chance on the downsampled views. The M19 runs therefore use
`--tag-size 8 --min-target-px 5` (the mock's floor then agrees with Qwen).

Mock, 10 scenes, persistent identity + proto-objects (`attend_proto.yaml`):

| Arm (same budget unless noted) | delivered | 95% CI |
|---|---|---|
| frames-full (~11× the budget) | 1.00 | |
| frames-uniform | 0.00 | |
| space-ior | 0.63 | [0.52, 0.75] |
| **object-ior** | **0.90** | [0.82, 0.97] |
| space-ior / object-ior with a perfect tracker | 1.00 / 1.00 | |
| oracle | 1.00 | |

**This is the H7 effect, model-free:** at the same budget, identity-keyed crops
deliver 0.90 of the codes and location-keyed crops 0.63 — non-overlapping
intervals — while budget-matched frames deliver none. The scanpaths say why:
with the small code the disks stay near-uniform at the 320-px processing scale
and identity holds (1.4 distinct labels per object, no off-object fixations);
object-based IOR wastes almost no fixations on already-seen objects (0.01 vs
0.21) and reaches new ones sooner (1.0 vs 2.5 frames). Both behaviors attend
every object; the difference is purely which crops each memory spends the
budget on.

Which part does it? The same 10 scenes (mock), one component at a time:

| Configuration | object-ior | space-ior | labels / object (object-ior) | off-object (object-ior) |
|---|---|---|---|---|
| thesis object files + tracking aids | 0.45 | 0.52 | 3.8 | 0.32 |
| + persistent identity | 0.48 | 0.50 | 3.9 | 0.28 |
| + proto-objects | 0.87 | 0.63 | 1.6 | 0.00 |
| + proto-objects + persistent identity | **0.90** | 0.63 | 1.4 | 0.00 |

Proto-object segmentation carries the effect: once each object is one clean
cluster, the object files hold identity and object-based IOR spends the budget
on distinct objects; persistent identity adds a little on top. Without
proto-objects neither memory helps — the thesis segmentation's fragments and
background clusters swamp both.

### With a real VLM (Qwen)

`qwen3.8:27b` (Q4_K_M, local Ollama), the same 10 scenes, six questions each,
≈ 530 real tokens per question in every arm:

| Arm (same budget) | Qwen accuracy (60 questions) | 95% CI | delivered (mock rule) |
|---|---|---|---|
| frames-uniform | 0.27 | [0.17, 0.38] | 0.00 |
| space-ior | 0.80 | [0.70, 0.90] | 0.63 |
| **object-ior** | **0.95** | [0.88, 1.00] | 0.90 |
| oracle | 1.00 | | 1.00 |

The real VLM confirms the model-free result: budget-matched frames sit at
chance (0.25), and identity-keyed crops beat location-keyed ones by 15 points
at the same budget. The 95% intervals (bootstrap over questions — scenes are
the natural unit, so optimistic) only just touch. Qwen does a little better
than the mock's legibility rule in the crop arms (it sometimes reads a partly
clipped code), but the ordering and the gap match.

## DAVIS-2017: real video (model-free)

`eval/vlm_video_davis.py` runs the same arms on the 30 val sequences (61
annotated objects, hand-labelled with categories from mask overlays in
`eval/datasets/davis2017.py`). Crops cover the attended object (the object
file's box plus a margin, capped at 224 px); an object is *delivered* when a
view shows ≥ 60% of its mask with its shorter side ≥ 24 px on screen.

| Arm | delivered (61 objects) | small objects (5) | tokens vs frames-full |
|---|---|---|---|
| frames-full | 1.00 | 1.00 | 1.00 |
| frames-uniform | 0.95 | 0.40 | 0.17 |
| space-ior | 0.93 | 0.20 | 0.14 |
| object-ior | 0.93 | 0.20 | 0.14 |
| perfect tracker (either behavior) | 0.95 | 0.40 | 0.12 |
| oracle (each object where it is largest) | 0.98 | 0.80 | 0.12 |

(Persistent identity; the thesis object files give nearly the same table.) At
480p DAVIS barely separates the arms: its objects are large, the overview
frame alone shows most of them, and only five count as small — too few to
compare, though the oracle row shows crops *can* deliver them. Identity on real video is poor in every configuration
(13–16 labels per attended object, 57–75% of fixations off the annotated
objects: real saliency lands on busy backgrounds), and proto-objects *hurt*
here — the colour fill traces only a homogeneous part of a textured object (a
shirt), so crops cover less of it (object-ior crop coverage 0.99 → 0.41).
Proto-objects are for flat-coloured objects; DAVIS runs use persistent
identity alone.

**With Qwen** (30 sequences, 46 "which of these can be seen" questions,
chance 0.25): frames-full 0.98 (1687 real tokens per question), frames-uniform
0.98 (348), space-ior 0.96 (330), object-ior 0.96 (328), oracle 0.98 (280).
At 480p the category question is easy from any arm — budget-matched frames
cost a fifth of full frames and lose nothing. The only misses are the scooter
in *scooter-black* (every arm) and a phone in *lab-coat* (both attention arms;
too small for their crops). DAVIS at this resolution is a real-video check of
the machinery, not a test of H7.

## Next

1. **Sweeps:** more seeds, speed × object count × K, and code sizes around
   the VLM's reading threshold — the effect should grow with busier scenes
   and smaller detail.
2. **Identity on real video** is the open problem: stronger appearance than
   mean colour (histograms / per-feature signatures), a segmentation that
   copes with texture, and fewer background fixations.
3. **Harder real video:** DAVIS full resolution, or video QA where small
   objects matter.
4. **Object files as working memory:** labels and trajectories handed to the
   VLM as text next to the crops (closes M13 Tier 3); re-send on change.

## Running it

```bash
# mock (legibility oracle), with the perfect-tracker decomposition arms
eval/vlm_video.py --backend mock --seeds 10 --gt-identity --config configs/attend_proto.yaml \
    --tag-size 8 --min-target-px 5
# real VLM (local Qwen via Ollama), real token counts
eval/vlm_video.py --seeds 10 --count-tokens --config configs/attend_proto.yaml \
    --tag-size 8 --min-target-px 5 --arms frames-uniform,space-ior,object-ior,oracle
# DAVIS-2017 val (data under data/DAVIS — see eval/datasets/davis2017.py)
eval/vlm_video_davis.py --backend mock --gt-identity --config configs/attend_identity.yaml
eval/vlm_video_davis.py --count-tokens --config configs/attend_identity.yaml
```

## Files

- `tools/make_dynamic_scene.py` — `--tags` (codes), `--late` (arrivals), `--tag-size`
- `eval/vlm_video.py` — the harness (arms, questions, legibility oracle, identity statistics)
- `eval/vlm_video_davis.py` — the DAVIS-2017 harness (object crops, mask-based
  delivery, category questions); categories in `eval/datasets/davis2017.py`
- `configs/attend_identity.yaml` — persistent identity switched on;
  `configs/attend_proto.yaml` — plus proto-objects
- `src/system/object_file.cpp` — `persistent_identity` correspondence and
  duplicate folding; `src/system/attention_system.cpp` — `segment_close`,
  `max_cluster_fraction`, proto-objects and the `attention_system:` config section
- Tests: `eval/tests/test_vlm_video.py`, `eval/tests/test_vlm_video_davis.py`;
  Catch2 `[identity]`, `[segment]`, `[proto]`, `[config]` cases in
  `tests/test_system.cpp`; CTest `vlm_video_smoke`
