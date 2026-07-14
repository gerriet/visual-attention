# Dynamic-IOR study (M12) — is object-based inhibition of return useful?

*First cut, 2026-07-12. Tests hypothesis **H1**: in multi-object dynamic scenes,
object-based inhibition of return (IOR) beats space-based and no IOR. This is
the thesis's raison d'être for its symbolic second stage.*

## Design

Three focus-selection **behaviors** that are identical except in *what* they
inhibit — so a scanpath comparison isolates the inhibition domain (all else:
features, fusion, object-file tracking, is shared):

| arm | behavior | inhibits |
|---|---|---|
| no-IOR | `greedy` | nothing — always the most salient object |
| space-IOR | `spatial-ior` | recently attended **locations** (decaying) |
| object-IOR | `object-ior` | recently attended **objects** (decaying) |

Run over synthetic scenes (`tools/make_dynamic_scene.py`: coloured disks moving
and bouncing, optional occlusion) via `attention --attend … --behavior <arm>`,
scored against ground truth by `eval/dynamic_ior.py`:

- **coverage** — distinct objects ever attended / total (↑)
- **mean latency** — frames from an object appearing to first being attended,
  never-attended = full penalty (↓)
- **revisit waste** — fixations that re-hit a seen object while an unseen one was
  visible (↓)
- **perseveration** — fixations on the same object as the previous frame (↓)

Reproduce:
```bash
eval/.venv/bin/python tools/make_dynamic_scene.py --out results/scene --frames 40 --objects 4 --speed 6 --occlude --seed 1
eval/.venv/bin/python eval/dynamic_ior.py --scene results/scene
```

## Results

Single scene (4 objects, 40 frames, speed 6, one occlusion, seed 1):

| behavior | coverage | mean latency | revisit waste | perseveration |
|---|---|---|---|---|
| greedy (none) | 0.750 | 13.75 | 0.875 | 0.750 |
| spatial-ior | 1.000 | 1.50 | 0.000 | 0.000 |
| object-ior | 1.000 | 2.00 | 0.069 | 0.138 |

Speed sweep (5 objects, 50 frames, 3 seeds averaged):

| speed | arm | coverage | revisit waste | perseveration |
|---|---|---|---|---|
| 6  | greedy | 0.867 | 0.853 | 0.566 |
| 6  | spatial-ior | 1.000 | 0.097 | 0.044 |
| 6  | object-ior | 1.000 | 0.276 | 0.189 |
| 16 | greedy | 0.733 | 0.880 | 0.505 |
| 16 | spatial-ior | 1.000 | 0.186 | 0.035 |
| 16 | object-ior | 1.000 | 0.141 | 0.159 |
| 26 | greedy | 0.800 | 0.876 | 0.437 |
| 26 | spatial-ior | 1.000 | 0.176 | 0.028 |
| 26 | object-ior | 1.000 | 0.376 | 0.190 |

## Reading (honest)

- **IOR matters, decisively.** No-IOR (`greedy`) is worst on every metric at
  every speed — it perseverates on the strongest object (perseveration
  0.44–0.57), never reaches full coverage, and ~0.85 of its fixations are wasted
  revisits. The *first* half of H1 — that inhibition of return is useful in
  dynamic scenes — is strongly supported.
- **Object-IOR does *not* dominate space-IOR here — H1 is refined, not
  confirmed.** Both reach full coverage; at these settings space-IOR has lower
  revisit waste and perseveration. Two concrete reasons, both real:
  1. **The inhibition scale is large relative to per-frame motion.** With a 60 px
     spatial tag and objects moving 6–26 px/frame, a moving object stays inside
     its own inhibited spot for several frames, so space-IOR suppresses it about
     as well as object-IOR. Object-IOR's theoretical edge — *following* a moving
     object — only bites when objects move fast relative to the inhibition scale
     (the speed-16 row, where object-IOR's waste dips below space-IOR's, is the
     first hint of that crossover).
  2. **Object-IOR is only as good as the tracking.** When the correspondence
     step switches an object's label (near-crossings, occlusion), the
     object-keyed inhibition no longer applies and the object is re-fixated —
     inflating object-IOR's waste. Space-IOR is immune because it inhibits
     locations, not identities.

So the useful, defensible finding today is: **IOR ≫ no-IOR; object- vs
space-based is regime-dependent, and naive object-IOR does not win by default.**
That is a sharper, more honest claim than the thesis's, and the framework can
now measure exactly where each regime lives.

## Follow-up: pushing into fast motion + occlusion (and it doesn't rescue H1)

We chased the regime where object-IOR *should* win — objects moving far enough to
escape a tight spatial tag, and occlusion where object-IOR should "remember" a
reappeared object — and added the machinery for it:

- a configurable spatial-tag radius (`--ior-radius`), so objects can be made to
  escape spatial inhibition;
- **motion-predicted correspondence** in the object-file store
  (`--motion-prediction`, `ObjectFileStore::Config::motion_prediction`, default
  off): match clusters to each file's *predicted* next centroid (last position +
  trajectory velocity), extrapolated over the gap for occluded files. This holds
  object identity through fast motion / short occlusion — the thing object-IOR
  depends on.

Single seeds *can* show object-IOR winning (e.g. 3 objects, speed 28,
`ior_radius` 15: object-IOR best on latency 5.33 and waste 0.000). But averaged
over seeds the advantage does not survive:

High speed (4 objects, speed 40, `ior_radius` 20, 4 seeds averaged):

| tracker | arm | revisit waste | mean latency |
|---|---|---|---|
| naive | spatial-ior | **0.086** | **4.81** |
| naive | object-ior | 0.413 | 8.19 |
| motion-predicted | spatial-ior | **0.086** | **4.81** |
| motion-predicted | object-ior | 0.311 | 6.50 |

Occlusion (4 objects, speed 20, `ior_radius` 18, occlude-len 10, 4 seeds averaged):
space-IOR waste 0.240 vs object-IOR 0.253 — a tie, space-IOR marginally ahead.

**Honest conclusion.** Across seeds and regimes, **space-based IOR is at least as
good as object-based IOR, and usually better** — the thesis's headline advantage
for object-based selection does *not* robustly materialize here. Motion-predicted
tracking helps object-IOR (high-speed waste 0.413 → 0.311, latency 8.19 → 6.50)
but does not close the gap. The reason is structural: **object-IOR is only as
good as the tracker.** Every time the object-file correspondence switches a
label — under fast, dense, or crossing motion — the object-keyed inhibition is
lost and the object is re-fixated as if new. Space-IOR has no such failure mode;
it inhibits locations, which are always "tracked." So object-IOR's theoretical
edge (inhibition that follows the object) is spent paying for tracking errors.

This refines H1 into a sharper, testable claim: **object-based IOR beats
space-based only to the extent the tracker keeps identities stable** — and the
thesis's simple nearest-centroid correspondence (even with velocity prediction)
is not stable enough at the speeds where space-IOR would otherwise fail.

## Better tracking from the features we already have (and it helps — a lot)

If object-IOR is bottlenecked by label-switches, the fix is a better tracker —
and we can build one almost for free. The first selection stage has already
chosen the regions and the feature pipeline has already computed colour there,
so each object file can carry an **appearance descriptor** (mean colour of its
region) and correspondence can match on **motion + appearance**, not position
alone — the DeepSORT idea, with our own features as the embedding. Two crossing
objects of different colour then keep their labels (unit test:
`appearance matching keeps object identity through a crossing`). Both are opt-in
(`ObjectFileStore::Config::{motion_prediction, appearance_matching}`,
`--motion-prediction --appearance-matching`), default off so the thesis
behaviour is unchanged.

The effect on object-IOR is large. High speed (4 objects, speed 40,
`ior_radius` 20, 4 seeds):

| tracker | object-IOR waste | object-IOR latency |
|---|---|---|
| naive (position only) | 0.413 | 8.19 |
| motion-predicted | 0.311 | 6.50 |
| **motion + appearance** | **0.126** | **5.88** |

Appearance matching cuts object-IOR's wasted revisits **3×** (0.413 → 0.126) —
direct confirmation that label-switches were the bottleneck. Object-IOR goes
from clearly-worse to **nearly tied** with space-IOR (0.086 / 4.81).

But *nearly tied* is the honest word: even with good tracking, **space-IOR is
still marginally ahead on these exploration metrics** (high speed and occlusion
alike). Why: coverage / revisit-waste / latency reward *spreading attention over
the scene*, which "don't look where you just looked" does inherently well.
Object-IOR's distinctive product — knowing you are re-visiting the *same*
identity, and never needing to forget it (spatial memory must decay because
locations get reused; object memory needn't) — is barely exercised by these
metrics. So the tracking upgrade closed the gap that tracking errors had opened,
but did not, on its own, flip the exploration comparison.

## What a genuine object-IOR win would need (next)

- **Identity-centric metrics — the most likely place object-IOR wins.**
  Coverage / waste reward *finding* objects; they under-credit object-IOR's real
  product: a stable, identity-consistent scanpath, and *never forgetting* an
  object was seen (spatial memory has to decay because locations are reused;
  object memory needn't). Score ID-persistence and same-label occlusion recovery,
  and let object-IOR use *persistent* object memory against space-IOR's
  necessarily-decaying spatial memory — the asymmetry is the point.
- **Even better tracking** if needed — promote the kalman-mot backend's Kalman +
  gated association fully into the store, or add a stronger appearance embedding
  (histogram / per-feature signature, not just mean colour).
- **Rigorous statistics**: ≥20 seeds/cell with bootstrap CIs (here: ≤5 seeds).
- **Real video**: DAVIS-2017 masks for exact "which object attended".

## Artifacts

- `tools/make_dynamic_scene.py` — scene + `gt.json` generator (`dynamic-scene-gt/v1`).
- `eval/dynamic_ior.py` — three-arm runner + scorer (`--ior-radius`,
  `--motion-prediction`, `--appearance-matching`).
- Behaviors `greedy` / `spatial-ior` / `object-ior` (src/system/behavior.cpp) via
  `attention --attend --behavior <name> [--ior-radius R] [--ior-decay D]
  [--motion-prediction] [--appearance-matching]`.
- Robust correspondence: `ObjectFileStore::Config::{motion_prediction,
  appearance_matching, appearance_weight}` — appearance from the mean colour of
  each selected region, computed in `AttentionSystem::segment`.
