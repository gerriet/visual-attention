# Dynamic-IOR study (M12) — is object-based inhibition of return useful?

*Track: **replication** (docs/adr/0005) — H1 is the thesis's own claim. **The evidence for H1 is the confirmatory run at the end of this document (2026-09-20, 30 fresh scenes per regime, thesis profile): object-based IOR reaches new objects sooner in every regime — supported for the quantity H1 names, conditional on held identity at high speed.** The sections before it are the exploration that led there, kept as history; their numbers were measured with a colour feature that was not the thesis's, a defective eccentricity, and ≤ 6 seeds.*

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

## Persistent identity (M19): object-IOR draws level, and ahead at speed

M19 (the video token cache, `docs/VLM_VIDEO.md`) needs identity-keyed memory,
which motivated opt-in **persistent identity** in the object-file store
(`attention_system.object_files.persistent_identity`, shipped as
`configs/attend_identity.yaml`): an unmatched cluster revives the inactive file
with the lowest position + colour cost inside a gate that widens with the time
unseen — or anywhere, if it is a clear colour look-alike; a clearly different
colour is never the same object; inactive files never age out. The thesis path
is unchanged by default. Rerunning this study's three regimes, 6 seeds each
(tracking aids on in both columns):

| Regime | arm | aids: latency / waste | persistent identity: latency / waste |
|---|---|---|---|
| standard (4 obj, speed 6, occlusion) | space-IOR | 5.54 / 0.145 | 5.54 / 0.145 |
| | object-IOR | 4.54 / 0.113 | 4.63 / 0.123 |
| fast (speed 40, `ior_radius` 20) | space-IOR | 5.58 / 0.088 | 5.58 / 0.088 |
| | object-IOR | 6.88 / 0.185 | **5.21** / 0.098 |
| occlusion (speed 20, length 10, `ior_radius` 18) | space-IOR | 5.63 / 0.101 | 5.63 / 0.101 |
| | object-IOR | 7.29 / 0.164 | **5.63** / 0.115 |

(The "aids" column differs from the earlier sections — there space-IOR led in
every regime, here object-IOR already leads in the standard one. The seed sets
differ (≤ 5 seeds then, 6 here) and neither has intervals; differences of this
size are within what a handful of seeds can produce, which is the reason the
confirmatory run is needed.)

Coverage is 1.00 for both IOR arms throughout; space-IOR is unaffected (it
doesn't inhibit by identity). Persistent identity takes object-IOR from clearly
worse to *ahead* of space-IOR on latency at high speed and level under
occlusion, with waste close to space-IOR's — the first regime map in which the
thesis's object-based advantage shows. Six seeds without CIs: a direction, not
yet a result. Identity is still far from stable (≈ 4.5 distinct labels per
attended object): the remaining loss is in *segmentation*, where a moving disk
falls apart into onset crescents that each get an object file. The opt-in
segmentation settings that address it (`segment_close`,
`max_cluster_fraction`) are documented with the M19 findings in
`docs/VLM_VIDEO.md`.

## The confirmatory run (2026-09-20)

Everything above is exploration: ≤ 6 hand-averaged seeds, no intervals, a
colour feature that was not the thesis's and a defective eccentricity, a
revisit metric that stops counting once every object has been seen, and a
spatial arm that never got a strengthened version. This section replaces it as
the evidence for H1 (`docs/HYPOTHESIS_CLOSURE_PLAN.md`, `docs/adr/0005`).

**Instrument.** `eval/dynamic_ior.py --regime all --seeds 30 --seed0 1000` —
one command, scenes generated per seed, every arm on identical scenes, paired
bootstrap differences over scenes. Profile: `configs/thesis/attend.yaml` (the
thesis's colour contrast, eccentricity, symmetry, exclusivity, plus onset).

| Arm | Inhibition rides on | Identity held by |
|---|---|---|
| `greedy` | nothing | — |
| `spatial-ior` | decaying location tags | — |
| `spatial-ior-mc` | location tags that drift with the velocity of the object they were left on (**the strengthened space-based baseline**) | — (dead reckoning) |
| `object-ior` | object files | the thesis's nearest-centroid correspondence |
| `object-ior+aids` | object files | + motion prediction, appearance |
| `object-ior+id` | object files | + persistent identity |

Regimes as above: *standard* (speed 6, 60-px tag, one occlusion), *fast*
(speed 40, 20-px tag), *occlusion* (speed 20, 18-px tag, 10-frame occlusion).

**Metrics.** Primary: **mean latency** (how soon a new object is attended) and
**staleness** — the mean, over all frames and visible objects, of the frames
since that object was last attended. Staleness is new: revisit waste only
accrues until every object has been seen once (2–8 frames of 40), so it is
decided by a handful of fixations; staleness scores how evenly attention keeps
cycling through the scene for the whole video, which is what inhibition of
return is *for*. Secondary: revisit waste, the share of fixations on no object
(previously dropped from every denominator), and distinct labels per attended
object (the tracker's identity switches).

### Prediction, written down before the run

Development seeds 0–9 (the run that shaped these predictions) are in
`results/h1_dev`; the test seeds 1000–1029 have not been generated at the time
of this commit. A difference "holds" if its paired 95% interval excludes zero.

1. **IOR ≫ no IOR.** `greedy` is worst on latency, staleness and coverage in
   every regime. *(H1's first half; refuted if any IOR arm fails to beat it.)*
2. **The thesis's object-based IOR does not beat space-based IOR.** `object-ior`
   (thesis correspondence) is no better than `spatial-ior` on staleness in any
   regime, and worse in *fast*. *(H1 as the thesis states it; refuted if
   `object-ior` beats `spatial-ior` on staleness in *fast* or *occlusion*.)*
3. **Better identity buys the first sweep, not the long run.** `object-ior+id`
   reaches new objects sooner (lower latency) and wastes fewer early revisits
   than `spatial-ior` in *fast* and *occlusion* — but its **staleness is not
   better** in any regime, because about a third of its fixations land on no
   object: object-based IOR works through *every* object file, including
   low-saliency background clusters, while a location tag that a moving object
   leaves behind simply frees that object again. *(Refuted if `object-ior+id`
   has lower staleness than `spatial-ior` in *fast* or *occlusion*, or if its
   off-object share is below 0.15.)*
4. **Motion compensation is not what space-based IOR was missing.**
   `spatial-ior-mc` does not differ from `spatial-ior` on latency or staleness
   in *fast* and *occlusion* (at these speeds a 20-px tag is outrun within a
   frame either way, and a dead-reckoned tag does not survive a bounce); it may
   help in *standard*. `object-ior+id` beats it on latency in *fast*.
   *(Refuted if `spatial-ior-mc` beats `spatial-ior` on staleness in *fast*.)*

If 2 and 3 hold, the verdict for H1 is: **not supported as stated; supported
under a condition and for one quantity** — object-based inhibition finds new
objects sooner once identity is held, and loses the sustained-coverage
comparison to a plain location tag as long as the object files contain things
that are not objects. That would move the open question from *tracking* (where
M12 put it) to *segmentation*: what deserves an object file.

### Outcome (30 fresh scenes per regime, seeds 1000–1029)

`results/h1_confirmatory`; arm minus `spatial-ior`, paired over scenes, 95% CI.
Coverage is 1.00 for every IOR arm (0.99 for the spatial arms in *standard*), so
it does not discriminate. **Bold** = interval excludes zero.

| Regime | Arm | latency | Δ latency | staleness | Δ staleness | off-object | labels / object |
|---|---|---|---|---|---|---|---|
| standard | greedy | 22.11 | **+19.17** | 11.81 | **+9.46** | 0.00 | 1.4 |
| | spatial-ior | 2.94 | | 2.36 | | 0.02 | 1.7 |
| | spatial-ior-mc | 2.71 | −0.23 [−0.80, +0.31] | 2.25 | −0.11 [−0.43, +0.21] | 0.03 | 1.7 |
| | object-ior (thesis) | 1.77 | **−1.17 [−1.98, −0.52]** | 2.65 | +0.29 [−0.05, +0.57] | 0.32 | 1.9 |
| | object-ior+id | 1.79 | **−1.15 [−1.98, −0.48]** | 2.58 | +0.22 [−0.10, +0.49] | 0.31 | 1.4 |
| fast | greedy | 16.23 | **+12.22** | 10.10 | **+7.08** | 0.00 | 8.0 |
| | spatial-ior | 4.00 | | 3.02 | | 0.00 | 6.4 |
| | spatial-ior-mc | 3.54 | −0.46 [−1.03, +0.12] | 2.85 | −0.17 [−0.49, +0.16] | 0.01 | 6.5 |
| | object-ior (thesis) | 3.65 | −0.35 [−1.27, +0.58] | 3.45 | **+0.43 [+0.07, +0.77]** | 0.15 | 7.6 |
| | object-ior+aids | 2.04 | **−1.96 [−2.75, −1.18]** | 3.15 | +0.12 [−0.19, +0.43] | 0.34 | 3.7 |
| | object-ior+id | 1.85 | **−2.15 [−2.92, −1.40]** | 2.64 | **−0.38 [−0.65, −0.11]** | 0.31 | 2.0 |
| occlusion | greedy | 17.77 | **+15.21** | 10.00 | **+7.64** | 0.00 | 3.4 |
| | spatial-ior | 2.57 | | 2.35 | | 0.02 | 3.3 |
| | spatial-ior-mc | 2.25 | **−0.32 [−0.57, −0.07]** | 2.32 | −0.03 [−0.17, +0.10] | 0.13 | 3.3 |
| | object-ior (thesis) | 2.05 | **−0.52 [−0.89, −0.17]** | 2.94 | **+0.59 [+0.39, +0.80]** | 0.34 | 3.6 |
| | object-ior+id | 1.82 | **−0.75 [−1.10, −0.42]** | 2.63 | **+0.28 [+0.12, +0.45]** | 0.33 | 1.7 |

Against the strengthened baseline, `object-ior+id` minus `spatial-ior-mc`:
latency **−0.92** (standard), **−1.69** (fast), **−0.43** (occlusion), all
intervals excluding zero; staleness **+0.34** (standard), −0.21 [−0.53, +0.05]
(fast), **+0.31** (occlusion).

**The predictions, scored.**

1. *IOR ≫ no IOR* — **holds**, in every regime and on every metric, by a wide
   margin.
2. *The thesis's object-based IOR does not beat space-based IOR on staleness,
   and is worse in fast* — **holds** (worse in *fast* and *occlusion*, no
   difference in *standard*). Not predicted either way, and worth stating: even
   with the thesis's own nearest-centroid correspondence it reaches new objects
   sooner in *standard* and *occlusion*.
3. *Better identity buys the first sweep, not the long run* — **partly
   refuted, in object-based IOR's favour.** The latency and early-waste
   advantage holds in every regime, and the off-object share is 0.31–0.33 as
   predicted. But in *fast* `object-ior+id` also has **lower staleness** than
   `spatial-ior` (−0.38 [−0.65, −0.11]) — the prediction said it would not. In
   *occlusion* it is worse, in *standard* there is no difference.
4. *Motion compensation is not what space-based IOR was missing* — **holds**:
   `spatial-ior-mc` does not differ from `spatial-ior` in *fast* or *standard*;
   it gains a little latency in *occlusion* (not predicted). `object-ior+id`
   beats it on latency in all three regimes.

### Verdict on H1

H1: *"object-based IOR yields higher object coverage and lower novel-object
detection latency than space-based IOR, and degrades gracefully with object
speed — while space-based IOR collapses."*

- **Novel-object latency: supported.** Object-based inhibition reaches new
  objects sooner than space-based inhibition in all three regimes — against the
  plain location tag and against the motion-compensated one. At low and medium
  speed the thesis's own correspondence is enough; at high speed the advantage
  exists only when identity is held (`+aids`, `+id`: about two labels per
  object instead of eight).
- **Graceful degradation with speed: supported, with held identity.** From
  *standard* to *fast*, `spatial-ior`'s latency rises 2.94 → 4.00 and the
  thesis's `object-ior` 1.77 → 3.65, while `object-ior+id` stays at 1.79 →
  1.85. Space-based IOR degrades; it does not collapse (coverage stays 1.0).
- **Coverage: no difference** — every IOR arm sees every object in 40 frames.
- **Sustained coverage (staleness) — not part of H1 as stated, and not an
  object-based win:** worse under occlusion, equal at low speed, better only at
  high speed with held identity. The cost is visible in the off-object share:
  about a third of the object-based arms' fixations go to object files that are
  not objects, against 0–2% for a location tag.

**Caveat added 2026-09-20 (replication dossier, finding A).** The symmetry
feature — a third of this profile — responds in rings *around* objects rather
than on them (`docs/replication/REPLICATION_DOSSIER.md`). On scenes of disks
that adds salient clusters in empty space, and is a plausible contributor to the
off-object third of the object-based arms' fixations (a location tag rarely
selects them because the true objects, freed again by their own motion, outrank
them). The latency result does not depend on it; the *staleness* comparison and
the "what deserves an object file" reading should be re-checked after symmetry
is ported from the original — H1 is cheap to re-run (one command, ~30 min).

So the earlier reading of this document — "space-based IOR is at least as good
as object-based, usually better" — **does not survive** the confirmatory run. It
rested on a revisit metric that a handful of fixations decided, a few seeds,
and a stage 1 that was not the thesis's. The thesis's claim holds for the
quantity it names, under the identity condition M12 identified; and the
remaining weakness has moved from *tracking* to *segmentation*: what deserves
an object file. (That is also what M19 found from the other side, where
proto-objects, not persistent identity, carried the effect.)

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
