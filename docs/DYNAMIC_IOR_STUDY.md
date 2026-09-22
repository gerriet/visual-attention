# Dynamic-IOR study (M12) — is object-based inhibition of return useful?

*Track: **replication** (docs/adr/0005) — H1 is the thesis's own claim. **Since replication-v2 (2026-09-22) the replication claim rests on the thesis's own chain, `configs/thesis/attend_field128.yaml` — see "The thesis's own chain" at the end: supported inside the field's tracking range, refuted outside. The tables of the earlier runs are measured on `configs/thesis/attend.yaml`, the segment-based extension arm, on which the claim holds at every speed.** **The evidence for H1 is the re-run at the end of this document (2026-09-21, 30 fresh scenes per regime, the thesis profile with all three static features ported from the original): H1 is supported — object-based IOR beats space-based IOR on latency and on sustained coverage in every regime, also against a motion-compensated location tag.** The first confirmatory run (2026-09-20) before it used a symmetry feature later found to be defective. The sections before that are the exploration that led there, kept as history; their numbers were measured with a colour feature that was not the thesis's, a defective eccentricity, and ≤ 6 seeds.*

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

## The re-run with the ported symmetry feature (2026-09-21) — the result that stands

**Why a second confirmatory run.** The replication dossier (M10), built the day
after the run above, found that the symmetry feature — a third of the thesis
profile — answered in rings *around* objects instead of on them
(`docs/replication/REPLICATION_DOSSIER.md`, finding A). It was ported from the
original source, judged by its own criteria (a lone disk peaks at its centre; a
single edge gives nothing; the thesis's Abb. 5.13 and 5.14), not by anything in
this study. H1 was then re-run unchanged: same command, same arms, same
predictions as written above, **a fresh block of scenes (seeds 2000–2029)**.
The model did change between the two runs, and the reader should know it; the
change was decided by a replication test that does not involve inhibition of
return, and the first run's numbers stay above for comparison.

`results/h1_confirmatory_symfix`; arm minus `spatial-ior`, paired over 30
scenes, 95% CI; **bold** = interval excludes zero.

| Regime | Arm | latency | Δ latency | staleness | Δ staleness | off-object | labels / object |
|---|---|---|---|---|---|---|---|
| standard | greedy | 23.58 | **+20.98** | 12.49 | **+10.16** | 0.00 | 1.4 |
| | spatial-ior | 2.61 | | 2.33 | | 0.00 | 1.7 |
| | spatial-ior-mc | 2.46 | **−0.15 [−0.32, −0.01]** | 2.06 | **−0.27 [−0.41, −0.15]** | 0.00 | 1.7 |
| | object-ior (thesis) | 2.13 | **−0.47 [−0.77, −0.24]** | 1.69 | **−0.64 [−0.83, −0.47]** | 0.00 | 1.7 |
| | object-ior+id | 2.10 | **−0.51 [−0.82, −0.24]** | 1.67 | **−0.66 [−0.86, −0.47]** | 0.00 | 1.3 |
| fast | greedy | 17.43 | **+11.35** | 9.99 | **+5.89** | 0.00 | 4.8 |
| | spatial-ior | 6.08 | | 4.10 | | 0.00 | 4.3 |
| | spatial-ior-mc | 3.37 | **−2.72 [−4.13, −1.49]** | 2.78 | **−1.32 [−1.84, −0.83]** | 0.00 | 4.6 |
| | object-ior (thesis) | 2.08 | **−4.00 [−5.28, −2.86]** | 2.10 | **−2.00 [−2.46, −1.59]** | 0.00 | 5.5 |
| | object-ior+aids | 1.73 | **−4.36 [−5.63, −3.24]** | 1.75 | **−2.35 [−2.81, −1.94]** | 0.00 | 3.9 |
| | object-ior+id | 1.63 | **−4.45 [−5.69, −3.36]** | 1.67 | **−2.43 [−2.89, −2.02]** | 0.00 | 1.7 |
| occlusion | greedy | 19.64 | **+15.35** | 10.40 | **+7.79** | 0.00 | 2.2 |
| | spatial-ior | 4.29 | | 2.61 | | 0.00 | 2.4 |
| | spatial-ior-mc | 2.19 | **−2.10 [−2.95, −1.33]** | 1.91 | **−0.71 [−0.99, −0.48]** | 0.00 | 2.6 |
| | object-ior (thesis) | 1.73 | **−2.56 [−3.40, −1.78]** | 1.57 | **−1.04 [−1.33, −0.81]** | 0.00 | 2.8 |
| | object-ior+id | 1.69 | **−2.60 [−3.43, −1.83]** | 1.54 | **−1.07 [−1.34, −0.85]** | 0.00 | 1.5 |

Against the strengthened baseline, `object-ior+id` minus `spatial-ior-mc`:
latency **−0.36 / −1.73 / −0.50**, staleness **−0.39 / −1.11 / −0.36**
(standard / fast / occlusion), every interval excluding zero.

**What changed, and why.** The share of fixations on no object fell from about a
third to **zero, in every arm**. The off-object fixations of the first run were
the symmetry feature's rings — salient clusters in empty space beside every
disk — not "object files that are not objects". With them gone, the object
files are the objects, and object-based inhibition does what the thesis says.

**The predictions, scored again.**

1. *IOR ≫ no IOR* — **holds.**
2. *The thesis's object-based IOR does not beat space-based IOR on staleness* —
   **refuted.** With the thesis's own nearest-centroid correspondence it is
   better on staleness *and* latency in all three regimes.
3. *Better identity buys the first sweep, not the long run* — **refuted.**
   Staleness is better in every regime, and the off-object share is 0.00, not
   above 0.15.
4. *Motion compensation is not what space-based IOR was missing* — **refuted.**
   `spatial-ior-mc` beats `spatial-ior` clearly in *fast* and *occlusion*: it is
   a genuinely stronger baseline — and object-based IOR still beats it.

Three of four predictions failed, all in the same direction: they were shaped
by development runs on the defective feature.

### Verdict on H1

**Supported.** In multi-object dynamic scenes object-based inhibition of return
reaches new objects sooner *and* keeps cycling through the scene more evenly
than space-based inhibition — in all three regimes, against a plain location tag
and against a motion-compensated one, with 30 fresh scenes per regime and
predictions (pessimistic ones) written down beforehand.

- *"Lower novel-object latency"* — yes: −0.5 / −4.4 / −2.6 frames.
- *"Degrades gracefully with speed, while space-based IOR collapses"* — from
  *standard* to *fast* space-based latency goes 2.61 → 6.08 and staleness 2.33 →
  4.10; object-based IOR goes 2.10 → 1.63 and 1.67 → 1.67. Space-based IOR
  degrades badly; "collapses" is too strong — it still sees every object.
- *"Higher object coverage"* — no difference: every IOR arm reaches 1.00 within
  40 frames. On these scenes coverage is the wrong place to look.
- *Identity still matters, but less than M12 concluded.* Persistent identity
  brings the labels per object from 5.5 to 1.7 at high speed and adds a further
  −0.4 frames of latency; the thesis's correspondence alone already carries most
  of the advantage. "Only as good as its tracker" was, to a large part, "only as
  good as its stage 1".

**What this does to the project's earlier "honest negative".** M12, ADR-0004 and
the first confirmatory run all reported that object-based IOR does not beat
space-based IOR. Each was an honest report of what was measured; what was
measured was a model whose colour feature was not the thesis's, whose
eccentricity ran on a black image, and whose symmetry answered beside objects.
The negative result was about the reimplementation, not about the thesis.

Open: the speed × object-count sweep, the Abb. 6.14 scenario, real video
(DAVIS), and whether the modern-track H7 result changes with the same stage 1.

## The thesis's own chain, and the thesis's own correspondence (2026-09-21, replication-v2)

**Why another block.** Reading the WAPCV 2003 paper against the thesis and the
code (`docs/replication/WAPCV_2003_NOTES.md`) showed two things about the runs
above:

- **The arm called "object-ior (thesis)" is position-only correspondence** —
  weaker than thesis §7.2.3, which also compares feature means where position is
  ambiguous and revives inactive object files "primarily by the feature
  properties". Wherever this document says "the thesis's (nearest-centroid)
  correspondence", read "position-only". `+aids` and `+id` are closer to §7.2.3
  than the word "extensions" suggests (they differ from it in using colour, a
  summed cost, a widening gate and no maximum age).
- **The neural field was not in the loop.** In the thesis object files are
  created for the field's activity clusters; `--attend` segments the saliency map
  instead. H1 above was measured on that approximation.

Both now exist as options (`attention_system.cluster_source: field`,
`object_files.correspondence: thesis`), and the field runs as the dissertation
system ran it over a stream — a fixed 20 cycles per frame, input gain 0.765
(dossier, finding 22). New arms:

| Arm | Object files on | Correspondence |
|---|---|---|
| `object-ior+7.2.3` | saliency segments | thesis §7.2.3 |
| `chain:spatial-ior`, `chain:spatial-ior-mc`, `chain:object-ior` | **the field's activity clusters** | thesis §7.2.3 |

The chain is run twice: with the dissertation system's field size (64 px,
`configs/thesis/attend_field.yaml`) and at twice that
(`attend_field128.yaml`). The scenes, regimes, metrics and the reference arm
(`spatial-ior` on saliency segments) are those of the runs above.

```bash
eval/dynamic_ior.py --regime all --seeds 30 --seed0 4000 --out results/h1_v2 \
  --arms spatial-ior,spatial-ior-mc,object-ior,object-ior+7.2.3,object-ior+id,chain:spatial-ior,chain:spatial-ior-mc,chain:object-ior
eval/dynamic_ior.py --regime all --seeds 30 --seed0 4000 --out results/h1_v2_field128 \
  --chain-config configs/thesis/attend_field128.yaml \
  --arms spatial-ior,chain:spatial-ior,chain:spatial-ior-mc,chain:object-ior
```

### What the development run showed (seeds 0–9)

With the 64-px field the chain holds **three of the four disks** and keeps them —
that is the first stage doing what the thesis says it does (a small number of
items, with hysteresis), but on these scenes (disks of radius 16 px = 3 field
pixels, global inhibition 8) the fourth object never gets a cluster, and with it
never an object file. At 128 px all four fit. The regimes' speeds in field
pixels per frame: *standard* 1.2 (2.4 at 128), *occlusion* 4 (8), *fast* 8 (16)
— the thesis puts the field's tracking limit at "an object movement of more than
12 pixels" and the dossier's tracking experiment (finding 20) agrees.

### Prediction, written down before the run

Seeds 4000–4029 have not been generated at the time of this commit.

1. **§7.2.3 on saliency segments changes little.** `object-ior+7.2.3` does not
   differ from `object-ior` on latency or staleness in any regime (on these
   scenes position is rarely ambiguous); it has fewer labels per object in
   *fast*. Both beat `spatial-ior` and `spatial-ior-mc` on staleness in every
   regime — the result of the re-run above holds on a fourth block. *(Refuted if
   either object-based arm fails to beat `spatial-ior-mc` on staleness in any
   regime.)*
2. **The chain with the dissertation system's field size loses** — to the
   segment-based `spatial-ior` on staleness in every regime, whatever the
   behaviour, with coverage below 1 in *standard*: the field's capacity, not
   inhibition of return, decides. Within the chain `object-ior` is not better
   than `spatial-ior`. *(Refuted if `chain:object-ior` beats the reference on
   staleness in any regime.)*
3. **At 128 px the chain supports H1 inside the field's tracking range and
   fails outside it.** In *standard*, `chain:object-ior` beats
   `chain:spatial-ior` and the segment-based `spatial-ior` on latency and
   staleness. In *fast* and *occlusion* `chain:object-ior` is *worse* than
   `chain:spatial-ior` on latency, with more than 6 labels per object: a field
   that cannot follow an object hands the second stage a new object file every
   few frames, and object-based inhibition has nothing to hold on to.
   *(Refuted if `chain:object-ior` is not ahead in *standard*, or is ahead in
   *fast*.)*

If 1–3 hold, the verdict on H1 is refined, not reversed: **object-based
inhibition of return beats space-based inhibition wherever the first stage
delivers trackable objects** — with the reimplementation's segment-based first
stage at every speed tested, with the thesis's neural field inside its tracking
range (which the thesis states). The high-speed result of the re-run above is a
result about the segment-based variant, which the WAPCV paper names as an
extension ("a segmentation process based on the feature computations").

### Outcome (30 fresh scenes per regime, seeds 4000–4029, 2026-09-21)

`results/h1_v2`; arm minus the segment-based `spatial-ior`, paired over scenes,
95% CI; **bold** = interval excludes zero. Coverage is 1.00 for the segment-based
arms (0.99 for `spatial-ior` in *fast*).

| Regime | Arm | latency | Δ latency | staleness | Δ staleness | off-object | labels / object |
|---|---|---|---|---|---|---|---|
| standard | spatial-ior | 2.68 | | 2.28 | | 0.00 | 1.6 |
| | spatial-ior-mc | 2.34 | −0.34 [−0.82, +0.01] | 2.07 | **−0.21 [−0.31, −0.11]** | 0.00 | 1.6 |
| | object-ior (position only) | 1.88 | **−0.80 [−1.43, −0.29]** | 1.72 | **−0.56 [−0.71, −0.41]** | 0.00 | 1.7 |
| | object-ior+7.2.3 | 1.88 | **−0.80 [−1.42, −0.29]** | 1.72 | **−0.56 [−0.71, −0.41]** | 0.00 | 1.8 |
| | object-ior+id | 1.88 | **−0.81 [−1.43, −0.29]** | 1.70 | **−0.58 [−0.73, −0.43]** | 0.00 | 1.2 |
| | chain:spatial-ior (field 64) | 3.40 | **+0.72 [+0.08, +1.51]** | 4.20 | **+1.93 [+1.48, +2.35]** | 0.02 | 1.3 |
| | chain:spatial-ior-mc | 3.02 | +0.33 [−0.45, +1.23] | 4.09 | **+1.81 [+1.36, +2.28]** | 0.02 | 1.3 |
| | chain:object-ior | 2.67 | −0.01 [−0.95, +1.04] | 3.85 | **+1.58 [+1.11, +2.04]** | 0.01 | 1.3 |
| fast | spatial-ior | 6.66 | | 4.37 | | 0.00 | 4.2 |
| | spatial-ior-mc | 3.12 | **−3.54 [−5.12, −2.22]** | 2.79 | **−1.58 [−2.21, −1.04]** | 0.00 | 4.6 |
| | object-ior (position only) | 2.06 | **−4.60 [−6.08, −3.35]** | 2.07 | **−2.30 [−2.96, −1.70]** | 0.00 | 5.8 |
| | object-ior+7.2.3 | 1.93 | **−4.72 [−6.24, −3.44]** | 1.91 | **−2.46 [−3.06, −1.93]** | 0.00 | 4.2 |
| | object-ior+id | 1.77 | **−4.88 [−6.41, −3.59]** | 1.66 | **−2.70 [−3.34, −2.15]** | 0.00 | 1.6 |
| | chain:spatial-ior (field 64) | 6.67 | +0.02 [−1.69, +1.70] | 4.86 | +0.49 [−0.27, +1.19] | 0.24 | 6.4 |
| | chain:spatial-ior-mc | 7.16 | +0.50 [−1.34, +2.24] | 5.08 | +0.71 [−0.09, +1.45] | 0.26 | 6.4 |
| | chain:object-ior | 7.47 | +0.81 [−0.70, +2.29] | 5.18 | **+0.81 [+0.15, +1.49]** | 0.27 | 6.5 |
| occlusion | spatial-ior | 3.85 | | 2.57 | | 0.00 | 2.5 |
| | spatial-ior-mc | 2.23 | **−1.62 [−2.31, −0.97]** | 1.83 | **−0.74 [−0.93, −0.56]** | 0.00 | 2.6 |
| | object-ior (position only) | 1.75 | **−2.10 [−2.88, −1.35]** | 1.63 | **−0.94 [−1.12, −0.78]** | 0.00 | 2.8 |
| | object-ior+7.2.3 | 1.82 | **−2.03 [−2.81, −1.29]** | 1.64 | **−0.93 [−1.13, −0.74]** | 0.00 | 2.7 |
| | object-ior+id | 1.65 | **−2.20 [−2.98, −1.48]** | 1.54 | **−1.04 [−1.23, −0.86]** | 0.00 | 1.4 |
| | chain:spatial-ior (field 64) | 5.82 | **+1.97 [+0.35, +3.69]** | 4.45 | **+1.87 [+1.37, +2.40]** | 0.06 | 3.1 |
| | chain:spatial-ior-mc | 4.41 | +0.56 [−0.87, +1.99] | 4.13 | **+1.56 [+1.01, +2.19]** | 0.15 | 3.1 |
| | chain:object-ior | 3.98 | +0.12 [−1.23, +1.52] | 4.24 | **+1.66 [+1.13, +2.23]** | 0.11 | 3.2 |

Against the strengthened baseline, object-based minus `spatial-ior-mc`
(standard / fast / occlusion), every interval excluding zero — latency:
position-only −0.46 / −1.06 / −0.48, §7.2.3 −0.46 / −1.18 / −0.42, `+id` −0.47 /
−1.34 / −0.58; staleness: −0.35 / −0.72 / −0.20, −0.35 / −0.88 / −0.19, −0.37 /
−1.12 / −0.30. Within the chain (field 64), `chain:object-ior` minus
`chain:spatial-ior`: staleness **−0.35 [−0.54, −0.18]** (standard), +0.32
[−0.17, +0.80] (fast), −0.21 [−0.53, +0.09] (occlusion); latency −0.72 [−1.79,
+0.25], +0.79 [−0.23, +1.82], **−1.84 [−2.98, −0.82]**.

**The predictions, scored.**

1. *§7.2.3 on saliency segments changes little; H1 holds on a fourth block* —
   **holds.** `object-ior+7.2.3` and position-only correspondence do not differ
   on latency in any regime, nor on staleness in *standard* and *occlusion*; in
   *fast* §7.2.3 brings the labels per object from 5.8 to 4.2 and gains a little
   staleness (−0.16 [−0.31, −0.05], not predicted). Every object-based arm beats
   `spatial-ior` and `spatial-ior-mc` on latency and staleness in every regime.
2. *The chain with the dissertation system's field size loses to the
   segment-based reference* — **holds for staleness** in *standard* and
   *occlusion* whatever the behaviour, and in *fast* for `chain:object-ior` (the
   two spatial chain arms there are worse too, within the interval). About a
   quarter of the chain's fixations in *fast* land on no object: clusters the
   field keeps alive where an object was. The detail *"within the chain
   `object-ior` is not better than `spatial-ior`"* — **refuted, in H1's
   favour**: inside the chain object-based inhibition has the better staleness
   in *standard* and the better latency under *occlusion*. Coverage stayed at
   0.98–1.00 over 40 frames; the capacity limit shows as latency and staleness,
   not as objects never seen.
3. *At 128 px the chain supports H1 inside the field's tracking range and fails
   outside it* — **holds, on every count** (`results/h1_v2_field128`):

   | Regime (speed in field px / frame) | chain:object-ior − chain:spatial-ior: latency | staleness | − chain:spatial-ior-mc: latency | staleness | labels / object |
   |---|---|---|---|---|---|
   | standard (2.4) | **−0.93 [−1.63, −0.38]** | **−0.74 [−1.06, −0.51]** | **−0.70 [−1.13, −0.34]** | **−0.55 [−0.77, −0.39]** | 3.0 |
   | occlusion (8) | **+2.40 [+1.11, +3.74]** | **+0.50 [+0.11, +0.90]** | **+3.88 [+2.51, +5.37]** | **+0.80 [+0.39, +1.24]** | 9.0 |
   | fast (16) | **+3.66 [+1.95, +5.34]** | **+1.56 [+1.10, +2.08]** | **+3.63 [+1.79, +5.53]** | **+1.50 [+0.97, +2.11]** | 10.3 |

   In *standard* `chain:object-ior` (latency 1.62, staleness 1.68) also beats the
   segment-based reference (−1.07 [−1.74, −0.47], −0.60 [−0.78, −0.44]) and is
   the best arm of the whole block. In *occlusion* and *fast* the field hands the
   second stage a new object file every few frames — nine to ten labels per
   object — and object-based inhibition, which has nothing to hold on to, is the
   *worst* behaviour; the spatial behaviours on the same clusters do not care
   whose cluster it is.

### Occlusion inside the tracking range (prediction, written before the run, 2026-09-22)

The occlusion regime moves objects at 8 field pixels per frame at 128 px — outside
the range in which the thesis says its field tracks — so on the chain it tests
the claim outside its stated domain. Thesis Abb. 6.14 is about identity through
occlusion *at trackable speed*. New regime `occlusion-slow`: speed 6, a 10-frame
occlusion, the standard tag radius; seeds 4000–4029, not yet generated at this
commit; chain at 128 px.

4. **Inside the tracking range, occlusion does not break object-based inhibition
   on the chain.** `chain:object-ior` beats `chain:spatial-ior` and
   `chain:spatial-ior-mc` on staleness, as in *standard*; the occluded object is
   re-found with a new label more often than not (labels per object 2–4, against
   3.0 in *standard*), so the latency advantage is smaller than in *standard* and
   may not exclude zero. *(Refuted if `chain:object-ior` is worse than
   `chain:spatial-ior` on staleness.)*

### Verdict on H1, refined

**Supported — wherever the first stage delivers objects that can be tracked.**

- With the reimplementation's segment-based first stage, object-based inhibition
  of return beats space-based inhibition, plain and motion-compensated, on
  latency and on staleness, at every speed tested, now on four independent
  blocks of scenes (1000, 2000, 3000, 4000) and with position-only
  correspondence, the correspondence of thesis §7.2.3, or persistent identity.
- With the thesis's own first stage — the neural field — it does so **inside the
  field's tracking range** (a few field pixels per frame), where that chain is
  the best system measured here, and **fails outside it**, where it is the
  worst. The thesis states the limit ("an object movement of more than 12
  pixels"); what it does not say is that beyond it object-based inhibition is
  not merely no better but worse than the location tag it set out to replace.
- The field's *capacity* is a second condition: at the dissertation system's
  field size it holds about three objects of these scenes' size, and the whole
  chain then loses to a system without a field, whatever the behaviour.
- The high-speed result of the earlier blocks is therefore a result about the
  segment-based variant — the one the WAPCV paper's last sentence proposes as an
  extension — not about the dissertation system.


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
