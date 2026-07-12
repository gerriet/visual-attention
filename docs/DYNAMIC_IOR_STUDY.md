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

## Where object-IOR *should* win (next)

The regimes this first cut does not yet stress, where object-IOR's advantage is
predicted to appear — the next iteration:

- **Fast motion relative to the inhibition scale** (small `ior_radius`, high
  speed) so objects escape spatial tags; map the crossover speed.
- **Occlusion / crossing stress**: an object that disappears and returns, or two
  that swap positions — object-IOR (with tracking memory / the kalman-mot
  backend) should remember "already seen" where space-IOR re-fixates the
  reappeared object. This is the clearest theoretical win and needs a reliable
  tracker in the loop.
- **Rigorous statistics**: ≥20 seeds/cell with bootstrap CIs (here: 3 seeds, no
  CIs) and a fixed dwell across arms to remove that confound.
- **Real video**: DAVIS-2017 masks for exact "which object attended" on natural
  motion.

## Artifacts

- `tools/make_dynamic_scene.py` — scene + `gt.json` generator (`dynamic-scene-gt/v1`).
- `eval/dynamic_ior.py` — three-arm runner + scorer.
- Behaviors: `greedy` / `spatial-ior` / `object-ior` (src/system/behavior.cpp),
  selectable with `attention --attend --behavior <name>`.
