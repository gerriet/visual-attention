# Replication dossier — the dissertation's findings, re-run (M10)

*Track: **replication** (docs/adr/0005). First version 2026-09-20; symmetry
ported and re-run, the stereo experiments and the field dynamics added,
2026-09-21. Every number here is printed by one command:*

```bash
eval/replication.py --all        # CPU only, ~15 min; figures land in docs/replication/figures
```

The dissertation (Backer 2004, ch. 5–6) supports its features and its selection
stage with small, concrete experiments: vary the property a feature is named
after and watch its response; add noise; sweep a parameter and look for the
range in which the result stays plausible. This dossier rebuilds those
experiments on the reimplementation and gives each a verdict:

- **replicated** — the claim holds, measured;
- **partially** — it holds in part, or only in a weaker form;
- **diverged** — it does not hold, with the reason;
- **not attempted** — and why.

The thesis's lab images are not available, so each experiment rebuilds the
*kind* of stimulus the figure describes; parameter sweeps on a natural image use
the thesis's own running example (`data/test_images/inputc.png`, the office
scene of Abb. 5.8–5.22). Feature responses are read on an absolute [0, 1] scale
(`attention --emit-features`), one feature at a time, with the thesis's
parameters and without exclusivity unless the experiment is about it.
"Divergences are findings, not failures" (roadmap M10): the first version of
this dossier found that the symmetry feature diverged (finding A); it has since
been ported from the original and the affected rows re-run.

## Summary

| # | Thesis | Claim | Verdict |
|---|---|---|---|
| 1 | Abb. 5.10 | Eccentricity is 0 for round shapes and high for elongated ones | **replicated** |
| 2a | Abb. 5.13 | Stretching an object raises its eccentricity response, monotonically | **replicated** — on the thesis's eq. 5.6 to within 0.02 |
| 2b | Abb. 5.13 | … and lowers its symmetry response | **replicated** after the symmetry port (0.67 → 0.15, monotone); *diverged* before it (finding A) |
| 3a | Abb. 5.14 | The eccentricity maximum stays on its object under added noise | **replicated** up to σ ≈ 38 grey levels; degrades beyond |
| 3b | Abb. 5.14 | The symmetry maximum stays on its object under added noise | **replicated** after the port — on the object at every noise level, up to σ ≈ 115; *diverged* before it (finding A) |
| 4 | Abb. 5.15 | Eccentricity segmentation is plausible for a growth threshold of 0.5–0.75 | **partially** — stable in 0.5–0.75 as claimed, but less flat than the thesis suggests (map correlation 0.64–0.78 with the default) |
| 5 | Abb. 5.16 | … and for a merge threshold of 12–36 (default 20) | **replicated** — correlation ≥ 0.88 over the whole range 4–48 |
| 6 | Abb. 5.20 | The colour-contrast response grows with the colour difference | **replicated** — monotone; saturates early |
| 7 | Abb. 5.21 | The colour-contrast maximum is robust to added noise | **replicated** — on the object at every noise level tested |
| 8 | Abb. 5.22 | Colour contrast is insensitive to `cc_add` / `cc_mult` over a broad range; only clearly different values change the salient regions (ball, picture) | **replicated** — and the same two regions are the salient ones |
| 9 | Abb. 5.34 | Exclusivity lowers the saliency of common orientations / colours relative to unique ones | **replicated** |
| 11 | Abb. 5.28 | The depth response follows an object's distance | **replicated** — the estimated disparity equals the true one at every step |
| 12 | Abb. 5.29 | Depth is found in a random-dot stereogram | **replicated** — the square exists only as a disparity and is recovered completely |
| 13 | Abb. 5.30 | Under independent noise the disparity estimate "changes only in a very small range" | **replicated** — 10.0 → 9.9 px at σ ≈ 90 grey levels, *without* the multi-scale scheme the thesis credits for it |
| 14 | Abb. 5.31 | Several near-vertical orientations help; beyond one, the choice matters little | **replicated** |
| 15 | Abb. 5.32 | Only very high variance thresholds drop correct results; very low ones admit wrong pixels on structureless surfaces | **replicated** on the synthetic scene; on real pairs (Middlebury) a low threshold is simply best — the default stays |
| 15b | on Abb. 5.25 | "In the great majority of regions the disparity is determined correctly" | **replicated** on independent real data: 82–85% of pixels within 1 px on three Middlebury pairs |
| 16 | Abb. 6.4 | Hysteresis: the activation cluster changes place only after the formerly weaker input is clearly stronger | **replicated** with the dissertation system's field parameters (switch at α = 0.55 rising, held down to 0.25 falling); *diverged* with the port's defaults (finding B) |
| 17 | Abb. 6.5 | Bifurcation: from a certain distance two maxima give two clusters | **replicated** |
| 18 | Abb. 6.6 | Noise suppression: for a pulse of amplitude 1, only signal-caused activation up to noise amplitude 1.2 | **replicated** exactly with the dissertation parameters (zero outside the pulse up to 1.2, first activation at 1.4; zero-mean noise); *diverged* with the port's defaults |
| 19 | Abb. 6.8 | A stable state is reached in about 10 cycles | **replicated** — 10 cycles |
| 20 | Abb. 6.9 | Tracking a target in noise: within limits ~10 update cycles per frame suffice; weak (0.5) and fast targets are the limit | **replicated** with the dissertation parameters; *diverged* with the port's defaults (most moving targets lost) |
| 21 | Abb. 6.10 | Two approaching maxima: repulsion, then merging below distance 5; no oscillation | **replicated** with the dissertation parameters (merge at 5, one clean transition, re-split only at 13); *diverged* with the port's defaults (never merge) |
| 10 | Abb. 6.14 | Object identity survives temporary occlusion when several objects are tracked | **partially** — different architecture; see the H1 occlusion regime |
| — | Abb. 5.33, 5.35, 6.11–6.13, §9.3 | superposition, 2D vs 3D integration, multi-field systems and the 3D field, flanker and early-vs-late selection | **not attempted** (below) |

## The findings

### 1 · Abb. 5.10 — simple shapes — replicated

| Shape | measured | eq. 5.6 |
|---|---|---|
| disk | 0.000 | 0 |
| square | 0.000 | 0 |
| ellipse, axes 2:1 | 0.354 | 0.360 |
| bar, 8:1 | 0.934 | 0.940 |

"0 for a round object and 1 for a line-shaped one and therefore already suitably
normalized as a saliency measure" — yes. (Before the 2026-09 port the same
ellipse scored 0.87, from a different formula, on segments that ignored the
image.)

### 2 · Abb. 5.13 — stretching an object

![eccentricity and symmetry while an object is stretched](figures/ecc_variation.png)

A reference disk on the left; on the right an object of constant area stretched
from 1:1 to 6:1. **Eccentricity** rises monotonically and lies on
((a² − 1)/(a² + 1))² — the thesis's eq. 5.6 for an ellipse — to within 0.02 at
every step. **Symmetry** falls monotonically, 0.67 → 0.62 → 0.52 → 0.37 → … →
0.15 ("immer geringere Salienzwerte") — the two features trade places exactly as
the thesis's figure shows. *(Before the symmetry port it went 0.67 → 0.50 →
0.87 → 0.82: not monotone, higher at the end than at the start.)*

### Finding A — the symmetry feature did not peak on symmetric objects (found 2026-09-20, fixed 2026-09-21)

*What the first version of this dossier found.* For a single bright disk on a grey ground the symmetry map has responses inside
the disk *and equally strong ones about 80 px to either side of it*, in empty
space; its global maximum is off the object for every disk size tried (radius
12, 24, 40), without any noise. This is the "symmetry fires in the empty space
between the disks" that the video study met (`docs/VLM_VIDEO.md`) — here
isolated.

The cause is not the summation core, which is a line-for-line port of the
original `symmetry_intern` (the sum over the two opposite boxes is *additive*, so
an edge on one side alone already contributes about half of a true symmetry).
It is what happens afterwards. The original works on an absolute scale — Gabor
magnitudes clipped at 255, then a fixed offset of 60 subtracted from each radius
band — which removes exactly those one-sided half-responses. The
reimplementation instead normalizes every radius band to its own maximum and
applies *relative* thresholds (0.3 / 0.5 / 0.65) plus a "≥ 2 radii" consistency
weight, over three pyramid scales with radii up to 30 px at 1/8 resolution (240
px in the image; the thesis's Tab. 5.1 has radii 6–15 at one scale). With no
absolute reference, a one-sided response at a coarse scale is as strong as a
true centre, and appears as a ring around every object.

*The fix.* The combination step was ported from the original
(`feature/symmetry.C`, thesis Tab. 5.1): a working image of at most 256 px,
quadrature Gabor energy on an absolute scale (a full-contrast step edge = 1),
radii 6–15 in adjoining bands of 3 at the working size and its two halvings, a
fixed clip offset of 60/255, scales combined by maximum. A lone disk now peaks
at its centre (0.43 / 0.82 / 0.99 for radius 12 / 24 / 40), a single straight
edge — the one-sided case — gives 0.08, and on the thesis's running example the
map peaks on the ball. One thing the sources do not settle is the gain of the
original Gabor implementation, which fixes what "60 of 255" means; the
calibration chosen is documented in `symmetry_feature.h`.

*Consequences.* Symmetry is a third of the thesis profile, so the H1
confirmatory run and the M11 human-scanpath run were repeated with the ported
feature (H1 on a fresh block of seeds); both documents say which numbers are
which.

### 3 · Abb. 5.14 / 5.21 — noise

![share of noise seeds for which a feature's maximum stays on its object](figures/noise.png)

Normally distributed noise, σ = level × 128 grey levels, five seeds per level.
The eccentricity maximum stays on the bar up to level 0.3 (σ ≈ 38) and is lost
for half the seeds beyond; the colour-contrast maximum stays on the blob at
every level, up to σ ≈ 115 (the blob-minus-ground contrast falls only from 1.00
to 0.91). The symmetry maximum stays on the disk at every level as well, with
the bar in the scene or without. (The figure is from the re-run.)

*A side finding:* eccentricity histogram-equalizes its input, as the original
does. On a synthetic image with three grey levels that maps a *bright* object
and a dominant ground to nearly the same grey (225 vs 128 become 255 vs ~237);
their difference then passes the merge criterion (max. 20) and the object
dissolves into the ground. The experiment uses a dark bar for that reason. On
natural images, with full histograms, this does not arise — but it is a trap for
synthetic stimuli, H1's scenes included.

### 4, 5 · Abb. 5.15 / 5.16 — the two eccentricity thresholds

![growth threshold](figures/grow_threshold.png) ![merge threshold](figures/merge_threshold.png)

On the running example, the correlation of the eccentricity map with the map at
the thesis's default. **Merge threshold** (max. mean-grey difference, default
20, thesis: plausible 12–36): ≥ 0.88 everywhere from 4 to 48 — insensitive, as
claimed, and over a wider range than claimed. **Growth threshold** (share of
pixels counted as area, default 0.65, thesis: plausible 0.5–0.75): 0.64–0.78
inside the claimed range, collapsing below 0.5 (0.06 at 0.4) and declining
slowly above 0.75. The claimed range is the right one; "plausible" in the
thesis is a visual judgement of the segmentation, and a pixel-wise map
correlation is a stricter reading than that — hence *partially*.

### 6 · Abb. 5.20 — colour contrast variation — replicated

![colour-contrast response while a blob's colour moves away from its ground](figures/colour_variation.png)

A blob whose colour moves from the ground's green to red: the response rises
monotonically, 0.00 → 0.32 → 0.81 → 0.98 → 1.00. It saturates after a third of
the way — the sigmoid of eq. 5.12 with β = 3 on a contrast scale of 32 MTM units
is steep — which the thesis's own argument ("high values are rare overall")
intends.

### 8 · Abb. 5.22 — cc_add and cc_mult — replicated

Mean response in the two regions the thesis names, on the running example
(default: cc_add 8, cc_mult 5):

| cc_add | 2 | 4 | **8** | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| ball | 0.57 | 0.70 | **0.71** | 0.72 | 0.71 | 0.70 | 0.69 |
| picture | 0.39 | 0.70 | **0.93** | 0.92 | 0.89 | 0.63 | 0.04 |
| map correlation with default | 0.68 | 0.90 | 1 | 0.93 | 0.91 | 0.86 | 0.50 |

| cc_mult | 0 | 2 | **5** | 8 | 12 | 20 |
|---|---|---|---|---|---|---|
| ball | 0.12 | 0.44 | **0.71** | 0.71 | 0.69 | 0.69 |
| picture | 0.20 | 0.91 | **0.93** | 0.89 | 0.86 | 0.82 |
| map correlation with default | 0.34 | 0.88 | 1 | 0.92 | 0.89 | 0.87 |

"Ball und Bild" are the salient regions here as in the thesis, and they stay so
for cc_add 4–16 and cc_mult 5–20; only clearly different values (cc_add 32: the
picture merges into the wall; cc_mult 0: the variance term is gone and textured
regions shatter) change the picture — "erst bei deutlich veränderter Wahl der
Parameter", as the thesis says. The variance term earns its place: without it
the ball's response drops from 0.71 to 0.12.

### 9 · Abb. 5.34 — exclusivity — replicated

| Scene | odd / common, plain | odd / common, c = 1.1 | thesis factor for 5 vs 1 |
|---|---|---|---|
| five vertical bars, one horizontal | 1.03 | **1.51** | 1.1⁴ = 1.46 |
| five muted green blobs, one red | 1.03 | **2.13** | 1.46 before the sigmoid |

The odd one out gains by the factor the formula predicts for orientation; for
colour the sigmoid that follows (eq. 5.12) stretches the ratio further.

### 11–15 · Abb. 5.28–5.32 — the depth feature (added 2026-09-21)

Rendered stereo pairs with known disparity: a random-dot ground at disparity 0
and a random-dot surface shifted by *d* pixels in the right image (a nearer
surface), 256 × 256, search range 16 px, three orientations — the parameters of
`configs/thesis/stereo.yaml`. The depth response is |d| / search range (eq.
5.16), so it reads back as a disparity estimate. Random dots carry no monocular
cue: every pair is also a random-dot stereogram (Abb. 5.29).

![depth response against the surface's disparity](figures/stereo_distance.png)

**Distance (5.28) and the random-dot stereogram (5.29).** For d = 0, 2, …, 14
the estimated disparity on the surface is 0.0, 2.0, …, 14.0 — exact — with 100%
of the surface pixels within one pixel of the truth, and the ground at 0. The
stereogram's square (its mean grey differs from the ground's by 3 levels out of
255; there is nothing to see in either image alone) is recovered completely.

![correct surface pixels under noise, for one, three and five orientations](figures/stereo_noise_orientations.png)

**Noise (5.30).** Independent normally distributed noise in the two images, five
seeds per level. The estimated disparity of a surface at 10 px is 10.0 up to
σ ≈ 38 grey levels, 9.99 at σ ≈ 64 (98% of pixels within 1 px) and 9.91 at
σ ≈ 90 (92%) — "the estimate changes only in a very small range", as the thesis
says. The thesis attributes this to its multi-scale scheme, which restricts the
search at high resolution by the result at low resolution; that scheme is *not*
ported (documented in `stereo_feature.h`), and the single-scale correlation with
the spatial integration of eq. 5.14 is already this robust on these stimuli. So
the finding replicates; the thesis's explanation for it is not tested here.

**Orientations (5.31).** At σ = 25 every orientation set is perfect, so the
comparison is made under strong noise. Share of surface pixels within 1 px /
share of ground pixels with a wrong disparity:

| noise σ | vertical only | + ±30° | + ±15°, ±30° |
|---|---|---|---|
| 90 | 0.84 / 0.16 | **0.92 / 0.08** | 0.91 / 0.09 |
| 115 | 0.71 / 0.31 | **0.82 / 0.20** | 0.80 / 0.22 |

"The method profits from several orientations, but apart from the case of a
single orientation the results differ only slightly" — exactly that.

![the variance threshold: what it admits and what it drops](figures/stereo_variance.png)

**Variance threshold (5.32).** The scene gets a textureless band and a second
surface whose texture has a tenth of the contrast; both images carry a little
sensor noise (σ = 3). Thresholds up to 5 let the textureless band through — 89%
of its pixels receive a wrong disparity, the thesis's "erroneously classified
pixels … on the surface of the ball"; from about 20 the band is rejected (1%);
from 40 upwards correct pixels start to go, the faint surface first (88% at 40,
49% at 80, none at 160), while the high-contrast surface survives everything up
to 240. Both halves of the thesis's statement hold, and there is a broad window
between them.

*Calibration — checked on real stereo pairs (2026-09-21), and the first reading
corrected.* On this synthetic scene the port's default threshold (3.0) lies below
the window: it lets a band of pure sensor noise through. That suggested raising
it (the dissertation system used 75, on its own Gabor scale). Real imagery says
otherwise. Three Middlebury 2001 pairs with ground truth — Tsukuba, Venus,
Sawtooth, the ones whose disparities fit the 16-px search range at the feature's
256-px working size (`eval/replication.py --only stereo-real`) — share of pixels
with a correct disparity (within 1 px) / a wrong one / none:

![correct and wrong disparities against the variance threshold, three real pairs](figures/stereo_real_threshold.png)

| Threshold | Tsukuba | Venus | Sawtooth |
|---|---|---|---|
| 0 | 0.86 / 0.14 / 0.00 | 0.84 / 0.10 / 0.06 | 0.83 / 0.12 / 0.06 |
| **3 (default)** | **0.85 / 0.15 / 0.00** | **0.82 / 0.12 / 0.06** | **0.82 / 0.12 / 0.06** |
| 10 | 0.82 / 0.16 / 0.02 | 0.74 / 0.14 / 0.12 | 0.79 / 0.15 / 0.06 |
| 20 | 0.77 / 0.18 / 0.05 | 0.67 / 0.13 / 0.20 | 0.74 / 0.17 / 0.10 |
| 40 | 0.70 / 0.20 / 0.10 | 0.59 / 0.14 / 0.27 | 0.64 / 0.18 / 0.19 |
| 80 | 0.55 / 0.21 / 0.25 | 0.48 / 0.14 / 0.39 | 0.50 / 0.20 / 0.31 |

Raising the threshold never buys fewer wrong pixels — their share stays at
12–20% — it only discards correct ones. The reason is visible when the pixels
are split by local texture: on the 40–50% of each image with little grey-value
variation, **81–86% of the disparities are correct at threshold 0**; the spatial
integration of eq. 5.14 carries them from their textured surroundings. The
errors sit at occlusions and depth edges, which no variance threshold addresses.
A band of pure noise, as in the synthetic scene, is the one case the threshold
exists for, and real scenes here do not contain it. **The default stays at 3.0**;
the thesis's statement about Abb. 5.32 holds as the synthetic experiment shows
it, and the practical advice is the opposite of what that experiment alone
suggested: keep the threshold low.

This is also an independent check of the thesis's summary of its depth feature —
"in the great majority of regions the disparity has been determined correctly"
(on Abb. 5.25): 82–85% of all pixels with ground truth, on real pairs the
thesis never saw, with the single-scale port.

### Text or code? — the two thesis profiles compared (2026-09-21)

Decision (Gerriet, 2026-09-21): the thesis **text's** values stay the documented
replication profile (`configs/thesis/thesis.yaml`, `attend.yaml`), because the
text is what a reader can check; the dissertation **system's** values are kept
runnable next to it (`thesis_esab2.yaml`, `attend_esab2.yaml`). Both use the
system's field parameters. They differ in the eccentricity thresholds (0.65 / 2
vs 0.78 / 1.5 and a saliency offset of 0.2), the colour thresholds (8 / 5 vs
12 / 6, sigmoid slope 3 vs 4), exclusivity (1.1 vs off) and the feature weights
(1 / 1 / 1 vs symmetry 1.2, eccentricity 0.35, colour 1.6).

Which fares better? Every benchmark the repository can run without a model,
each paired on identical items; "code − text", 95% CI:

| Benchmark | text | code | code − text |
|---|---|---|---|
| **Thesis's own feature findings** (`eval/replication.py --only text-vs-code`) | all reproduced; eccentricity on eq. 5.6 (2:1 ellipse 0.35) | all reproduced qualitatively; the saliency offset takes eccentricity off eq. 5.6 (0.19); colour loses the blob at the strongest noise level | text is the closer fit |
| V\*Bench target coverage, top 3 (191 items) | 0.152 | 0.105 | −0.047 [−0.099, +0.010] |
| … `fovea` accuracy at K = 3, legibility oracle | 0.147 | 0.099 | |
| COCO-Search18 bottom-up, found@10 (150 trials) | 0.37 | 0.34 | −0.027 [−0.100, +0.040] |
| … with the category prior | 0.42 | 0.43 | +0.013 [−0.047, +0.073] |
| H1, object-ior+id latency, standard / fast / occlusion (30 scenes each, seeds 3000–3029) | 1.73 / 1.79 / 1.92 | 1.62 / 1.77 / 1.93 | −0.12 / −0.03 / +0.01, all intervals include 0 |
| H1, object-ior+id staleness | 1.83 / 1.63 / 1.58 | 1.81 / 1.63 / 1.57 | −0.01 / +0.01 / −0.00 |
| MIT1003, the model's own scanpath, ScanMatch (200 images) | 0.545 | 0.529 | **−0.016 [−0.023, −0.008]** |
| … generic readout of the map | 0.674 | 0.668 | **−0.005 [−0.009, −0.001]** |
| … object-file readout, position agreement | 0.791 | 0.799 | **+0.008 [+0.001, +0.015]** |

**Answer: neither is decisively better; where they differ, the text's values
are ahead — slightly.** The text profile is closer to the thesis's own
equations, a little better at putting early fixations on small targets and at
agreeing with human scanpaths, and tied on search and on the dynamic-scene
study. The only place the system's values lead is the object-file readout's
position agreement on stills, by 0.008. On the running example the two colour
maps correlate at 0.92 and the eccentricity maps at 0.59 — eccentricity is where
the variants really differ, and it is also the feature with the smallest weight
in the system's profile (0.35), which is probably why so little of the
difference reaches the benchmarks.

Two things this comparison settles beyond the question asked:

- **H1 holds under both profiles, on a third fresh block of scenes.**
  Object-based IOR beats space-based IOR on latency and staleness in every
  regime under either parameter set (latency −1.2 / −4.8 / −1.4 frames with the
  text's, −1.2 / −4.4 / −1.5 with the system's). The verdict does not depend on
  which set of values is "the thesis".
- **The feature weights the dissertation system used do not matter much** at
  this resolution of measurement: down-weighting eccentricity to 0.35 and
  up-weighting colour to 1.6 changes no benchmark by more than its interval.

### 10 · Abb. 6.14 — tracking several objects through occlusion — partially

The thesis compares variants of its neural fields. The reimplementation tracks
identity in the *object files* of the second stage, not in the field, so the
figure is not reproduced as such; its question — does identity survive a
temporary occlusion — is measured by the H1 study's occlusion regime (speed 20,
10-frame occlusion, 30 scenes): **3.6 distinct labels per attended object with
the thesis's nearest-centroid correspondence, 1.7 with persistent identity**
(`docs/DYNAMIC_IOR_STUDY.md`). With the thesis's correspondence identity is
*not* reliably kept; the opt-in extensions are what make it hold.

## Not attempted, and why

| Thesis | Finding | Why not yet |
|---|---|---|
| Abb. 5.33, 5.35 | superposition; 2D vs 3D integration | qualitative figures on lab stereo imagery |
| Abb. 6.11, 6.12 | systems of several 2D fields with global inhibition and per-feature weights | not implemented: the reimplementation has the single 2D field and the 3D field only |
| Abb. 6.13 | the 3D field with local inhibition on a stereo scene | the 3D field is ported and tested; a dynamics experiment like the 2D ones needs the harness extended to the depth volume |
| §9.3.2, §9.3.3 | flanker compatibility; early vs late selection | qualitative demonstrations |

## What the dossier says about the reimplementation

- The three static features, all ported from the original sources in 2026-09 —
  **eccentricity, colour contrast and symmetry — reproduce every thesis finding
  tested**, quantitatively where the thesis gives an equation. Exclusivity does
  too, and so does the **depth feature** (a port from the start): disparity is
  recovered exactly, robustly under noise, and its two parameters behave as the
  thesis describes.
- The **neural field reproduces the thesis's dynamics — with the parameters the
  dissertation system actually used.** The port had taken the defaults of the
  original's setter functions instead; with those the clusters never merge, noise
  leaks in early and moving targets are lost (finding B).
- The dossier earned its keep twice: it found that symmetry answered beside
  objects instead of on them (finding A), and that the field ran on the wrong
  parameters (finding B). Both looked like properties of the 2004 model until
  they were tested against the thesis's own figures.
- Identity through occlusion holds only with the extensions, not with the
  thesis's correspondence — consistent with H1.

Replication-track definition of done (ADR-0005): ~~fix symmetry~~ → ~~re-run H1
and M11 with it~~ → ~~the stereo experiments~~ → ~~the field dynamics~~ (Abb.
6.4–6.10 done; 6.11–6.13 documented as not implemented / not attempted) →
~~decide on the open parameter questions~~ (decided 2026-09-21: the text's values
are the profile, the system's are a runnable sibling; compared above) →
~~the stereo variance threshold~~ (checked on real pairs 2026-09-21: the default
is right) → tag `replication-v1`.
