# How good are the feature implementations? (assessment, 2026-09-19)

*Two questions, one per paper. For the **replication** paper: do the features
implement what the thesis describes? For the **front-end** paper: are they —
or the six M9 alternatives — a useful source of crops for a vision-language
model, and what should be added? Based on a read-through of `src/features/`,
`src/fusion/`, `src/selection/neural_field*` against thesis ch. 5 (local copy),
with small synthetic probes. Items marked **[verified]** were re-checked by
hand after the audit.*

## Status (2026-09-20): what has been fixed since this assessment

Acted on, on `module/thesis-features` (ported from the surviving original
sources `feature/color.C` and `feature/eccentricity.C` plus thesis ch. 5):

| Finding | Now |
|---|---|
| Colour is Itti–Koch, not the thesis's | **`color-munsell` added** — MTM transform, centroid-linkage region growing with the variance-adaptive threshold, border-weighted contrast, sigmoid (eq. 5.7–5.12). `color` (Itti–Koch) is kept unchanged; `configs/thesis.yaml` selects `color-munsell`. |
| Eccentricity: other segmentation, other formula, and the all-black-image bug | **Replaced by a port**: Sobel-histogram threshold, region growing, merging + dilation, Jähne's ε, 12 + 1 orientation classes. A 2:1 ellipse scores 0.36, a disk 0, a bar > 0.85 (thesis Abb. 5.10). |
| No exclusivity weighting (§5.5.3) | **Added, inside the features** as the thesis describes (it is not a fusion step): divide by c^n for the n segments sharing an orientation / hue class; per disparity level for stereo. Thesis form c^n (1.1) by default, the source's n^p selectable. Off unless configured; on in `thesis.yaml`. |
| Goldens lock defects in; no behavioural tests | **`tests/test_thesis_features.cpp`**: 13 cases checking each feature against the property it is named after (shape ordering, segmentation follows the image, flat map when nothing is there, colour contrast grows with colour distance, odd-one-out pops out under exclusivity). Goldens regenerated after review — only `feature_eccentricity` and the fused saliency changed. |
| Bottom-up = random, measured with a scratch script | **Permanent**: `eval/vlm_frontend.py` now always runs a `fovea-random` arm and reports the chance level of target coverage next to the pipeline's. |

Both ported features now emit an **absolute** saliency in [0, 1] instead of a
min-max-stretched map — the thesis's integration presupposes comparable ranges,
and an image with nothing elongated / no colour contrast must give a flat map.

Where thesis text and surviving source disagree, the source fixes the *units*
and the thesis the *parameters*; each choice is documented in the header:

| Item | Thesis text | Original source | Used |
|---|---|---|---|
| MTM lightness | L = 0.23·V(Y) | L = V(Y), 0–100 tristimulus scale | source (it fixes what "threshold 8" means) |
| MTM S2 | cos φ | sin φ | source (and the MTM paper) |
| Colour threshold | cc_add + cc_mult·σ² | + cc_mult·σ (3×3 window) | source |
| Colour sigmoid β | 3 | 4 | thesis |
| Eccentricity growth share | 0.65 | 0.75 | thesis |
| Eccentricity variance ratio k | 2 | 1.12 | thesis |
| Eccentricity saliency offset | none | 0.2 | thesis |
| Exclusivity | ÷ c^n, c = 1.1 | ÷ n^p, p = 0 (off) | thesis; source form selectable |
| Stereo exclusivity | ÷ c^(np_i/np), *np* undefined | not in the surviving file | reconstructed: np = matched pixels / disparity levels |

One deliberate deviation: eccentricity takes the variance ratio on
(variance + 1), so two perfectly flat segments (synthetic stimuli; 0/0 in the
original) count as alike.

**Checked:** the H7 mock study (10 scenes) is unchanged by the new eccentricity
(object-ior 0.92 / space-ior 0.62, before 0.90 / 0.63) — the stage-2 findings
did not depend on the defective feature.

**Still open from this assessment:**
- The Itti–Koch `color` keeps its defects (polarity lost by `abs()` before
  center–surround; per-level min-max). It was asked to be *kept*, not repaired;
  it is what `default.yaml` and the `attend*.yaml` study configs still run.
- `configs/attend*.yaml` (H1, H7) still use `color`, not `color-munsell` — see
  ADR-0005, open point 1.
- ~~Symmetry~~ — **ported 2026-09-21** (absolute scale, clip offset, thesis
  Tab. 5.1 radii; a lone disk peaks at its centre, a single edge gives 0.08; four
  behavioural tests). What was wrong:
  (`docs/replication/REPLICATION_DOSSIER.md`, finding A): for a lone disk its
  maximum lies ~80 px *beside* the disk. The summation core is a faithful port;
  the combination step is not — per-band normalization and relative thresholds
  over three scales replaced the original's absolute scale and clip offset (60
  of 255), which is what removed one-sided responses. Also: single-phase Gabor
  bank, hot-path `std::cerr`, blind borders.
- Fusion still min-max-stretches maps that are not absolute (the Itti-style
  ones); no N(·).
- `--batch` ignores `--config` and always runs the default feature set.
- Everything in Part 3 (tiling, better crop sources).

## Summary

- **The second stage and the dynamic features are the faithful part.** Stereo
  is an equation-level port (§5.4), the neural field a credible port with
  documented deviations, onset an honestly documented reconstruction; all
  three have behavioural tests. Object files/behaviors were reconstructed from
  ch. 7–8 and say so.
- **The three static thesis features are the weak part.** Colour is a
  different algorithm (undisclosed), eccentricity has a different
  segmentation and formula *and a bug that makes its segments meaningless*,
  symmetry has the right core with untested additions. None has a behavioural
  test; the characterization goldens currently lock the defects in.
- **Fusion lacks the thesis's exclusivity weighting** (§5.5.3) and any
  peak-promoting normalization, so broad, empty maps swamp sparse peaks.
- **Consequence for the studies so far:** H1 and H7 ran on synthetic disks,
  where stage 1 mostly has to find blobs — their conclusions are about stage 2
  and are little affected. H6's "bottom-up crops rarely land" and H2's
  "allocation-limited" cases were measured with this stage 1, so they say less
  about *bottom-up saliency* than the docs imply.
- **As a crop selector for small targets, no bottom-up source in the repo can
  work as configured** — thesis or M9 — because every stage downsamples a
  36-px target to 1–4 px. Resolution handling comes before any new feature.

## Part 1 — thesis features, for the replication paper

| Feature | Thesis (ch. 5) | Code | Verdict |
|---|---|---|---|
| **Colour** §5.3.2 | RGB→XYZ→Adams→**MTM/Munsell** space; centroid-linkage region growing with variance-adaptive threshold; per-segment saliency = border-length-weighted colour distance to neighbours; sigmoid β = 3; 12 hue classes feeding exclusivity | **Itti–Koch** opponent channels (RG, BY) with center–surround pyramids — the header says so; "Munsell"/"MTM" appear nowhere in code or docs **[verified]** | **Different algorithm, undisclosed.** `configs/thesis.yaml` lists it as an original feature. |
| **Eccentricity** §5.2.3 | Region growing (Sobel threshold at 65th percentile), iterative merging (Δμ ≤ 20, variance ratio 2, ≤ 4 iterations); Jähne's ε = ((m20−m02)² + 4m11²)/(m20+m02)²; 12 orientation-class maps + 1 | Sobel → fixed threshold → distance-transform markers → `cv::watershed` → sqrt(1 − λmin/λmax) | **Different segmentation and formula** (a 2:1 ellipse scores 0.87 vs the thesis's 0.36) **plus a bug** (below). |
| **Symmetry** §5.2.2 | Tangential Gabor edge energy in boxes at ±r, max over radii/scales; radii 6, 9, 12, 15 (Tab. 5.1) | Same core, cites `symmetry_intern`; **added**: per-scale global-max-relative thresholds, a "≥ 2 radii above threshold" consistency weight, radii 5–30 step 2; Gabor bank is a single odd-phase kernel + `abs()`, not a quadrature pair | **Core faithful, additions untested.** The consistency weight penalises perfect circles; ~90% of runtime. |
| **Stereo** §5.4 | eq. 5.13–5.16 | Port with equation-level provenance; multiscale variant not ported (documented) | **Faithful.** |
| **Onset** | original lost (empty stub) | Rectified positive edge-energy difference | **Honest reconstruction.** |
| **Fusion** §5.5 | Weighted superposition **with exclusivity weighting** (c_exkl = 1.1) | Weighted sum of min-max-normalised maps, min-max again; no exclusivity, no N(·) | **Incomplete.** A map with nothing in it is still stretched to max 1. |
| **Neural field** ch. 6 | Amari field, Backer kernel | Port; documented deviations | **Credible.** Minor: threshold comment says 0.01·N, `thesis.yaml` says 0.02. |

### Defects (not just deviations)

1. **Eccentricity segments an almost black image [verified].**
   `eccentricity_feature.cpp:156`: `gray.convertTo(gray_8u, CV_8U)` on the
   [0, 1] float pyramid level (`frame.cpp:40` scales by 1/255) without ×255.
   The watershed floods an image of 0s and 1s, so segments are
   marker-Voronoi cells, not image regions. Probe: two bars on grey — large
   polygonal background wedges light up, the bars do not.
2. **Colour loses opponent polarity and normalises inconsistently.** `abs()`
   before center–surround; every pyramid level min-max-normalised
   independently (center and surround in different units); every
   center–surround map and the output stretched again, so numerical residue
   always reaches 1.0. Probe: red disk on green → bright uniform background,
   dark holes not at the disk. This is the "colour channel is a smooth
   gradient" of `docs/VLM_VIDEO.md` — an artefact, not the algorithm.
3. **Symmetry**: unconditional `std::cerr` in the hot path; a blind border of
   radius + box at every scale (on a 256-px image the coarsest scale
   contributes nothing); assumes square maps in one allocation.
4. Dead config keys: `normalize_channels` (colour), `variance_threshold`
   (eccentricity). `docs/ARCHITECTURE.md` lists a "normalization" *fusion*
   strategy that does not exist (it is a selection strategy).

### Which M19 observations were inherent, which were artefacts

| Observation (`docs/VLM_VIDEO.md`) | Cause |
|---|---|
| "symmetry fires in the empty space between the disks" | **Inherent** to the additive formulation (never requires edges on *both* sides or polarity agreement); aggravated by the enlarged radii |
| "onset draws a hollow ring" | **Inherent** to edge-energy differencing |
| "eccentricity is blank" | **Inherent** for disks (ε = 0, as the thesis says) — but on other shapes the bug above applies |
| "the colour channel is a smooth gradient" | **Artefact** (defect 2) |

### What the replication paper needs

The project's bar is *loose behavioral equivalence*, and the default path is
protected by goldens — both fine, but a reader comparing with the thesis will
object to "faithful" as things stand. Options, in order of preference:

1. **Port the thesis colour feature** (MTM transform → region growing →
   border-weighted contrast → sigmoid) as `color-thesis`, and **the thesis
   eccentricity** (region growing + merging, Jähne's ε, orientation classes) —
   the thesis text specifies both down to the constants. Add **exclusivity
   weighting** to fusion. Keep the current Itti-style colour as `color-itti`.
   Then `thesis.yaml` means what it says. This is also what M10's
   feature-variation curves (Abb. 5.13, 5.20) need in order to be replicable
   at all — they are curves *of these features*.
2. At minimum: fix the eccentricity bug (goldens regenerate — an intentional
   change), and **disclose** the substitutions in `ARCHITECTURE.md`, the
   README and the dossier ("stage 1 colour is Itti–Koch, not the thesis's
   Munsell segmentation contrast").
3. Either way: **behavioural tests** for colour, symmetry, eccentricity (red
   blob on green peaks at the blob; a symmetric shape beats an asymmetric one;
   an elongated bar beats a disk, with the thesis's ε values), so goldens stop
   being the only guard.

Since the first-stage colour feature segments the image anyway, the thesis
version has a side benefit for stage 2: its segments are close to the
proto-objects that M19 had to add.

## Part 2 — the M9 alternatives

`configs/modern.yaml` does **not** select them (it is the thesis features with
default parameters); only `configs/alternative.yaml` does. The name
misleads.

| Operator | Fidelity to the paper |
|---|---|
| spectral-residual (Hou & Zhang 2007) | Faithful (64 px, log-spectrum average, squaring, blur) |
| frequency-tuned (Achanta 2009) | Faithful; minor: mean taken from the blurred image |
| phase-spectrum (Guo 2008) | **PFT, not PQFT** (documented): per-channel whitening gives near-empty channels full weight — amplifies sensor noise on static video |
| boolean-map (Zhang & Sclaroff 2013) | **Most simplified**: no Lab whitening, threshold step 20 (paper: 8) so colour is split by 2–4 thresholds only; no opening/dilation; 256 px (paper ~400) |
| image-signature (Hou 2012) | Faithful |
| minimum-barrier (Zhang 2015) | Core FastMBD correct; no backgroundness cue, no post-processing (documented) |

All six are 2007–2015 hand-crafted operators — "post-thesis classical", not
modern by 2026 standards. Each has one synthetic pop-out test (a 40-px square
on 200 px); none is tested on natural images or small targets. They work at
64–256 px: a 36-px target in a 2250-px image becomes 1–4 px. The DeepGaze
adapter uses a flat centre bias, no rescaling to its training resolution, and
CPU only (no `mps`).

## Part 3 — better sources for the front-end paper

### First: is the current bottom-up map better than chance?

**No.** Random fixations cover the V\*Bench targets exactly as often as the
pipeline's (top-3: 0.069 vs 0.068; top-10: 0.219 vs 0.220, all 191 items) —
see "Measured" at the end of this document.

### Resolution comes before features

Colour accumulates at pyramid level 4 (1/16), symmetry starts below 256 px,
eccentricity runs at quarter resolution, the field at 128 px with a 9-px blind
margin; the harness processes at ≤ 1024 px and peaks are ≥ 30 px apart, ≤ 10.
No new feature helps until saliency is computed **in overlapping
native-resolution tiles** (rank-merged) — a harness-level change
(`emit_fixations`) or a `--tiles` mode.

### Shortlist (ranked)

| # | Addition | Effort | Where it plugs in | Expected effect |
|---|---|---|---|---|
| 1 | **Native-resolution tiled saliency** | S–M | `eval/vlm_frontend.py: emit_fixations`, later a C++ `--tiles` mode | Prerequisite for everything below; makes small targets representable at all |
| 2 | **Open-vocabulary detector as the top-down source** (OWLv2; YOLO-World has ONNX exports) — noun phrase from the question → boxes → the existing `relevance_map` → `top_down_map` | M | Python-side, the M17 slot; replaces the VLM grounding call of `fovea-td` (which costs 0.17 of the budget) | Highest. Question-conditioned, runs on tiles, cost not charged in VLM tokens. The attention pipeline stays the controller (saliency + IOR order the candidates). |
| 3 | **Scene-text feature** (DB text detector, `cv::dnn::TextDetectionModel_DB`, opencv_zoo ONNX) + **face feature** (YuNet, `cv::FaceDetectorYN`) | S–M | C++ features at default weight 0 — same pattern as `dnn-classify` | High on the OCR/text share of V\*Bench and HR-Bench; small text is exactly what downsampling destroys. YuNet is also the standing M11 face-channel reminder. |
| 4 | **Self-supervised objectness**: DINOv2/v3 patch-feature rarity on tiles, optionally intersected with FastSAM/MobileSAM small masks as proto-objects | M | Python-side model behind the interchange format or `top_down_map` | Medium; query-blind but finds "things", and gives texture-capable proto-objects — the named blocker for identity on DAVIS |
| 5 | **Motion-contrast feature** (DIS/Farnebäck flow, global motion compensated) replacing frame differencing | S | C++ feature for the M19 video runs | Medium for video; removes the hollow-ring onset problem; also the ingredient for motion-compensated spatial IOR (H1's strengthened baseline) |

Also worth one table row as honest baselines, not as hopes: **DeepGaze IIE**
(learned free-viewing saliency predicts gist/faces/centre — small targets are
non-salient by construction, so expect "learned bottom-up doesn't fix it
either"), and **ViCrop/FOCUS-style crops from the VLM's own attention** — the
published training-free competitor; needs model internals (HF/MLX, not
Ollama).

Fusion, for both papers: add the thesis's **exclusivity weighting** and/or
Itti's N(·) as opt-in fusion strategies, so a sparse strong peak outranks a
broad pedestal.

### Do this first (model-free, ~an afternoon)

Coverage@K (K = 1, 3, 5, 10) of the annotated V\*Bench targets, all 191 items,
bootstrap intervals, stratified by target size, for: random fixations ·
current pipeline at 1024 / 2048 / native · tiled 2×2 and 3×3 · 
`alternative.yaml` tiled · (then) each shortlist source. **If tiled bottom-up
does not beat random, stop investing in bottom-up sources for stills** and go
straight to #2 and #3; the paper then states the negative with a number.

## Measured: bottom-up vs random fixations on V\*Bench

All 191 V\*Bench items, default thesis pipeline (`--proc-max-side 1024`), 336-px
fovea window, the harness's own hit criterion (`target_fixation_rank`, 0.6
overlap). Random = 10 uniformly placed fixations per image, 200 draws.

| Targets covered by the top K | K = 1 | K = 3 | K = 5 | K = 10 |
|---|---|---|---|---|
| pipeline, all (191) | 0.021 | 0.068 | 0.126 | 0.220 |
| **random, all** | 0.023 | 0.069 | 0.115 | 0.219 |
| pipeline, direct attributes (115) | 0.026 | 0.087 | 0.148 | 0.261 |
| random, direct attributes | 0.030 | 0.088 | 0.143 | 0.264 |
| pipeline, relative position (76) | 0.013 | 0.039 | 0.092 | 0.158 |
| random, relative position | 0.013 | 0.041 | 0.071 | 0.151 |

**The bottom-up fixations are indistinguishable from random placement.** The
2 / 7 / 22% reported in `docs/VLM_FRONT_END.md` is the base rate of a 336-px
window landing on a small target, not a (weak) saliency signal: on this
benchmark the current stage 1 carries no information about where the
question's target is.

### After the feature port (2026-09-20): the thesis profile beats chance, the default does not

Same 191 items, same criterion, now from the permanent harness
(`eval/vlm_frontend.py --vstar --backend mock --limit 0 --config …`; the mock
makes it model-free). Difference to the per-item chance level, paired
bootstrap over items:

| Profile | K = 1 | K = 3 | K = 5 | K = 10 | top-10 minus chance |
|---|---|---|---|---|---|
| random fixations (chance) | 0.02 | 0.07 | 0.11 | 0.22 | — |
| `default.yaml` (Itti colour, intensity, orientation + ported eccentricity, symmetry; NMS) | 0.04 | 0.09 | 0.13 | 0.23 | +0.010 [−0.048, +0.070] |
| `thesis.yaml` (**colour contrast in MTM**, ported eccentricity, exclusivity, symmetry; neural field) | 0.05 | **0.13** | **0.24** | **0.39** | **+0.172 [+0.106, +0.241]** |

(thesis profile, top-3: +0.061 [+0.015, +0.111]; top-5: +0.122 [+0.066, +0.185].)

- **The dissertation's own stage 1 carries real information about where
  question targets are** — about twice the chance coverage at K = 5–10 — and the
  Itti-style default carries none. V\*Bench targets are small, distinctly
  coloured things (a cup, a sign, a piece of clothing); a segment-level colour
  contrast with odd-one-out weighting is a reasonable detector for exactly
  that, where a multi-scale center–surround average washes them out.
- It is still far from enough: 61% of targets are never covered by ten
  fixations, and in the legibility-oracle accuracy the `fovea` arm (0.12)
  remains below the same-budget uniform downsample (0.18). Bottom-up attention
  alone does not rescue H6 on stills; it stops being *useless*.
- Which difference between the two profiles does it — features, selection or
  exclusivity — is answered by the ablation below: mostly the *selection*.
- Everything in Part 3 still applies — above all tiling: this was measured at
  ≤ 1024 px processing size with a 256-px colour segmentation.

### Ablation (2026-09-20): what carries the effect — and the choice of the modern base

One change at a time against the thesis profile (`configs/ablation/`), all 191
V\*Bench items, model-free; `eval/coverage_table.py` gives coverage by the top
K fixation windows, the chance level, and paired bootstrap differences over
items (top 10):

| Profile | top-3 | top-5 | top-10 | minus chance | minus thesis |
|---|---|---|---|---|---|
| chance (random fixations) | 0.07 | 0.11 | 0.22 | | |
| colour contrast (MTM) alone, neural field | 0.13 | 0.24 | 0.42 | +0.20 [+0.14, +0.27] | +0.03 [−0.05, +0.11] |
| thesis + Itti intensity & orientation | 0.13 | 0.24 | 0.41 | +0.19 [+0.13, +0.26] | +0.02 [−0.05, +0.09] |
| thesis − symmetry | 0.14 | 0.20 | 0.40 | +0.19 [+0.12, +0.26] | +0.02 [−0.06, +0.09] |
| thesis − exclusivity | 0.12 | 0.21 | 0.39 | +0.18 [+0.11, +0.24] | +0.01 [−0.07, +0.08] |
| **thesis** | 0.13 | 0.24 | 0.39 | +0.17 [+0.11, +0.24] | — |
| thesis − eccentricity | 0.10 | 0.20 | 0.37 | +0.15 [+0.08, +0.22] | −0.02 [−0.08, +0.04] |
| thesis, **Itti colour for MTM colour** | 0.11 | 0.16 | 0.29 | +0.08 [+0.02, +0.14] | **−0.09 [−0.16, −0.03]** |
| colour contrast alone, NMS (plateau-aware) | 0.13 | 0.23 | 0.33 | +0.11 [+0.05, +0.18] | −0.06 [−0.15, +0.02] |
| thesis, **NMS for the neural field** | 0.13 | 0.15 | 0.22 | −0.00 [−0.06, +0.06] | **−0.17 [−0.25, −0.10]** |
| thesis-extended (the old default) | 0.09 | 0.13 | 0.23 | +0.01 [−0.05, +0.07] | −0.16 [−0.25, −0.08] |
| old default with MTM colour, NMS | 0.09 | 0.16 | 0.22 | −0.00 [−0.06, +0.06] | −0.17 [−0.25, −0.09] |

Reading:

1. **The selection stage is decisive, and that was not expected.** The same
   three feature maps are at chance read out by NMS and at twice chance read
   out by the neural field. Part of it was a defect: on a piecewise-constant
   map every pixel of a salient segment is a "local maximum", so NMS tiled the
   largest segment with peaks; NMS is now plateau-aware (one interior peak per
   plateau), which lifts colour-contrast-alone from chance to +0.11. With
   several features fused it stays at chance even so — pixel-wise maxima follow
   the smooth maps (symmetry ridges, Itti channels), while the field integrates
   over area and reads out cluster centres. *The thesis's argument for a
   field-based, region-level selection stage shows up as a measurable
   difference on a 2024 benchmark.*
2. **Colour contrast in MTM space carries the information.** Replacing it with
   the Itti–Koch colour feature is the only feature change with a measurable
   cost. Alone, with the field, it is as good as the full profile.
3. **Exclusivity, symmetry, eccentricity and the Itti channels: no measurable
   effect here** (all intervals include zero).

**Validation on a second benchmark** — COCO-Search18 target-present search,
bottom-up arm, 150 validation trials, paired over trials
(`eval/coco_search.py --config <profile>`):

| Profile | mean fixations-to-target | found@10 | with category prior |
|---|---|---|---|
| human | 2.58 | 0.92 | |
| thesis-extended (old default) | 9.07 | 0.29 | 7.86 / 0.47 |
| two-feature profile (MTM colour + eccentricity, field) | 8.34 | 0.38 | 6.66 / 0.56 |
| **thesis** | **7.97** | **0.47** | **6.35 / 0.59** |

thesis − old default: found@10 **+0.18 [+0.09, +0.27]**; thesis − two-feature:
found@10 +0.09 [+0.01, +0.17]. The ordering found on V\*Bench holds on a
different task and dataset; the simplification that V\*Bench could not tell
apart (dropping symmetry and exclusivity) costs targets here. (This also
revises M17's bottom-up baseline: 9.03 / 0.27 was measured with the old
default.)

**Decision (ADR-0005, "start the modern default from the best base that can be
measured"):** `configs/modern.yaml` starts as the dissertation's feature set
with the neural field — the best base on both benchmarks — and is free to leave
it. The modern second-stage profiles (`configs/attend*.yaml`) moved to the same
stage 1; on the H7 dev scenes that lifts object-file crops from 0.92 to 0.98
and identity to exactly one label per object (mock, 10 scenes, to be confirmed
on fresh seeds).

Not done: the compiled-in base feature set (what runs with no `--config`, and
what every YAML is layered on) is still the old five features with NMS.
Changing it touches every profile that disables features explicitly; it needs
the configs to stop inheriting a hidden base first.

### Tiling (2026-09-20): a modest gain, not yet a significant one

`eval/vlm_frontend.py --tiles 2` — the whole image plus a 2 × 2 grid of
overlapping native-resolution tiles, fixations merged by rank — on the modern
profile, same 191 items:

| | top-1 | top-3 | top-5 | top-10 | minus chance | minus untiled |
|---|---|---|---|---|---|---|
| untiled | 0.05 | 0.13 | 0.24 | 0.39 | +0.17 [+0.11, +0.24] | — |
| 2 × 2 tiles | 0.05 | **0.18** | 0.26 | **0.45** | +0.23 [+0.16, +0.30] | +0.06 [−0.02, +0.14] |

The direction is right (top-3, the crop budget actually used, goes 0.13 → 0.18;
the legibility-oracle accuracy of the `fovea` arm 0.12 → 0.15, against 0.10 for
random crops and 0.18 for the same-budget uniform downsample), but the paired
interval still includes zero, and it costs five pipeline runs per image. Tiling
also yields more fixations (33 per image on average): the targets are covered
by the top 20 on 66% and by the top 30 on 81% of items — the information is
there, the *ranking* across tiles is what is weak (per-tile saliency values are
not comparable, so the merge is a plain round-robin). Better ranking — or a
question-conditioned source to do the ranking — is where the next gain is;
finer grids are not worth running before that.

### After the symmetry port (2026-09-21)

Thesis profile, same 191 items: top-3 0.13 → 0.16, top-10 0.39 → **0.42**
(+0.21 over chance [+0.14, +0.28]); paired against the pre-port profile +0.04
[−0.03, +0.10] — the right direction, within noise. Legibility-oracle accuracy
of the `fovea` arm 0.12 → 0.14 (uniform 0.18). The port matters for fidelity and
for the dynamic studies (H1's off-object fixations went from a third to zero);
on V\*Bench's small coloured targets colour contrast still carries the effect.

### The dissertation system's field parameters (2026-09-21): fewer, better fixations

The replication dossier found that the field had been running on the default
arguments of the original's setter functions, not on what the dissertation
system configured (dossier, finding B). With the system's parameters the thesis
profile on the same 191 items:

| Thesis profile, field parameters | fixations per image | top-1 | top-3 | top-5 | top-10 | `fovea` accuracy (K = 3, legibility oracle) |
|---|---|---|---|---|---|---|
| port defaults (Backer kernel, global inhibition 1) | 9.2 | 0.04 | 0.16 | 0.22 | 0.42 | 0.14 |
| dissertation system (DoG kernel, global inhibition 8) | **3.9** | **0.06** | 0.15 | 0.16 | 0.16 | 0.15 |
| chance (10 random fixations) | | 0.02 | 0.07 | 0.11 | 0.22 | 0.10 |

The field now decides, as the thesis intends, *how many* things deserve
attention — about four per image, one per object — and its first fixation is
more often right (2.6 × chance instead of 1.5 ×). It simply has no tenth
fixation to offer on a single frame, so coverage at K = 10 is capped near its
coverage at K = 4; at the crop budget the front-end actually uses (K = 3)
nothing is lost. More fixations are what the model's *temporal* side is for:
inhibition of return across frames, which a single still does not exercise.

For the tracks this means: `configs/thesis/thesis.yaml` follows the dissertation
system (fidelity); `configs/modern.yaml` keeps the port's field parameters, with
which it was measured and which give a ranking of ten candidates — useful when a
downstream stage (a crop budget, a detector) does the final choosing. Which is
better *per crop spent* is an open modern-track measurement (K = 1–3, where the
two now differ in opposite directions).

### What the first (pre-port) measurement does and does not show

What this does and does not show. It does not show that bottom-up saliency
*cannot* help — given Part 1 (a broken colour map, meaningless eccentricity
segments, no peak-promoting fusion) and the resolution collapse, this stage 1
was never in a position to. It does show that every H6 "bottom-up fovea"
number so far measured *random crops plus a global view*. The tiled rerun
(shortlist #1), with the feature defects fixed, is the real test of bottom-up
crop selection; random placement is the baseline every crop source must now
be reported against.

*Method note: measured 2026-09-19 with a scratch script; the random-fixation
arm belongs in `eval/vlm_frontend.py` as a permanent baseline
(`docs/HYPOTHESIS_CLOSURE_PLAN.md`, H6).*
