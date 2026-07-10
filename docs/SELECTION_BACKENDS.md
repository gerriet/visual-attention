# Selection backends — alternatives to the neural field

*Drafted 2026-07-10. Motivation: the 2D/3D Amari neural field
(`neural-field`, `neural-field-3d`) is the thesis's selection mechanism and it
works, but it is **hard to parametrize** — a documented obstacle to DNF
adoption in technical systems, not a quirk of this implementation (cedar/DFT:
Lomp, Richter, Zibner & Schöner 2016). This doc collects alternatives for the
same job — tracking multiple high-activity blobs in 2D and 3D — that are more
robust, simpler, or more efficient. Each is a drop-in `SelectionStrategy`
behind the existing registry, so they A/B against the field on identical
saliency streams.*

## Why the field is hard to tune

The field conflates five jobs into one nonlinear dynamical system:

1. **Detect** the salient blobs (how many? which?),
2. **Compete** — soft winner-take-all via lateral inhibition; nearby blobs
   suppress each other; global inhibition caps the count,
3. **Persist** — a selected blob stays active when its input dips (hysteresis
   = short-term memory = tracking through noise/brief occlusion),
4. **Inhibit return** — decaying suppression of visited locations,
5. **Repel** — active blobs push apart (thesis Abb. 6.10).

These are controlled by ~13 coupled parameters (`alpha, beta, resting,
global_mult, input_mult, kernel_k, kernel_s, kernel_size, max_cycles,
change_thresh, min_cluster_size, ior_decay, border_margin`), and — the real
pain — the *ignition* of the field depends on the **absolute scale** of the
saliency input, so a tuning that works on one image class fails on another.

**Design principle for the alternatives: decouple the five jobs.** Give each an
independent knob with an obvious meaning and a monotone effect. Then the
backend is tunable by inspection, not by search.

## The menu (simplest → most principled)

All operate on the fused saliency map `cv::Mat` and return `std::vector<Peak>`,
persisting per-track state in `RunState` across frames (as the field already
does with `field_activity`). The **3D** column is the disparity/depth axis from
the stereo cue (`RunState::depth_map`), the same z the 3D field uses.

### A. Mode-seeking (mean-shift / CAMShift) — *simplest*
Treat the saliency map as a density; its **modes are the blobs**. Mean-shift
climbs to each mode from seed points; nearby seeds merge at the bandwidth
scale, so the number of blobs *emerges* and competition is free. CAMShift adapts
the window to blob size → scale changes handled. Track modes across frames by
nearest-mode association.
- **Knobs (2):** bandwidth = expected object radius (has physical meaning);
  min density = detection floor.
- **3D:** mean-shift in (x, y, disparity), one bandwidth per axis.
- **Persist / IOR:** carry mode identities across frames; IOR = per-mode
  refractory counter.
- **Lit:** Comaniciu & Meer 2002 (mean-shift); Bradski 1998 (CAMShift).
- **Cost:** O(seeds × iterations), no pixel-wise cycles. Robust, tiny.

### B. Blob detection + Kalman MOT (SORT-style) — *the pragmatic workhorse*
Split detection from tracking cleanly:
- **Detect:** percentile-threshold the saliency → connected components / local
  maxima with non-max suppression → blob list (centroid, size, peak).
- **Track:** a bank of constant-velocity **Kalman filters**, one per blob;
  greedy/Hungarian association by distance or IoU; a track lifecycle
  (tentative → confirmed → coasting → dead).
- This gives **persistence, occlusion tolerance, and object-based IOR** for
  almost free: hysteresis = confirmation + coast counts; IOR = a
  "recently-selected" flag that decays per track; repulsion = trivial if
  wanted. The Kalman process/measurement-noise pair is far more intuitive than
  the field's gain/kernel coupling, and it self-adapts to motion.
- **Knobs (~4):** detection percentile, min blob size, coast frames
  (occlusion tolerance), IOR decay — each independent and monotone.
- **3D:** Kalman state (x, y, z, ẋ, ẏ, ż); z = disparity. *Same math* — no 3D
  interaction kernel to tune. This is the big win over `neural-field-3d`.
- **Lit:** Bewley et al. 2016 (SORT: Kalman + Hungarian MOT); Kalman 1960.
- **Cost:** O(blobs), not O(pixels × cycles) — orders of magnitude cheaper.
- **Note:** this partly overlaps the M6 object-file layer; run it *at
  selection* to get sub-object blob tracking, or fold the two — a good
  simplification to evaluate.

### C. Attention particles — *cheap, keeps the "emergent" flavor*
Represent attention as N interacting "particles" that gradient-ascend the
saliency landscape with a short-range **repulsive potential** and a per-particle
**refractory** (IOR). N hill-climbers with mutual repulsion.
- **Knobs (3):** N = max foci; repulsion radius; refractory decay. Repulsion
  reproduces the field's Abb. 6.10 blob-repulsion directly.
- **3D:** particles live in (x, y, disparity); repulsion is a 3D distance.
- **Lit:** particle/point-process attention; akin to N-body soft-WTA.
- **Cost:** O(N × neighborhood). Retains repulsion/hysteresis feel without
  pixel-field dynamics.

### D. Normalization-model competition — *keeps a field, kills the scale problem*
Keep continuous, biologically-grounded competition but replace tuned global
inhibition with **divisive normalization**: `r_i = a_i^n / (σ^n + Σ_j w_ij a_j^n)`.
This does soft-WTA + count-limiting + contrast-invariance *by construction*,
and it is **input-scale invariant** — the #1 field tuning pain vanishes.
- **Knobs (2–3):** semi-saturation σ (competition strength), pool width,
  exponent n. Each has a clean, monotone meaning.
- **3D:** normalize over a 3D neighborhood pool.
- **Lit:** Reynolds & Heeger 2009 (normalization model of attention); Carandini
  & Heeger 2012 (normalization as a canonical computation).
- **Cost:** one pass per frame (no convergence loop needed).

### E. Self-tuning field (intrinsic plasticity) — *smallest change to what exists*
If the emergent field dynamics are worth keeping (repulsion, sustained
activity, the thesis fidelity), make it **auto-tune** the two worst knobs:
adapt each unit's gain/threshold online to hold a target mean activity
(intrinsic plasticity), and **normalize the input** each frame to a fixed range
so ignition no longer depends on image brightness.
- **Removes:** hand-tuning of `resting` and `beta` (sigmoid slope), and the
  input-scale dependence — the biggest tuning sources.
- **Keeps:** the Amari dynamics and thesis behavior; minimal code delta over
  the current `NeuralFieldSelection`.
- **Lit:** "Dynamic Neural Fields with Intrinsic Plasticity" (Frontiers 2017).

### F. GM-PHD random-finite-set filter — *the principled MOT answer*
For the hardest job the 3D field was built for — an **unknown, time-varying
number of targets under clutter and occlusion** (thesis Abb. 6.14) — the modern
principled tool is a random-finite-set filter. **GM-PHD** propagates a Gaussian
mixture over the target set in closed form, with explicit birth/death/clutter
models and **no explicit data association**.
- **3D:** Gaussian components live in (x, y, z, velocity); z = disparity.
- **Lit:** Vo & Ma 2006 (GM-PHD); Mahler 2003 (PHD).
- **Cost:** heavier than A–E, but it is the rigorous baseline for the
  dynamic-IOR study (H1) and directly models occlusion.

## Summary

| Backend | Knobs | 2D | 3D | Persist/occlusion | Cost | Keeps field behavior |
|---|---|---|---|---|---|---|
| Neural field (current) | ~13, coupled | ✓ | ✓ | ✓ (tuned) | O(px·cycles) | — |
| A. Mean-shift/CAMShift | 2 | ✓ | ✓ | via re-ID | low | no |
| B. Blob + Kalman MOT | ~4 | ✓ | ✓ (trivial) | ✓ (coast) | very low | no |
| C. Attention particles | 3 | ✓ | ✓ | partial | low | repulsion |
| D. Normalization competition | 2–3 | ✓ | ✓ | with hysteresis add-on | low | soft-WTA |
| E. Self-tuning field | ~13 → ~4 effective | ✓ | ✓ | ✓ | O(px·cycles) | **yes** |
| F. GM-PHD | moderate, semantic | ✓ | ✓ | ✓ (modeled) | medium–high | no |

## The experiment the framework already enables

Because selection is a registry and every model emits the interchange format,
these are **swappable behind one config key** and comparable on *identical*
saliency streams. That yields a clean study:

> Same fused saliency (thesis features), vary only the selection backend →
> measure, on the M12 dynamic-scene benchmark: tracking accuracy (coverage,
> ID persistence, occlusion recovery), **parameter sensitivity** (how wide is
> the working parameter range — the whole point), and runtime.

This turns "the field was hard to tune" into a quantified result and gives H1
(dynamic IOR) a set of rigorous baselines.

## Recommendation

> **Decision (Gerriet, 2026-07-10): agreed — prototype B first**, as a
> `kalman-mot` `SelectionStrategy`, to A/B against the neural field on identical
> saliency streams (e.g. `vtest.avi`). To be built after the M9 commit lands.

Build **B (blob + Kalman MOT)** first — biggest robustness/efficiency win,
simplest to reason about, and its Kalman occlusion handling is exactly what the
dynamic-IOR study needs; it also generalizes to 3D for free. Add **D
(normalization competition)** as the "keep it field-like but scale-invariant"
option, and **E (self-tuning field)** as the minimal-risk upgrade path for
users who want thesis fidelity without the tuning. Keep **F (GM-PHD)** as the
principled MOT baseline for the H1 study. **A** and **C** are cheap enough to
add as lightweight extras. The thesis field stays the default; all are opt-in.
