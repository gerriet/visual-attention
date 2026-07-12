# Research positioning — this system vs. current visual-attention research

*Written 2026-07-10. An assessment of where the framework and the v3 plan
(`docs/V3_ROADMAP.md` — the H1 dynamic-IOR study, M11 scanpaths-vs-human, M13
recognition gating) sit relative to the 2025–26 literature in human and
computer vision, and how the work could interact with large language / vision
models. Sources are listed at the end; the CV claims are grounded in cited
2025–26 work rather than memory.*

## Where the system sits relative to current research

### What still relates — and is current framing

- **The two-stage architecture itself.** Preattentive feature maps → an
  attentive, object-based selection stage is the Treisman/Wolfe lineage, and it
  has *not* been superseded — it has been renamed and generalized. The modern
  unifying construct is the **priority map**: a single map combining bottom-up
  salience, top-down task relevance, and *selection history / reward*. The
  "master saliency map" here is a priority map missing two of its three modern
  inputs. This is the most important conceptual bridge: the right object already
  exists; it needs two more terms.
- **Object files + object-based IOR + tracking through occlusion.** The part of
  the thesis that was ahead of its time and is *live research right now* —
  multiple-object tracking, unequal attention allocation across tracked objects,
  and object- vs space-based selection are active in 2025–26 neuroscience. **H1**
  (object-based IOR beating spatial IOR in dynamic scenes) is a real, contestable
  scientific claim, not a reimplementation exercise. This is the defensible
  niche.
- **Active / overt vision.** The `move_sensor`/scanpath ambitions and the
  "virtual fovea" (M15) map onto a *revived* topic: foveated active perception
  for navigation and, newly, foveated compute for large models (see the LLM
  section).
- **Neural fields / DFT.** The Amari-field lineage is still a living paradigm
  (Schöner's Dynamic Field Theory, the `cedar` framework) in embodied cognition
  and robotics — niche relative to deep learning, but not dead.
  `docs/SELECTION_BACKENDS.md` already diagnoses its real weakness
  (input-scale-dependent ignition, ~13 coupled knobs) honestly, and the plan to
  A/B it against decoupled backends (Kalman MOT first) is the right instinct.

### What is dated

- **Hand-crafted features as the accuracy mechanism.** Gabor/opponent/symmetry/
  eccentricity channels are historically interesting but not where predictive
  accuracy lives. The M7 benchmark already shows the thesis map diverging from
  human fixations. Free-viewing saliency — "where do people look on a still
  image, aggregated" — is essentially **saturated**: DeepGaze III and successors
  sit near the inter-observer ceiling on the MIT/Tübingen benchmark. Do **not**
  try to win there.
- **Pure bottom-up emphasis.** The field moved to task-driven / top-down
  attention and then to *selection history and value* as a third priority source
  (Awh, Belopolsky & Theeuwes; Wolfe's Guided Search 6.0). A model with no
  top-down/task input and no learned prior looks 2004 precisely because it stops
  at bottom-up salience.
- **WTA + spatial IOR as the whole selection story.** Modern free-viewing
  readouts are probabilistic/generative, and human scanpath data now demands
  modeling *variability*, not a single deterministic path (2025 diffusion-based
  scanpath models exist for exactly this). The deterministic argmax-with-
  inhibition here is a fine mechanism but should be scored as one sample of a
  distribution, not as "the" scanpath.

## Main topics right now

**Human vision (psychology / neuroscience):**
- **Priority maps** as the unifying substrate, with **selection history and
  reward / value** as first-class priority sources alongside bottom-up and
  top-down.
- **Rhythmic / oscillatory attention** — attention as ~4–8 Hz theta sampling
  rather than a steady spotlight. Genuinely *new since the thesis*, and a
  different temporal story from continuous field relaxation; recent work ties
  rhythmic sampling to awareness and distractor suppression.
- **Object-based attention and MOT** — how attention is allocated (unequally)
  across multiple tracked objects; tracking through occlusion.
- **Attention ↔ awareness / consciousness** interplay; feature-based attention
  stable across saccades (eye-centered feature representations).

**Computer vision:**
- **Scanpath prediction** has replaced static saliency as the frontier — HAT
  (Human Attention Transformer), Gazeformer (goal-directed search), TPP-Gaze
  (temporal point processes), and 2025 **diffusion** models (ScanDiff, DiffEye)
  modeling scanpath *variability*. Metrics: **MultiMatch** and **ScanMatch** —
  which M11 already adopts.
- **Goal-directed / task-driven visual search** rather than free viewing.
- **Efficiency of large vision models** via token pruning and foveation — the
  topic that most directly connects this work to LLMs.

## Vision systems ↔ LLMs: the modernization bridge

Four concrete interaction modes; three map onto architecture that already
exists here.

1. **Attention as a token-budget allocator for VLMs (strongest, most timely).**
   Modern VLMs choke on high-resolution images — LLaVA-NeXT-class models spend
   up to ~2,880 visual tokens per image, and downsampling makes them blind to
   small objects. The hot response is foveated / gaze-driven pruning: **GazeVLM**
   extracts fovea-like ROIs plus a low-res global view and reportedly cuts visual
   tokens by up to ~93%; a parallel line ("Eye Gaze Tells You Where to Compute")
   uses gaze to decide *where* the model spends compute. This system produces
   exactly that input — a priority map, ranked attended ROIs, object files over
   time — as an interpretable, model-free front-end. The **M8 processor-plugin**
   architecture is already the right shape: a processor that crops the attended
   ROI and hands it to a VLM is a small addition, not a redesign.

2. **LLM as the top-down source for the priority map (closes the biggest gap).**
   The missing top-down term is naturally supplied by language: a prompt ("find
   the exit sign") → target embedding → per-feature weight vector or a semantic
   relevance channel (open-vocabulary detector / CLIP-style map) fused into the
   master map. That is Guided Search with a language front-end, and it turns a
   bottom-up model into a task-driven one with minimal architectural change.

3. **"Thinking with images" / agentic visual search — this system as controller
   or baseline.** VLMs now iteratively zoom and crop (ZoomEye, Chain-of-Focus,
   DeepEyes, Thyme) — a *learned overt-attention loop*, essentially learned
   scanpaths. The two-stage object-based model is a fast, interpretable,
   bottom-up controller for the same crop-and-look loop, and a scientifically
   interesting **baseline**: do RL-learned visual-search policies resemble
   human / object-based scanpaths, or not?

4. **LLM as recognition processor and reasoner over object files (neuro-symbolic
   story).** M13's recognition tier can be a VLM captioning attended ROIs, with
   labels accumulating on object files by majority vote. More interestingly, the
   symbolic second stage — object files with labels, trajectories, saliency
   history — *is a scene graph / working memory* an LLM can query and update.
   "Attention maintains a small symbolic object store; the LLM reasons over it"
   is a current framing, and it is exactly what stage 2 already builds.

## Where to move it to become relevant

The v3 plan is already well-aimed; sharpen three things rather than redirect:

- **Commit hard to the dynamic, object-based scanpath niche (H1 + M11)** — where
  learned models are weakest and a real mechanism exists here. Frame H1 as a
  contribution to the MOT / dynamic-attention debate; score with MultiMatch /
  ScanMatch (planned); add a **generative / variability baseline** (a
  ScanDiff-style probabilistic peer) and probabilistic scoring, so the
  deterministic scanpath is evaluated as a sample, not compared unfairly.
- **Add the two missing priority-map terms:** a **top-down / task channel** (best
  sourced from an LLM, mode 2) and **selection-history / value**. Even a simple
  version moves the model from "2004 bottom-up" to "current priority-map framing"
  and gives an ablation axis.
- **Build the VLM-front-end demo (mode 1)** — lowest-effort, highest-timeliness
  result: same instrument, reframed as "interpretable attention that saves VLM
  compute," with a measurable token/FLOP-vs-accuracy curve. That single demo
  repositions the project from "reimplementing a 2004 thesis" to "a stateful,
  object-based attention front-end for large vision models."

One caution: don't let it drift into "another VLM," or into chasing DeepGaze on
free-viewing salience. The edge is the interpretable, stateful, object-based
*dynamic controller* — the thing large learned models don't have and
increasingly need.

## Sources

- [State-of-the-Art in Human Scanpath Prediction (Kümmerer & Bethge)](https://arxiv.org/pdf/2102.12239)
- [ScanDiff: Modeling Human Gaze Behavior with Diffusion Models (ICCV 2025)](https://aimagelab.github.io/ScanDiff/) · [paper](https://arxiv.org/abs/2507.23021)
- [Gazeformer: Goal-Directed Human Attention](https://arxiv.org/pdf/2303.15274)
- [Human Scanpath Prediction in Target-Present Visual Search (Semantic-Foveal Bayesian Attention)](https://arxiv.org/pdf/2507.18503)
- [GazeVLM: Eye Gaze Tells You Where to Compute — Gaze-Driven Efficient VLMs](https://arxiv.org/html/2509.16476v1)
- [Thinking with Images for Multimodal Reasoning (survey)](https://arxiv.org/pdf/2506.23918)
- [CVSearch: Cognitive Visual Search for High-Resolution Image Perception](https://arxiv.org/html/2605.23655)
- [Nature Index — Visual Attention and Cognitive Processing Mechanisms](https://www.nature.com/nature-index/topics/l4/visual-attention-and-cognitive-processing-mechanisms)
- [Journal of Neuroscience — attention keyword feed](https://www.jneurosci.org/keyword/attention?page=1)
- [Visual awareness enhances attentional rhythmic sampling (2025)](https://medicalxpress.com/news/2025-12-visual-awareness-interplay-attention-consciousness.html)
