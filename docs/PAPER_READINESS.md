# How far from a paper? (assessment, 2026-09-16)

An honest stock-take of what this repository could publish, what a reviewer
would object to, and what is missing. Two candidate papers — the plan
(`docs/V3_ROADMAP.md`, and the papers-phase decision of 2026-07-26) has paper
A leading; this assessment says B is closer to done and A needs one specific
result first.

## What exists (assets either paper can lean on)

- A C++17/OpenCV reimplementation of the 2004 two-stage attention model, at
  the stated bar of *loose behavioral equivalence*, with the second stage
  (object files, object-based IOR, behaviors) intact.
- An interchange format as the single integration boundary, so every model —
  C++ or Python — is scored by the same harness.
- Honest negatives, documented: object-based IOR does **not** beat
  space-based IOR on exploration metrics (M12/H1) because it is only as good
  as its tracker; on stills, bottom-up crops lose to a budget-matched uniform
  downsample (M18/H6).
- Decomposition instruments that separate *policy* from *tracker quality*:
  oracle arms (crops on the annotated target) and perfect-tracker arms
  (fixations deduplicated by ground-truth identity).
- Real token accounting against a real model (local Qwen via Ollama), not a
  proxy; every study runs end to end on a mock backend in CI.
- A dynamic result with a real VLM (M19/H7, synthetic scenes): at a matched
  token budget, identity-keyed crops 0.95 vs location-keyed 0.80 vs
  budget-matched frames 0.27 (chance 0.25).

## Paper B — replication / retrospective (closer)

*"A 2004 attention dissertation, reimplemented and re-tested."* Reproduce the
thesis's own findings, then ask which of its claims survive modern scrutiny.
The distinctive asset is the honest H1 result plus its follow-up: persistent,
identity-keyed object memory moves object-based IOR from clearly worse to
level with space-based IOR. The confirmatory run (2026-09-20) sharpened it:
object-based IOR reaches new objects sooner in every regime, against a
motion-compensated location tag too; it does not win on sustained coverage
(`docs/DYNAMIC_IOR_STUDY.md`).

Missing:

0. ~~H1 at full statistics~~ — **done 2026-09-20**: 30 fresh scenes per
   regime, thesis profile, pre-registered predictions, a strengthened spatial
   baseline. H1 is supported for novel-object latency (conditional on held
   identity at high speed), not for sustained coverage
   (`docs/DYNAMIC_IOR_STUDY.md`, "The confirmatory run").
1. **M10 replication dossier** — the thesis figures reproduced one by one,
   with a verdict per finding (replicated / partial / diverged).
2. **M11 scanpaths vs human fixations** — behavioral evidence that the
   scanpaths are plausible (MultiMatch/ScanMatch against generative baselines).
3. Writing, and a venue that values negative and reproduction results
   (ReScience-style journals, reproducibility workshops).

Mostly running and writing rather than new research: roughly 2–4 weeks.

## Paper A — the attention front-end for VLMs (more interesting, not yet defensible)

*"An interpretable, model-free attention front-end that decides where a
vision-language model spends its visual tokens."* What a reviewer would say
today:

1. **The win is synthetic.** The H7 effect is demonstrated on coloured disks
   carrying codes whose size was calibrated to the VLM's reading threshold. On
   the one real dataset tried (DAVIS-2017 at 480p) every arm ties. As it
   stands: "the effect exists where it was constructed to exist".
2. **The obvious baseline is missing.** Why not pick crops with SAM2 /
   ByteTrack / an open-vocabulary detector? Interpretable and model-free is a
   fair argument, but it has to be *measured* — accuracy and controller cost.
3. **The still-image half currently loses.** On V\*Bench, bottom-up crops
   trail the same-budget uniform arm, and question-conditioned crops don't
   overtake it either (`docs/VLM_FRONT_END.md`). The oracle arm shows the
   ceiling is real, which makes this an instructive negative, not a headline.
4. **Statistics are thin.** 10 scenes; intervals bootstrapped over questions
   rather than scenes; one VLM at one quantization; pilot-scale V\*Bench and
   HR-Bench runs.
5. **Identity on real video is unsolved** — 13–16 object-file labels per
   attended object on DAVIS, 57–75% of fixations off the annotated objects.

What would make it publishable, in order:

1. **A real-video setting where it wins** — the crux, still open. The
   pre-registered attempt (DAVIS at full resolution, small-object questions)
   was **run on 2026-09-18 and refuted the prediction**: at ~8× the pixels per
   side every budget arm scores alike, because the category question survives
   downsampling (`docs/VLM_VIDEO.md`). The next attempt needs a *task* whose
   answer requires detail — reading a label or number, an attribute of a small
   object, counting small instances — authored per sequence and verified
   against crops. Until that exists, the H7 evidence stays synthetic, and
   paper A's honest form is the workshop version below.
2. **Identity on textured video**: a stronger appearance descriptor than mean
   colour (histogram / per-feature signature), and fewer background fixations.
3. **A modern crop-selection baseline**, with the controller's own compute
   counted on both sides.
4. **The full keyed runs** (all of V\*Bench and HR-Bench), a budget *sweep*
   rather than single points, and a second VLM so the result isn't a
   Qwen artefact.

If the crux works: 1–2 months to a main-conference submission. If it doesn't:
a shorter, honest workshop paper — *where an interpretable attention
front-end helps a VLM and where it does not* — carried by the
oracle-vs-actual decomposition, which is a genuinely useful instrument.

## Recommendation

Spend a bounded effort on the crux (the prepared full-resolution test). If
object-file crops win there, paper A leads as planned. If they don't, publish
B first and fold the front-end work into it as the chapter on what the old
model buys a modern system.

## Risks to name in either paper

- The controller is not free: token savings are measured on the VLM side; the
  attention pipeline costs its own compute (`docs/PERFORMANCE.md`).
- Results are from one local open-weights VLM at 4-bit quantization.
- The synthetic scenes are constructed; their code size is calibrated to the
  model that reads them. Say so plainly and report the calibration.
