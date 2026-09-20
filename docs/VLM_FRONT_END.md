# Attention as a VLM token-budget allocator (M18, H6)

*Status: instrument built and verified end-to-end on a mock backend, 2026-07.
Since 2026-09 the default backend is a local open-weights VLM (Ollama,
`qwen3.8:27b`), so the real V\*Bench measurement needs no credentials (see
"Running it for real"). Numbers so far are pilot scale (20 items per V\*Bench
category, 10 per HR-Bench category); the full V\*Bench run and a budget sweep
are open. The HR-Bench pilot rows were rerun on 2026-09-19 after a scoring
defect (correct option always "A") was fixed — see the note under the table.*

**H6 — Attention as a VLM token budget.** *Feeding a vision-language model only
the attended ROIs (fovea) plus a low-res global view preserves task accuracy at
a large fraction of the visual tokens/FLOPs of the full-resolution image, and
the saving grows with input resolution.*

This is the milestone that reframes the whole instrument: not "a 2004 thesis
reimplementation" but a **stateful, interpretable, model-free attention
front-end for large vision models**. Modern VLMs pay for every visual token, and
on high-resolution inputs they either burn tokens on the full image or downsample
and go blind to small objects. The attention pipeline is a cheap controller that
says *where to spend the tokens* — the GazeVLM / "gaze tells you where to
compute" recipe, with an interpretable mechanism in the driver's seat. The
non-goal (guarded): stay the controller, don't become another VLM.

## How it works

The whole front-end rides the existing interchange output — **no new C++ mode**
for this first cut (the full C++ virtual fovea is M15's job). Per image+question:

1. Run the pipeline (`attention <image> --emit-json`) to get the attention
   fixations — the top-K salient points, ranked.
2. Crop a fixed-size **fovea window** at native resolution around each of the
   top-K fixations, plus one low-res **global view** of the whole image.
3. Hand the VLM *only* those images + the question.

Every arm answers the same multiple-choice question, so the trade is honest:

| Arm | What the VLM sees | Role |
|---|---|---|
| `full-res` | the whole image, capped to a practical VLM size | accuracy ceiling |
| `uniform` | the whole image uniformly downsampled **to the fovea arm's token budget** | same-budget baseline (small objects vanish) |
| `fovea` (ours) | low-res global view + K native-res attention crops | the front-end |
| `fovea-random` | the same global view + K crops at uniformly random positions (seeded per question) | the floor: what the front-end is worth with no attention at all |
| `fovea-oracle` (`--oracle`) | the same, crops centred on the annotated targets | upper bound: perfect attention |
| `fovea-td` (`--top-down`) | the same, crops from a priority map with a question-conditioned top-down channel | H5×H6 |

Crops in the `fovea` arm are **bottom-up** (saliency). The `fovea-td` arm makes
them *question-conditioned* through the M17 `top_down_map` slot — the H5×H6
combination, "attend the thing the question asks about". The VLM is asked, on
a low-res view of the whole image (the fovea arm's global view by default;
`--ground-side` for a wider one), where the object the question is about is.
That box is rendered as a relevance plateau and fused into the priority map;
the C++ pipeline then picks the fixations, so bottom-up saliency still chooses
the point *within* the relevant region and IOR the order. The attention
pipeline stays the controller — the VLM only supplies the task term, like any
other top-down source — and the grounding call's tokens are charged to the
arm. (qwen3.8 answers grounding requests in 0–1000 normalized coordinates,
measured on V\*Bench.) The `fovea-oracle` arm swaps the attention source for
the annotated boxes at the same crop count, bounding what *any* attention
source can reach at this budget; both opt-in arms share the fovea arm's global
view, so they differ from it only in where the crops sit.

**Token accounting, two ways.** A provider-independent patch estimate (≈ one
visual token per 28×28 px, Qwen2-VL-style) is always computed, so the curve
draws in CI without any API. When a backend has a real tokenizer (Claude's
`count_tokens()`, or the `prompt_eval_count` Ollama returns with every answer),
that authoritative number is recorded too. Only the token *fraction* vs full-res
is reported, so the patch constant cancels.

**Target diagnostics.** V\*Bench annotates each question's targets (a sidecar
JSON per image with one box per target; two-object questions carry two). Where
boxes exist, every row records per arm whether the targets were *delivered*
legibly (the mock's criterion below), whether a fovea *crop* — not the global
view — covered them (`crop_hit`), and the rank of the first fixation whose
window covers them (`target_rank`, so coverage by the top K can be read off for
any K). That separates the two ways the fovea arm can fail — attention missed
the target, or the VLM misread a crop it was given. Next to the coverage the
harness prints its **chance level** — the expected coverage of uniformly
random fixations (200 draws per item): for small targets the base rate of a
336-px window is not negligible, and a crop source is informative only to the
extent that it beats it.

The VLM is pluggable (`eval/vlm_backends.py`): a `VLMBackend` interface with an
`ollama` default (a local VLM over Ollama's HTTP API, `qwen3.8:27b`, stdlib
only), a `claude` backend (anthropic SDK, `claude-opus-5`, base64 image blocks),
and the `mock` the CI smoke runs on. This keeps the core dependency-free and
CI-safe — the model lives Python-side behind the interchange boundary, exactly
like the other modern models in this repo.

## The mock is a real test

The `mock` backend answers correctly **iff the target is delivered at usable
resolution** — the harness computes, per arm, whether the target region is
present in some view *and* large enough to read there. This makes the whole
pipeline testable end to end without a model: the fovea arm scores only when an
attention crop actually lands on the target, which is the H6 effect itself. It
also models the failure mode honestly — a uniform downsample that shrinks the
target below the legibility floor is scored blind, just as a real VLM would be.
On V\*Bench the same rule makes the mock a model-free *legibility oracle*: its
accuracy per arm is that arm's delivered rate.

Synthetic demo (one high-res image, a small salient marker among clutter, mock
backend):

![synthetic demo](images/vlm_frontend_demo.png)

| arm | accuracy | token-fraction |
|---|---|---|
| full-res | 100% | 100% |
| uniform (same budget) | **0%** | 27% |
| fovea (ours) | **100%** | 27% |

The attention pipeline finds the salient marker, a crop covers it, and the VLM
answers — at **27% of the full-resolution tokens**, where the token-matched
uniform downsample has lost the marker entirely. This is a single constructed
item (the point is to exercise the pipeline, not to prove H6); the real evidence
is the V\*Bench run below.

```bash
# the end-to-end demo (mock; no dataset, no model) — also the CI smoke via --check
eval/vlm_frontend.py --demo --check --backend mock
```

## Running it for real (V\*Bench, HR-Bench)

[V\*Bench](https://huggingface.co/datasets/craigwu/vstar_bench) is
high-resolution VQA where the answer hinges on a small region — the regime where
uniform downsampling makes VLMs blind, and where the attention front-end should
shine. It's public and small (~200 items). Adapter:
`eval/datasets/vstar.py` (download documented there, never redistributed).

```bash
# 1. get the data
huggingface-cli download craigwu/vstar_bench --repo-type dataset \
    --local-dir data/vstar_bench
# 2. a real VLM — the default is local and free (Ollama, open weights)
ollama pull qwen3.8:27b          # any vision model works: --model <tag>
# 3. run the study (+ the oracle and top-down arms) and draw the curve
eval/vlm_frontend.py --vstar --count-tokens --oracle --top-down --limit 0
eval/plot_vlm_frontend.py results/vlm_frontend/summary.json \
    --out docs/images/vlm_frontend_tradeoff.png

# or with Claude: install the SDK + provide a key
eval/.venv/bin/pip install anthropic
export ANTHROPIC_API_KEY=...     # or `ant auth login` with the Anthropic CLI
eval/vlm_frontend.py --vstar --backend claude --count-tokens --limit 0
```

**Ollama specifics.** qwen3.8 under Ollama spends one visual token per 32×32 px
(measured: a 448² image is 196 image + ~44 text tokens), and Ollama silently
downscales any image beyond a 64×64-token grid (~2048 px/side). The full-res
arm's default cap (`--full-max-side 1512`) stays under that, so full-res really
is full-res — raise it past ~2048 and it no longer is. The backend asks for an
8192-token context and warns if a prompt reaches it (Ollama truncates
silently). A 4-bit 27B is weaker in absolute terms than a frontier API model;
H6 compares arms *within* one model, so report the exact model tag and
quantization alongside the numbers. Results are written after every item, and
`--resume` continues a run that was killed midway (a 27B model plus 8K images
can run a 32 GB machine short of memory).

[HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench) is the
higher-resolution regime: 4K and 8K images (200 questions each, 4 options;
"single"-instance attributes/OCR and "cross"-instance maps/charts/relations),
far past what a VLM ingests natively — where a uniform downsample should truly
blind the model. Adapter: `eval/datasets/hrbench.py`. The parquet files carry
each question four times (option rotations for CircularEval); only the first
rotation is scored (plain accuracy, chance 0.25). In that rotation the correct
option is "A" for every question, so the adapter re-orders the options per
question (seeded by the question id — the same order in every arm and run);
without that, a model that answers "A" whenever it cannot see the target scores
1.0. There are no target boxes,
so no oracle arm or target diagnostics. The first use extracts the images
(needs `pyarrow`); the 8K extraction peaks at ~5.8 GB of RAM, so unload the VLM
first (`ollama stop qwen3.8:27b`).

```bash
huggingface-cli download DreamMr/HR-Bench --repo-type dataset --local-dir data/hr_bench
eval/.venv/bin/pip install pyarrow
eval/vlm_frontend.py --hrbench 8k --count-tokens --top-down --full-max-side 2000 --limit 0
```

The prediction (H6): **fovea tracks full-res accuracy while using a small
fraction of the tokens, and uniform-at-the-same-budget lags well behind** —
because the answer-bearing region survives at native resolution in a crop but
dissolves under uniform downsampling. This section will carry the measured
numbers and the real-backend figure once that run completes.

### First real numbers (2026-09, local Qwen, pilot scale)

`qwen3.8:27b` (Q4_K_M) via Ollama; default settings (K = 3 crops of 336 px,
512-px global view; full-res cap 1512 on V\*Bench, 2000 on HR-Bench 8K). The
first 20 items per V\*Bench category and the first 10 per HR-Bench category —
95% CIs are about ±0.3, so these are directions, not results. "@" is the real
token fraction of full-res (Ollama's own count).

| Dataset (n) | full-res | uniform | fovea (bottom-up) | fovea-oracle | fovea-td |
|---|---|---|---|---|---|
| V\*Bench direct attributes (20) | 0.85 | 0.80 @ 0.36 | 0.45 @ 0.37 | **1.00** @ 0.34 | 0.70 @ 0.53 |
| V\*Bench relative position (20) | 0.90 | 0.80 @ 0.37 | 0.60 @ 0.38 | 0.70 @ 0.37 | 0.65 @ 0.54 |
| HR-Bench 8K single (10) | 0.90 | 0.60 @ 0.22 | 0.70 @ 0.23 | — | 0.70 @ 0.32 |
| HR-Bench 8K cross (10) | 0.80 | 0.50 @ 0.23 | 0.40 @ 0.24 | — | 0.40 @ 0.34 |

**Caveats on this table (added 2026-09-19).** (1) *The two HR-Bench rows were
rerun on 2026-09-19 with the option order fixed.* The first version was scored
with the options as stored, where the correct answer is always "A", and read
1.00 / 0.70 / 0.50 / 0.70 (single) and 0.70 / 0.80 / 0.50 / 0.40 (cross). The
bias was real and favoured the blind arm: with options re-ordered the uniform
arm falls 0.70 → 0.60 and 0.80 → 0.50 (on the cross items it still answers "A"
six times in ten — it guesses "A" when it cannot see), and the apparent
"uniform beats full-res on cross" disappears. At n = 10 per row the intervals
are about ±0.3; the ordering full-res > budget arms is the only thing these
rows show. (2) *Chance differs per
row:* V\*Bench relative-position questions are two-way, so chance there is 0.5
and 0.60–0.70 at n = 20 is not distinguishable from guessing; direct-attribute
questions are mostly four-way. The harness now prints the chance level of the
item mix. (3) "1.00 vs 0.85" for the oracle arm is 3 discordant items of 20.

What they say:

- **The front-end works when attention lands.** With crops on the annotated
  target, a third of the tokens beats full resolution on single-target
  questions (1.00 vs 0.85) — the upper bound H6 promises.
- **Bottom-up attention rarely lands** (crops cover the target on 10% of
  direct-attribute items), so the bottom-up fovea arm trails everything.
- **Question-conditioned grounding closes part of the gap** — crops cover the
  target on 65% of direct-attribute items, accuracy 0.45 → 0.70 — but it does
  not yet beat the same-budget uniform arm, and its grounding call raises its
  cost (the arm is not budget-matched; a fair comparison gives uniform its
  budget).
- **Relations may favour the whole view** — two-object (relative position)
  questions lose their spatial layout in crops — but at two-way chance and
  n = 20 the pilot cannot show it. On HR-Bench "cross" the crop arms (0.40) do
  trail uniform (0.50) and full-res (0.80), which points the same way, at
  n = 10.
- **Qwen reads small detail from downsampled images better than the 24-px
  legibility floor assumes**, which keeps the uniform arm strong at these
  budgets (V\*Bench rows; the synthetic video probe in `docs/VLM_VIDEO.md`
  found the same).

## Honest limitations (to report with the real numbers)

- **Bottom-up crops can miss the target.** Saliency isn't task-relevance: if the
  question asks about a low-salience object, the crops won't cover it and fovea
  accuracy drops. This is precisely the gap the M17 top-down channel closes —
  and the reason the H5×H6 arm (question-conditioned crops) is the natural next
  step, not an afterthought. Report the bottom-up ceiling honestly first.
  *Measured (2026-09, all 191 V\*Bench items, model-free target
  diagnostics):* the first fixation's 336-px window covers the annotated target
  on 2% of items, the top 3 on 7%, the top 10 on 22% — 78% of targets are never
  covered by the pipeline's 10 fixations (V\*Bench targets are small: median
  36 px on the short side). By the legibility oracle, bottom-up crops deliver
  the target *less* often than a same-budget uniform downsample (12% vs 17% at
  ~20% of full-res tokens). On V\*Bench, bottom-up crops alone are not
  expected to beat uniform; the top-down channel is the necessary next step.
  *Random baseline (2026-09-19):* uniformly random fixations cover the targets
  just as often — 2% / 7% / 22% for the top 1 / 3 / 10 — so these figures are
  the base rate of a 336-px window, not a saliency signal; the bottom-up
  `fovea` arm has so far been equivalent to random crops. Why (first-stage
  defects, resolution collapse) and what to try: `docs/FEATURE_ASSESSMENT.md`.
- **The controller isn't free.** The token *saving* is measured on the VLM side;
  the attention pipeline that picks the crops costs its own compute
  (`docs/PERFORMANCE.md`). The argument scales as the VLM gets more expensive per
  token relative to the (fixed, cheap) controller — which is the real-world
  regime for large models on high-res inputs.
- **K and the fovea size are knobs.** Too few/small crops miss the target; too
  many erase the saving. The curve is drawn *across* these, not at one point.

## Files

- `eval/vlm_backends.py` — the `VLMBackend` interface, `ollama` + `claude` + `mock` backends, token estimate
- `eval/vlm_frontend.py` — the three-arm harness (crop / assemble / score), `--demo` + `--vstar`
- `eval/datasets/vstar.py` — V\*Bench adapter (questions + target boxes)
- `eval/datasets/hrbench.py` — HR-Bench 4K/8K adapter (streams the parquet, extracts once)
- `eval/plot_vlm_frontend.py` — the accuracy-vs-token-budget figure
- CTest `vlm_frontend_smoke` (`--demo --check --backend mock`) + help tests;
  `eval/tests/test_vlm_backends.py` (the ollama backend against a faked HTTP API),
  `eval/tests/test_vlm_frontend.py` (target diagnostics, V\*Bench boxes),
  `eval/tests/test_vlm_arms.py` (oracle crops, relevance map, HR-Bench extraction)
