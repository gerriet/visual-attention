# Hypothesis closure plan (2026-09-19)

*Follows `docs/CRITICAL_REVIEW_2026-09.md`. The review's central finding: seven
hypotheses are open and only one sub-result (H5 synthetic) meets the roadmap's
own bar. This plan opens nothing new. Each hypothesis gets one confirmatory
run and leaves with a verdict — **supported**, **refuted**, **supported under
a named condition**, or **parked** with the reason written down.*

## Rules for every confirmatory run

1. **Frozen config, fresh seeds.** Mechanisms and knobs are tuned on *dev
   seeds* (0–9) only. The confirmatory run uses *test seeds* 1000 … 1000+n−1,
   touched once, with the config committed beforehand.
2. **Prediction first.** A short block in the study doc before the run:
   expected ordering, what would refute it, the analysis. Template: the
   full-resolution DAVIS block in `docs/VLM_VIDEO.md`.
3. **Paired statistics over the independent unit.** All arms see the same
   scenes/items, so report the *paired* difference with a bootstrap interval,
   resampling **scenes** (or images), never questions within a scene.
   `eval/study_common.py` gets `paired_bootstrap(a, b)`; with a unit test.
4. **n ≥ 20** units per cell (30 where it is cheap: everything synthetic).
5. **One external or trivial baseline, and one strengthened loser.** Before
   building a mechanism for the arm we like, run the dumbest thing that could
   explain the effect, and give the arm we expect to lose its best version.
6. **One command reproduces the table**; rows carry git SHA + config hash.
7. **The README quotes a number only after its confirmatory run.**

## What needs what

| Work | VLM / API key | Dataset download | Compute |
|---|---|---|---|
| H1 confirmatory | no | no (synthetic) | minutes–1 h, CPU |
| H7 model-free part (mock = legibility oracle) | no | no | minutes |
| H7 with Qwen | local Ollama, no key | no | hours, unattended |
| H6 coverage@K experiments (model-free) | no | V\*Bench (on disk) | ~10 min per arm |
| H6 full V\*Bench + budget sweep, HR-Bench rerun | local Ollama, no key | on disk | hours, unattended |
| H4 / M11 human scanpaths | **no** | **MIT1003: 3 zips, ~1 GB** (`ALLSTIMULI`, `ALLFIXATIONMAPS`, `DATA`) | ~1 h CPU; DeepGaze arm optional (torch + weights) |
| M10 replication dossier | **no** | **no** — thesis text (local) + the binary + synthetic stimuli | CPU only |
| H2 controls | no | no (vtest in repo; DAVIS on disk) | minutes |
| H5 control | no | COCO-Search18 (on disk) | ~15 min |

Nothing in this plan needs an API key. Only H6/H7's real-VLM rows need the
local model.

## The runs

### H1 — object-based vs space-based IOR *(the centerpiece; do first)*

- **Instrument fixes:** `eval/dynamic_ior.py --seeds N --seed0 S` with the
  regime presets (standard / fast / occlusion) and paired intervals; a revisit
  metric that does not saturate once coverage is complete (*redundant-revisit
  rate over the whole video*: share of fixations on an object attended within
  the last *T* frames while another has been unattended longer); off-object
  fixations counted, not dropped.
- **Arms:** greedy · spatial-ior · **motion-compensated spatial-ior** (shift
  the inhibition map by the attended cluster's velocity — the strengthened
  loser) · object-ior (thesis) · object-ior + aids · object-ior + persistent
  identity · object-ior with ground-truth identity (ceiling).
- **Prediction to write down:** object-ior + persistent identity beats plain
  spatial-ior on latency in *fast*; ties in *standard*; **does not beat
  motion-compensated spatial-ior anywhere** unless identity is near-perfect.
  Refuted if it beats the compensated baseline at p < 0.05 paired in any
  regime.
- **Second figure (the paper-B figure):** object-IOR advantage vs measured
  ID-switch rate across the tracker ladder (thesis → +motion → +appearance →
  persistent → kalman-mot → ground truth). Turns "only as good as its tracker"
  into a dose–response curve, and finally uses the M10b backends.
- **Verdict forms:** "supported under condition: ID-switches < x per object
  per 100 frames" or "refuted: a spatial baseline with motion compensation
  matches it".

### H7 — object files as a video token cache

- **Model-free first** (mock, 30 test seeds, frozen `attend_proto.yaml`):
  arms frames-uniform · random crops · **colour-keyed dedup** (one crop per
  new colour cluster; no object files) · space-ior with **decaying** location
  memory (radius × decay tuned on dev seeds) · object-ior · gtid · oracle.
- **The discriminating stimulus:** `--same-colour` scenes (all disks one
  colour, question keyed by code position/order of arrival or by a per-disk
  glyph), where an appearance hash cannot carry identity and spatiotemporal
  continuity must. This is the experiment that makes H7 about *object files*.
- **Then Qwen** on the arms that survive, same 30 seeds.
- **Prediction:** on distinct-colour scenes colour-keyed dedup ≈ object-ior
  (both near oracle); on same-colour scenes object-ior > colour-keyed and
  > decaying-location; the gap shrinks with ID-switch rate. Refuted if
  decaying-location ties object-ior on same-colour scenes.
- **Real video** stays open as a separate question (paper A go/no-go): first a
  *model-free legibility check* on a candidate benchmark (HRVideoBench,
  EgoTextVQA, SoccerNet-GSR) — does uniform downsampling at budget destroy the
  evidence region? — and only then VLM time.

### H6 — attention as a VLM token budget (stills)

- ~~Rerun the HR-Bench pilot with the option order fixed~~ — done 2026-09-19
  (uniform fell 0.80 → 0.50 on "cross": the bias was real). The full 200-item
  splits remain; the 8K run needs the machine to itself (a 27B model plus 8K
  images was stopped once for low memory — `--resume` picks up).
- **Model-free coverage@K on all 191 V\*Bench items**, per crop source, against
  a **random-fixation baseline** — make it a permanent arm of
  `eval/vlm_frontend.py`. First measurement (2026-09-19): the current pipeline
  equals random (top-3 0.068 vs 0.069), so the bottom-up `fovea` arm has been
  random crops. Sources to test, in order: tiled native-resolution saliency with
  the stage-1 defects fixed → open-vocabulary detector → text/face features
  (`docs/FEATURE_ASSESSMENT.md`). This is where new saliency features are
  evaluated, without VLM cost.
- Full V\*Bench with Qwen: full-res · uniform · fovea (best bottom-up source) ·
  fovea-td · oracle, at **three budgets** (the roadmap's curve), paired by item.
- **Verdict forms:** "refuted for bottom-up sources (coverage ≤ random + ε);
  supported for question-conditioned sources at budgets ≤ b" — or plain
  refuted. Either is publishable next to the oracle ceiling.

### H4 — scanpaths vs humans (M11)

Download MIT1003 → rebase `module/scanpath-human` → run
`eval/scanpath_vs_human.py --mit1003` → table on the floor↔ceiling axis, the
WTA-vs-object-file readout ablation → merge. Add the stage-1 face channel
ablation when the real run happens (standing reminder).

### H2, H5 — one control each, then close

- **H2:** random-ROI and frame-difference-ROI gating at the same pixel budget
  on vtest; per-track recovery instead of per-detection. Verdict wording drops
  "near-full-frame": *"recovers x% of tracks at y% of pixels; z points above a
  motion-gated baseline"* (or not).
- **H5:** pooled, category-agnostic prior as a control; random sample of
  validation trials instead of the first 150. If the pooled prior recovers most
  of the gain, the finding is centre bias.

### H3 — park

No stereo-video work is planned; mark H3 and M15 *deferred* in the roadmap.

## Order

1. H1 instrument + confirmatory run (no dependencies; CPU only).
2. H6 model-free coverage@K with the random baseline — decides how much to
   invest in features.
3. H7 model-free (baselines, same-colour scenes); then the Qwen rows for H6/H7
   and the HR-Bench rerun as one unattended batch.
4. M11 as soon as MIT1003 is on disk.
5. H2/H5 controls.
6. M10 replication dossier — writing-heavy; runs in parallel with everything.
7. `docs/STATUS.md`: one table (hypothesis · verdict · n · date of
   confirmatory run · doc) that the README links instead of restating numbers.
