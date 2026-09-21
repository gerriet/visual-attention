# Paper B — claims and their evidence

Every quantitative statement in `main.tex`, with the document it is copied from,
the command that prints it, and the data it was measured on. State of the
repository: tag `replication-v1`. Rule: if a source document changes, change the
row here first, then the paper. A claim without a row does not go into the paper.

Status: **ok** — measured, documented, reproducible by the command ·
**check** — true as far as documented, but something needs a second look before
submission · **open** — not yet measured.

## Study 1 — the dossier

Source for all rows: `docs/replication/REPLICATION_DOSSIER.md`; command
`eval/replication.py --all` (CPU, ~15 min; one experiment: `--only <name>`).

| # | Claim in the paper | Value | `--only` | Status |
|---|---|---|---|---|
| 1.1 | 24 findings: 22 replicated, 2 partially | summary table (24 rows; #4 and #10 partial) | — | ok |
| 1.2 | Eccentricity on eq. 5.6: 2:1 ellipse 0.354 vs 0.360, 8:1 bar 0.934 vs 0.940; disk, square 0.000 | finding 1 | `shapes` | ok |
| 1.3 | Stretching: eccentricity within 0.02 of eq. 5.6; symmetry 0.67 → 0.15 monotone; before the port 0.67 → 0.50 → 0.87 → 0.82 | finding 2 | `ecc-variation` | ok (the "before" numbers need commit `a506646~1` to regenerate) |
| 1.4 | Eccentricity maximum on its object up to σ ≈ 38; colour and symmetry at every level up to σ ≈ 115; 5 seeds per level | finding 3 | `gray-noise`, `colour-noise` | check — 5 seeds per level is thin for a paper figure; raise to 20 |
| 1.5 | Growth threshold: map correlation 0.64–0.78 in 0.5–0.75, 0.06 at 0.4 | finding 4 | `grow-threshold` | ok |
| 1.6 | Merge threshold: correlation ≥ 0.88 over 4–48 | finding 5 | `merge-threshold` | ok |
| 1.7 | Colour response 0.00 → 0.32 → 0.81 → 0.98 → 1.00 | finding 6 | `colour-variation` | ok |
| 1.8 | cc_add / cc_mult tables; ball and picture stay the salient regions for cc_add 4–16, cc_mult 5–20 | finding 8 | `colour-thresholds` | ok |
| 1.9 | Exclusivity: odd/common 1.03 → 1.51 (bars; predicted 1.1⁴ = 1.46), 1.03 → 2.13 (colour) | finding 9 | `exclusivity` | ok |
| 1.10 | Stereo: disparity exact for d = 0…14; RDS recovered; 9.99 px at σ ≈ 64, 9.91 at σ ≈ 90 (92% within 1 px) | findings 11–13 | `stereo-distance`, `stereo-rds`, `stereo-noise` | ok |
| 1.11 | Orientations under noise: 0.84 → 0.92 → 0.91 (σ 90) | finding 14 | `stereo-orientations` | ok |
| 1.12 | Variance threshold, synthetic: ≤ 5 admits the textureless band (89% wrong), ≥ 20 rejects it, ≥ 40 starts dropping correct pixels | finding 15 | `stereo-variance` | ok |
| 1.13 | Middlebury (Tsukuba, Venus, Sawtooth): 82–85% within 1 px at threshold 3; wrong share 12–20% at any threshold; 81–86% correct on low-texture pixels at threshold 0 | finding 15b | `stereo-real` (needs `data/Middlebury`) | ok |
| 1.14 | Finding A: lone disk maximum off the object for radius 12/24/40 before; after the port peaks 0.43/0.82/0.99 at the centre, single edge 0.08 | finding A | `shapes` + `tests/test_thesis_features.cpp` | check — the paper wants the ratio one-sided : closed contour stated; the header says "about a quarter", the dossier "about half". Measure once and use one number everywhere |
| 1.15 | Finding B table: merge at 5 / re-split at 13; noise 0.000 up to 1.2, 0.7% at 1.4; hysteresis 0.55 / 0.25; convergence 10 cycles; port defaults never merge, switch at 0.35 | findings 16–21 | `field` (wraps `build/field_dynamics --params esab2\|port`) | ok; depends on the input blob shape (σ = 3 px), say so |
| 1.16 | Tracking table (cycles needed by amplitude × speed) | finding 20 | `field` | ok |
| 1.17 | Text vs code: V\*Bench top-3 0.152 vs 0.105 (−0.047 [−0.099, +0.010]); COCO-Search18 found@10 0.37 vs 0.34; MIT1003 own-scanpath ScanMatch 0.545 vs 0.529 (−0.016 [−0.023, −0.008]) on 200 images | "Text or code?" | `text-vs-code`; `eval/coverage_table.py`, `eval/coco_search.py --config`, `eval/scanpath_vs_human.py --profile` | ok; only the MIT1003 and H1 rows are used in the paper |
| 1.18 | Appendix A parameter table | "The parameters of the original system" | — (values read from the non-public sources) | check — **author to confirm each value against the sources by hand**; this table is the paper's only window onto them |

## Study 2 — H1

Source: `docs/DYNAMIC_IOR_STUDY.md`. Profile `configs/thesis/attend.yaml`.

| # | Claim | Value | Command / data | Status |
|---|---|---|---|---|
| 2.1 | Predictions committed before the first confirmatory run | 4 predictions with refutation criteria | commit `ab7d75e` (2026-09-20); results commit `a568fa0` | ok |
| 2.2 | First confirmatory run: latency win in all regimes, staleness not; off-object 0.31–0.34 | table "Outcome" | `eval/dynamic_ior.py --regime all --seeds 30 --seed0 1000`; `results/h1_confirmatory` | ok (appendix C) |
| 2.3 | Re-run, Table 4 of the paper: every Δ vs `spatial-ior` excludes zero in favour of the object arms; off-object 0.00 | table "The re-run" | same command, `--seed0 2000`; `results/h1_confirmatory_symfix`; commit `7b944e4` | ok |
| 2.4 | vs strengthened baseline: latency −0.36 / −1.73 / −0.50, staleness −0.39 / −1.11 / −0.36 | same section | same | ok |
| 2.5 | Third block, both parameter sets, `object-ior+id − spatial-ior` latency: −1.18 [−2.11, −0.42] / −4.82 [−6.13, −3.59] / −1.43 [−2.20, −0.77] (text); −1.17 [−2.10, −0.41] / −4.41 [−5.69, −3.18] / −1.52 [−2.54, −0.68] (system); staleness −0.79 / −2.76 / −1.04 and −0.73 / −2.56 / −1.08, all excluding zero | dossier, "Text or code?" (rounded there); `results/h1_text_vs_code_attend.log`, `…_attend_esab2.log` | `--seed0 3000 --arms spatial-ior,object-ior,object-ior+id`, `--config configs/thesis/attend.yaml` / `attend_esab2.yaml` | ok — this block has no `spatial-ior-mc` arm; say so |
| 2.6 | Labels per object, occlusion: 2.8 (thesis correspondence) vs 1.5 (persistent identity) | re-run table | same as 2.3 | ok. Note: dossier finding 10 still quotes the *first* run's 3.6 / 1.7 |
| 2.7 | Scores of the four predictions in the re-run: 1 holds, 2–4 refuted | "The predictions, scored again" | — | ok |
| 2.8 | Exploration phase reported space-based IOR "at least as good, usually better" | sections before "The confirmatory run"; `docs/adr/0004` | ≤ 6 seeds, no intervals | ok (history; quote, do not re-measure) |
| 2.9 | **Real video (DAVIS 2017)** | — | not built | **open** — see README, "Still to run" |

## Study 3 — H4

Source: `docs/SCANPATH_VS_HUMAN.md`. Command
`eval/scanpath_vs_human.py --mit1003 --profile thesis-extended=configs/thesis-extended.yaml --out results/scanpath_vs_human`
(+ `--resume --refresh-field` for the field-parameter update). Data: `data/MIT1003` (stimuli, fixation maps, raw DATA archive).

| # | Claim | Value | Status |
|---|---|---|---|
| 3.1 | Predictions committed before the run | commit `36761d8` (2026-09-20); six images had been looked at while debugging — say so | ok |
| 3.2 | Table 5 (unmatched, 1003 images, 10 fixations) | "Outcome" table | ok |
| 3.3 | Table 6 (matched length k, 964 images) | "Update" table; commit `caae285` | ok — a post-hoc control, labelled as such |
| 3.4 | thesis-field@k − random@k +0.076 [+0.072, +0.081]; − center@k −0.101 [−0.105, −0.098]; − thesis-wta@k +0.003 [+0.000, +0.006] | same | ok |
| 3.5 | objfile − wta −0.022 [−0.025, −0.020] | paired table | ok |
| 3.6 | ScanMatch: constant centre 0.754 vs inter-observer 0.759; direction 0.501 vs 0.630 | paired table | ok; novelty of the observation **unchecked** (literature) |
| 3.7 | Unmatched score of a 4-fixation path falls below random (0.544 vs 0.636) while position/shape/length rise (+0.037 / +0.019 / +0.055) | "Update" | ok |
| 3.8 | I-DT fixation extraction assumptions (240 Hz, column order) | "Honest caveats" | check — state the dispersion and duration thresholds in the paper |

## Statements that are not numbers, and who vouches for them

| Statement | Basis |
|---|---|
| "The summation core was a faithful port; what followed it was not" (Finding A) | comparison against the original `feature/symmetry.C`, not publishable — author to confirm |
| "The port took the field parameters from the default arguments of the setter functions" (Finding B) | original `nf2d.h` / `esab2.C`, not publishable — author to confirm |
| "The thesis's laboratory images did not survive" | author |
| "The thesis argues its central claim from demonstrations, without statistics" | thesis ch. 9 — author to give page references |
| Rights to the original sources are not the author's | author |
