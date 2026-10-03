# Status — one row per hypothesis

*Last updated 2026-10-03. This is the single place that says where each
hypothesis stands. The README links here instead of restating numbers, so a
result cannot drift between the two.*

The project's own bar, from `docs/HYPOTHESIS_CLOSURE_PLAN.md`: a verdict counts
as **confirmatory** only if it comes from a frozen config, fresh seeds never
touched before, a prediction written down in advance, paired statistics over the
independent unit, and n ≥ 20. Anything else is a **first pass** or a **pilot** —
useful, directional, and not quotable as a result. The distinction is the point
of this table.

## Replication track

*Closed and frozen at the tag `replication-v2` (2026-09-22). Runs on
`configs/thesis/`; see [ADR-0005](adr/0005-two-tracks-replication-and-modern.md).*

| | Hypothesis | Verdict | Basis | n | Date | Detail |
|---|---|---|---|---|---|---|
| **H1** | Object-based IOR beats space-based in dynamic scenes | **Supported**, under stated conditions | confirmatory | 30 scenes × 3 regimes, on four independent blocks (seeds 1000/2000/3000/4000) | 2026-09-21 | [DYNAMIC_IOR_STUDY.md](DYNAMIC_IOR_STUDY.md) |
| **H1** | …the same question on real video | **Inconclusive**, for a stated reason | confirmatory design, uninformative outcome | 17 DAVIS-2017 validation sequences | 2026-09-22 | [DYNAMIC_IOR_STUDY.md](DYNAMIC_IOR_STUDY.md) |
| **H4** | Scanpaths land above random/centre toward the human ceiling | **Split: above random weakly; _not_ above centre; no stage-2 ordering benefit** | confirmatory | all 1003 MIT1003 stimuli | 2026-09-21 | [SCANPATH_VS_HUMAN.md](SCANPATH_VS_HUMAN.md) |
| **M10** | The thesis's findings, figure by figure | **22 of 25 replicate, 3 partially** | dossier | 25 findings | 2026-09-22 | [REPLICATION_DOSSIER.md](replication/REPLICATION_DOSSIER.md) |

**The conditions on H1 matter and are not footnotes.** With the
reimplementation's segment-based first stage it holds at every speed tested.
With the thesis's *own* first stage — the neural field — it holds **inside the
field's tracking range** (a few field pixels per frame, occlusion included) and
**fails outside it**, where object-based inhibition is not merely no better but
worse than the location tag it replaces. The field's capacity is a second
condition. On real video the test could not decide: 42–61% of fixations land on
salient things the ground truth does not annotate, so no inhibition-domain
comparison survives. That is a limitation of the test, not evidence either way —
and it is what [M21](V3_ROADMAP.md) exists to fix.

## Modern track

*Open-ended, judged by usefulness. Runs on whatever works best; the thesis
profile is one arm, not the subject.*

| | Hypothesis | Verdict | Basis | n | Date | Detail |
|---|---|---|---|---|---|---|
| **H2** | Gated recognition reaches near-full-frame accuracy at a fraction of the compute | **Supported** — 51% of full-frame detections at 5.8% of the pixels | first pass; one control open | vtest sequence | 2026-07-19 | [GATED_RECOGNITION.md](GATED_RECOGNITION.md) |
| **H5** | A priority map beats salience-only for task-driven search | **Supported** — the top-down channel is decisive; the history term buys hold, not acquisition | first pass; one control open | synthetic search + COCO-Search18 | 2026-07-20 | [PRIORITY_MAP.md](PRIORITY_MAP.md) |
| **H6** | Attention as a VLM token budget | **Partly** — right-region crops beat full resolution at a third of the tokens, but bottom-up crops find the target on 10% of V\*Bench items (65% question-conditioned) and still lose to a same-budget uniform downsample | **pilot** (20 items per V\*Bench category, 10 per HR-Bench category) | pilot scale | 2026-09 | [VLM_FRONT_END.md](VLM_FRONT_END.md) |
| **H7** | Object files as a video token cache | **Supported on synthetic** (0.95 identity-keyed vs 0.80 location-keyed vs 0.27 frames, chance 0.25); **ties on DAVIS** at 480p and full resolution | **first pass — reported on the seeds the mechanisms were developed on (0–9)** | 10 synthetic scenes × 6 questions = 60; 30 DAVIS val sequences | 2026-09-18 | [VLM_VIDEO.md](VLM_VIDEO.md) |
| **H8** | Learned top-down weights beat hand-built channels | not started (M20) | — | — | — | [V3_ROADMAP.md](V3_ROADMAP.md) |
| **H9** | A real object source makes the object/space question decidable | not started (M21) | — | — | — | [V3_ROADMAP.md](V3_ROADMAP.md) |
| **H10** | Scale-specific objectness pays for the controller | not started (M22, later) | — | — | — | [V3_ROADMAP.md](V3_ROADMAP.md) |

## Parked

| | Hypothesis | Why |
|---|---|---|
| **H3** | Depth priority: near/approaching objects attended earlier with the stereo channel and 3D field | Parked 2026-09-20 in favour of closing the open hypotheses rather than opening another. The machinery exists and is exercised — stereo features, a 3D field that now integrates over time, stereo streams in `--attend` — so this is a question waiting for a run, not for code. |

## What is honestly still owed

Ranked by how much a reviewer would care:

1. **H7's confirmatory run.** The headline is on the development seeds. It needs
   fresh seeds, a frozen config, paired intervals, and the baselines the study
   already names (colour-keyed dedup, decaying location memory, random crops).
   This is the biggest gap between what is claimed and what is established.
2. **H6 at full scale.** Everything is pilot-scale: the full V\*Bench run, three
   budgets, `fovea-random` as the floor, HR-Bench at full size. No credentials
   needed — the backend is a local Ollama model.
3. **The H2 and H5 controls**, one each, both cheap and both already specified
   in the closure plan.
4. **Two cheap things that would move H4**, and are named in its own verdict: a
   centre prior on the priority map (the `top_down_map` slot takes one as it is)
   and a face channel at weight 0 by default — MIT1003 is full of faces and text.

Nothing here needs an API key or a dataset that is not already on disk.
