# The WAPCV 2003 paper, read against the thesis and the reimplementation

*2026-09-21. Track: replication (docs/adr/0005). The paper is not in the
repository (proceedings, third-party rights); it was read from the author's
copy of the full proceedings volume.*

> G. Backer, B. Mertsching: **Two selection stages provide efficient
> object-based attentional control for dynamic vision.** In: L. Paletta, G. W.
> Humphreys, R. B. Fisher (eds.), *Proc. International Workshop on Attention and
> Performance in Computer Vision (WAPCV 2003)*, Graz, Austria, 3 April 2003 (in
> conjunction with ICVS'03; co-sponsored by Joanneum Research and ECVision),
> pp. 9–16.

It is the last publication from the thesis work: written after the thesis's
first version (end of 2002), before its revision (2003) and publication (May
2004). Eight pages, in English — the one place where the model's second stage
and its evaluation are described for a reader who cannot read the thesis.

## 1 · What the paper adds to, or says differently from, the thesis

Mostly it condenses. The differences that matter here:

| Topic | Thesis | WAPCV 2003 |
|---|---|---|
| Name of the middle stage | "semiattentiv" appears (5 places), the architecture is described as two selection stages | **semiattentive computations** is the organising idea: three stages — preattentive (features, saliency), semiattentive (neural field, tracking, object files = the world model), attentive (FOA, recognition) — and the abstract is built on it |
| What an object file holds | §7.2 | stated compactly: position, size, trajectory, history of selection, mean feature values, results of high-level computations |
| Correspondence of object files and activity clusters | §7.2.3 in detail (below) | one sentence, but unambiguous: "**spatial distance and similarity of features** inside the activity areas determine the correspondence", within boundaries given by the field's tracking limits |
| Exploration behaviour | §8 | "priority levels … according to the time they were last selected. Unselected items receive the highest priority. Within a priority level … ordered by their saliency. Dynamic IOR for moving objects is implicit to this behavior" — this is exactly `Exploration::select_focus` |
| The comparison experiment (below) — the conventional model's selection | "Maximumssuche" on the shared saliency map | **blurs the input first** ("mimicking the selection by neural fields and finding the center of the input"), then takes the maximum after inhibition. A detail the thesis omits and a replication needs |
| … its optimum | "6.25 recognized objects" for the conventional model at 5 + 5 objects (3 frames per object), 5 for the new model (4 frames) | only the new model's optimum is given |
| Scanpath of Fig. 6 / thesis ch. 9 demo | — | claims an **optimal dynamic scanpath**: "the first re-checking of an item occurs only after all objects were subject to focal attention" — a checkable property |
| Outlook | — | "To provide even better object candidates by the first selection stage, **a segmentation process based on the feature computations would be a suggestive extension**" |
| Depth | multi-scale, eq. 5.13–5.16 | "Results from the lower resolution scales limit the correlation range"; saliency monotonic with disparity ("first react to close objects") — same as the thesis; the multi-scale part is what the port leaves out |

Nothing in the paper contradicts the thesis. Its reference list also names where
parts of the thesis were published in English: the field dynamics (Abb. 6.4–6.10
of the thesis) in Backer & Mertsching, ICANN 2002, pp. 1237–1242; the simulation
framework evaluation in Journal of WSCG 10 (2002), pp. 32–39; the gaze-control
system in TPAMI 23(12), 2001.

## 2 · What reading it exposes in the reimplementation

Three things. None was visible from the dossier, because the dossier was built
from chapters 5 and 6.

### Finding C — the thesis has a quantitative test of its central claim, and the dossier does not contain it

Paper §IV-B, Fig. 4–5; **thesis §9.2, Abb. 9.1–9.2**. The dossier lists §9.3.2
and §9.3.3 as "qualitative demonstrations, not attempted" and misses §9.2
entirely; `paper/replication/main.tex` said the claim "was argued from a handful
of demonstrations". That is wrong, and it is corrected there.

The experiment, as specified by the two sources together:

| | |
|---|---|
| Input | a simulated 2D master map per frame — no features. Objects are 5 × 5 px squares, pairwise ≥ 14 px apart at every moment; static, or moving on a straight path by at most 2 px in x and in y per frame. Uniformly distributed noise, half the objects' amplitude |
| Conditions | 1, 3, 5 static objects × 0–5 dynamic objects; **50 runs of 40 frames** per data point |
| Task | keep a world model: identity and position of as many objects as possible, at every moment. Recognition is simulated: correct identity for the attended pixel, after a fixed cost in frames |
| Conventional model (after Koch & Ullman) | blur, inhibit, take the maximum; recognition takes **3 frames**; then mark an **8 × 8** area in an inhibition map that decays **×0.8 per frame** (chosen favourably: large and slow, which the ≥ 14 px spacing allows); the identity is bound to **the location where it was recognized** |
| Two-stage model | the simplest variant: **one 2D neural field of the local-inhibition type** → activity clusters → object files; behaviour "Exploration"; recognition takes **4 frames** (one extra, to charge for the field); the identity is bound to the object file and follows its cluster |
| Measures | mean number of recognized objects over all frames and runs (an object whose believed position is off by more than 20 px counts as not recognized); mean position error of the recognized ones |
| Result | without dynamic objects the conventional model is ahead (faster recognition); with any dynamic object the two-stage model is ahead and "scales much better"; position error **< 0.5 px** for the two-stage model in every condition, **0.5–5 px** for the conventional one, ≥ 3 px once dynamic objects outnumber static ones. At 5 + 5 the two-stage model "nearly reaches" its optimum of 5 |

Why it matters beyond completeness: this experiment measures something the
project's H1 study does not. H1 (`docs/DYNAMIC_IOR_STUDY.md`) scores *where
attention goes* — latency, staleness. §9.2 scores *what the system knows
afterwards* — a world model whose identities stay attached to moving objects.
That is the thesis's actual argument for object files ("extracted information
cannot be bound to moving objects" is the second of its three objections to
conventional models), and it is the property the modern track's H7 (object files
as a video token cache) relies on.

It is fully specified, CPU-only and small. **It should be dossier finding 22,**
replicated with the thesis's numbers as the target: 50 runs per point, both
measures, plus intervals (the original has none).

### Finding D — the "thesis correspondence" arm is less than the thesis's correspondence

Thesis §7.2.3 (single field of local inhibition — the variant in question):

1. candidates within the radius *x_a* that the field's tracking limits imply;
   unique matches are made;
2. for the rest, all assignments are scored **on centroid error and on feature
   distance**; a cluster gets the file that is best on both;
3. merged clusters get a new file pointing at both predecessors, resolved by
   feature comparison after 4 frames;
4. before a new file is created, **the inactive files are searched for a
   correspondence, "primarily by the feature properties"**; inactive files keep
   all their information and are dropped after a maximum age that "is determined
   primarily by the characteristics of task and environment" (§7.2.4).

The reimplementation's default (`ObjectFileStore::Config`) is position only,
with a radius-gated revival of inactive files and ageing after 30 frames; its
header calls that "the thesis's simple nearest-centroid correspondence".
Feature similarity in the match (`appearance_matching`) and feature-led revival
(`persistent_identity`, which also removes the maximum age) exist — as **opt-in
extensions**,
introduced in M12 and M19 and described in the H1 study as "aids" and as modern
additions.

Consequences:

- In the H1 tables the arm labelled "object-ior (thesis)" is a *weakened* thesis
  model, and `object-ior+aids` / `object-ior+id` are closer to what §7.2.3
  describes than to extensions of it. The verdict does not move — every
  object-based arm beats both spatial arms — but the labels and the sentence
  "the thesis's correspondence alone already carries most of the advantage"
  need rewording, and the plain arm becomes the ablation ("position only").
- Dossier finding 10 (Abb. 6.14, identity through occlusion: *partially*, "holds
  only with the extensions, not with the thesis's correspondence") called
  position-only correspondence the thesis's. *Measured since (seeds 4000–4029):*
  §7.2.3 as written gives 2.7 labels per object under occlusion against 2.8
  position-only and 1.4 with persistent identity — the verdict stands, now with
  the right attribution; this note's first guess, that feature-led revival alone
  would reach 1.5, was wrong.
- The two implementations are not identical to §7.2.3 either (mean colour as the
  only feature; a cost sum instead of "best on both criteria"; the colour veto,
  the widening gate and the missing maximum age are additions). A faithful version is a bounded piece of
  work; whether to build it is a replication-track decision (below).

The surviving original sources do not contain the second stage (no object files
or behaviours in them), so the thesis text and this paper are the only record.

### Finding E — in the reimplementation the object files do not come from the neural field

Thesis and paper: the first selection stage *is* the neural field; its activity
clusters — robust, with hysteresis, tracking their input — are what object files
are created for ("tracking is integrated into this first selection stage").
Reimplementation: `AttentionSystem` thresholds and segments the fused saliency
map (35% of its maximum) and corresponds those segments; its header says so
("an approximation of the neural field's activation clusters at this bar"). The
neural field is used for the still-image scanpath (`thesis-field`), not in
`--attend`, so **H1 was measured without the neural field in the loop**.

This is a known, documented approximation rather than a defect, and it happens
to be the variant the paper's last sentence proposes as an extension. But a
replication paper has to say it plainly, and the §9.2 experiment is the natural
place to close the gap: it is defined on the single 2D field → clusters → object
files chain, and `build/field_dynamics` already drives that field on synthetic
input with the dissertation system's parameters.

## 3 · What was done about it (2026-09-21)

All three findings were acted on the same day, on `module/replication-v2`:
finding C → dossier finding 22 (`build/world_model`; partially replicated, the
two-stage model's curve is the thesis's); finding D → `object_files.correspondence:
thesis`; finding E → `attention_system.cluster_source: field` and
`configs/thesis/attend_field*.yaml`; H1 re-measured with both on a fresh block
(`docs/DYNAMIC_IOR_STUDY.md`, "The thesis's own chain"). The section below is the
plan as it stood before.

## 4 · What follows (as planned)

Documentation (done with this note): the paper draft no longer says the claim
was untested; `CLAIMS.md` and the paper's README carry the three findings; the
bibliography entry is checked against the proceedings themselves.

Decisions for the replication track — it is frozen at `replication-v1`, and
ADR-0005 allows changes "with a stated replication reason and a re-run of the
dossier"; all three findings are such reasons:

1. **Replicate §9.2 / Fig. 5** (finding C) — a harness that feeds simulated
   master maps to (a) the conventional model as specified, (b) the 2D field →
   activity clusters → object files → Exploration. Needs a cluster source for
   the second stage that reads the field's activity instead of segmenting
   saliency — which also addresses finding E, as a sibling option, leaving the
   current path untouched. *Recommended: yes, and before the DAVIS check — it is
   the thesis's own experiment, and the paper is weaker without it than without
   real video.*
2. **Correspondence as in §7.2.3** (finding D) — either relabel only (the plain
   arm is an ablation; `+aids`/`+id` are "after §7.2.3, with differences
   listed"), or implement the rule as written as a third option and re-run H1
   on a fresh seed block. *Recommended: relabel now; implement together with
   (1), since the §9.2 experiment needs a correspondence anyway; one fresh H1
   block afterwards.*
3. A `replication-v2` tag when (1) and (2) are in, with the dossier at 25
   findings.
