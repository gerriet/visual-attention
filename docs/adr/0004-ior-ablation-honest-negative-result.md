# ADR-0004 — Inhibition of return as a controlled ablation, and an honest negative result

**Status:** accepted · **Date:** 2026-07

## Context

The thesis's reason for its symbolic second stage is hypothesis **H1**: in
multi-object dynamic scenes, *object-based* inhibition of return (IOR) — inhibit
the recently attended object and let that inhibition *follow* it as it moves —
beats *space-based* IOR (inhibit the location) and no IOR at all. This is the
claim the whole two-stage model exists to justify. It had never been isolated
and measured. The failure mode to avoid was the tempting one: assume H1, tune
parameters until a single plot agrees, and present the assumption as a result.

## Decision

Two coupled decisions.

**1. Realize IOR as three behaviors that differ only in what they inhibit.**
`greedy` (nothing), `spatial-ior` (recently attended *locations*, decaying),
`object-ior` (recently attended *objects*, decaying) — sharing every other stage
(features, fusion, object-file tracking). See [ADR-0002](0002-registry-config-driven-strategies.md).
The object-vs-space question then becomes a controlled ablation over one variable
(`src/system/behavior.cpp`, scored by `eval/dynamic_ior.py`), not a fork of the
pipeline.

**2. Report whatever the ablation shows, including the negative half.** Do not
tune object-IOR until it wins; measure it, keep the seeds honest (multiple seeds,
never a cherry-picked single scene), and record where each regime actually lives.

## Consequences

- **IOR ≫ no-IOR — confirmed, decisively.** `greedy` is worst on every metric at
  every speed: it perseverates on the strongest object and wastes ~85% of
  fixations on revisits. The first half of H1 is strongly supported.
- **object-IOR does *not* beat space-IOR by default — H1 is refined, not
  confirmed.** Both reach full coverage; across seeds and regimes space-based IOR
  is at least as good and usually marginally better on the exploration metrics
  (coverage, revisit-waste, latency). The thesis's headline advantage does not
  robustly materialize here.
- **The reason is structural, and worth stating plainly: object-IOR is only as
  good as its tracker.** Every time correspondence switches an object's label —
  under fast, dense, or crossing motion — the object-keyed inhibition no longer
  applies and the object is re-fixated as if new. Space-IOR has no such failure
  mode: it inhibits locations, which are always "tracked." Object-IOR's
  theoretical edge is spent paying for tracking errors.
- **Better tracking narrows but does not close the gap.** Adding motion-predicted
  and appearance-based correspondence — the DeepSORT idea using features the
  pipeline already computes, opt-in and default-off to preserve thesis fidelity
  (`ObjectFileStore::Config::{motion_prediction, appearance_matching}`) — cut
  object-IOR's wasted revisits ~3× (0.413 → 0.126), confirming label-switches were
  the bottleneck. It reached *nearly tied* — not a win. The exploration
  metrics reward *spreading attention over the scene*, which "don't look where you
  just looked" does inherently well.
- **Where a genuine object-IOR win should live — the asymmetry that is the point.**
  Object memory can be *persistent*; spatial memory *must* decay, because
  locations get reused. Exploration metrics barely exercise this. Settling H1 needs
  identity-centric metrics (ID-persistence, same-label occlusion recovery,
  redundant re-inspection under a *persistent* object model vs a decaying spatial
  one) — the queued identity-metrics work (H1 in [../V3_ROADMAP.md](../V3_ROADMAP.md)).
- **Cost: an honest, sharper claim instead of a headline.** What we can defend is
  *"IOR ≫ no-IOR; object- vs space-based is regime-dependent and tracking-limited,
  and naive object-IOR does not win by default"* — more useful than the thesis's
  unqualified claim, and the framework can now say exactly where each regime lives.
  Full data and reproduction: [../DYNAMIC_IOR_STUDY.md](../DYNAMIC_IOR_STUDY.md).

## Update (2026-09)

The identity-centric follow-up ran as part of M19: opt-in *persistent identity*
in the object-file store. In a six-seed pilot per regime it takes object-IOR
from clearly worse to level with space-IOR, and ahead on latency at high speed
(`docs/DYNAMIC_IOR_STUDY.md`, "Persistent identity"). Six seeds without
intervals: a direction, not yet a result — the statement above ("nearly tied —
not a win") stands until the confirmatory run (≥ 20 fresh seeds per regime,
paired intervals; `docs/HYPOTHESIS_CLOSURE_PLAN.md`). Note for that run: since
this ADR the object arm has gained several mechanisms and the spatial arm one
knob, so the fair comparison adds a strengthened spatial baseline
(motion-compensated spatial IOR).

## Update (2026-09-20): the confirmatory run

Run on the thesis profile with 30 fresh scenes per regime, predictions written
down beforehand, and the strengthened spatial baseline (`spatial-ior-mc`). The
statement above — "across seeds and regimes space-based IOR is at least as good
and usually marginally better" — **does not survive it**: it rested on a revisit
metric decided by a handful of fixations, ≤ 6 seeds, and a stage 1 that was not
the thesis's. Object-based IOR reaches new objects sooner in every regime
(supported for the quantity H1 names; at high speed only with held identity —
"only as good as its tracker" stands), and does not win on sustained coverage.
The decision recorded here held up better than the result: *report whatever the
ablation shows* is what made the correction possible.

## Update (2026-09-21): the negative result was about the reimplementation

The replication dossier found the symmetry feature defective (it answered beside
objects); after porting it, the confirmatory study on fresh seeds supports H1 in
full — object-based IOR beats space-based IOR on latency and on sustained
coverage in every regime, with the thesis's own correspondence
(`docs/DYNAMIC_IOR_STUDY.md`). Together with the colour and eccentricity ports
this means every measurement this ADR summarizes was taken on a stage 1 that was
not the thesis's. The ADR's decision stands and is the reason the record is
straight: the ablation design made the comparison cheap to repeat, and reporting
the negative plainly is what kept anyone from building on a tuned positive.
The lesson added: *an honest negative about a replication is a claim about the
replication until its fidelity has been tested* — which is what ADR-0005's
fidelity tests and the dossier are for.
