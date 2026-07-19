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
