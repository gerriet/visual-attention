# ADR-0005 — Two tracks: a finite replication, an open-ended modern system

**Status:** proposed (2026-09-20) · awaiting Gerriet's decision on the open points below

## Context

The repository serves two goals that pull in different directions:

1. **Replication.** Reproduce the 2004 dissertation: its model, and then its
   findings (M10). This goal is *finite* — once the dossier is written and the
   profile reproduces the thesis behaviour, it is done and should stop moving.
2. **A modern system on the thesis's core ideas** — two-stage selection, object
   files, object-based inhibition, attention as a controller for expensive
   perception. This goal is *open-ended*: it should be as useful as possible and
   may one day be far from the thesis (learned features, detectors as top-down
   sources, other trackers).

Until now one rule covered both: *"the thesis model is the protected default;
everything new is opt-in"*. It served the replication well and has started to
cost the modern system:

- The default path cannot improve. The review found the default stage 1 no
  better than random at choosing crops on V\*Bench — and "fixing the default"
  was a contradiction in terms while the default had to stay the thesis.
- "Thesis" and "default" were never the same thing anyway: `default.yaml` runs
  Itti–Koch colour, intensity and orientation — none of them in the thesis —
  while `thesis.yaml` is the dissertation profile. The 2026-09 feature audit
  showed what the ambiguity cost: a colour feature that was not the thesis's
  sat in the thesis profile unnoticed for a year, because no test asked "is this
  the thesis?" — only "did this change?".
- Every study had to decide ad hoc which profile it was about. H1 (a claim *of
  the thesis*) and H6/H7 (a claim *about usefulness*) ran on the same configs.

## Decision (proposed)

Separate the tracks by **profile, tests and definition of done — not by
repository or branch.** They share one codebase, one registry architecture and
one interchange format; that sharing is the project's main asset (every modern
component can be A/B-ed against the thesis one, which is what both papers need).

**1. The replication track is a named, frozen profile.**
- `configs/thesis/` holds the dissertation profiles (`thesis.yaml`, the stereo
  and second-stage variants). Only components that implement the dissertation
  may appear in them, and each carries a provenance note (thesis section +
  original source file).
- Its guard is *fidelity tests*, not only goldens: behavioural tests that check
  a component against the thesis's own statements (`tests/test_thesis_features.cpp`
  is the first), and the M10 dossier's scripted experiments.
- **Definition of done:** the M10 dossier has a verdict per finding; H1 has its
  confirmatory run; the fidelity tests pass. Then the profile is tagged
  (`replication-v1`), its goldens are frozen, and changes to thesis components
  need a stated replication reason (a found deviation, a bug). After that the
  track costs nothing but CI time.

**2. The modern track owns the default.**
- `configs/default.yaml` becomes "the best system we know how to build", free to
  change whenever a measured improvement says so. Its goldens are regenerated as
  a matter of course; its guard is the *benchmarks* (target coverage vs random,
  the H6/H7 harnesses), not similarity to 2004.
- New components are judged by usefulness and cost, with the thesis component as
  the baseline arm — never by fidelity.

**3. Every study declares its track.** H1, H4 and M10 are replication-track: they
run `configs/thesis/*` and their question is "was the thesis right?". H2, H5, H6,
H7 are modern-track: they run whatever works best and their question is "is this
useful?", with the thesis profile as one arm. A study document states its track
in its header.

**4. Shared code stays shared, and track-neutral.** Registries, pipeline,
interchange format, object files, behaviors. A modern need never edits a thesis
component in place — it adds a sibling (`color` next to `color-munsell`,
`kalman-mot` next to `neural-field`, proto-objects next to thesis segmentation).
This is already the pattern; the ADR makes it the rule.

## Consequences

- Paper B is written from the replication track and cannot be destabilised by
  modern work; paper A is free to replace stage 1 entirely.
- The README's "thesis model is the protected default" becomes "the thesis model
  is a protected *profile*".
- One-time cost: moving configs, re-pointing tests and docs, and one more golden
  regeneration when the default starts to move.
- Risk: the tracks drift apart until the shared code serves neither. Mitigation:
  the thesis profile stays an arm in every modern benchmark, so it is exercised
  by the same harnesses forever.

## Open points for the decision

1. **Second-stage configs.** `configs/attend*.yaml` (used by H1 *and* H7) still
   run the Itti-style `color`. Split them into a thesis variant (`color-munsell`,
   exclusivity) for H1 and a modern variant for H7?
2. **Where the modern default starts.** Keep today's five-feature default as the
   starting point, or start from the best measured crop source once the coverage
   experiments are in?
3. **Naming.** `thesis` / `modern` is already taken by `configs/modern.yaml`
   (which is merely the thesis features plus Itti's, and misleads); rename it
   when the split happens.
4. **How strict is "frozen".** Tag + frozen goldens (proposed), or additionally a
   CI job that fails when files under a `thesis/` path change without a
   `Replication:` line in the commit message?
