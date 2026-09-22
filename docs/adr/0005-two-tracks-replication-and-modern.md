# ADR-0005 — Two tracks: a finite replication, an open-ended modern system

**Status:** accepted (2026-09-20; proposed the same day, open points decided by Gerriet — see "Decisions")

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

## Decision

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

## Decisions on the open points (Gerriet, 2026-09-20)

1. **Second-stage configs: split — thesis on H1, modern on H7.**
   `configs/thesis/attend.yaml` (colour contrast in MTM, exclusivity) is the
   profile `eval/dynamic_ior.py` runs by default; `configs/attend.yaml` and its
   `attend_identity` / `attend_proto` variants are modern-track and free to
   change. Consequence for the order of work: **optimise the modern stage 1
   before the H7 confirmatory run**, so that H7 is confirmed once, on the system
   it will be reported for — not on a profile that is about to be replaced.
2. **The modern default starts from the best base we can measure**, not from
   the historical five-feature set. The one-at-a-time ablation between the
   profiles (`configs/ablation/`, docs/FEATURE_ASSESSMENT.md) decides it; the
   choice is validated on a second benchmark before it becomes the default.
3. **Naming: "modern" means the best modern profile.** The old
   `configs/modern.yaml` (the thesis features plus Itti–Koch channels — never
   modern, and at chance on V\*Bench) is now `configs/thesis-extended.yaml`;
   `configs/modern.yaml` is reserved for the best profile and changes with it.
   Intermediate variants get descriptive names (`thesis-extended`, …).
4. **"Frozen" = tag + frozen goldens**, as proposed: when the replication track
   is done it is tagged `replication-v1`; no commit-message CI rule.

## The replication track is closed: tag `replication-v1` (2026-09-21)

The definition of done is met — the dossier has a verdict for each of 24
findings (22 replicated, 2 partially; `docs/replication/REPLICATION_DOSSIER.md`),
H1 has its confirmatory run (supported, three independent seed blocks, under the
thesis text's parameters and under the dissertation system's), H4 has its run on
all of MIT1003, and the fidelity tests pass. The state is tagged
`replication-v1`.

From here on:

- **Frozen:** everything under `configs/thesis/`, the components it selects
  (`color-munsell`, `eccentricity`, `symmetry`, `stereo`, `onset`, the neural
  fields, the object-file stage and its thesis behaviors) in their default
  behaviour, and the thesis goldens (`thesis_inputc`, `stereo`, `thesis_attend`).
  A change to any of them needs a stated replication reason — a deviation from
  the thesis found, or a bug — and re-runs the dossier
  (`eval/replication.py --all`).
- **Free:** everything else. Modern-track work adds siblings and new profiles,
  and may change `configs/modern.yaml`, `configs/attend*.yaml` and the defaults
  at will; the thesis profile stays an arm in every modern benchmark.
- Not attempted, and not blocking: the multi-field systems of Abb. 6.11/6.12
  (not implemented), a dynamics experiment for the 3D field (6.13), the
  qualitative ch. 9 demonstrations.

### CI state at the tag

`replication-v1` was cut with the Linux CI red, noticed only after the tag
was pushed: `scanpath_attend` had failed there since the symmetry port
(green locally on macOS). It is a modern-track test — `--attend` under the
compiled-in default profile — and the three frozen thesis goldens pass on both
platforms, so the tag stands. The cause: frame 0 of `motion_seq` is pure noise,
the first focus falls on whichever noise fragment rounding favours, the two
platforms disagree, and the dwell carries the choice through all three frames.
The golden pinned an arbitrary choice (and, regenerated on macOS after the port,
one that never reached the moving patch — the Linux run did). The test now
checks what the sequence determines (`eval/check_scanpath.py`: a focus on every
frame; the moving patch has become an object file). Lesson for the way of
working: look at CI after a push, not only at the local suite.

## Reopened once, with a replication reason: `replication-v2` (2026-09-21)

The freeze allows changes "with a stated replication reason and a re-run of the
dossier". The WAPCV 2003 paper, read against thesis and code
(`docs/replication/WAPCV_2003_NOTES.md`), gave three: the thesis's own
quantitative test of its central claim was missing from the dossier; the
correspondence called "the thesis's" was weaker than thesis §7.2.3; object files
were not formed from the neural field's activity clusters. What changed:

- **Added, as siblings — nothing frozen was edited:** `build/world_model` and
  dossier finding 22; `object_files.correspondence: thesis`;
  `attention_system.cluster_source: field`; `NeuralFieldSelection::track`;
  `configs/thesis/attend_field.yaml`, `attend_field128.yaml`; H1 arms
  `object-ior+7.2.3` and `chain:*`. Defaults, `configs/thesis/{thesis,attend,
  stereo}.yaml` and the three thesis goldens are as tagged.
- **Found on the way:** over a stream the dissertation system ran its field a
  fixed 20 cycles per frame with input gain 0.765 (dossier, finding 22). The new
  field profiles use that; `thesis.yaml` (stills, relaxed from rest to
  convergence) is unchanged.
- **Decided (Gerriet, 2026-09-22):** the replication claim for H1 rests on the
  thesis's own chain, `configs/thesis/attend_field128.yaml` — object files on the
  neural field's activity clusters, correspondence of §7.2.3 — because that is
  the dissertation system; the claim holds on it inside the tracking range the
  thesis states and fails outside. `attend.yaml` (saliency segments, position-only
  correspondence) stays as the *extension* arm — the segmentation-based first
  stage the WAPCV 2003 paper proposes in its last sentence — on which the claim
  holds at every speed tested. Both are reported, and every number says which
  profile it comes from. The field size 128 is a choice the thesis does not fix
  (the deployed 64-px field holds about three objects of the H1 scenes' size);
  the dependence is stated wherever the chain's numbers are.

The dossier then has 25 findings: 22 replicated, 3 partially.

## Implemented so far

- `configs/thesis/{thesis,stereo,attend}.yaml`; a test
  (`[thesis-track]` in `tests/test_config.cpp`) fails if a profile there enables
  anything but dissertation components.
- Fidelity tests: `tests/test_thesis_features.cpp`; behavioural goldens for all
  three thesis profiles (`thesis_inputc`, `stereo`, `thesis_attend`).
- `eval/dynamic_ior.py` (H1) defaults to the thesis second-stage profile.
- `configs/ablation/` — the arms that decide the modern default.
