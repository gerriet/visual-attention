# ADR-0002 — Registry- and config-driven strategies

**Status:** accepted · **Date:** 2026-07

## Context

The project's purpose is comparison: thesis features vs. modern operators,
neural-field selection vs. simpler trackers, one behaviour vs. another. Hard-
wiring a pipeline (or expressing variants through an inheritance hierarchy)
would make every comparison a code change and a recompile, and would couple the
pipeline to the specific set of stages that existed when it was written.

## Decision

Model each replaceable stage — **feature extractor, fusion strategy, selection
strategy, behaviour, ROI processor** — behind an interface registered in a
**string-keyed registry**. Composition is driven entirely by YAML: a profile
names the strategies and their parameters. Adding a variant means registering a
key, not editing the pipeline.

## Consequences

- Classic (`thesis.yaml`) and modern (`modern.yaml`, `alternative.yaml`)
  configurations are data, not branches — the same binary runs both.
- New operators (the six post-2004 saliency features, the Kalman-MOT selection
  backend) landed as additive registrations with no change to existing
  profiles.
- Cost: construction is dynamic (factory + parameter parsing), so errors like a
  misspelled strategy surface at load time rather than compile time. Mitigated
  by explicit, listing error messages and config tests.
