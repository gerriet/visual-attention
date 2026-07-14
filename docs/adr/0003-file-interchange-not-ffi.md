# ADR-0003 — File-based interchange instead of FFI

**Status:** accepted · **Date:** 2026-07

## Context

The C++ pipeline and the Python evaluation layer ([ADR-0001](0001-cpp-core-python-eval.md))
need to exchange results: saliency maps and ordered fixations. The obvious
options are language bindings (pybind11 / a C API) or a file-based exchange.

## Decision

Exchange results as **files**: a saliency map (16-bit PNG) plus a result JSON
(ordered fixations with position, value, iteration; per-feature timing; a config
hash). Every model — the C++ pipeline, a C++ variant, or a Python model — emits
the **same** format, documented in [INTERCHANGE_FORMAT.md](../INTERCHANGE_FORMAT.md).

## Consequences

- The evaluation harness consumes one format and never special-cases a model;
  adding a model (even a Python learned one) never touches the harness.
- Runs are reproducible and inspectable by construction — the artifact on disk
  *is* the record, diffable and archivable.
- No binding build, no ABI coupling, no shared-memory lifetime questions across
  the language boundary.
- Cost: disk I/O per result and a format to version. Both are cheap for an
  offline evaluation workflow, and versioning is handled by a `schema` field.
