# Contributing

Thanks for looking. This is a research repository with an unusual constraint, so
please read the first section before opening a pull request — it decides whether
a change is even possible, not just whether it is wanted.

## The one rule that is different here: two tracks

The repository holds **two things at once** ([ADR-0005](docs/adr/0005-two-tracks-replication-and-modern.md)):

- a **replication** of a 2004 dissertation's attention model — finite, finished,
  and **frozen** at the tag `replication-v2`. It lives in `configs/thesis/` and
  the code paths those configs reach. Its job is to be faithful, not good.
- a **modern system** built on the same core — open-ended, judged by usefulness.
  Everything else: `configs/modern.yaml`, the priority map, the VLM front-end,
  the evaluation layer.

**A change to the frozen replication needs a stated replication reason** — that
is, evidence from the dissertation's text or its surviving sources that the
current behaviour is *unfaithful*. "This is better" is a modern-track change, and
a welcome one; it just does not go there. A replication change also means
re-running the dossier (`eval/replication.py`) and recording the effect in
`docs/replication/REPLICATION_DOSSIER.md`.

If you are unsure which track your change is on, say so in the issue. It is the
first thing a reviewer will decide anyway.

## Build and test

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build
```

Dependencies: CMake, a C++17 compiler, OpenCV 4, yaml-cpp. Catch2 v3 is fetched
by CMake at configure time. On macOS, Homebrew's `opencv` is 5.x — install
`opencv@4`. The Python evaluation layer needs a venv, which CTest picks up
automatically:

```bash
python3 -m venv eval/.venv && eval/.venv/bin/pip install -r eval/requirements.txt
```

CI runs the same thing on Ubuntu. **Local green is not CI green**: the project
has had platform-dependent failures that only Linux showed (a first focus on a
pure-noise frame; `std::uniform_real_distribution` differing between libstdc++
and libc++). Anything involving random numbers or tie-breaking deserves a
suspicious look.

## What a change needs

- **Style**: `docs/CODE_STYLE.md` (a Google–Allman hybrid). Run
  `cmake --build build --target format` before committing, and
  `format-check` to verify. Note that CI does **not** currently enforce this:
  clang-format's output shifts between major versions, so an unpinned check
  would fail on a tree that is clean locally. Please run it yourself, and if a
  reviewer's diff disagrees with yours, say which version you used.
- **Code quality**: `docs/DEVELOPMENT_GUIDELINES.md` — small units, platform
  independence, documented invariants and pre/postconditions where they matter.
  Comments say *why*; the code already says what.
- **Tests.** C++ characterization goldens (Catch2) plus behavioural scanpath
  goldens through the CLI and `eval/compare_scanpaths.py`. The bar for the
  replication is *loose behavioural equivalence* to the thesis, not
  pixel-exactness.
- **A `--help` for every new entry point**, binary or Python script, plus a
  `help_<name>` test in `tests/CMakeLists.txt`. This is a hard contract.
- **Performance is measured, not guessed.** A hot loop gets a benchmark before
  it gets an optimisation. Current numbers: `docs/PERFORMANCE.md`.

### Changing an algorithm on purpose

Goldens will fail. That is the point — review the diffs first, convince yourself
each one is the change you meant, *then* regenerate:

```bash
ATTENTION_UPDATE_GOLDEN=1 ./build/tests/characterization_tests
```

Never regenerate goldens to make a red build green without reading the diff. Two
of this project's worst bugs survived for months behind a rescaling that made
their output look plausible.

## Adding a feature, fusion, selection strategy or processor

These are **registries**, and composition is driven entirely by `configs/*.yaml`
([ADR-0002](docs/adr/0002-registry-config-driven-strategies.md)). Adding one
means implementing the interface, registering it, and adding a config — not
touching the pipeline. If your change requires editing the pipeline to
accommodate it, that is worth a conversation first.

Anything that emits results must speak the interchange format
(`docs/INTERCHANGE_FORMAT.md`); it is the only thing the evaluation harness
consumes ([ADR-0003](docs/adr/0003-file-interchange-not-ffi.md)).

## Claims, numbers and the experimental bar

If your change is an **experimental result** rather than code, the project has a
standing bar, and it is strict on purpose (`docs/HYPOTHESIS_CLOSURE_PLAN.md`):

1. Mechanisms and knobs are tuned on development seeds only. The confirmatory
   run uses fresh seeds, touched once, with the config committed beforehand.
2. The prediction is written down *before* the run, including what would refute
   it.
3. Paired statistics over the independent unit (scenes or images), with a
   bootstrap interval — never questions within a scene.
4. n ≥ 20 units per cell.
5. One trivial or external baseline, and a *strengthened* version of the arm you
   expect to lose.
6. One command reproduces the table.
7. **The README quotes a number only after its confirmatory run.** Where a
   result stands is one table: `docs/STATUS.md`.

Negative results are kept and reported. Several are load-bearing here.

## What cannot be accepted

- **The original 2003–2005 dissertation sources.** They are not bundled and
  cannot be — the author does not hold the rights to redistribute them. Do not
  commit them, and do not quote them at length. Provenance comments cite them by
  original filename and thesis section; that is the right level.
- Large binaries or datasets. `results/` is gitignored; keep it that way.
- Changes to frozen replication behaviour without a replication reason (above).

## Reporting a problem

An issue is most useful with the config, the command, the platform, and what you
expected instead. If it is a numerical difference, say which of the two tracks it
is on — a divergence from the thesis is a bug in the replication and a feature
request in the modern track, and the fix is different in each case.

## Licence

MIT (see `LICENSE`). By contributing you agree your contribution is licensed the
same way.
