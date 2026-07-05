# Evaluation layer

Python-side evaluation for the attention framework (roadmap M4). Consumes
only the interchange format (`docs/INTERCHANGE_FORMAT.md`) — a result JSON
plus its sibling 16-bit saliency PNG — so C++ pipelines and Python models are
compared identically.

## Setup

```bash
python3 -m venv eval/.venv
eval/.venv/bin/pip install -r eval/requirements.txt
```

CTest picks up `eval/.venv` automatically for the Python unit tests;
`compare_scanpaths.py` needs no dependencies and runs with any python3.

## Usage

```bash
# Loose behavioral equivalence of two results (used by the golden tests)
python3 eval/compare_scanpaths.py golden.json actual.json

# Full comparison report: reference result vs. one or more others
eval/.venv/bin/python -m attention_eval.report \
    results/classic/result.json results/modern/result.json \
    --names classic modern
```

The report contains per-result stats, map metrics against the reference
(CC, SIM, KL), NSS/AUC-Judd of each map against the reference fixations, and
scanpath agreement (matched fraction, mean distance, gridded Levenshtein).

## Modules

- `attention_eval.io` — load interchange results (JSON + 16-bit PNG map)
- `attention_eval.metrics` — AUC-Judd, NSS, CC, SIM, KL
- `attention_eval.scanpath` — fixation matching, gridded Levenshtein
- `attention_eval.report` — comparison report CLI
- `datasets/mit1003.py` — MIT1003 dataset adapter (download instructions
  inside; map-based metrics work without scipy)

Run the tests: `cd eval && .venv/bin/python -m unittest discover -s tests`
