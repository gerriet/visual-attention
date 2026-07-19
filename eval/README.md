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

## Modern models and benchmarking (M7)

Modern saliency models emit the same interchange format as the C++ pipeline,
so they are drop-in peers of the thesis model:

```bash
# Run a modern model on an image (writes JSON + 16-bit PNG)
eval/.venv/bin/python -m attention_eval.models spectral-residual img.png --emit-json out.json
eval/.venv/bin/python -m attention_eval.models --list

# Benchmark: the C++ thesis model vs modern models, cross-model over an image set
eval/.venv/bin/python -m attention_eval.benchmark \
    --images data/test_images \
    --out results/bench \
    --model cpp:build/attention:configs/thesis.yaml=thesis \
    --model spectral-residual --model center-bias --reference thesis

# Or against a public fixation dataset (ground truth), once downloaded
eval/.venv/bin/python -m attention_eval.benchmark --dataset mit1003 \
    --out results/bench --model spectral-residual --model center-bias

# Regenerate the "thesis, 20 years later" report + montage in docs/
cd eval && .venv/bin/python report_thesis_vs_modern.py
```

## Modules

- `attention_eval.io` — load interchange results (JSON + 16-bit PNG map)
- `attention_eval.metrics` — AUC-Judd, NSS, CC, SIM, KL
- `attention_eval.scanpath` — fixation matching, gridded Levenshtein
- `attention_eval.report` — single-set comparison report CLI
- `attention_eval.models` — modern saliency models (spectral-residual,
  center-bias; torch-gated DeepGaze IIE adapter) emitting the interchange format
- `attention_eval.benchmark` — run models over an image set, aggregate
  cross-model or against ground-truth fixations
- `attention_eval.plots` — saliency montage + metric bar charts (matplotlib,
  optional)
- `datasets/mit1003.py`, `datasets/middlebury.py`, `datasets/davis2017.py` —
  dataset adapters (download instructions inside; the DAVIS adapter carries the
  hand-curated person↔object-id map used by the H2 study)
- `dynamic_ior.py` — the M12 dynamic-IOR study (H1), one command per scene
- `gated_recognition.py`, `plot_gated_recognition.py` — the M13
  gated-recognition study (H2): gated vs full-frame detection, accuracy vs
  compute (see `docs/GATED_RECOGNITION.md`)

Run the tests: `cd eval && .venv/bin/python -m unittest discover -s tests`
