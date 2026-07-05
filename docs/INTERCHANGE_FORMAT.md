# Result Interchange Format (`attention-result/v1`)

Every model in this project — the C++ classic pipeline, future C++ variants,
and Python-side learned models — emits its results in this one format, and the
Python evaluation layer only ever consumes this format. Adding a new model
never requires touching the harness.

A result is two files side by side:

1. **`<name>.json`** — run metadata and the ordered fixation sequence
2. **`<name>_saliency.png`** — the saliency map as 16-bit grayscale PNG,
   values linearly mapped from [0, 1] to [0, 65535], same resolution as the
   source image

## Producing results

- C++: `attention <image> --no-display --emit-json out/result.json`
  (also works in `--config` mode), or `attention::io::ResultWriter::write()`
  from code.
- Python models: write the same JSON/PNG pair directly.

## Consuming results

- `eval/compare_scanpaths.py golden.json actual.json` — loose behavioral
  equivalence check (position tolerance, order tolerance); used by the
  behavioral CTest suite against `tests/golden/<image>/result.json`.
- Future: metric computation and report generation in `eval/` (roadmap M4).

## JSON schema

```json
{
  "schema": "attention-result/v1",
  "source": {
    "image": "path as given to the generator (informational only)",
    "width": 256,
    "height": 256
  },
  "generator": {
    "name": "attention-framework",
    "variant": "classic-weighted-sum"
  },
  "params": { "free-form, generator-specific": "echo of the configuration" },
  "saliency_map": "result_saliency.png",
  "fixations": [
    { "n": 0, "x": 42, "y": 57, "value": 1.0 }
  ],
  "timing_ms": { "free-form, generator-specific": 0 }
}
```

Field notes:

- **`fixations`** is the scanpath: ordered by attention sequence (`n`
  ascending). `x`/`y` are pixel coordinates in the source image. `value` is
  the generator's salience measure for that fixation — the map value at the
  peak for nms/ior, the cluster mean for neural-field. Comparators must rank
  by it, not interpret its absolute scale. This is the field consumers rely
  on.
- **`saliency_map`** is a path relative to the JSON file.
- **`source.image`** is informational; comparators must not interpret it
  (golden files may contain paths from another machine).
- **`params`** and **`timing_ms`** are generator-specific and not part of the
  comparison contract.
- Schema changes bump the version (`attention-result/v2`); consumers check the
  prefix.

## Scanpath format (`attention-scanpath/v1`)

The AttentionSystem (M6) runs the second selection stage and a behavior over a
*stream* and emits a scanpath — a sibling schema to the single-frame result
above, written by `--attend --emit-scanpath`:

```json
{
  "schema": "attention-scanpath/v1",
  "generator": { "name": "attention-framework", "behavior": "exploration" },
  "frames": 3,
  "scanpath": [
    { "frame": 0, "label": 1, "x": 29, "y": 142, "saliency": 0.62,
      "bbox": [x, y, w, h] }
  ],
  "objects": [
    { "label": 1, "x": 29, "y": 142, "size": 640, "saliency": 0.62,
      "avg_saliency": 0.60, "created_frame": 0, "last_selected_frame": 2 }
  ]
}
```

- **`scanpath`** is the ordered sequence of foci, one per frame that produced
  one. `label` is the object file focused; `x`/`y` its centroid. Comparators
  should match by frame + position within a tolerance — object-file `label`s
  are creation-order dependent and not stable across algorithm changes.
- **`objects`** is the final active world model (object files at stream end).
- Compared by `eval/compare_scanpath_json.py` (focus position per frame).
