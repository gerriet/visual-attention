"""HR-Bench adapter (Wang et al., "Divide, Conquer and Combine", 2024).

High-resolution VQA at 4K and 8K — fine-grained single-instance ("single":
attributes, OCR) and cross-instance ("cross": maps, charts, spatial relations)
perception. Images are far beyond what a VLM ingests natively, so this is the
regime where a uniform downsample should blind the model and native-res
attention crops should pay off (roadmap M18, H6).

Layout expected under data/hr_bench/ (gitignored):

    huggingface-cli download DreamMr/HR-Bench --repo-type dataset \\
      --local-dir data/hr_bench

Each split is one parquet file with the image inline as base64: 800 rows =
200 questions x 4 option rotations (`cycle_category`, HR-Bench's CircularEval).
Only cycle 0 is scored here — plain accuracy, chance 0.25 — so each question
counts once. In cycle 0 the correct option is *always* "A", so the options are
re-ordered per question (seeded by the question id: the same order in every arm
and every run); served as stored, a model that says "A" whenever it cannot see
the target would score 1.0. The first use extracts the cycle-0 images to
data/hr_bench/extracted_<split>/ (needs pyarrow: eval/.venv/bin/pip install
pyarrow), streaming one row at a time so the 2.8 GB 8K file never sits in
memory whole. There are no target boxes, so the front-end's target
diagnostics stay off. Data is pointed to, never redistributed.
"""

import base64
import json
import os
import random
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "hr_bench"
SPLITS = ("4k", "8k")


def available(split, root=DEFAULT_ROOT):
    root = Path(root)
    return (root / ("extracted_%s" % split) / "items.jsonl").exists() or \
        (root / ("hr_bench_%s.parquet" % split)).exists()


def extract(split, root=DEFAULT_ROOT):
    """Write the cycle-0 images (their original encoded bytes) plus an
    items.jsonl of the question fields to extracted_<split>/."""
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("HR-Bench extraction needs pyarrow: eval/.venv/bin/pip install pyarrow") from e
    root = Path(root)
    out = root / ("extracted_%s" % split)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["index", "question", "A", "B", "C", "D", "answer", "category", "cycle_category"]
    # Unbuffered page reads: one row group holds the whole file, and the
    # default pre-buffering would load all of it.
    parquet = pq.ParquetFile(root / ("hr_bench_%s.parquet" % split), pre_buffer=False, buffer_size=1 << 23)
    records = []
    for batch in parquet.iter_batches(batch_size=1, columns=fields + ["image"]):
        row = batch.to_pylist()[0]
        if int(row["cycle_category"]) != 0:
            continue
        data = base64.b64decode(row.pop("image"))
        name = "%d%s" % (row["index"], ".png" if data[:4] == b"\x89PNG" else ".jpg")
        (out / name).write_bytes(data)
        row["image"] = name
        records.append(row)
    listing = out / "items.jsonl"
    with open(str(listing) + ".tmp", "w") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")
    os.replace(str(listing) + ".tmp", listing)  # a partial extraction reruns
    return len(records)


def shuffled_choices(choices, question_id):
    """The options in an order fixed by the question id alone. Cycle 0 stores
    the correct option first for every question; this spreads it evenly over
    the letters without making the order depend on the run or the arm."""
    order = list(choices)
    random.Random("hrbench-%s" % question_id).shuffle(order)
    return order


def iter_items(split, category=None, root=DEFAULT_ROOT):
    """Yield dicts: {image (Path), question, choices, answer (str), answer_letter,
    category ('single' | 'cross'), question_id, target_boxes (None)}."""
    if split not in SPLITS:
        raise ValueError("HR-Bench split must be one of %s" % ", ".join(SPLITS))
    root = Path(root)
    out = root / ("extracted_%s" % split)
    if not (out / "items.jsonl").exists():
        extract(split, root)
    with open(out / "items.jsonl") as fh:
        for line in fh:
            record = json.loads(line)
            if category and record["category"] != category:
                continue
            stored = [record[k] for k in "ABCD" if record.get(k) is not None]
            letter = record["answer"].strip()
            index = ord(letter) - 65
            answer = stored[index] if 0 <= index < len(stored) else letter
            choices = shuffled_choices(stored, record["index"])
            if answer in choices:
                letter = chr(65 + choices.index(answer))
            yield {
                "image": out / record["image"],
                "question": record["question"],
                "choices": choices,
                "answer": answer,
                "answer_letter": letter,
                "category": record["category"],
                "question_id": record["index"],
                "target_boxes": None,
            }
