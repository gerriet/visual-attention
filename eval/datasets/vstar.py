"""V*Bench adapter (Wu & Xie, "V*: Guided Visual Search", CVPR 2024).

High-resolution VQA where the answer depends on a small region — exactly the
regime where uniform downsampling makes a VLM blind, and where an attention
front-end that crops the right region should keep accuracy at a fraction of the
visual tokens (roadmap M18, H6).

Layout expected under data/vstar_bench/ (gitignored):

    data/vstar_bench/
      test_questions.jsonl
      direct_attributes/*.jpg
      relative_position/*.jpg

Download (small — ~200 items, a few hundred MB of images):

    huggingface-cli download craigwu/vstar_bench --repo-type dataset \\
      --local-dir data/vstar_bench

or clone the HF dataset repo. Data is pointed to, never redistributed.

Each record's `text` bundles the question, the lettered options, and an
instruction line; parse() splits them. There is no target bounding box in this
mirror — the mock VLM's visibility path is exercised by a synthetic item in
the front-end harness, not by these records.
"""

import json
import re
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "vstar_bench"

_OPTION = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def available(root=DEFAULT_ROOT):
    return (Path(root) / "test_questions.jsonl").exists()


def parse(text):
    """Split a V*Bench `text` field into (question, choices) where choices is a
    list of option strings in letter order."""
    question_lines, choices = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _OPTION.match(line)
        if match:
            choices.append(match.group(2).strip())
        elif line.lower().startswith("answer with"):
            continue  # the trailing instruction line
        elif not choices:
            question_lines.append(line)
    return " ".join(question_lines), choices


def iter_items(root=DEFAULT_ROOT, category=None):
    """Yield dicts: {image (Path), question, choices, answer (str), answer_letter,
    category, question_id}. `category` filters to 'direct_attributes' or
    'relative_position'."""
    root = Path(root)
    listing = root / "test_questions.jsonl"
    if not listing.exists():
        raise FileNotFoundError(
            "V*Bench not found under %s — see this module's docstring for download steps" % root)
    with open(listing) as fh:
        for line in fh:
            record = json.loads(line)
            if category and record["category"] != category:
                continue
            question, choices = parse(record["text"])
            letter = record["label"].strip()
            index = ord(letter) - 65
            answer = choices[index] if 0 <= index < len(choices) else letter
            yield {
                "image": root / record["image"],
                "question": question,
                "choices": choices,
                "answer": answer,
                "answer_letter": letter,
                "category": record["category"],
                "question_id": record["question_id"],
            }
