#!/usr/bin/env python3
"""Compare two attention-result JSON files for loose behavioral equivalence.

Two results are considered equivalent when the model attends to essentially
the same locations in a similar order -- the replication bar of the v2 phase
(see docs/V2_ROADMAP.md). Pixel-level agreement is explicitly not required.

Usage:
    compare_scanpaths.py golden.json actual.json [--pos-tol PX] [--order-tol N]
                         [--max-fixations N] [--count-tol N]

Exit code 0 when equivalent, 1 when not, 2 on bad input.
"""

import argparse
import json
import math
import sys


def load_fixations(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        sys.exit(2)
    schema = data.get("schema", "")
    if not schema.startswith("attention-result/"):
        print(f"error: {path} is not an attention-result file (schema={schema!r})", file=sys.stderr)
        sys.exit(2)
    return data.get("fixations", [])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("golden", help="golden/reference result JSON")
    parser.add_argument("actual", help="result JSON to check")
    parser.add_argument("--pos-tol", type=float, default=20.0,
                        help="max distance (px) between corresponding fixations (default: 20)")
    parser.add_argument("--order-tol", type=int, default=1,
                        help="max allowed rank shift for a matched fixation (default: 1)")
    parser.add_argument("--max-fixations", type=int, default=5,
                        help="compare at most the first N golden fixations, 0 = all (default: 5)")
    parser.add_argument("--count-tol", type=int, default=2,
                        help="max allowed difference in fixation counts (default: 2)")
    args = parser.parse_args()

    golden = load_fixations(args.golden)
    actual = load_fixations(args.actual)

    failures = []

    if abs(len(golden) - len(actual)) > args.count_tol:
        failures.append(f"fixation count differs too much: golden={len(golden)}, actual={len(actual)}")

    n = len(golden) if args.max_fixations == 0 else min(len(golden), args.max_fixations)
    for i, g in enumerate(golden[:n]):
        best = None
        for j, a in enumerate(actual):
            d = math.hypot(g["x"] - a["x"], g["y"] - a["y"])
            if d <= args.pos_tol and (best is None or d < best[1]):
                best = (j, d)
        if best is None:
            failures.append(
                f"golden fixation #{i} at ({g['x']}, {g['y']}) has no counterpart within {args.pos_tol}px")
        elif abs(best[0] - i) > args.order_tol:
            failures.append(
                f"golden fixation #{i} at ({g['x']}, {g['y']}) matched actual #{best[0]} "
                f"(rank shift {abs(best[0] - i)} > {args.order_tol})")
        else:
            print(f"  ok: golden #{i} ({g['x']}, {g['y']}) ~ actual #{best[0]} (dist {best[1]:.1f}px)")

    if failures:
        print(f"\nNOT equivalent ({args.golden} vs {args.actual}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print(f"\nequivalent: first {n} fixations match within {args.pos_tol}px, order tolerance {args.order_tol}")


if __name__ == "__main__":
    main()
