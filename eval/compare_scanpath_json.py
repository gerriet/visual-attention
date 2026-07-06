#!/usr/bin/env python3
"""Compare two attention-scanpath/v1 JSON files by focus position over frames.

Standalone (stdlib only), like compare_scanpaths.py — used by the M6 golden
test. Object-file *labels* are creation-order dependent and deliberately not
compared; what matters at the project's loose-equivalence bar is where the
focus went on each frame, within a position tolerance.

Usage: compare_scanpath_json.py GOLDEN.json ACTUAL.json [--pos-tol N]
Exit 0 if equivalent, 1 otherwise.
"""
import argparse
import json
import math
import sys


def load_foci(path):
    with open(path) as f:
        data = json.load(f)
    if not data.get("schema", "").startswith("attention-scanpath/"):
        raise SystemExit(f"{path}: not an attention-scanpath file")
    return {f["frame"]: (f["x"], f["y"]) for f in data.get("scanpath", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("golden", help="golden attention-scanpath/v1 JSON")
    ap.add_argument("actual", help="actual attention-scanpath/v1 JSON")
    ap.add_argument("--pos-tol", type=float, default=20.0,
                    help="max focus-position distance in px (default: %(default)s)")
    args = ap.parse_args()

    golden = load_foci(args.golden)
    actual = load_foci(args.actual)

    if set(golden) != set(actual):
        print(f"not equivalent: focus frames differ (golden {sorted(golden)}, actual {sorted(actual)})")
        return 1

    ok = True
    for frame in sorted(golden):
        gx, gy = golden[frame]
        ax, ay = actual[frame]
        dist = math.hypot(gx - ax, gy - ay)
        status = "ok" if dist <= args.pos_tol else "FAIL"
        if dist > args.pos_tol:
            ok = False
        print(f"  {status}: frame {frame} golden ({gx},{gy}) ~ actual ({ax},{ay}) (dist {dist:.1f}px)")

    if ok:
        print(f"\nequivalent: all foci match within {args.pos_tol}px")
        return 0
    print(f"\nnot equivalent: some foci exceed {args.pos_tol}px")
    return 1


if __name__ == "__main__":
    sys.exit(main())
