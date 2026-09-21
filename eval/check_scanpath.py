#!/usr/bin/env python3
"""Check an attention-scanpath/v1 JSON against stated behavioural expectations.

Standalone (stdlib only). For sequences where a per-frame golden would pin an
arbitrary choice: on `data/test_images/motion_seq` the first frame is pure
noise, so *which* noise fragment wins the first focus is decided by rounding
and differs between platforms (macOS/Accelerate vs Linux CI), and the dwell
then carries that choice through the whole three-frame sequence. What the
sequence does determine is checked here instead: there is a focus on every
frame, and the moving patch has become an object file by the last frame.

Usage: check_scanpath.py ACTUAL.json [--frames N] [--object-near X,Y] [--tol PX]
Exit 0 if every expectation holds, 1 otherwise.
"""
import argparse
import json
import math
import sys


def parse_point(text):
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y, got '{text}'")
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("actual", help="attention-scanpath/v1 JSON to check")
    ap.add_argument("--frames", type=int, default=None,
                    help="expect a focus on each of the frames 0..N-1")
    ap.add_argument("--object-near", type=parse_point, action="append", default=[], metavar="X,Y",
                    help="expect an object file (or a focus) within --tol of this point; repeatable")
    ap.add_argument("--tol", type=float, default=25.0, help="distance in px for --object-near (default: %(default)s)")
    args = ap.parse_args()

    with open(args.actual) as f:
        data = json.load(f)
    if not data.get("schema", "").startswith("attention-scanpath/"):
        raise SystemExit(f"{args.actual}: not an attention-scanpath file")

    ok = True
    foci = {f["frame"]: (f["x"], f["y"]) for f in data.get("scanpath", [])}
    if args.frames is not None:
        missing = [i for i in range(args.frames) if i not in foci]
        status = "ok" if not missing else "FAIL"
        ok = ok and not missing
        print(f"  {status}: focus on every frame 0..{args.frames - 1}" + (f" (missing {missing})" if missing else ""))

    # Object files that ended the run, plus every focus position: an object that
    # was attended and then merged or dropped still counts as found.
    candidates = [(o["x"], o["y"]) for o in data.get("objects", [])] + list(foci.values())
    for px, py in args.object_near:
        dist, (cx, cy) = min(((math.hypot(px - x, py - y), (x, y)) for x, y in candidates),
                             default=(math.inf, (None, None)))
        found = dist <= args.tol
        ok = ok and found
        print(f"  {'ok' if found else 'FAIL'}: object near ({px:g},{py:g}): closest ({cx},{cy}), {dist:.1f}px")

    print("\nas expected" if ok else "\nnot as expected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
