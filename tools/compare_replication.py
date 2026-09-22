#!/usr/bin/env python3
"""Compare two replication.json files (eval/replication.py) number by number.

Used to check the dossier across platforms: the numbers in
docs/replication/REPLICATION_DOSSIER.md were measured on macOS, and the paper
reports them, so they are checked once on Linux (.github/workflows/replication.yml).

Walks both documents in parallel and reports every numeric leaf whose absolute
difference exceeds --tol, every boolean that differs, and every key present on
one side only. Exit 1 if anything differs beyond the tolerance, 0 otherwise.

Usage: compare_replication.py A.json B.json [--tol 0.02] [--top N] [--exclude PREFIX]
"""
import argparse
import json
import sys


def walk(node, path, out):
    if isinstance(node, dict):
        for key, value in node.items():
            walk(value, path + [str(key)], out)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            walk(value, path + [str(i)], out)
    else:
        out["/".join(path)] = node
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.02, help="absolute difference that counts (default: %(default)s)")
    ap.add_argument("--top", type=int, default=25, help="how many differences to print (default: %(default)s)")
    ap.add_argument("--exclude", action="append", default=[], metavar="PREFIX",
                    help="skip keys starting with this (repeatable)")
    args = ap.parse_args()

    with open(args.a) as fh:
        a = walk(json.load(fh), [], {})
    with open(args.b) as fh:
        b = walk(json.load(fh), [], {})

    if args.exclude:
        a = {k: v for k, v in a.items() if not any(k.startswith(p) for p in args.exclude)}
        b = {k: v for k, v in b.items() if not any(k.startswith(p) for p in args.exclude)}
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    differences = []
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) \
                or not isinstance(y, (int, float)):
            if x != y:
                differences.append((float("inf"), key, x, y))
            continue
        delta = abs(x - y)
        if delta > args.tol:
            differences.append((delta, key, x, y))

    differences.sort(key=lambda d: -d[0])
    print("%s vs %s: %d shared values, %d differ by more than %g"
          % (args.a, args.b, len(set(a) & set(b)), len(differences), args.tol))
    for delta, key, x, y in differences[:args.top]:
        shown = "differ" if delta == float("inf") else "%.4g" % delta
        print("  %-60s %s vs %s  (%s)" % (key[-60:], x, y, shown))
    if len(differences) > args.top:
        print("  ... and %d more" % (len(differences) - args.top))
    for key in only_a:
        print("  only in %s: %s" % (args.a, key))
    for key in only_b:
        print("  only in %s: %s" % (args.b, key))
    return 1 if differences or only_a or only_b else 0


if __name__ == "__main__":
    sys.exit(main())
