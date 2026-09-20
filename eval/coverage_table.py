#!/usr/bin/env python3
"""Target coverage of attention fixations against chance, per profile.

Reads the per-item rows that eval/vlm_frontend.py writes (results.json; run it
model-free with --backend mock) and tabulates, for each result directory: the
share of annotated targets covered by the top K fixation windows, the same for
uniformly random fixations (the chance level the harness records per item), and
the paired difference with a bootstrap 95% CI over items. With --versus, also
the paired difference of every profile to that one.

A crop source is informative only to the extent it beats chance — this is the
model-free way to compare stage-1 profiles before any VLM time is spent.

    eval/vlm_frontend.py --vstar --backend mock --limit 0 --config configs/thesis/thesis.yaml \\
        --out results/coverage/thesis
    eval/coverage_table.py results/coverage/* --versus results/coverage/thesis
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import paired_bootstrap  # noqa: E402

KS = (1, 3, 5, 10)


def load(directory):
    """{question_id: item row} for the items with annotated targets."""
    with open(os.path.join(directory, "results.json")) as fh:
        rows = json.load(fh)
    return {r["item"]["question_id"]: r["item"] for r in rows if r["item"].get("random_coverage")}


def hits(items, ids, k):
    return [int(items[i]["target_rank"] is not None and items[i]["target_rank"] <= k) for i in ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="vlm_frontend.py output directories")
    ap.add_argument("--versus", default=None, help="also report each profile's paired difference to this one")
    ap.add_argument("--k", type=int, default=10, choices=KS, help="K for the paired differences (default 10)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    profiles = {os.path.basename(os.path.normpath(d)): load(d) for d in args.results}
    reference = load(args.versus) if args.versus else None
    table = []
    for name, items in sorted(profiles.items()):
        ids = sorted(items)
        row = {"profile": name, "n": len(ids),
               "coverage": {k: sum(hits(items, ids, k)) / len(ids) for k in KS},
               "chance": {k: sum(items[i]["random_coverage"][str(k)] for i in ids) / len(ids) for k in KS}}
        row["minus_chance"] = paired_bootstrap(hits(items, ids, args.k),
                                               [items[i]["random_coverage"][str(args.k)] for i in ids])
        if reference is not None:
            shared = sorted(set(ids) & set(reference))
            row["minus_reference"] = paired_bootstrap(hits(items, shared, args.k), hits(reference, shared, args.k))
        table.append(row)

    if args.json:
        print(json.dumps(table, indent=2))
        return
    header = "%-26s %4s  %s   top-%d minus chance%s" % (
        "profile", "n", "  ".join("top-%-2d" % k for k in KS), args.k,
        "      minus %s" % os.path.basename(os.path.normpath(args.versus)) if args.versus else "")
    print(header)
    print("-" * len(header))
    print("%-26s %4s  %s" % ("(chance: random fixations)", "", "  ".join(
        "%6.3f" % table[0]["chance"][k] for k in KS)))
    for row in sorted(table, key=lambda r: -r["minus_chance"][0]):
        line = "%-26s %4d  %s   %+.3f [%+.3f, %+.3f]" % (
            row["profile"], row["n"], "  ".join("%6.3f" % row["coverage"][k] for k in KS), *row["minus_chance"])
        if "minus_reference" in row:
            line += "   %+.3f [%+.3f, %+.3f]" % row["minus_reference"]
        print(line)


if __name__ == "__main__":
    main()
