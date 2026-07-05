"""Comparison report over attention results (interchange format).

Usage:
    python -m attention_eval.report REFERENCE.json OTHER.json [...]
                                    [--names a b ...] [--pos-tol PX] [--grid N]

The first result is the reference. For every other result the report gives
map metrics against the reference map (CC, SIM, KL), point metrics of each
map against the *reference* fixations (NSS, AUC-Judd), and scanpath
agreement. Output is GitHub-flavored markdown on stdout.
"""

import argparse
import sys

from . import io, metrics, scanpath


def _fmt(value, digits=3):
    if value != value:  # NaN
        return "–"
    return f"{value:.{digits}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="result JSONs; first = reference")
    parser.add_argument("--names", nargs="*", default=None, help="display names")
    parser.add_argument("--pos-tol", type=float, default=20.0)
    parser.add_argument("--grid", type=int, default=5, help="scanpath grid size")
    args = parser.parse_args(argv)

    results = [io.load_result(p) for p in args.results]
    names = args.names if args.names and len(args.names) == len(results) else \
        [r.path.parent.name or f"result{i}" for i, r in enumerate(results)]

    reference = results[0]

    print(f"# Attention result comparison\n")
    print(f"Reference: **{names[0]}** (`{reference.variant}`, "
          f"{len(reference.fixations)} fixations)\n")

    print("| result | variant | fixations | CC | SIM | KL | NSS@ref | AUC@ref "
          "| matched | mean dist | rank shift | grid edit |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for result, name in zip(results, names):
        row = [name, f"`{result.variant}`", str(len(result.fixations))]

        if result is reference:
            row += ["1.000", "1.000", "0.000"]
        elif reference.saliency is not None and result.saliency is not None \
                and reference.saliency.shape == result.saliency.shape:
            row += [_fmt(metrics.cc(reference.saliency, result.saliency)),
                    _fmt(metrics.sim(reference.saliency, result.saliency)),
                    _fmt(metrics.kl_div(reference.saliency, result.saliency))]
        else:
            row += ["–", "–", "–"]

        if result.saliency is not None and reference.fixations:
            row += [_fmt(metrics.nss(result.saliency, reference.fixations)),
                    _fmt(metrics.auc_judd(result.saliency, reference.fixations))]
        else:
            row += ["–", "–"]

        if result is reference:
            row += ["1.00", "0.0", "0.0", "0.000"]
        else:
            stats = scanpath.match_stats(reference.fixations, result.fixations,
                                         pos_tol=args.pos_tol)
            edit = scanpath.gridded_levenshtein(reference.fixations, result.fixations,
                                                reference.size, grid=args.grid)
            row += [_fmt(stats["matched_fraction"], 2),
                    _fmt(stats["mean_distance"], 1),
                    _fmt(stats["mean_rank_shift"], 1),
                    _fmt(edit)]

        print("| " + " | ".join(row) + " |")

    print("\nMetrics: CC/SIM/KL compare saliency maps to the reference map; "
          "NSS/AUC score each map against the reference fixation points; "
          "scanpath columns compare fixation sequences "
          f"(pos-tol {args.pos_tol}px, {args.grid}x{args.grid} grid).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
