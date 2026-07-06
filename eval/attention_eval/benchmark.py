"""Benchmark harness: run saliency models over an image set and aggregate.

Models are named specs, run to the interchange format (so the C++ thesis model
and the Python modern models are peers):

  - a Python model name:            spectral-residual   (see attention_eval.models)
  - the C++ pipeline:               cpp:<binary>:<config>
  - an explicit display name:       <spec>=<name>

Two aggregation modes:

  - cross-model (default): one model is the reference; each other model's map
    and scanpath are scored against it (CC/SIM/KL, NSS/AUC@ref, scanpath), then
    averaged over the image set. Needs no ground truth.
  - against ground truth (--dataset): every model is scored against human
    fixation maps/points (AUC-Judd, NSS, CC, SIM, KL), averaged over the set.

Usage:
  python -m attention_eval.benchmark --images DIR_OR_FILES --out DIR \\
      --model cpp:../build/attention:../configs/thesis.yaml=thesis \\
      --model spectral-residual --model center-bias [--reference thesis]
  python -m attention_eval.benchmark --dataset mit1003 --out DIR \\
      --model spectral-residual --model center-bias
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import io, metrics, scanpath
from .models import get_model, run_model

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# --- model specs ----------------------------------------------------------

def parse_spec(spec):
    """('python', name, None, display) or ('cpp', binary, config, display)."""
    display = None
    if "=" in spec:
        spec, display = spec.rsplit("=", 1)
    if spec.startswith("cpp:"):
        _, binary, config = spec.split(":", 2)
        return ("cpp", binary, config, display or Path(config).stem)
    return ("python", spec, None, display or spec)


def run_spec(spec, image_path, out_json):
    kind, a, b, _ = spec
    if kind == "python":
        run_model(get_model(a), image_path, out_json)
    else:
        subprocess.run([a, "--config", b, str(image_path), "--no-display", "--emit-json", str(out_json)],
                       check=True, capture_output=True)


def collect_images(paths):
    images = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            images.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS))
        elif p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def run_all(specs, images, out_dir):
    """Run every spec on every image; return {display: {image_stem: Result}}."""
    out_dir = Path(out_dir)
    results = {}
    for spec in specs:
        display = spec[3]
        results[display] = {}
        for image in images:
            out_json = out_dir / display / (Path(image).stem + ".json")
            run_spec(spec, image, out_json)
            results[display][Path(image).stem] = io.load_result(out_json)
    return results


def _mean(values):
    values = [v for v in values if v == v]  # drop NaN
    return float(np.mean(values)) if values else float("nan")


# --- aggregation ----------------------------------------------------------

def aggregate_cross_model(results, reference, pos_tol=20.0, grid=5):
    """Average each model's agreement with the reference model over images."""
    rows = {}
    for display, per_image in results.items():
        cc, sim, kl, nss, auc, matched, dist, edit = ([] for _ in range(8))
        for stem, res in per_image.items():
            ref = results[reference].get(stem)
            if ref is None:
                continue
            if display == reference:
                continue
            if ref.saliency is not None and res.saliency is not None and ref.saliency.shape == res.saliency.shape:
                cc.append(metrics.cc(ref.saliency, res.saliency))
                sim.append(metrics.sim(ref.saliency, res.saliency))
                kl.append(metrics.kl_div(ref.saliency, res.saliency))
            if res.saliency is not None and ref.fixations:
                nss.append(metrics.nss(res.saliency, ref.fixations))
                auc.append(metrics.auc_judd(res.saliency, ref.fixations))
            stats = scanpath.match_stats(ref.fixations, res.fixations, pos_tol=pos_tol)
            matched.append(stats["matched_fraction"])
            dist.append(stats["mean_distance"])
            edit.append(scanpath.gridded_levenshtein(ref.fixations, res.fixations, ref.size, grid=grid))
        rows[display] = dict(CC=_mean(cc), SIM=_mean(sim), KL=_mean(kl), NSS=_mean(nss),
                             AUC=_mean(auc), matched=_mean(matched), dist=_mean(dist), edit=_mean(edit))
    return rows


def aggregate_vs_gt(results, gt_maps, gt_points):
    """Average each model's scores against ground-truth fixation maps/points.

    gt_maps/gt_points are {image_stem: ndarray / [(x,y),...]}."""
    rows = {}
    for display, per_image in results.items():
        auc, nss, cc, sim, kl = ([] for _ in range(5))
        for stem, res in per_image.items():
            if res.saliency is None:
                continue
            pts = gt_points.get(stem)
            gmap = gt_maps.get(stem)
            if pts:
                auc.append(metrics.auc_judd(res.saliency, pts))
                nss.append(metrics.nss(res.saliency, pts))
            if gmap is not None and gmap.shape == res.saliency.shape:
                cc.append(metrics.cc(gmap, res.saliency))
                sim.append(metrics.sim(gmap, res.saliency))
                kl.append(metrics.kl_div(gmap, res.saliency))
        rows[display] = dict(AUC=_mean(auc), NSS=_mean(nss), CC=_mean(cc), SIM=_mean(sim), KL=_mean(kl))
    return rows


def _fmt(v, d=3):
    return "–" if v != v else f"{v:.{d}f}"


def cross_model_markdown(rows, reference, n_images):
    out = ["# Benchmark — cross-model agreement\n",
           f"Reference model: **{reference}** — {n_images} images. Higher CC/SIM/NSS/AUC/matched "
           "and lower KL/dist/edit mean closer agreement with the reference.\n",
           "| model | CC | SIM | KL | NSS@ref | AUC@ref | matched | mean dist | grid edit |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        marker = " (ref)" if name == reference else ""
        if name == reference:
            out.append(f"| {name}{marker} | 1.000 | 1.000 | 0.000 | – | – | 1.00 | 0.0 | 0.000 |")
        else:
            out.append(f"| {name} | {_fmt(r['CC'])} | {_fmt(r['SIM'])} | {_fmt(r['KL'])} | {_fmt(r['NSS'])} "
                       f"| {_fmt(r['AUC'])} | {_fmt(r['matched'],2)} | {_fmt(r['dist'],1)} | {_fmt(r['edit'])} |")
    return "\n".join(out) + "\n"


def gt_markdown(rows, dataset_name, n_images):
    out = ["# Benchmark — against ground-truth fixations\n",
           f"Dataset: **{dataset_name}** — {n_images} images. AUC-Judd/NSS/CC/SIM higher is better; KL lower.\n",
           "| model | AUC | NSS | CC | SIM | KL |",
           "|---|---|---|---|---|---|"]
    for name, r in rows.items():
        out.append(f"| {name} | {_fmt(r['AUC'])} | {_fmt(r['NSS'])} | {_fmt(r['CC'])} | {_fmt(r['SIM'])} "
                   f"| {_fmt(r['KL'])} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, dest="models", help="model spec (repeatable)")
    parser.add_argument("--images", nargs="*", default=[], help="image files/directories (cross-model mode)")
    parser.add_argument("--dataset", default=None, help="dataset adapter name for GT mode (e.g. mit1003)")
    parser.add_argument("--dataset-root", default=None, help="dataset root (adapter default if omitted)")
    parser.add_argument("--limit", type=int, default=0, help="cap number of dataset images (0 = all)")
    parser.add_argument("--out", required=True, help="output directory for per-model interchange files")
    parser.add_argument("--reference", default=None, help="reference model display name (cross-model)")
    parser.add_argument("--pos-tol", type=float, default=20.0)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--report", default=None, help="write the markdown report here (default: stdout)")
    args = parser.parse_args(argv)

    specs = [parse_spec(s) for s in args.models]

    if args.dataset:
        # The local dataset adapters live in eval/datasets; prepend the eval
        # root so `import datasets` resolves to them rather than an installed
        # top-level package of the same name (e.g. HuggingFace `datasets`).
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        import datasets
        images, gt_maps, gt_points = datasets.load_for_benchmark(args.dataset, args.dataset_root, args.limit)
        results = run_all(specs, images, args.out)
        rows = aggregate_vs_gt(results, gt_maps, gt_points)
        report = gt_markdown(rows, args.dataset, len(images))
    else:
        images = collect_images(args.images)
        if not images:
            parser.error("no images found (pass --images or --dataset)")
        reference = args.reference or specs[0][3]
        display_names = [s[3] for s in specs]
        if reference not in display_names:
            parser.error(f"--reference '{reference}' is not one of the models: {', '.join(display_names)}")
        results = run_all(specs, images, args.out)
        rows = aggregate_cross_model(results, reference, pos_tol=args.pos_tol, grid=args.grid)
        report = cross_model_markdown(rows, reference, len(images))

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report)
        print(f"wrote {args.report}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
