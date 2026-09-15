#!/usr/bin/env python3
"""Attention as a VLM token-budget allocator (roadmap M18, H6).

The front-end feeds a vision-language model only the *attended* fovea crops plus
one low-res global view, instead of the full-resolution image. On a task where
the answer hinges on a small region (V*Bench), that should hold accuracy at a
fraction of the visual tokens — the "gaze tells you where to compute" recipe,
with the model-free attention pipeline as the interpretable controller (the
project's edge; the non-goal is becoming another VLM).

Three arms per question, all answered by the same VLM backend:

  full-res    the whole image (capped to a practical VLM size) — the ceiling
  uniform     the whole image uniformly downsampled to the fovea arm's token
              budget — the honest same-budget baseline (small objects vanish)
  fovea       one low-res global view + K native-res crops around the top-K
              attention fixations (ours)

Crops come from the C++ pipeline's saliency fixations (`--emit-json`); the M17
`top_down_map` slot can later make them question-conditioned (H5xH6) with no
change here. Token cost is reported two ways: a provider-independent patch
estimate (always) and the backend's real count_tokens() (when it has one).
Only the token *fraction* vs full-res is compared, so the patch constant
cancels.

The VLM is pluggable (eval/vlm_backends.py). The default `ollama` backend runs
a local open-weights VLM (qwen3.8:27b) — free, reproducible, and its real token
count comes back with every answer; `--backend claude` swaps in Claude. The
`mock` backend answers correctly iff the target is delivered at usable
resolution, so the harness is testable end to end without any model (the CI
smoke runs on it); on V*Bench it becomes a model-free legibility oracle.

Where targets are annotated (V*Bench ships boxes), each row also records per
arm whether the targets were delivered legibly, whether a fovea *crop* covered
them (crop_hit), and the rank of the first fixation whose window covers them
(target_rank) — separating "attention missed the target" from "the VLM misread
it".

  eval/vlm_frontend.py --vstar --limit 50 --count-tokens    # local Ollama
  eval/vlm_frontend.py --vstar --limit 50 --backend claude
  eval/vlm_frontend.py --demo --backend mock                # synthetic, no model
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import bootstrap_ci  # noqa: E402
from vlm_backends import create_backend  # noqa: E402


def _pil():
    from PIL import Image
    return Image


def emit_fixations(binary, image_path, config, proc_max_side):
    """Run the C++ pipeline on a bounded-size copy and return attention
    fixations as (x, y, value) in *native* image coordinates, ordered by
    attention sequence."""
    Image = _pil()
    with Image.open(image_path) as native:
        native = native.convert("RGB")
        w, h = native.size
        scale = min(1.0, proc_max_side / max(w, h))
        proc = native.resize((max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1.0 else native.copy()

    with tempfile.TemporaryDirectory() as tmp:
        proc_path = os.path.join(tmp, "proc.png")
        proc.save(proc_path)
        result_path = os.path.join(tmp, "result.json")
        cmd = [binary, proc_path, "--no-display", "--emit-json", result_path]
        if config:
            cmd += ["--config", config]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(result_path) as fh:
            fixations = json.load(fh)["fixations"]

    inv = 1.0 / scale if scale < 1.0 else 1.0
    fixations.sort(key=lambda f: f["n"])
    return [(f["x"] * inv, f["y"] * inv, f["value"]) for f in fixations]


def make_view(image, source_box, scale):
    """A view = an image plus what native region it shows and at what scale."""
    return {"image": image, "source_box": source_box, "scale": scale}


def visual_tokens(backend, views):
    return sum(backend.estimate_visual_tokens(v["image"].size) for v in views)


def target_visible(views, target_boxes, min_target_px, overlap=0.6):
    """True if every target region is shown at usable resolution by some view:
    enough of the native target box lies inside the view's source region, and
    the *visible portion* is on-screen at least min_target_px. Models
    'downsampling makes the target too small to read', and 'a crop that clips
    most of the target off doesn't count'. Every box must pass — a
    relative-position question needs both objects. None when there is no
    ground-truth box (visibility unknown)."""
    if not target_boxes:
        return None
    return all(_box_visible(views, box, min_target_px, overlap) for box in target_boxes)


def target_fixation_rank(fixations, target_boxes, fovea_side, size, overlap=0.6):
    """Crop-hit diagnostic independent of K: the 1-based rank of the first
    fixation whose fovea window covers the target (for several boxes, the rank
    by which all are covered), or None if no fixation's window ever does.
    'Covered by top-K' is then rank <= K, ignoring build_arms' crop dedup."""
    if not target_boxes:
        return None
    w, h = size
    half = fovea_side // 2
    worst = 0
    for box in target_boxes:
        for n, (fx, fy, _value) in enumerate(fixations, 1):
            cx, cy = int(round(fx)), int(round(fy))
            window = (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))
            if _box_visible([make_view(None, window, 1.0)], box, 0, overlap):
                worst = max(worst, n)
                break
        else:
            return None
    return worst


def _box_visible(views, box, min_target_px, overlap):
    tx0, ty0, tx1, ty1 = box
    t_area = max(1, (tx1 - tx0) * (ty1 - ty0))
    for v in views:
        sx0, sy0, sx1, sy1 = v["source_box"]
        ix0, iy0 = max(tx0, sx0), max(ty0, sy0)
        ix1, iy1 = min(tx1, sx1), min(ty1, sy1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        if (iw * ih) / t_area < overlap:
            continue
        # On-screen size of the *visible* part, not the full native box.
        if min(iw * v["scale"], ih * v["scale"]) >= min_target_px:
            return True
    return False


def resized(image, long_side):
    Image = _pil()
    w, h = image.size
    if max(w, h) <= long_side:
        return image.copy(), 1.0
    scale = long_side / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS), scale


def build_arms(image, fixations, params):
    """Return {arm_name: list_of_views} for one image."""
    Image = _pil()
    w, h = image.size

    # full-res: whole image capped to a practical VLM size.
    full_img, full_scale = resized(image, params["full_max_side"])
    full = [make_view(full_img, (0, 0, w, h), full_scale)]

    # fovea: global thumbnail + up to K native-res crops around the top
    # fixations, deduplicated so near-coincident fixations don't spend the
    # budget twice on overlapping pixels.
    global_img, global_scale = resized(image, params["global_side"])
    views = [make_view(global_img, (0, 0, w, h), global_scale)]
    half = params["fovea_side"] // 2
    kept_centres = []
    for (fx, fy, _value) in fixations:
        if len(kept_centres) >= params["k"]:
            break
        cx, cy = int(round(fx)), int(round(fy))
        if any(abs(cx - kx) < half and abs(cy - ky) < half for kx, ky in kept_centres):
            continue  # overlaps an already-kept crop
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(w, cx + half), min(h, cy + half)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        views.append(make_view(image.crop((x0, y0, x1, y1)), (x0, y0, x1, y1), 1.0))
        kept_centres.append((cx, cy))
    fovea = views

    # uniform: whole image downsampled to match the fovea arm's token budget.
    dummy = type("B", (), {"estimate_visual_tokens": staticmethod(
        lambda size: -(-size[0] // 28) * (-(-size[1] // 28)))})()
    budget = visual_tokens(dummy, fovea)
    patch = 28
    import math
    r = min(1.0, patch * math.sqrt(budget) / math.sqrt(w * h))
    uni_img = image.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
    uniform = [make_view(uni_img, (0, 0, w, h), r)]

    return {"full-res": full, "uniform": uniform, "fovea": fovea}


def run_item(backend, image, fixations, item, params):
    """Score all three arms on one (image, question). Returns per-arm dict."""
    arms = build_arms(image, fixations, params)
    full_tokens = visual_tokens(backend, arms["full-res"])
    boxes = item.get("target_boxes")
    rows = {}
    for name, views in arms.items():
        tokens = visual_tokens(backend, views)
        visible = target_visible(views, boxes, params["min_target_px"])
        payload = {
            "images": [v["image"] for v in views],
            "question": item["question"],
            "choices": item["choices"],
            "oracle": {"target_visible": visible, "answer": item["answer"]},
        }
        letter = backend.answer(payload)  # may be None (abstain -> scored wrong)
        chosen = None
        if letter is not None and 0 <= ord(letter) - 65 < len(item["choices"]):
            chosen = item["choices"][ord(letter) - 65]
        rows[name] = {
            "correct": int(chosen is not None and chosen == item["answer"]),
            "tokens": tokens,
            "token_fraction": tokens / full_tokens if full_tokens else 0.0,
            "real_tokens": backend.count_tokens(payload) if params["count_tokens"] else None,
            "target_visible": visible,
        }
    # Where the targets are annotated: did an attention crop (not the global
    # view) cover them, and how deep in the scanpath does coverage come? This
    # separates "attention missed the target" from "the VLM misread it".
    if boxes:
        rows["fovea"]["crop_hit"] = target_visible(arms["fovea"][1:], boxes, 0)
    rows["item"] = {
        "question_id": item.get("question_id"),
        "category": item.get("category"),
        "n_fixations": len(fixations),
        "target_rank": target_fixation_rank(fixations, boxes, params["fovea_side"], image.size),
    }
    return rows


def synthetic_item(params):
    """A high-res image with one small coloured target the question asks about;
    used by --demo and the smoke test (no dataset, no key)."""
    Image = _pil()
    from PIL import ImageDraw
    W, H = 1600, 1200
    img = Image.new("RGB", (W, H), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    for i in range(120):  # clutter so uniform-downsample genuinely loses the target
        x = (i * 137) % (W - 20)
        y = (i * 91) % (H - 20)
        draw.rectangle([x, y, x + 12, y + 12], fill=(90, 90, 90))
    # The target: a small red square in the lower right — the answer is its colour.
    tx, ty = 1360, 980
    draw.rectangle([tx, ty, tx + 26, ty + 26], fill=(220, 40, 40))
    item = {
        "question": "What is the colour of the small square marker?",
        "choices": ["red", "green", "blue", "yellow"],
        "answer": "red",
        "target_boxes": [(tx, ty, tx + 26, ty + 26)],
    }
    return img, item


def summarize(results):
    summary = {}
    have_real = bool(results) and all(r["full-res"]["real_tokens"] for r in results)
    for arm in ("full-res", "uniform", "fovea"):
        acc = [r[arm]["correct"] for r in results]
        frac = [r[arm]["token_fraction"] for r in results]
        lo, hi = bootstrap_ci(acc)
        row = {
            "accuracy": sum(acc) / len(acc) if acc else 0.0,
            "accuracy_ci": [lo, hi],
            "mean_token_fraction": sum(frac) / len(frac) if frac else 0.0,
            "n": len(acc),
        }
        if have_real:
            # Real token fraction vs the same item's full-res real tokens.
            real_frac = [r[arm]["real_tokens"] / r["full-res"]["real_tokens"] for r in results]
            row["mean_real_tokens"] = sum(r[arm]["real_tokens"] for r in results) / len(results)
            row["mean_real_token_fraction"] = sum(real_frac) / len(real_frac)
        # Where targets are annotated: how often this arm delivered them
        # legibly, and accuracy split on that.
        seen = [r[arm] for r in results if r[arm].get("target_visible") is not None]
        if seen:
            row["delivered_rate"] = _mean([int(x["target_visible"]) for x in seen])
            row["accuracy_if_delivered"] = _mean([x["correct"] for x in seen if x["target_visible"]])
            row["accuracy_if_missed"] = _mean([x["correct"] for x in seen if not x["target_visible"]])
        summary[arm] = row
    boxed = [r for r in results if r["fovea"].get("crop_hit") is not None]
    if boxed:
        ranks = [r["item"]["target_rank"] for r in boxed]
        summary["fovea"]["crop_hit_rate"] = _mean([int(r["fovea"]["crop_hit"]) for r in boxed])
        summary["targets"] = {
            "n": len(boxed),
            "covered_by_top": {str(k): _mean([int(x is not None and x <= k) for x in ranks])
                               for k in (1, 3, 5, 10)},
            "never_covered": _mean([int(x is None) for x in ranks]),
        }
    return summary


def _mean(values):
    return sum(values) / len(values) if values else None


def format_table(summary):
    have_real = "mean_real_token_fraction" in summary["full-res"]
    have_boxes = "delivered_rate" in summary["full-res"]
    header = "%-10s %10s %18s %16s%s%s" % (
        "arm", "accuracy", "95% CI", "token-fraction", "  real-fraction" if have_real else "",
        "  delivered" if have_boxes else "")
    lines = ["VLM front-end (H6): accuracy vs visual-token budget", header, "-" * len(header)]
    for arm in ("full-res", "uniform", "fovea"):
        s = summary[arm]
        extra = "  %13.3f" % s["mean_real_token_fraction"] if have_real else ""
        extra += "  %9.3f" % s["delivered_rate"] if have_boxes else ""
        lines.append("%-10s %10.3f  [%5.3f,%5.3f] %16.3f%s" % (
            arm, s["accuracy"], s["accuracy_ci"][0], s["accuracy_ci"][1], s["mean_token_fraction"], extra))
    if "targets" in summary:
        t, top = summary["targets"], summary["targets"]["covered_by_top"]
        lines.append("fovea crop-hit %.3f; target covered by fixation #1 %.2f, top-3 %.2f, top-5 %.2f, "
                     "top-10 %.2f, never %.2f (n=%d)" % (summary["fovea"]["crop_hit_rate"], top["1"],
                                                         top["3"], top["5"], top["10"],
                                                         t["never_covered"], t["n"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--out", default="results/vlm_frontend")
    ap.add_argument("--backend", default="ollama", help="ollama (default; local) | claude | mock")
    ap.add_argument("--model", default=None,
                    help="backend model (defaults: ollama qwen3.8:27b, claude claude-opus-5)")
    ap.add_argument("--vstar", action="store_true", help="run over V*Bench (data/vstar_bench)")
    ap.add_argument("--demo", action="store_true", help="run one synthetic item (no dataset; with --backend mock, no model)")
    ap.add_argument("--limit", type=int, default=50, help="max V*Bench items (0 = all)")
    ap.add_argument("--category", default=None, help="V*Bench category filter")
    ap.add_argument("--k", type=int, default=3, help="number of fovea crops")
    ap.add_argument("--fovea-side", type=int, default=336, help="fovea crop size (px)")
    ap.add_argument("--global-side", type=int, default=512, help="low-res global view long side (px)")
    ap.add_argument("--full-max-side", type=int, default=1512, help="full-res arm cap (px)")
    ap.add_argument("--proc-max-side", type=int, default=1024, help="attention processing cap (px)")
    ap.add_argument("--min-target-px", type=int, default=24, help="legibility floor (on-screen px) for the delivered / mock oracle")
    ap.add_argument("--config", default=None, help="pipeline config for --emit-json")
    ap.add_argument("--count-tokens", action="store_true", help="also record the backend's real count_tokens()")
    ap.add_argument("--check", action="store_true",
                    help="with --demo: exit non-zero unless fovea beats the token-matched uniform arm")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    backend = create_backend(args.backend, **({"model": args.model} if args.model else {}))
    params = {
        "k": args.k, "fovea_side": args.fovea_side, "global_side": args.global_side,
        "full_max_side": args.full_max_side, "min_target_px": args.min_target_px,
        "count_tokens": args.count_tokens,
    }

    results = []
    if args.demo:
        img, item = synthetic_item(params)
        os.makedirs(args.out, exist_ok=True)
        demo_path = os.path.join(args.out, "demo.png")
        img.save(demo_path)
        fixations = emit_fixations(args.binary, demo_path, args.config, args.proc_max_side)
        results.append(run_item(backend, img, fixations, item, params))
    elif args.vstar:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from datasets import vstar
        if not vstar.available():
            sys.exit("V*Bench not found under data/vstar_bench — see eval/datasets/vstar.py")
        if args.backend == "mock":
            print("NOTE: the mock on V*Bench is a legibility oracle — an arm scores iff its views "
                  "deliver the annotated targets (--min-target-px); use --backend ollama or claude "
                  "for real accuracy.", file=sys.stderr)
        Image = _pil()
        items = vstar.iter_items(category=args.category)
        for n, item in enumerate(items):
            if args.limit and n >= args.limit:
                break
            if not item["image"].exists():
                continue
            fixations = emit_fixations(args.binary, str(item["image"]), args.config, args.proc_max_side)
            with Image.open(item["image"]) as im:
                results.append(run_item(backend, im.convert("RGB"), fixations, item, params))
            if (n + 1) % 25 == 0:
                print("  %d items" % (n + 1), file=sys.stderr)
    else:
        sys.exit("nothing to do: pass --demo or --vstar")

    summary = summarize(results)
    print(format_table(summary))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"config": {k: v for k, v in vars(args).items()},
                   "backend": args.backend, "model": getattr(backend, "model", None),
                   "summary": summary}, fh, indent=2)
    # Per-item rows too (real_tokens land here when --count-tokens is on).
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    if args.json:
        print(json.dumps(summary, indent=2))

    if args.check:
        # End-to-end sanity: on the synthetic item the attention crop must
        # deliver the target the token-matched uniform downsample loses. Runs
        # *before* the success banner so the exit code governs the CTest gate.
        fovea, uniform = summary["fovea"]["accuracy"], summary["uniform"]["accuracy"]
        if not (fovea > uniform):
            sys.exit("check failed: fovea accuracy %.3f did not beat uniform %.3f" % (fovea, uniform))
        if summary["fovea"]["mean_token_fraction"] >= 0.9:
            sys.exit("check failed: fovea used %.3f of full-res tokens (no saving)"
                     % summary["fovea"]["mean_token_fraction"])

    print("summary: %s" % os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
