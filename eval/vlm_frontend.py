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

The VLM is pluggable (eval/vlm_backends.py). The default `mock` backend answers
correctly iff the target is delivered at usable resolution — so the harness is
testable end to end without a model, and the real Claude run just swaps the
backend. On V*Bench (no target box) the mock scores at chance across arms; use
--backend claude for real numbers.

  eval/vlm_frontend.py --vstar --limit 50 --backend claude
  eval/vlm_frontend.py --demo            # synthetic, mock, no dataset/key
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


def target_visible(views, target_box, min_target_px, overlap=0.6):
    """True if any view shows the target region at usable resolution: enough of
    the native target box lies inside the view's source region, and the *visible
    portion* is on-screen at least min_target_px. Models 'downsampling makes the
    target too small to read', and 'a crop that clips most of the target off
    doesn't count'. None when there is no ground-truth box (visibility unknown)."""
    if target_box is None:
        return None
    tx0, ty0, tx1, ty1 = target_box
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
    rows = {}
    for name, views in arms.items():
        tokens = visual_tokens(backend, views)
        payload = {
            "images": [v["image"] for v in views],
            "question": item["question"],
            "choices": item["choices"],
            "oracle": {
                "target_visible": target_visible(views, item.get("target_box"),
                                                 params["min_target_px"]),
                "answer": item["answer"],
            },
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
        "target_box": (tx, ty, tx + 26, ty + 26),
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
        summary[arm] = row
    return summary


def format_table(summary):
    have_real = "mean_real_token_fraction" in summary["full-res"]
    header = "%-10s %10s %18s %16s%s" % (
        "arm", "accuracy", "95% CI", "token-fraction", "  real-fraction" if have_real else "")
    lines = ["VLM front-end (H6): accuracy vs visual-token budget", header, "-" * len(header)]
    for arm in ("full-res", "uniform", "fovea"):
        s = summary[arm]
        extra = "  %13.3f" % s["mean_real_token_fraction"] if have_real else ""
        lines.append("%-10s %10.3f  [%5.3f,%5.3f] %16.3f%s" % (
            arm, s["accuracy"], s["accuracy_ci"][0], s["accuracy_ci"][1], s["mean_token_fraction"], extra))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default="build/attention")
    ap.add_argument("--out", default="results/vlm_frontend")
    ap.add_argument("--backend", default="mock", help="mock | claude")
    ap.add_argument("--vstar", action="store_true", help="run over V*Bench (data/vstar_bench)")
    ap.add_argument("--demo", action="store_true", help="run one synthetic item (mock, no dataset/key)")
    ap.add_argument("--limit", type=int, default=50, help="max V*Bench items (0 = all)")
    ap.add_argument("--category", default=None, help="V*Bench category filter")
    ap.add_argument("--k", type=int, default=3, help="number of fovea crops")
    ap.add_argument("--fovea-side", type=int, default=336, help="fovea crop size (px)")
    ap.add_argument("--global-side", type=int, default=512, help="low-res global view long side (px)")
    ap.add_argument("--full-max-side", type=int, default=1512, help="full-res arm cap (px)")
    ap.add_argument("--proc-max-side", type=int, default=1024, help="attention processing cap (px)")
    ap.add_argument("--min-target-px", type=int, default=24, help="mock: target must show at least this big")
    ap.add_argument("--config", default=None, help="pipeline config for --emit-json")
    ap.add_argument("--count-tokens", action="store_true", help="also record the backend's real count_tokens()")
    ap.add_argument("--check", action="store_true",
                    help="with --demo: exit non-zero unless fovea beats the token-matched uniform arm")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    backend = create_backend(args.backend)
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
            print("WARNING: the mock backend cannot score V*Bench — it carries no target boxes, "
                  "so every arm reports chance. Use --backend claude for real numbers.", file=sys.stderr)
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
                   "backend": args.backend, "summary": summary}, fh, indent=2)
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
