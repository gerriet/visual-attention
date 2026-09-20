#!/usr/bin/env python3
"""Attention as a VLM token-budget allocator (roadmap M18, H6).

The front-end feeds a vision-language model only the *attended* fovea crops plus
one low-res global view, instead of the full-resolution image. On a task where
the answer hinges on a small region (V*Bench, HR-Bench), that should hold
accuracy at a fraction of the visual tokens — the "gaze tells you where to
compute" recipe, with the model-free attention pipeline as the interpretable
controller (the project's edge; the non-goal is becoming another VLM).

Four arms per question, all answered by the same VLM backend:

  full-res    the whole image (capped to a practical VLM size) — the ceiling
  uniform     the whole image uniformly downsampled to the fovea arm's token
              budget — the honest same-budget baseline (small objects vanish)
  fovea       one low-res global view + K native-res crops around the top-K
              attention fixations (ours, bottom-up)
  fovea-random  the same global view + K crops at uniformly random positions
              (seeded per question) — the floor every crop source has to beat:
              a crop arm that does not beat it has an attention source that
              carries no information about the target. Where targets are
              annotated, the expected coverage of random fixations is reported
              next to the pipeline's ("random fixations cover the target").

Two opt-in arms keep the fovea arm's global view and crop count and change only
*where* the crops sit:

  fovea-oracle  (--oracle) crops centred on the annotated target boxes —
                perfect attention, the upper bound any attention source can
                reach at this budget. Needs target boxes (V*Bench).
  fovea-td      (--top-down) question-conditioned attention (H5xH6): the VLM
                localizes what the question asks about on the same low-res
                global view; that box becomes the M17 `top_down_map`, fused
                into the pipeline's priority map, and the crops come from the
                resulting fixations. The attention pipeline stays the
                controller; the grounding call's tokens are charged to this arm.

Token cost is reported two ways: a provider-independent patch estimate (always)
and the backend's real count_tokens() (when it has one). Only the token
*fraction* vs full-res is compared, so the patch constant cancels.

The VLM is pluggable (eval/vlm_backends.py). The default `ollama` backend runs
a local open-weights VLM (qwen3.8:27b) — free, reproducible, and its real token
count comes back with every answer; `--backend claude` swaps in Claude. The
`mock` backend answers correctly iff the target is delivered at usable
resolution, so the harness is testable end to end without any model (the CI
smoke runs on it); on V*Bench it becomes a model-free legibility oracle.

Where targets are annotated (V*Bench ships boxes), each row also records per
arm whether the targets were delivered legibly, whether a *crop* covered them
(crop_hit), and the rank of the first fixation whose window covers them
(target_rank, target_rank_td) — separating "attention missed the target" from
"the VLM misread it".

  eval/vlm_frontend.py --vstar --limit 50 --count-tokens --oracle --top-down
  eval/vlm_frontend.py --hrbench 8k --limit 20 --count-tokens --top-down
  eval/vlm_frontend.py --vstar --limit 50 --backend claude
  eval/vlm_frontend.py --demo --backend mock                # synthetic, no model
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import bootstrap_ci  # noqa: E402
from vlm_backends import create_backend  # noqa: E402

ARMS = ("full-res", "uniform", "fovea", "fovea-random", "fovea-oracle", "fovea-td")

# Random-fixation baseline: crops per question come from one seeded draw; the
# model-free coverage figure averages this many draws of RANDOM_FIXATIONS each.
RANDOM_DRAWS = 200
RANDOM_FIXATIONS = 10
COVERAGE_KS = (1, 3, 5, 10)

# Appended to the pipeline config for the fovea-td arm (M17 dense top-down).
PRIORITY_YAML = """
priority:
  top_down_weight: %(weight)s
  top_down_map: %(map)s
"""


def _pil():
    from PIL import Image
    return Image


def emit_fixations(binary, native, proc_max_side, config=None, top_down=None):
    """Run the C++ pipeline on a bounded-size copy of `native` (a PIL image) and
    return attention fixations as (x, y, value) in *native* image coordinates,
    ordered by attention sequence. `top_down` = (native-px boxes, weight) adds
    a question-conditioned relevance map through the M17 `top_down_map` slot."""
    w, h = native.size
    scale = min(1.0, proc_max_side / max(w, h))
    proc = native.resize((max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1.0 else native

    with tempfile.TemporaryDirectory() as tmp:
        proc_path = os.path.join(tmp, "proc.png")
        proc.save(proc_path)
        result_path = os.path.join(tmp, "result.json")
        cmd = [binary, proc_path, "--no-display", "--emit-json", result_path]
        if top_down:
            boxes, weight = top_down
            map_path = os.path.join(tmp, "relevance.png")
            relevance_map(proc.size, [[v * scale for v in box] for box in boxes]).save(map_path)
            text = ""
            if config:
                with open(config) as fh:
                    text = fh.read()
            config = os.path.join(tmp, "config.yaml")
            with open(config, "w") as fh:
                fh.write(text + PRIORITY_YAML % {"weight": weight, "map": map_path})
        if config:
            cmd += ["--config", config]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(result_path) as fh:
            fixations = json.load(fh)["fixations"]

    inv = 1.0 / scale if scale < 1.0 else 1.0
    fixations.sort(key=lambda f: f["n"])
    return [(f["x"] * inv, f["y"] * inv, f["value"]) for f in fixations]


def tile_boxes(size, tiles, overlap=0.25):
    """An N x N grid of equal windows covering (w, h), neighbours overlapping by
    `overlap` of a window so an object on a seam is whole in at least one."""
    w, h = size
    tw = w / (tiles - (tiles - 1) * overlap)
    th = h / (tiles - (tiles - 1) * overlap)
    boxes = []
    for row in range(tiles):
        for col in range(tiles):
            x0, y0 = col * tw * (1 - overlap), row * th * (1 - overlap)
            boxes.append((round(x0), round(y0), min(w, round(x0 + tw)), min(h, round(y0 + th))))
    return boxes


def tiled_fixations(binary, native, proc_max_side, config=None, tiles=2, min_distance=96):
    """Attention at native scale: the pipeline runs once on the whole image (the
    gist) and once per tile of an N x N grid, so a small target is not shrunk
    away before stage 1 sees it — every stage of the pipeline works on a bounded
    image, and a 36-px object in a 2000-px photo is 1-4 px at that size.

    Saliency values are normalized per run and not comparable across tiles, so
    the lists are merged by *rank*: round-robin over the runs (whole image
    first, then tiles by the strength of their top fixation), skipping a
    fixation closer than `min_distance` native px to one already taken — the
    overlap would otherwise report an object twice."""
    runs = [emit_fixations(binary, native, proc_max_side, config)]
    for box in tile_boxes(native.size, tiles):
        local = emit_fixations(binary, native.crop(box), proc_max_side, config)
        runs.append([(x + box[0], y + box[1], v) for x, y, v in local])
    head, tail = runs[0], sorted(runs[1:], key=lambda run: -run[0][2] if run else 0.0)
    merged = []
    for rank in range(max(len(run) for run in runs)):
        for run in [head] + tail:
            if rank < len(run):
                x, y, v = run[rank]
                if all(math.hypot(x - mx, y - my) >= min_distance for mx, my, _ in merged):
                    merged.append((x, y, v))
    return merged


def bottom_up_fixations(args, image):
    """The fovea arm's attention source, as configured on the command line."""
    if args.tiles > 1:
        return tiled_fixations(args.binary, image, args.proc_max_side, args.config, args.tiles)
    return emit_fixations(args.binary, image, args.proc_max_side, args.config)


def relevance_map(size, boxes):
    """Grayscale top-down relevance raster: each box filled, padded by a quarter
    of its size and softly blurred. A plateau, not a peak — so *within* the
    relevant region the bottom-up saliency still picks the fixation point."""
    from PIL import Image, ImageDraw, ImageFilter
    raster = Image.new("L", size, 0)
    draw = ImageDraw.Draw(raster)
    blur = 2.0
    for x0, y0, x1, y1 in boxes:
        pad = max(4.0, 0.25 * max(x1 - x0, y1 - y0))
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=255)
        blur = max(blur, pad / 2)
    return raster.filter(ImageFilter.GaussianBlur(blur))


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


def random_fixations(size, n, seed):
    """`n` uniformly placed fixations (x, y, value) — the no-attention baseline.
    Seeded by the caller (the question id), so a rerun and every arm of one run
    see the same positions."""
    import random
    rng = random.Random("random-fixations-%s" % (seed,))
    w, h = size
    return [(rng.uniform(0, w), rng.uniform(0, h), 0.0) for _ in range(n)]


def random_coverage(size, target_boxes, fovea_side, seed, draws=RANDOM_DRAWS):
    """Expected share of random scanpaths whose top K fixation windows cover the
    targets, {K: rate} for COVERAGE_KS — the chance level of `target_rank`.
    None without target boxes."""
    if not target_boxes:
        return None
    hits = dict.fromkeys(COVERAGE_KS, 0)
    for draw in range(draws):
        rank = target_fixation_rank(random_fixations(size, RANDOM_FIXATIONS, "%s-%d" % (seed, draw)),
                                    target_boxes, fovea_side, size)
        for k in COVERAGE_KS:
            hits[k] += int(rank is not None and rank <= k)
    return {str(k): hits[k] / draws for k in COVERAGE_KS}


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


def _intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def resized(image, long_side):
    Image = _pil()
    w, h = image.size
    if max(w, h) <= long_side:
        return image.copy(), 1.0
    scale = long_side / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS), scale


def fovea_views(image, global_view, fixations, params):
    """The global view + up to K native-res crops around the given fixations,
    deduplicated so near-coincident fixations don't spend the budget twice on
    overlapping pixels."""
    w, h = image.size
    views = [global_view]
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
    return views


def oracle_views(image, global_view, target_boxes, fovea, params):
    """Perfect attention at the fovea arm's budget: one crop centred on each
    annotated target (the window grown to fit a large box, then resized back to
    fovea_side), padded with the fovea arm's own crops to the same crop count."""
    Image = _pil()
    w, h = image.size
    side = params["fovea_side"]
    n_crops = max(len(fovea) - 1, 1)
    views = [global_view]
    for x0, y0, x1, y1 in target_boxes[:n_crops]:
        win = max(side, int(math.ceil(1.25 * max(x1 - x0, y1 - y0))))
        sx0 = int(max(0, min(w - win, (x0 + x1 - win) / 2)))
        sy0 = int(max(0, min(h - win, (y0 + y1 - win) / 2)))
        box = (sx0, sy0, min(w, sx0 + win), min(h, sy0 + win))
        crop, scale = image.crop(box), 1.0
        if win > side:
            scale = side / win
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                               Image.LANCZOS)
        views.append(make_view(crop, box, scale))
    for view in fovea[1:]:
        if len(views) - 1 >= n_crops:
            break
        if not any(_intersects(view["source_box"], kept["source_box"]) for kept in views[1:]):
            views.append(view)
    return views


def build_arms(image, fixations, params, oracle_boxes=None, td_fixations=None, seed=None):
    """Return {arm_name: list_of_views} for one image. The opt-in arms reuse the
    fovea arm's global view and crop count, so they differ from it only in
    where the crops sit."""
    Image = _pil()
    w, h = image.size

    # full-res: whole image capped to a practical VLM size.
    full_img, full_scale = resized(image, params["full_max_side"])
    full = [make_view(full_img, (0, 0, w, h), full_scale)]

    global_img, global_scale = resized(image, params["global_side"])
    global_view = make_view(global_img, (0, 0, w, h), global_scale)
    fovea = fovea_views(image, global_view, fixations, params)

    # uniform: whole image downsampled to match the fovea arm's token budget.
    dummy = type("B", (), {"estimate_visual_tokens": staticmethod(
        lambda size: -(-size[0] // 28) * (-(-size[1] // 28)))})()
    budget = visual_tokens(dummy, fovea)
    patch = 28
    r = min(1.0, patch * math.sqrt(budget) / math.sqrt(w * h))
    uni_img = image.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
    uniform = [make_view(uni_img, (0, 0, w, h), r)]

    arms = {"full-res": full, "uniform": uniform, "fovea": fovea}
    # Same global view and crop count as the fovea arm, crops placed at random:
    # what the front-end is worth with no attention at all.
    arms["fovea-random"] = fovea_views(image, global_view,
                                       random_fixations(image.size, RANDOM_FIXATIONS, seed), params)
    if oracle_boxes:
        arms["fovea-oracle"] = oracle_views(image, global_view, oracle_boxes, fovea, params)
    if td_fixations is not None:
        arms["fovea-td"] = fovea_views(image, global_view, td_fixations, params)
    return arms


def ground(backend, image, item, params):
    """The top-down source for fovea-td: ask the VLM, on a low-res view of the
    whole image (by default the same global view the fovea arm sends; wider
    with --ground-side), where the object the question is about is. Returns
    native-px boxes plus the call's token cost."""
    w, h = image.size
    global_img, _ = resized(image, params.get("ground_side") or params["global_side"])
    oracle = [(x0 / w, y0 / h, x1 / w, y1 / h) for x0, y0, x1, y1 in item.get("target_boxes") or []]
    norm, real = backend.locate({"images": [global_img], "question": item["question"],
                                 "oracle": {"target_boxes": oracle}})
    return {"boxes": [(x0 * w, y0 * h, x1 * w, y1 * h) for x0, y0, x1, y1 in norm],
            "tokens": backend.estimate_visual_tokens(global_img.size),
            "real_tokens": real}


def run_item(backend, image, fixations, item, params):
    """Score every arm on one (image, question). Returns per-arm dicts plus an
    'item' record of the target diagnostics."""
    boxes = item.get("target_boxes")
    grounding, td_fixations = None, None
    if params.get("top_down"):
        grounding = ground(backend, image, item, params)
        # No box back (unparseable reply) -> the arm falls back to bottom-up.
        td_fixations = fixations
        if grounding["boxes"]:
            td_fixations = emit_fixations(params["binary"], image, params["proc_max_side"],
                                          params.get("config"),
                                          top_down=(grounding["boxes"], params["top_down_weight"]))
    arms = build_arms(image, fixations, params,
                      oracle_boxes=boxes if params.get("oracle") else None,
                      td_fixations=td_fixations, seed=item.get("question_id"))
    full_tokens = visual_tokens(backend, arms["full-res"])
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
        real = backend.count_tokens(payload) if params["count_tokens"] else None
        if name == "fovea-td":
            # The grounding call is part of this arm's cost.
            tokens += grounding["tokens"]
            real = real + grounding["real_tokens"] if real is not None and grounding["real_tokens"] else real
        rows[name] = {
            "correct": int(chosen is not None and chosen == item["answer"]),
            "letter": letter,  # with item.answer_letter: audits position bias after the fact
            "tokens": tokens,
            "token_fraction": tokens / full_tokens if full_tokens else 0.0,
            "real_tokens": real,
            "target_visible": visible,
        }
        # Where the targets are annotated: did a crop (not the global view)
        # cover them? Separates "attention missed" from "the VLM misread".
        if boxes and name.startswith("fovea"):
            rows[name]["crop_hit"] = target_visible(views[1:], boxes, 0)
    rows["item"] = {
        "question_id": item.get("question_id"),
        "category": item.get("category"),
        "answer_letter": item.get("answer_letter"),
        "n_choices": len(item["choices"]),  # chance differs per item (V*Bench has 2-way questions)
        "n_fixations": len(fixations),
        "target_rank": target_fixation_rank(fixations, boxes, params["fovea_side"], image.size),
        "random_coverage": random_coverage(image.size, boxes, params["fovea_side"], item.get("question_id")),
    }
    if grounding is not None:
        rows["fovea-td"]["grounded"] = bool(grounding["boxes"])
        rows["item"]["grounding_boxes"] = [[round(v) for v in box] for box in grounding["boxes"]]
        rows["item"]["target_rank_td"] = target_fixation_rank(td_fixations, boxes, params["fovea_side"],
                                                              image.size)
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
    arms = [a for a in ARMS if results and a in results[0]]
    for arm in arms:
        acc = [r[arm]["correct"] for r in results]
        frac = [r[arm]["token_fraction"] for r in results]
        lo, hi = bootstrap_ci(acc)
        row = {
            "accuracy": sum(acc) / len(acc) if acc else 0.0,
            "accuracy_ci": [lo, hi],
            "mean_token_fraction": sum(frac) / len(frac) if frac else 0.0,
            "n": len(acc),
        }
        if all(r[arm]["real_tokens"] and r["full-res"]["real_tokens"] for r in results):
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
        hits = [r[arm]["crop_hit"] for r in results if r[arm].get("crop_hit") is not None]
        if hits:
            row["crop_hit_rate"] = _mean([int(x) for x in hits])
        grounded = [r[arm]["grounded"] for r in results if "grounded" in r[arm]]
        if grounded:
            row["grounded_rate"] = _mean([int(x) for x in grounded])
        summary[arm] = row
    sizes = [r["item"]["n_choices"] for r in results if r["item"].get("n_choices")]
    if sizes:
        # Guessing accuracy for this item mix (V*Bench mixes 2- and 4-way questions).
        summary["chance"] = _mean([1.0 / n for n in sizes])
    boxed = [r for r in results if r["fovea"].get("crop_hit") is not None]
    if boxed:
        summary["targets"] = _rank_summary([r["item"]["target_rank"] for r in boxed])
        chance = [r["item"]["random_coverage"] for r in boxed if r["item"].get("random_coverage")]
        if chance:
            summary["targets_random"] = {
                "n": len(chance), "draws": RANDOM_DRAWS,
                "covered_by_top": {str(k): _mean([c[str(k)] for c in chance]) for k in COVERAGE_KS}}
        if "target_rank_td" in boxed[0]["item"]:
            summary["targets_td"] = _rank_summary([r["item"]["target_rank_td"] for r in boxed])
    return summary


def _rank_summary(ranks):
    return {
        "n": len(ranks),
        "covered_by_top": {str(k): _mean([int(x is not None and x <= k) for x in ranks]) for k in (1, 3, 5, 10)},
        "never_covered": _mean([int(x is None) for x in ranks]),
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def format_table(summary):
    arms = [a for a in ARMS if a in summary]
    have_real = all("mean_real_token_fraction" in summary[a] for a in arms)
    have_boxes = "delivered_rate" in summary["full-res"]
    header = "%-12s %10s %18s %16s%s%s" % (
        "arm", "accuracy", "95% CI", "token-fraction", "  real-fraction" if have_real else "",
        "  delivered" if have_boxes else "")
    lines = ["VLM front-end (H6): accuracy vs visual-token budget", header, "-" * len(header)]
    for arm in arms:
        s = summary[arm]
        extra = "  %13.3f" % s["mean_real_token_fraction"] if have_real else ""
        extra += "  %9.3f" % s["delivered_rate"] if have_boxes else ""
        lines.append("%-12s %10.3f  [%5.3f,%5.3f] %16.3f%s" % (
            arm, s["accuracy"], s["accuracy_ci"][0], s["accuracy_ci"][1], s["mean_token_fraction"], extra))
    if "chance" in summary:
        lines.append("chance (guessing, this item mix): %.2f" % summary["chance"])
    hits = ["%s %.2f" % (a, summary[a]["crop_hit_rate"]) for a in arms if "crop_hit_rate" in summary[a]]
    if hits:
        lines.append("crop-hit (a crop covers the target): " + ", ".join(hits))
    if "grounded_rate" in summary.get("fovea-td", {}):
        lines.append("fovea-td grounded (VLM returned a box): %.2f" % summary["fovea-td"]["grounded_rate"])
    for key, label in (("targets", "bottom-up"), ("targets_td", "top-down")):
        if key in summary:
            t, top = summary[key], summary[key]["covered_by_top"]
            lines.append("%s fixations cover the target: #1 %.2f, top-3 %.2f, top-5 %.2f, top-10 %.2f, "
                         "never %.2f (n=%d)" % (label, top["1"], top["3"], top["5"], top["10"],
                                                t["never_covered"], t["n"]))
    if "targets_random" in summary:
        top = summary["targets_random"]["covered_by_top"]
        lines.append("random fixations cover the target:    #1 %.2f, top-3 %.2f, top-5 %.2f, top-10 %.2f "
                     "(chance level of the rows above; %d draws)"
                     % (top["1"], top["3"], top["5"], top["10"], summary["targets_random"]["draws"]))
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
    ap.add_argument("--hrbench", choices=("4k", "8k"), default=None,
                    help="run over HR-Bench 4K or 8K (data/hr_bench; option cycle 0)")
    ap.add_argument("--demo", action="store_true", help="run one synthetic item (no dataset; with --backend mock, no model)")
    ap.add_argument("--limit", type=int, default=50, help="max dataset items (0 = all)")
    ap.add_argument("--category", default=None,
                    help="category filter (V*Bench: direct_attributes | relative_position; "
                         "HR-Bench: single | cross)")
    ap.add_argument("--k", type=int, default=3, help="number of fovea crops")
    ap.add_argument("--fovea-side", type=int, default=336, help="fovea crop size (px)")
    ap.add_argument("--global-side", type=int, default=512, help="low-res global view long side (px)")
    ap.add_argument("--full-max-side", type=int, default=1512, help="full-res arm cap (px)")
    ap.add_argument("--proc-max-side", type=int, default=1024, help="attention processing cap (px)")
    ap.add_argument("--min-target-px", type=int, default=24, help="legibility floor (on-screen px) for the delivered / mock oracle")
    ap.add_argument("--oracle", action="store_true",
                    help="add the fovea-oracle arm: crops on the annotated targets (needs boxes: V*Bench, --demo)")
    ap.add_argument("--top-down", action="store_true",
                    help="add the fovea-td arm: VLM grounding on the global view -> M17 top_down_map -> crops")
    ap.add_argument("--top-down-weight", type=float, default=1.5, help="M17 top_down_weight for fovea-td")
    ap.add_argument("--ground-side", type=int, default=None,
                    help="long side (px) of the view the VLM grounds on for fovea-td (default: --global-side)")
    ap.add_argument("--config", default=None, help="pipeline config for --emit-json")
    ap.add_argument("--tiles", type=int, default=1,
                    help="N > 1: also run attention on an N x N grid of overlapping native-resolution "
                         "tiles and merge the fixations by rank (small targets survive; N*N+1 pipeline runs)")
    ap.add_argument("--count-tokens", action="store_true", help="also record the backend's real count_tokens()")
    ap.add_argument("--resume", action="store_true",
                    help="continue a killed dataset run: keep the rows in --out/results.json and skip "
                         "their items (use the same settings)")
    ap.add_argument("--check", action="store_true",
                    help="with --demo: exit non-zero unless fovea beats the token-matched uniform arm")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit("binary not found: %s (build first: cmake --build build)" % args.binary)
    if args.oracle and args.hrbench:
        sys.exit("--oracle needs annotated target boxes; HR-Bench has none (use --vstar or --demo)")
    if args.top_down and args.config:
        with open(args.config) as fh:
            if re.search(r"(?m)^priority\s*:", fh.read()):
                sys.exit("--top-down appends its own priority: section; use a --config without one")
    backend = create_backend(args.backend, **({"model": args.model} if args.model else {}))
    params = {
        "k": args.k, "fovea_side": args.fovea_side, "global_side": args.global_side,
        "full_max_side": args.full_max_side, "min_target_px": args.min_target_px,
        "count_tokens": args.count_tokens, "oracle": args.oracle, "top_down": args.top_down,
        "top_down_weight": args.top_down_weight, "ground_side": args.ground_side,
        "binary": args.binary, "config": args.config,
        "proc_max_side": args.proc_max_side,
    }

    results = []
    if args.demo:
        img, item = synthetic_item(params)
        os.makedirs(args.out, exist_ok=True)
        img.save(os.path.join(args.out, "demo.png"))
        fixations = bottom_up_fixations(args, img)
        results.append(run_item(backend, img, fixations, item, params))
    elif args.vstar or args.hrbench:
        from datasets import hrbench, vstar
        if args.vstar:
            if not vstar.available():
                sys.exit("V*Bench not found under data/vstar_bench — see eval/datasets/vstar.py")
            items = vstar.iter_items(category=args.category)
        else:
            if not hrbench.available(args.hrbench):
                sys.exit("HR-Bench not found under data/hr_bench — see eval/datasets/hrbench.py")
            items = hrbench.iter_items(args.hrbench, category=args.category)
        if args.backend == "mock":
            print("NOTE: the mock is a legibility oracle — an arm scores iff its views deliver the "
                  "annotated targets (--min-target-px); without boxes (HR-Bench) it guesses. Use "
                  "--backend ollama or claude for real accuracy.", file=sys.stderr)
        Image = _pil()
        os.makedirs(args.out, exist_ok=True)
        results_path = os.path.join(args.out, "results.json")
        done = set()
        if args.resume and os.path.exists(results_path):
            with open(results_path) as fh:
                results = json.load(fh)
            # Rows from an older harness cannot be mixed in: before the answer
            # letter was recorded the HR-Bench options were served as stored
            # (correct option always "A"), and rows without the random-crop arm
            # would leave that arm with a different item set. Rescore them.
            def current(row):
                return "answer_letter" in row["item"] and "fovea-random" in row
            stale = len(results) - sum(1 for r in results if current(r))
            if stale:
                print("resuming: dropping %d rows scored by an older harness" % stale, file=sys.stderr)
                results = [r for r in results if current(r)]
            done = {r["item"]["question_id"] for r in results}
            print("resuming: %d items already scored in %s" % (len(done), results_path), file=sys.stderr)
        for n, item in enumerate(items):
            if args.limit and n >= args.limit:
                break
            if item["question_id"] in done or not item["image"].exists():
                continue
            with Image.open(item["image"]) as im:
                image = im.convert("RGB")
            fixations = bottom_up_fixations(args, image)
            results.append(run_item(backend, image, fixations, item, params))
            # Written after every item, so a run killed midway (a long local-VLM
            # run under memory pressure) keeps its rows for --resume.
            with open(results_path, "w") as fh:
                json.dump(results, fh, indent=2)
            if (n + 1) % 25 == 0:
                print("  %d items" % (n + 1), file=sys.stderr)
    else:
        sys.exit("nothing to do: pass --demo, --vstar or --hrbench")

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
        # deliver the target the token-matched uniform downsample loses, and
        # the opt-in arms' crops must land on it. Runs *before* the success
        # banner so the exit code governs the CTest gate.
        fovea, uniform = summary["fovea"]["accuracy"], summary["uniform"]["accuracy"]
        if not (fovea > uniform):
            sys.exit("check failed: fovea accuracy %.3f did not beat uniform %.3f" % (fovea, uniform))
        if not (fovea > summary["fovea-random"]["accuracy"]):
            sys.exit("check failed: attention crops did not beat random crops")
        if summary["fovea"]["mean_token_fraction"] >= 0.9:
            sys.exit("check failed: fovea used %.3f of full-res tokens (no saving)"
                     % summary["fovea"]["mean_token_fraction"])
        for arm in ("fovea-oracle", "fovea-td"):
            if arm in summary and summary[arm].get("crop_hit_rate") != 1.0:
                sys.exit("check failed: the %s crops missed the demo target" % arm)

    print("summary: %s" % os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
