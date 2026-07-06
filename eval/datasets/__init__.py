"""Fixation/stereo dataset adapters and a benchmark loader dispatcher.

`load_for_benchmark(name, root, limit)` returns
(image_paths, gt_maps_by_stem, gt_points_by_stem) for a fixation dataset, so
attention_eval.benchmark can score models against ground truth. Datasets are
not shipped; each adapter module documents its download steps.
"""

from pathlib import Path


def load_for_benchmark(name, root=None, limit=0):
    if name == "mit1003":
        from . import mit1003
        kwargs = {"root": root} if root else {}
        images, gt_maps, gt_points = [], {}, {}
        for i, (stimulus, fix_map, fix_pts) in enumerate(mit1003.iter_stimuli(**kwargs)):
            if limit and i >= limit:
                break
            stem = Path(stimulus).stem
            images.append(stimulus)
            gt_maps[stem] = mit1003.load_fixation_map(fix_map)
            gt_points[stem] = mit1003.load_fixation_points(fix_pts)
        return images, gt_maps, gt_points
    raise ValueError(f"Unknown dataset '{name}'. Available: mit1003")
