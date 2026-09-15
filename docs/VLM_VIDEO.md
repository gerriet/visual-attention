# Object files as a video token cache (M19, H7)

*Status: v1 harness built and verified on the mock backend, 2026-09-15; first
findings below are about what reaches the VLM (the mock is a legibility
oracle). Real-VLM (Qwen) runs, larger sweeps and DAVIS-2017 are next.*

**H7 — Object files as a video token cache (H1 × H6).** *At a matched
visual-token budget per video, an object-file front-end (object-based IOR +
persistent object files) answers object-centric questions better than
budget-matched uniform frame sampling and better than a spatial-IOR front-end;
the gap grows with object count and speed, and shrinks with tracking errors
(every identity switch is a re-send).*

## Why video

The still-image front-end (`docs/VLM_FRONT_END.md`, M18) leaves the second
stage idle: a single image is a stream of length one, so object files, IOR and
identity never engage. M12 (`docs/DYNAMIC_IOR_STUDY.md`) found that
object-based IOR only ties space-based IOR on *exploration*; the predicted win
was always persistent, identity-keyed memory. A video VLM front-end is the task
that needs exactly that: a video VLM pays tokens for every frame, and deciding
which pixels to (re)send requires knowing *what* has been seen, not *where*.
Spatial memory cannot tell "the same object moved" from "a new object arrived",
so it re-sends moved objects (wasted tokens) and skips newcomers that appear
where an earlier crop was (misses).

## How it works

- **Scenes:** `tools/make_dynamic_scene.py --tags --late` — coloured disks
  that move and bounce at 1280×960, some arriving mid-video, each carrying a
  small code ("K7", ~14 px tall) legible only at native resolution. The flags
  are additive; M12's scenes are unchanged without them.
- **Attention:** the system runs `--attend` on a 320-px copy — exactly M12's
  scene geometry (radius 16, speed 6 px/frame) — under the `spatial-ior` and
  `object-ior` behaviors; crops come from the native frames.
- **Questions:** per object, "what code is written on the ⟨colour⟩ disk?"
  (four options) — accuracy is the share of objects whose detail reached the
  VLM.

| Arm | What the VLM sees (per question) | Tokens |
|---|---|---|
| `frames-full` | 4 evenly spaced native frames | reference (~11×) |
| `frames-uniform` | the same 4 frames, downsampled to the crop budget | matched |
| `space-ior` | overview frame + K crops at spatial-IOR fixations, one per new *location* | matched |
| `object-ior` | overview frame + K crops, one per *object file* | matched |
| `oracle` | overview frame + one crop per ground-truth object | matched |

`--gt-identity` adds `space-ior-gtid` / `object-ior-gtid`: the behaviors'
fixations deduplicated by *ground-truth* identity — a perfect tracker. The gap
between `object-ior` and `object-ior-gtid` is what the object-file tracker
loses to label switches; the gap to `space-ior` is what identity memory is
worth. M12's scanpath scores (coverage, latency, waste) and two identity
statistics (distinct labels per attended object; share of fixations on no
object) ride along.

## First findings (mock — what reaches the VLM)

Five scenes (6 objects, 2 arriving late, 40 frames, K = 6), 30 questions per
arm, thesis object files with M12's tracking aids:

| Arm | delivered | tokens / question |
|---|---|---|
| frames-full | 1.00 | 6440 |
| frames-uniform | **0.00** | 616 |
| space-ior | 0.50 | 560 |
| object-ior | 0.43 | 560 |
| space-ior-gtid (perfect tracker) | 0.97 | 560 |
| object-ior-gtid (perfect tracker) | 0.97 | 560 |
| oracle | 1.00 | 560 |

- **Budget-matched frames are blind** to the codes; crops aren't.
- **Attention finds every object** (scanpath coverage 1.00 for both
  behaviors), and with a perfect tracker either behavior's crops deliver 97%
  of the codes. **Identity memory is worth ~+0.5 of all codes at the same
  budget** — H7's lever is real and large.
- **The object files lose all of it:** 4.4 distinct labels per attended object
  and ~a quarter of fixations on no object leave `object-ior` slightly *below*
  location-deduplicated crops. The same holds on a plain native-320 M12 scene,
  so neither the codes nor the downscaling cause it.

**Diagnosis.** The annotated object-file frames show two sources, both in
*segmentation*, not correspondence: (1) each moving disk falls apart into
fragments — a moving uniform disk is salient mostly at its leading and
trailing edges (the onset channel), so the two crescents become two clusters,
two object files, and the focus hops between them; (2) diffuse background
saliency is segmented as one giant "object" (the off-object fixations).
Correspondence adds a third: an object unobserved for a few frames drifts past
the fixed 30-px revival radius and comes back as a new file.

**Fixes (opt-in; thesis defaults unchanged, goldens untouched).**

- *Persistent identity* (`ObjectFileStore`, `attention_system.object_files.
  persistent_identity`): an unmatched cluster revives the inactive file with
  the lowest position + colour cost, inside a gate that widens with the time
  unseen, or anywhere if it is a clear colour look-alike; a clearly different
  colour is never the same object (so a newcomer where an old object vanished
  gets its own file); inactive files never age out. Shipped as
  `configs/attend_identity.yaml`.
- *Segmentation* (`attention_system.segment_close`, `max_cluster_fraction`):
  a morphological closing that bridges an object's fragments, and a size cap
  that drops regions too large to be an object.

Persistent identity alone barely moves the video result (object-ior 0.43 →
0.47; labels per object unchanged) — the churn is mostly fragmentation. The
segmentation sweep (3 scenes, persistent identity on, mock):

| Segmentation | object-ior (perfect tracker) | space-ior (perfect tracker) | labels / object (obj, space) | off-object (obj, space) |
|---|---|---|---|---|
| thesis | 0.39 (0.94) | 0.44 (0.94) | 4.4, 2.9 | 0.22, 0.28 |
| closing 4 px | 0.39 (0.94) | 0.39 (0.94) | 3.7, 2.7 | 0.24, 0.35 |
| closing 8 px | 0.50 (0.78) | 0.39 (0.78) | 1.7, 1.5 | 0.37, 0.30 |
| size cap 15% | 0.44 (1.00) | 0.44 (1.00) | 4.1, 2.6 | 0.22, 0.20 |
| closing 8 + cap 15% | **0.56** (0.78) | 0.39 (0.67) | 1.6, 1.5 | 0.34, 0.26 |

Closing fixes the fragmentation (4.4 → 1.6 labels per object) and, with it,
`object-ior` overtakes `space-ior` for the first time (0.56 vs 0.39) — a hint
of H7 on 18 questions per arm, not a result. But blind closing also merges
neighbouring objects (the perfect-tracker ceiling drops to 0.78), and a third
of fixations still land on no object. The size cap has no downside.

**H1 side effect.** On M12's own exploration study (6 seeds per regime,
`eval/dynamic_ior.py`), persistent identity moves object-based IOR from worse
to tied-or-better in the regimes it was built for — see
`docs/DYNAMIC_IOR_STUDY.md`, "Persistent identity (M19)".

## Next

1. **Segmentation that separates touching objects** — colour-connected
   proposals or a watershed split instead of blind closing — and a
   confirmation rule for crops (an object file must be seen for a few frames
   before it earns a crop), so junk regions stop consuming the budget.
2. **Real accuracy:** the same arms on Qwen (local, `--backend ollama`), more
   seeds, sweeps over speed × object count × K.
3. **Real video:** DAVIS-2017 with templated questions from the masks.
4. **Object files as working memory:** labels and trajectories handed to the
   VLM as text next to the crops (closes M13 Tier 3); re-send on change.

## Running it

```bash
# mock (legibility oracle), with the perfect-tracker decomposition arms
eval/vlm_video.py --backend mock --seeds 5 --gt-identity
# persistent identity + segmentation knobs
eval/vlm_video.py --backend mock --seeds 5 --gt-identity --config configs/attend_identity.yaml
# real VLM (local Qwen via Ollama), real token counts
eval/vlm_video.py --seeds 5 --count-tokens --config configs/attend_identity.yaml
```

## Files

- `tools/make_dynamic_scene.py` — `--tags` (codes), `--late` (arrivals), `--tag-size`
- `eval/vlm_video.py` — the harness (arms, questions, legibility oracle, identity statistics)
- `configs/attend_identity.yaml` — persistent identity switched on
- `src/system/object_file.cpp` — `persistent_identity` correspondence;
  `src/system/attention_system.cpp` — `segment_close` / `max_cluster_fraction`
  and the `attention_system:` config section
- Tests: `eval/tests/test_vlm_video.py`; Catch2 `[identity]`, `[segment]`,
  `[config]` cases in `tests/test_system.cpp`; CTest `vlm_video_smoke`
