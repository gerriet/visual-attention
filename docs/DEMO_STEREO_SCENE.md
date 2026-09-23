# Plan: a stereo demonstration of the attention mechanism

*2026-09-23. Track: **modern** (docs/adr/0005) — nothing here touches the frozen
replication profiles. This is a plan, not an implementation. Status of every
engineering item below was checked against the code today; where something does
not exist yet it says so.*

## What is being built

A 15-second synthetic 3D scene, filmed by a static stereo camera at a mobile
robot's eye height, with two or three people and one or two conspicuous objects
that get handed between them. The system runs on the stereo stream, and the
result is a composited video showing, frame by frame, what the model computes:
the five feature maps, the fused saliency, the neural field's activity, the
selected focus, and the object files — each drawn as a card beside the image
with an arrow to the thing it refers to.

Two audiences, one pipeline: a dense "instrument panel" version for the papers
and a reduced version for the README.

## Why this is worth doing beyond the picture

The scene is *authored*, so its ground truth names everything that matters — every
object's identity, 3D position and mask at every frame. That is exactly what
DAVIS could not give us: the real-video test of H1 was inconclusive because 42–61%
of fixations landed on salient things the annotation does not name
(`docs/DYNAMIC_IOR_STUDY.md`). A scene we build ourselves has no such mismatch,
and it is 3D and stereo, which DAVIS is not. Section 7 below sketches that
follow-up; the demonstration comes first and stands on its own.

---

## 1 · Decisions to take before any work starts

| # | Decision | Recommendation | Why it matters |
|---|---|---|---|
| D1 | Renderer | **Blender**, headless, driven by a Python script (`blender -b -P scene.py`) | Free, scriptable, exact ground truth (object-index and depth passes), built-in stereo rig. **Not currently installed** (`brew install --cask blender`, ~1 GB). |
| D2 | Human figures | **Articulated primitives** (capsules and spheres on a simple armature), keyframed | CC0 by construction, no downloads, fully reproducible from the script; at 3–5 m they read as people. Rigged characters (MakeHuman, Mixamo) can be dropped in later — licence to check first. |
| D3 | Selection stage | **2D neural field, with depth as one fused feature** for v1 | The 2D field persists across frames (`RunState::field_activity`), which is the whole point — hysteresis and tracking. The 3D field currently does **not** persist its volume (see G3). |
| D4 | Recognition labels on the cards | Optional, off by default | The M13 processors can put "person" on an object file, but need model weights (`tools/fetch_models.py`). Nice, not necessary. |
| D5 | Output | 1920×1080, 25 fps, H.264, plus stills and a short GIF | Matches the README-GIF goal in the papers plan. |

## 2 · The camera and the geometry that constrains the scene

These numbers are not decoration: two of our features only work in a band, and
the scene has to be built inside it.

**Rig.** Two cameras, **baseline 120 mm** (a robot's head, a little wider than
human interocular), **parallel** convergence — no toe-in — so the rendered pair
is already rectified and disparity is horizontal only, which is what
`StereoFeature` assumes. Height 1.35 m, horizontal FOV 60°, static for the whole
take.

**Render size 640×480**, so *f* ≈ 554 px; the stereo feature works internally at
a 256-px long side, where *f* ≈ 222 px.

**Disparity budget.** `configs/thesis/stereo.yaml` searches 16 px at the 256-px
working size, so

```
d(256) = f·B/Z = 222 × 0.12 / Z = 26.6 / Z   px
```

| Distance Z | disparity at 256 px | verdict |
|---|---|---|
| 1.7 m | 15.6 px | at the limit |
| 2.5 m | 10.6 px | good |
| 4.0 m |  6.7 px | good |
| 8.0 m |  3.3 px | shallow but usable |
| 13 m  |  2.0 px | floor of usefulness |

**So: keep everything between about 2 m and 8 m.** Put the back wall at 7–8 m so
it still has a measurable disparity and the depth feature separates people from
background.

**Object size and the symmetry feature.** Symmetry uses radii 6–15 px at each of
three pyramid levels of a 256-px working image, i.e. it responds to object radii
of roughly 6–60 px at 256 — about 2–23% of image width. At 640 px that is a
diameter of roughly **30–300 px**. A 0.30 m ball at 2.5 m images at ≈66 px:
comfortably mid-band. A 0.15 m cup at 4 m is ≈21 px — below the band, so if the
second object is small, either bring it closer or make it bigger. People at
3.5 m are ≈270 px tall, well inside.

**Colour.** The room should be desaturated (grey-beige walls, muted clothing) and
the salient objects strongly saturated (a red ball, a yellow box). Colour
contrast in MTM is the strongest channel; if everything is colourful the feature
maps become mush and the demo shows nothing.

## 3 · Choreography (15 s at 25 fps = 375 stereo pairs)

Every phase exists to exercise one part of the mechanism. Lateral separation
keeps occlusion low; the handover happens with an arm's length of gap and the
people never cross in front of each other.

| Time | What happens | What it demonstrates |
|---|---|---|
| 0–2 s | A (left, 3.0 m) holds the red ball; B (right, 3.5 m) stands still; C (centre-back, 5.5 m) idle | Bottom-up salience: the ball dominates colour contrast and symmetry; the field settles on a small number of clusters |
| 2–4 s | A raises and waves the ball | Onset fires; selection moves to the ball |
| 4–7 s | **Handover A → B**: the ball travels laterally between them | Object-file correspondence across a handover: the ball must keep its identity while changing hands, and the *hands* must not steal it |
| 7–9 s | C waves both arms | Competing event: inhibition of return should let attention leave the ball, visit C, and come back — the visible signature of the mechanism |
| 9–12 s | B walks from 3.5 m to 2.2 m carrying the ball | Depth change: disparity grows, the depth feature tracks it, the object file's depth field updates |
| 12–15 s | A uncovers a yellow box on a table at 4 m | A genuinely new object: latency to first fixation is H1's primary metric, made visible |

Ground truth exported per frame: object id, 3D position, image-space centroid,
bounding box, instance mask, visibility. Written in the study's own
`dynamic-scene-gt/v1` schema so the existing scorer can read it unchanged.

## 4 · Engineering, with today's status

### G1 — `--attend` cannot read a stereo stream *(does not exist yet)*
`StereoImageSource` exists in `frame_source.{h,cpp}` and sets `Frame::stereo_right`,
but `process_attend` in `src/main.cpp` only ever builds an `ImageListSource` or a
`VideoFrameSource`. Needed: `--attend <left_dir> --right <right_dir>` wiring the
stereo source in. Small and self-contained; it also makes stereo streams
available to every study, not just this demo.

### G2 — per-frame trace output *(does not exist yet)*
`--emit-features` writes feature maps but is wired for the single-image, `--config`
and `--stereo` paths; the attend loop writes only `objects.png` and the final
scanpath. Needed: `--emit-trace <dir>`, writing per frame
- `frame_%04d.json`: focus; **all** active *and* inactive object files with label,
  centroid, bbox, size, saliency, avg_saliency, created/last_seen/last_selected,
  selection_count, appearance colour, per-feature means (`ObjectFile::features`
  already exists), trajectory, recognition labels;
- `feature_<name>_%04d.png`, `saliency_%04d.png`, `field_%04d.png` as 16-bit maps
  on the fixed [0,1] scale the writer already uses.

This is the only substantial C++ work, and it is reusable: the same trace feeds
README GIFs and paper figures.

### G3 — the 3D field does not integrate over time *(known limitation)*
`NeuralField3DSelection` writes only the depth-collapsed activity into
`RunState::field_activity` and starts from rest each frame; there is no persisted
volume. For a dynamic demo that removes exactly the property worth showing. Hence
D3 (use the 2D field for v1). Making the 3D field stateful — a `field_volume` in
`RunState` — is a worthwhile separate piece of work and would let the demo show
selection in (x, y, disparity), which is the thesis's ch. 6.4 architecture.

### G4 — parameters to tune on a 3-second development clip
Field `field_max_size` (64 is coarse for 640×480; 96–128 may read better),
`cycles_per_frame` (20, from the dissertation system), the object-file
correspondence radius against the observed per-frame motion in pixels, and the
feature weights so that no single channel saturates the fused map. Tune on a
short clip, then freeze before the final render.

### G5 — performance
Stereo is the most expensive feature. Measure on 25 frames first; 375 frames at
even 1 s/frame is a 6-minute run, so this is very unlikely to be a problem, but
measure rather than assume.

## 5 · The composite visualisation

Rendered in Python from the trace (`eval/visualize_demo.py`), frames encoded with
ffmpeg. Python rather than the C++ visualiser because the layout will be iterated
on many times and none of it belongs in the library.

```
┌──────────────────────────────────────────────┬─────────────────────────┐
│                                              │  OBJECT FILES           │
│   left camera image                          │  ┌───────────────────┐  │
│   · object-file outlines, one stable colour  │  │ #3  ball          │  │
│     per identity                             │◄─┤ seen 4.2 s · depth │  │
│   · current focus: white outline + spotlight │  │ 2.4 m · sel ×5     │  │
│   · arrows from each card to its object      │  │ [colour|ecc|sym|d] │  │
│                                              │  └───────────────────┘  │
│                                              │  ┌───────────────────┐  │
│                                              │  │ #1  person A      │  │
│                                              │◄─┤ …                 │  │
│                                              │  └───────────────────┘  │
├──────────────────────────────────────────────┴─────────────────────────┤
│ colour  eccentricity  symmetry  depth  onset │ saliency │ neural field │
├────────────────────────────────────────────────────────────────────────┤
│ timeline: one row per object file, marked where it was the focus    ▲  │
└────────────────────────────────────────────────────────────────────────┘
```

Design decisions that make it readable:

- **One colour per object-file identity**, held for the file's whole life. When
  correspondence keeps identity through the handover, the colour does not change
  — that is the claim, shown rather than stated. If identity breaks, the colour
  changes and the failure is equally visible. No cheating either way.
- **Cards leave, they do not vanish**: an inactive file greys out and slides down
  rather than disappearing, so "the model still knows about it" is visible.
- **The timeline is the money shot.** One row per object file, a mark at every
  frame where it held the focus. Inhibition of return appears as a pattern:
  attention does not return to a row it has just left. Fifteen seconds of it says
  more than the prose does.
- **Feature maps keep a fixed colour scale** across the whole video (the emitter
  already writes on an absolute [0,1] scale), so brightness changes mean
  something. Per-frame normalisation would be a lie of exactly the kind Finding A
  was about.
- **The field panel** shows activity with the surviving clusters outlined, so the
  step from "a map" to "a handful of discrete things" is visible — that is the
  first selection stage, and it is the part nobody can see in a saliency paper.

**Reduced README version**: camera view with outlines and focus, plus saliency and
the timeline. No feature strip, no cards.

## 6 · Phases and rough effort

| Phase | Work | Effort |
|---|---|---|
| 0 | Decisions D1–D5; install Blender | — |
| 1 | `tools/make_stereo_scene.py` (Blender): room, three figures, two objects, choreography, stereo rig, ground-truth export | 1 milestone |
| 2 | G1 + G2: stereo `--attend`, `--emit-trace` | 1 milestone |
| 3 | `eval/visualize_demo.py`: layout, cards, arrows, timeline, encode | 1 milestone |
| 4 | G4 tuning on a 3-second clip; freeze parameters | small |
| 5 | Final render, stills, GIF, a paragraph in the README | small |

Phases 1 and 2 are independent and can be done in either order; phase 3 needs a
trace from phase 2, which can be produced from any existing sequence
(`data/test_images/motion_seq`) before phase 1 exists.

## 7 · The follow-up this makes possible

Once the generator exists, the scene is a parameterised family, not one clip:
number of people, object count, speeds, handover timing, depth range, clutter.
That gives a **controlled 3D stereo benchmark with complete ground truth**, which
is what the real-video test of H1 needed and DAVIS could not provide. The same
arms (`eval/dynamic_ior.py`) would run on it unchanged, since the ground truth is
written in the schema they already read. If that works, the paper's weakest
section — "inconclusive on natural video" — gains a companion result on data where
the question is decidable, with the honest caveat that we authored the scene.

Worth stating plainly in advance, so it cannot be claimed later as a surprise: a
scene built by the same people who built the model is not independent evidence
about the world. It is evidence about the mechanism under conditions we control,
which is what the synthetic H1 regimes already are — with the difference that
this one is 3D, stereo, and has people in it.
