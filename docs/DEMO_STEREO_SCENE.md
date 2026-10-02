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
| D2 | Human figures | **Articulated primitives** — decided 2026-10-02 | CC0 by construction, no downloads, reproducible from the script. They also read as what this is: a controlled simulation. Photoreal humans would add texture and clothing folds that produce spurious salience, and would invite the misreading that the video shows real-world performance, which the DAVIS result says it does not. `build_figure()` is the one place to change. |
| D3 | Selection stage | **Open again since the G3 fix**: 3D field (x, y, disparity), as the dissertation system ran it, with the 2D field as the fallback | Both now persist across frames. Decide on the development clip: the 3D field is the faithful choice and shows depth competition; the 2D one is cheaper and easier to read. |
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

### G1 — stereo streams in `--attend` *(**done**, 2026-09-23)*
`--attend <left_dir> --right <right_dir>` builds a `StereoImageSource`, so the
depth feature runs on every frame of a sequence. The two directories must hold
the same number of images. Covered by the `attend_stereo_trace` test on a
four-frame generated sequence (`data/test_images/stereo_seq`).

### G2 — per-frame trace output *(**done**, 2026-09-23)*
`--attend --emit-trace <dir>` writes `attention-trace/v1`: per frame, the fused
saliency, the neural field's activity, every feature map, and a `state.json`
with the focus and **every** object file, active and inactive, including
per-feature means and a trajectory tail. Schema and decoding rules:
`docs/INTERCHANGE_FORMAT.md`. Maps are on fixed scales and each frame records
the field's true range, so a reader can see whether the encoding clipped.
Reusable beyond this demo: the same trace feeds README GIFs and paper figures.

Two things learned while building it, both now pinned by tests: a temporal
feature (onset) is absent on frame 0, so the feature list has to be collected
across frames rather than from the first; and `process_frame()` advances the
frame index before returning, so trace directories are numbered by the writer
and the system's index is recorded inside the record.

### G3 — the 3D field does not integrate over time *(a port defect, investigated 2026-09-23)*
`NeuralField3DSelection::select()` constructs a fresh `NeuralField3D` **inside
the call**, so the volume starts from rest on every frame; only the
depth-collapsed activity reaches `RunState::field_activity`. Within one frame it
relaxes normally, which is why nothing ever failed — every test of it is on a
single pair.

This is not a simplification the thesis licenses. See
`docs/replication/REPLICATION_DOSSIER.md`, finding C: the thesis's §6.4 exists
*for* tracking through occlusion over time, its Abb. 6.14 measures exactly that,
and the original `NeuralField3D` holds its activity across frames like the 2D one
— the deployed system's default architecture. Fixing it is a replication-track
change with a stated reason; it does not affect any frozen golden, since those
are single pairs and a single pair has no previous state.

**Fixed on 2026-09-23** (dossier, finding C): `RunState::field_volume` carries
the volume across frames, `NeuralField3D::set_activity()` continues from it, and
the 3D field gained `cycles_per_frame`. So **D3 can be revisited**: the demo can
use `neural-field-3d`, selecting in (x, y, disparity) as the dissertation system
actually ran it, which is a better demonstration than the 2D fallback. Worth
tuning both on the development clip and keeping whichever reads more clearly.

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
| 1 | ~~`tools/make_stereo_scene.py` (Blender): room, three figures, two objects, choreography, stereo rig, ground-truth export~~ — **done** | — |
| 2 | ~~G1 + G2: stereo `--attend`, `--emit-trace`~~ — **done** | — |
| 3 | `eval/visualize_demo.py`: layout, cards, arrows, timeline, encode | 1 milestone |
| 4 | G4 tuning on a 3-second clip; freeze parameters | small |
| 5 | Final render, stills, GIF, a paragraph in the README | small |

Phases 1 and 2 are independent and can be done in either order; phase 3 needs a
trace from phase 2, which can be produced from any existing sequence
(`data/test_images/motion_seq`) before phase 1 exists.

## 6a · What the first render settled (2026-10-02)

The generator exists and the scene is validated against the two constraints the
plan derived, by running the system on a rendered pair rather than by assertion:

| Measured at | Depth-feature response | Implied disparity | Planned |
|---|---|---|---|
| A, 3.0 m | 0.56 | 9.0 px | 8.9 px |
| B, 3.4 m | 0.50 | 8.0 px | 7.8 px |
| C, 6.8 m | 0.25 | 4.0 px | 3.9 px |
| back wall, 7.5 m | 0.00 | — | 3.5 px |

The rig and the disparity budget hold. The wall reads zero because it is an
untextured plane and the variance threshold rejects it — the behaviour finding 15
describes, and welcome here: a flat wall should not be salient in depth.

Three things the staging had to fix, all found by rendering and looking:

- **Cast layout is an image-space problem, not a world-space one.** Placed by
  world coordinates, C stood exactly behind A (both near x≈100 px) and B walked
  out of the right edge. The cast is now positioned by where it lands in the
  640-px frame, and B walks *along its own line of sight* so it grows without
  sliding out.
- **Props compete.** The table and cover started brighter than the wall, which
  made them salient in their own right; they are now at the wall's own tone, so
  the yellow box appearing is the event. They also had to move out of the column
  B ends up occupying, or the ball would have sat on the box at the moment the
  box was revealed.
- A rounding bug in the frame helper (`round()`'s banker's rounding) produced an
  empty frame range at half-second boundaries, rendering a scene with nothing
  animated in it. Fixed, and worth knowing: a preview that silently contains only
  static props looks like a scene bug and is not one.

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
