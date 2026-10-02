"""Render the stereo demonstration scene (docs/DEMO_STEREO_SCENE.md) in Blender.

A static stereo rig at a mobile robot's eye height watches three figures and a
conspicuous object that is handed between two of them. Everything is built from
primitives by this script, so the scene is reproducible from source with no
downloaded assets and no licence question; see the "figures" note below.

    blender -b -P tools/make_stereo_scene.py -- --out data/demo_scene

Output:
    <out>/left/0000.png, <out>/right/0000.png, ...   the rectified pair sequence
    <out>/gt.json                                     dynamic-scene-gt/v1
    <out>/scene.json                                  rig and choreography, for the record

The pair feeds the system directly:

    ./build/attention --attend <out>/left --right <out>/right \
        --config configs/demo_stereo.yaml --emit-trace results/demo_trace

Geometry is constrained by two features and the plan states both: the depth
feature searches 16 px at a 256-px working size, so everything lives between
about 2 m and 8 m; and symmetry responds to objects roughly 30-300 px across, so
the ball is 0.30 m at 2.5-3.5 m (47-66 px).

On the figures: capsules and spheres on a keyframed skeleton, not rigged
characters. They segment cleanly into regions, they carry muted colour so the
ball dominates the colour channel, and they read as what this is — a controlled
simulation. Photoreal humans would add texture detail and clothing folds that
produce spurious salience, and would invite the misreading that this shows
real-world performance, which the DAVIS result (docs/DYNAMIC_IOR_STUDY.md) says
it does not. `build_figure()` is the one place to change if that trade is ever
worth making differently.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

# --- the rig, from the plan ---------------------------------------------------

BASELINE = 0.12          # m between the two cameras: a robot head, a bit wider than eyes
CAMERA_HEIGHT = 1.35     # m
CAMERA_PITCH = -5.0      # deg, slightly down, so the nearest figure keeps its feet
FOV_H = 60.0             # deg across the long side
WIDTH, HEIGHT = 640, 480
FPS = 25

BACK_WALL_Y = 7.5

# --- the cast -----------------------------------------------------------------
#  x is right, y is depth away from the camera, z is up; the camera sits at
#  (0, 0, CAMERA_HEIGHT) looking down +y.

#  Staged in *image* coordinates, not world ones: at 640 px wide and 554 px of
#  focal length, A lands near x=80 px, the handover crosses the middle, B sits
#  near x=530 px and C near x=215 px — so nobody stands behind anybody and the
#  plan's "avoid too much occlusion" holds for the whole take.
PERSON_A = {"name": "A", "x": -1.30, "y": 3.00, "shirt": (0.30, 0.34, 0.42)}  # slate
PERSON_B = {"name": "B", "x": 1.30, "y": 3.40, "shirt": (0.36, 0.38, 0.30)}   # olive
PERSON_C = {"name": "C", "x": -1.30, "y": 6.80, "shirt": (0.42, 0.38, 0.36)}  # warm grey

BALL_RADIUS = 0.15        # 0.30 m across
BOX_SIZE = 0.35
TABLE = {"x": -0.20, "y": 5.60, "top": 0.75}

SKIN = (0.52, 0.44, 0.38)
ROOM = (0.55, 0.52, 0.48)
FLOOR = (0.38, 0.36, 0.34)
BALL_COLOUR = (0.75, 0.05, 0.04)    # saturated red: the colour channel should own this
BOX_COLOUR = (0.80, 0.68, 0.05)     # saturated yellow
COVER_COLOUR = (0.47, 0.45, 0.43)   # muted, so the box is the event when it slips off


def seconds(t):
    """Frame index of a time in seconds (frame 0 = t 0). Floor-with-half rather
    than round(), whose banker's rounding turns a half-frame boundary into an
    empty range and renders a scene with nothing animated in it."""
    return int(math.floor(t * FPS + 0.5))


# --- small Blender helpers ----------------------------------------------------

def material(name, rgb, roughness=0.75):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def shade(obj, mat):
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def cylinder(name, radius, depth, mat, vertices=16):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, vertices=vertices)
    obj = bpy.context.object
    obj.name = name
    return shade(obj, mat)


def sphere(name, radius, mat, segments=20):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=segments, ring_count=segments // 2)
    obj = bpy.context.object
    obj.name = name
    return shade(obj, mat)


def cube(name, size, mat):
    bpy.ops.mesh.primitive_cube_add(size=size)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def place_segment(obj, start, end, thickness):
    """Put a unit-depth cylinder along the segment start->end (a limb)."""
    start, end = Vector(start), Vector(end)
    direction = end - start
    length = max(direction.length, 1e-4)
    obj.location = (start + end) * 0.5
    obj.rotation_euler = Vector((0, 0, 1)).rotation_difference(direction.normalized()).to_euler()
    obj.scale = (thickness, thickness, length)


def key(obj, frame, scale=False):
    obj.keyframe_insert("location", frame=frame)
    obj.keyframe_insert("rotation_euler", frame=frame)
    if scale:
        obj.keyframe_insert("scale", frame=frame)


# --- the figures --------------------------------------------------------------

HIP_Z, SHOULDER_Z, HEAD_Z = 1.00, 1.48, 1.68
SHOULDER_DX = 0.20


def build_figure(spec):
    """A figure from primitives: hips, torso, head, two arms, two legs. The limbs
    are unit cylinders repositioned each frame by place_segment()."""
    shirt = material("shirt_" + spec["name"], spec["shirt"])
    skin = material("skin_" + spec["name"], SKIN)
    parts = {
        "hips": cylinder("%s_hips" % spec["name"], 0.15, 0.22, shirt),
        "torso": cylinder("%s_torso" % spec["name"], 0.18, 0.52, shirt),
        "head": sphere("%s_head" % spec["name"], 0.115, skin),
        "arm_l": cylinder("%s_arm_l" % spec["name"], 1.0, 1.0, skin, vertices=12),
        "arm_r": cylinder("%s_arm_r" % spec["name"], 1.0, 1.0, skin, vertices=12),
        "leg_l": cylinder("%s_leg_l" % spec["name"], 1.0, 1.0, shirt, vertices=12),
        "leg_r": cylinder("%s_leg_r" % spec["name"], 1.0, 1.0, shirt, vertices=12),
        "hand_l": sphere("%s_hand_l" % spec["name"], 0.055, skin, segments=12),
        "hand_r": sphere("%s_hand_r" % spec["name"], 0.055, skin, segments=12),
    }
    return parts


def pose_figure(parts, spec, frame, x, y, hands, feet_phase=0.0):
    """Place one figure for a frame. `hands` is (left, right) world positions."""
    bob = 0.015 * math.sin(feet_phase * 2.0)
    parts["hips"].location = (x, y, HIP_Z + bob)
    parts["torso"].location = (x, y, HIP_Z + 0.33 + bob)
    parts["head"].location = (x, y, HEAD_Z + bob)
    for part in ("hips", "torso", "head"):
        key(parts[part], frame)

    shoulder_l = Vector((x - SHOULDER_DX, y, SHOULDER_Z + bob))
    shoulder_r = Vector((x + SHOULDER_DX, y, SHOULDER_Z + bob))
    place_segment(parts["arm_l"], shoulder_l, hands[0], 0.050)
    place_segment(parts["arm_r"], shoulder_r, hands[1], 0.050)
    parts["hand_l"].location = hands[0]
    parts["hand_r"].location = hands[1]
    key(parts["hand_l"], frame)
    key(parts["hand_r"], frame)

    # Legs: hips to feet, with a gentle swing while walking.
    swing = 0.18 * math.sin(feet_phase)
    hip_l = Vector((x - 0.09, y, HIP_Z - 0.10 + bob))
    hip_r = Vector((x + 0.09, y, HIP_Z - 0.10 + bob))
    place_segment(parts["leg_l"], hip_l, Vector((x - 0.09, y - swing, 0.02)), 0.075)
    place_segment(parts["leg_r"], hip_r, Vector((x + 0.09, y + swing, 0.02)), 0.075)
    for part in ("arm_l", "arm_r", "leg_l", "leg_r"):
        key(parts[part], frame, scale=True)


def rest_hands(x, y, bob=0.0):
    return (Vector((x - 0.30, y - 0.05, HIP_Z + 0.05 + bob)),
            Vector((x + 0.30, y - 0.05, HIP_Z + 0.05 + bob)))


# --- choreography -------------------------------------------------------------
#  Each phase exercises one part of the mechanism; see the plan's table.

def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def phase_mix(frame, t0, t1):
    """0 before t0 seconds, 1 after t1, smooth in between."""
    if t1 <= t0:
        return 1.0
    return smoothstep((frame / FPS - t0) / (t1 - t0))


def choreograph(frame):
    """Everything that moves, for one frame: hand positions, body positions, the
    ball, and the cover over the box."""
    t = frame / FPS

    # --- A: raises the ball (2-4 s), reaches out for the handover (4-5.4 s)
    raise_amount = phase_mix(frame, 2.0, 3.0) * (1.0 - phase_mix(frame, 5.6, 6.4))
    reach = phase_mix(frame, 4.0, 5.2) * (1.0 - phase_mix(frame, 6.0, 7.0))
    wave_a = 0.10 * math.sin(t * 7.0) * raise_amount
    a_hands = list(rest_hands(PERSON_A["x"], PERSON_A["y"]))
    a_hands[1] = Vector((
        PERSON_A["x"] + 0.30 + 0.62 * reach,
        PERSON_A["y"] - 0.15 - 0.10 * raise_amount,
        HIP_Z + 0.05 + 0.48 * raise_amount + wave_a))

    # --- B: walks forward 9-12 s, carrying the ball; reaches for the handover
    walk = phase_mix(frame, 9.0, 12.0)
    b_y = PERSON_B["y"] + (2.90 - PERSON_B["y"]) * walk
    # Walk straight toward the camera: holding x/y fixed keeps B at the same
    # image column while it grows, instead of sliding off the right edge.
    b_x = PERSON_B["x"] * b_y / PERSON_B["y"]
    b_phase = t * 5.0 if 9.0 <= t <= 12.0 else 0.0
    b_hands = list(rest_hands(b_x, b_y))
    b_reach = phase_mix(frame, 4.4, 5.2) * (1.0 - phase_mix(frame, 6.2, 7.2))
    b_hands[0] = Vector((
        b_x - 0.30 - 0.60 * b_reach,
        b_y - 0.15,
        HIP_Z + 0.05 + 0.44 * b_reach))

    # --- C: waves both arms 7-9 s — a competing event while the ball is held
    wave_c = phase_mix(frame, 7.0, 7.4) * (1.0 - phase_mix(frame, 8.6, 9.2))
    swing = 0.38 * math.sin(t * 8.0) * wave_c
    c_hands = list(rest_hands(PERSON_C["x"], PERSON_C["y"]))
    c_hands[0] = Vector((PERSON_C["x"] - 0.34 - 0.10 * wave_c, PERSON_C["y"] - 0.12,
                         HIP_Z + 0.05 + 0.62 * wave_c + swing))
    c_hands[1] = Vector((PERSON_C["x"] + 0.34 + 0.10 * wave_c, PERSON_C["y"] - 0.12,
                         HIP_Z + 0.05 + 0.62 * wave_c - swing))

    # --- the ball: in A's hand, then passed to B's, then carried by B
    pass_t = phase_mix(frame, 5.2, 6.0)
    hold_a = a_hands[1] + Vector((0.18, -0.22, 0.0))
    hold_b = b_hands[0] + Vector((-0.18, -0.22, 0.0))
    ball = hold_a.lerp(hold_b, pass_t)
    ball.z += 0.10 * math.sin(math.pi * pass_t)  # a slight arc across the gap

    # --- the cover slips off the box at 12 s: a genuinely new object appears
    cover_slide = phase_mix(frame, 12.0, 12.5)
    cover = Vector((TABLE["x"] + 0.95 * cover_slide,
                    TABLE["y"],
                    TABLE["top"] + BOX_SIZE * 0.5 - 0.62 * cover_slide))

    return {"a_hands": a_hands, "b_hands": b_hands, "c_hands": c_hands,
            "b_x": b_x, "b_y": b_y, "b_phase": b_phase, "ball": ball, "cover": cover,
            "box_visible": cover_slide > 0.55}


# --- scene --------------------------------------------------------------------

def build_room():
    room_mat = material("room", ROOM, roughness=0.9)
    floor_mat = material("floor", FLOOR, roughness=0.95)
    bpy.ops.mesh.primitive_plane_add(size=30)
    floor = bpy.context.object
    floor.name = "floor"
    floor.data.materials.append(floor_mat)
    bpy.ops.mesh.primitive_plane_add(size=16, location=(0, BACK_WALL_Y, 2.5),
                                     rotation=(math.radians(90), 0, 0))
    wall = bpy.context.object
    wall.name = "back_wall"
    wall.data.materials.append(room_mat)
    # Soft, broad light: hard shadows would be salient in their own right.
    bpy.ops.object.light_add(type='AREA', location=(-1.5, 1.0, 3.2))
    bpy.context.object.data.energy = 300
    bpy.context.object.data.size = 4.0
    bpy.ops.object.light_add(type='AREA', location=(2.0, 2.5, 3.0))
    bpy.context.object.data.energy = 190
    bpy.context.object.data.size = 3.0
    scene = bpy.context.scene
    if scene.world is None:   # an empty factory scene has none
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.35, 0.36, 0.38, 1)
    background.inputs[1].default_value = 0.6


def build_camera():
    data = bpy.data.cameras.new("stereo")
    data.lens_unit = 'FOV'
    data.angle = math.radians(FOV_H)
    data.stereo.convergence_mode = 'PARALLEL'   # rectified: disparity is horizontal only
    data.stereo.pivot = 'CENTER'
    data.stereo.interocular_distance = BASELINE
    cam = bpy.data.objects.new("stereo", data)
    bpy.context.collection.objects.link(cam)
    cam.location = (0.0, 0.0, CAMERA_HEIGHT)
    cam.rotation_euler = Euler((math.radians(90.0 + CAMERA_PITCH), 0.0, 0.0))
    bpy.context.scene.camera = cam
    return cam


def project(cam, point, eye_sign):
    """World point -> pixel in one eye's image, by hand (the stereo views are a
    lateral offset of the same camera, so projecting with an offset centre is
    exact and avoids rendering a pass just to recover coordinates)."""
    offset = cam.matrix_world.to_quaternion() @ Vector((eye_sign * BASELINE * 0.5, 0.0, 0.0))
    local = cam.matrix_world.inverted() @ (Vector(point) - offset)
    if local.z >= -1e-6:
        return None, None
    focal_px = (WIDTH * 0.5) / math.tan(math.radians(FOV_H) * 0.5)
    return (WIDTH * 0.5 + focal_px * (local.x / -local.z),
            HEIGHT * 0.5 - focal_px * (local.y / -local.z))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/demo_scene")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--start", type=float, default=0.0, help="skip ahead (for previews)")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.0, help="render size multiplier")
    parser.add_argument("--no-render", action="store_true", help="ground truth only")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = int(WIDTH * args.scale)
    scene.render.resolution_y = int(HEIGHT * args.scale)
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.eevee.taa_render_samples = args.samples
    scene.render.use_multiview = True
    scene.render.views_format = 'STEREO_3D'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.views_format = 'INDIVIDUAL'   # separate _L/_R files
    scene.render.views["left"].file_suffix = "_L"
    scene.render.views["right"].file_suffix = "_R"

    build_room()
    cam = build_camera()

    figures = {spec["name"]: build_figure(spec) for spec in (PERSON_A, PERSON_B, PERSON_C)}
    ball = sphere("ball", BALL_RADIUS, material("ball", BALL_COLOUR, roughness=0.45), segments=28)
    box = cube("box", BOX_SIZE, material("box", BOX_COLOUR, roughness=0.6))
    box.location = (TABLE["x"], TABLE["y"], TABLE["top"] + BOX_SIZE * 0.5)
    cover = cube("cover", BOX_SIZE * 1.18, material("cover", COVER_COLOUR, roughness=0.9))
    table = cube("table", 1.0, material("table", (0.47, 0.45, 0.43), roughness=0.9))
    table.location = (TABLE["x"], TABLE["y"], TABLE["top"] * 0.5)
    table.scale = (0.55, 0.45, TABLE["top"] * 0.5)

    first = seconds(args.start)
    last = max(first, seconds(args.start + args.seconds) - 1)
    scene.frame_start, scene.frame_end = first, last

    tracked = {"ball": ball, "box": box, "A": figures["A"]["torso"],
               "B": figures["B"]["torso"], "C": figures["C"]["torso"]}
    ground_truth = {name: [] for name in tracked}

    for frame in range(first, last + 1):
        state = choreograph(frame)
        pose_figure(figures["A"], PERSON_A, frame, PERSON_A["x"], PERSON_A["y"], state["a_hands"])
        pose_figure(figures["B"], PERSON_B, frame, state["b_x"], state["b_y"], state["b_hands"],
                    feet_phase=state["b_phase"])
        pose_figure(figures["C"], PERSON_C, frame, PERSON_C["x"], PERSON_C["y"], state["c_hands"])
        ball.location = state["ball"]
        key(ball, frame)
        cover.location = state["cover"]
        key(cover, frame)

        # Ground truth by projection: exact, and no extra render pass.
        for name, obj in tracked.items():
            if name == "ball":
                centre, radius = Vector(state["ball"]), BALL_RADIUS
            elif name == "box":
                centre, radius = Vector(box.location), BOX_SIZE * 0.5
            else:
                spec = {"A": PERSON_A, "B": PERSON_B, "C": PERSON_C}[name]
                y = state["b_y"] if name == "B" else spec["y"]
                x = state["b_x"] if name == "B" else spec["x"]
                centre, radius = Vector((x, y, (HIP_Z + HEAD_Z) * 0.5)), 0.42
            px, py = project(cam, centre, -1.0)   # the left image is the stream
            depth = (cam.matrix_world.inverted() @ centre).length
            scale = (WIDTH * 0.5) / math.tan(math.radians(FOV_H) * 0.5) / max(depth, 1e-3)
            half = radius * scale
            visible = (px is not None and -half < px < WIDTH + half and -half < py < HEIGHT + half
                       and (name != "box" or state["box_visible"]))
            ground_truth[name].append({
                "frame": frame,
                "x": int(round(px)) if px is not None else 0,
                "y": int(round(py)) if py is not None else 0,
                "w": int(round(2 * half)), "h": int(round(2 * half)),
                "visible": bool(visible),
                "position_3d": [round(c, 4) for c in centre],
                "distance_m": round(depth, 4),
            })

    gt = {
        "schema": "dynamic-scene-gt/v1",
        "source": "blender-stereo-demo",
        "width": scene.render.resolution_x, "height": scene.render.resolution_y,
        "frames": last - first + 1, "fps": FPS,
        "baseline_m": BASELINE, "camera_height_m": CAMERA_HEIGHT, "fov_deg": FOV_H,
        "objects": [{"id": i, "name": name, "positions": positions}
                    for i, (name, positions) in enumerate(sorted(ground_truth.items()))],
    }
    with open(os.path.join(out, "gt.json"), "w") as fh:
        json.dump(gt, fh)
    with open(os.path.join(out, "scene.json"), "w") as fh:
        json.dump({"cast": [PERSON_A, PERSON_B, PERSON_C], "table": TABLE,
                   "ball_radius_m": BALL_RADIUS, "box_size_m": BOX_SIZE,
                   "back_wall_y_m": BACK_WALL_Y, "camera_pitch_deg": CAMERA_PITCH}, fh, indent=2)
    print("ground truth: %s (%d frames)" % (os.path.join(out, "gt.json"), gt["frames"]))

    if args.no_render:
        return
    raw = os.path.join(out, "_raw")
    os.makedirs(raw, exist_ok=True)
    scene.render.filepath = os.path.join(raw, "f")
    bpy.ops.render.render(animation=True)

    for eye, suffix in (("left", "_L"), ("right", "_R")):
        os.makedirs(os.path.join(out, eye), exist_ok=True)
    for name in sorted(os.listdir(raw)):
        if not name.endswith(".png"):
            continue
        stem = name[:-4]
        for eye, suffix in (("left", "_L"), ("right", "_R")):
            if stem.endswith(suffix):
                index = int(stem[: -len(suffix)].lstrip("f"))
                os.replace(os.path.join(raw, name),
                           os.path.join(out, eye, "%04d.png" % index))
    os.rmdir(raw)
    print("rendered %d stereo pairs into %s" % (gt["frames"], out))


main()
