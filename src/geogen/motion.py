"""Motions: declarative interaction animation (geogen.103; design: docs/dev/characters.md "Motions").

A motion (``kind: motion``, ``assets/motions/*.yaml``) is a small state machine of *beats*. A beat
is written like a body pose (``pose``, ``bones``, ``ik``) except that an IK target may name a point
on a thing (a role's contact, e.g. ``button``, or ``tv.screen``) instead of body-frame coordinates,
plus where the body stands (``root``), where it looks (``look``) and how long it takes. Several
motions run at once on a body's *channels* (``channels:`` on the skeleton: hips, spine, legs,
left_hand, right_hand, gaze, root): each says which it ``require``s, ``claim``s or only
``prefer``s, and each channel follows its strongest claim. Every beat says what an interruption
does to it (``interrupt: yield | finish | hold | drop``, ``exit``).

``MotionRig`` is the reference solver: deterministic, fixed-step, in Python, for offline review
(``geogen.motion_review``), QA and tests. Each step it advances the beats, arbitrates channels,
lays each beat's pose over its channels (eased from where the body was when the beat began),
places the body (``root``), solves IK for hands and planted feet and turns the head, and
inertializes every channel whose owner changed (the old pose's offset decays; nothing is
authored between any pair of motions). Targets are resolved once to a node plus a local
transform and followed, so a moved chair or a swinging door is tracked. ``trace`` records
beats, owners, emits and interruptions; ``samples`` the per-frame numbers review metrics use.

A run that loses a claimed channel to an overlay pauses and picks up from its beat's ``resume``;
``break_`` drops everything into a reaction (geogen.103.5). Runs that ``share`` a channel (the
gaze) take turns at it by weight; a motion's ``while`` says what it does to a walk, and a role
that is ``held`` borrows the hand already holding its thing (geogen.103.6).
runtime/godot/scripts/motion_rig.gd is the runtime's port (geogen.103.7).

Not yet (later geogen.103 tasks): ``when``/``until``/``do``/``bend``/``layer``, and locomotion
between places (the body slides onto ``root`` targets).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import roots
from .core.ik import LIMBS, solve_limb
from .core.node import SceneNode
from .core.skeleton import Pose, Skeleton
from .core.transform import Transform, matrix_from_quat, quat_from_matrix, slerp
from .fields import known_keys

MOTION_KIND = "motion"
# Grips that are not a hand holding the thing: `grip: any` on a role takes the first other one.
NOT_HELD = ("sit", "lie", "stand", "use", "look", "press", "put", "rest")
MOTIONS_DIR = "motions"

MOTION_KEYS = {
    "kind": "motion",
    "name": "the motion's name (defaults to the file's)",
    "category": "where a catalogue lists it: idle | emote | dance | exercise | action | reaction (motion_catalogue)",
    "summary": "one line on what it looks like, for catalogues and galleries",
    "props": "{role: {asset, on, at} | {asset, behind}}: what a demonstration stages for its roles (a cup on a table in front; a seat `behind` m behind the person, facing the way they do)",
    "demo": "[{at, start: <motion>, <role>: <role> | <role>@start, ...}]: how a demonstration shows it when it needs "
            "others around it (take, carry and set_down: take it, carry it, put it back where it stood); roles name "
            "this motion's staged props, `@start` where one stood when the demonstration began",
    "roles": "{role: {grip, optional, held}}: the things it acts on, bound by whoever runs it; `held: true`: a hand must "
             "already hold the thing and the motion borrows that hand (written for the right hand, it runs mirrored "
             "when the left has it); its `attach` re-grips the thing where it is, its `release` lets go for good",
    "params": "{name: {default}}: build-time values for {expr} in its beats",
    "holds": "{require: [channels], claim: [...], prefer: [...], share: {channel: weight}}",
    "posture": "{from, to}: its edge in the posture graph (stand -> sit)",
    "while": "{walk: {speed: 0..1, arm_swing: 0..1}}: what it does to the body's walk while it runs, as shares of "
             "the walk's own (eating slows it; several motions multiply)",
    "layer": "true: it outlives the action that started it (carry; not yet)",
    "initial": "the first beat (default: the first listed)",
    "beats": "{name: beat}: its named moments",
    "extends": "another motion whose keys this one starts from",
    "mirror": "true: the motion for the other side of the body (with `extends`: take_left from take)",
    "priority": "breaks ties between equal claims (default 0; reactions higher)",
}

BEAT_KEYS = {
    "pose": "a pose of the body (body.poses) laid over the channels the motion holds",
    "bones": "{Bone: [x, y, z] degrees}: body-frame rotations on top of the pose",
    "ik": "{End: {target: <ref> | [x, y, z], offset, pole, rest, grip, holding}, mirror}: hands and feet onto points; "
          "a hand takes the shape `grip` names (a hand shape, or a role: its contact's grip), else its "
          "target's grip when that is a hand shape (skeleton hand_shapes: press, power, pinch, ...); "
          "`holding: <role>` puts the held thing on the target; `fingers: [x, y, z]` and `palm: [x, y, z]` "
          "turn the hand (body-frame directions the fingers point and the palm faces; mirrored with the target; "
          "`frame: target` gives them in the target's frame instead: into a button's face; `tip: index` puts "
          "that fingertip on the target rather than the wrist; `arc: m` lifts the way there by m, a third of the way along)",
    "root": "{at: <ref>, offset: [x, y, z], facing: away | toward | <deg>, posture: <pose>}: where the body stands",
    "look": "<ref> | {at: <ref>, weight}: where the head turns",
    "plant": "feet: the feet stay where they are on the floor (legs by IK)",
    "lean": "{by: deg, at: 0..1}: a forward lean of the spine that peaks at `at` of the beat and is gone by its end",
    "attach": "{role: hand | {hand, level: true}}: the thing goes with that hand (`level`: kept flat)",
    "release": "<role>: let go of it; motions holding it (carry) end",
    "over": "seconds the beat takes to arrive (default: from the distance its targets travel)",
    "ease": "smooth (default) | settle | linear",
    "emit": "an event fired on arrival (the world changes in step with the body)",
    "then": "the next beat, or done",
    "after": "seconds to dwell once arrived, or `duration` (the runner's duration)",
    "interrupt": "yield (default: go to `exit`) | finish (this beat first) | hold (refuse) | drop (stop here)",
    "exit": "the beat to go to when asked to yield",
    "resume": "the beat to pick up from after an overlay borrowed a channel (default: this one)",
    "when": "skip the beat unless this holds (not yet)",
    "until": "dwell until this holds (not yet)",
    "do": "a whole motion as one beat (not yet)",
    "bend": "auto: low targets blend toward reach_low (not yet)",
    "hips": "[x, y, z]: m the pelvis moves in the body frame on top of the pose (a squat drops it; "
            "planted feet bend the knees to keep up)",
    "loop": "{to: <beat>, times: n}: once this beat is done, go back to `to` until the stretch from "
            "there to here has run `times` times, then on to `then` (repetitions of an exercise)",
}
NOT_YET = {"when", "until", "do", "bend"}
INTERRUPTS = ("yield", "finish", "hold", "drop")
STRENGTHS = {"prefer": 1, "claim": 2, "require": 3}
COUPLINGS = {"walk": ("speed", "arm_swing")}      # what a motion's `while` can change, and of what
EASES = ("smooth", "settle", "linear", "spring")

FPS = 30
FALL = 9.81                # m/s2: what a broken hold lets go of falls to the floor (the runtime's physics
                           # does this properly; the reference rig only drops it straight down)
REACH_SPEED = 0.9          # m/s an effector or the body travels when a beat gives no `over`
MIN_OVER = 0.35            # s: the shortest arrival
TRAVEL_EPS = 1e-4          # m: closer than this a beat's end or root is already there
HALF_LIFE = {"gaze": 0.08, "left_hand": 0.12, "right_hand": 0.12}   # s, inertialization per channel
DEFAULT_HALF_LIFE = 0.18
REACH_SLACK = 0.96         # of the arm's length: a hand reaches no further without bending
BEND_BONES = ("Spine", "Chest")
MAX_BEND = 75              # degrees, shared by BEND_BONES
HANG_SHARE = 0.7           # of the way a free arm swings back to hanging plumb when the spine bends
LOOK_IN = 0.3              # s: a gaze turns fully onto a new look in this long
LOOK_OUT = 0.4             # s: a look that ends (nothing to look at any more) eases off in this long
LOOK_FOLLOW = 0.1          # s: half-life of the look point following a change of what is looked at
MORPH_HALF_LIFE = 0.12     # s: drape correctives settle onto a new owner's pose
ROOT_LEAD = 0.7            # over planted feet the root arrives by this fraction of the beat
SHARE_TURN = 2.0           # s a sharer of weight 1 keeps a shared channel before passing it on
YAW_LIMIT, PITCH_LIMIT, NECK_SHARE = math.radians(70), math.radians(35), 0.4


class MotionError(ValueError):
    """A motion file, or a request to run one, that doesn't make sense."""


# --- loading -------------------------------------------------------------------------------------

def load_motion(name: str | Path, assets_dir: Path | None = None) -> dict:
    """A motion by name (``motions/<name>.yaml`` through the content roots) or path, validated,
    with ``extends`` merged."""
    from .layout.yaml_utils import safe_load_path

    path = Path(name)
    if path.suffix != ".yaml":
        path = roots.find(f"{MOTIONS_DIR}/{name}.yaml", assets_dir or roots.CORE)
    spec = safe_load_path(path)
    if not isinstance(spec, dict) or spec.get("kind") != MOTION_KIND:
        raise MotionError(f"{path}: expected kind: {MOTION_KIND}")
    if "extends" in spec:
        base = load_motion(spec["extends"], assets_dir)
        merged = {**base, **{k: v for k, v in spec.items() if k != "extends"}}
        merged["beats"] = {**base.get("beats", {}), **(spec.get("beats") or {})}
        spec = merged
    spec.setdefault("name", path.stem)
    validate_motion(spec, str(path))
    return mirror_motion(spec) if spec.pop("mirror", False) else spec


MOTIONS_FILE = "motions.json"
MOTIONS_FORMAT = "geogen-motions"
MOTIONS_VERSION = 1


def motion_names(assets_dir: Path | None = None) -> list[str]:
    """Every motion the content roots define (``motions/*.yaml``), by name."""
    return [path.stem for path in roots.glob(f"{MOTIONS_DIR}/*.yaml", assets_dir or roots.CORE)]


CATEGORIES = ("idle", "emote", "dance", "exercise", "action", "reaction")
# Beat keys that name things in the world or move the body through it: a motion using one runs
# only where the world is (the reference rig; the runtime rig refuses them until it is ported).
WORLD_KEYS = ("root", "attach", "release")


def body_only(spec: dict) -> bool:
    """Whether ``spec`` needs nothing but the body: no roles, no beat placing the body or taking
    things. The runtime rig (runtime/godot/scripts/motion_rig.gd) runs these."""
    return not spec.get("roles") and not any(k in (beat or {}) for beat in spec["beats"].values() for k in WORLD_KEYS)


def motion_catalogue(assets_dir: Path | None = None) -> list[dict]:
    """Every motion as a catalogue entry, by category then name: what a gallery or a tool lists
    (``python -m geogen.motion_review --list``). ``params`` are the defaults, ``roles`` what it
    must be given, ``props`` what a demonstration stages for them (on a table in front),
    ``demo`` the script it plays when it needs other motions around it, ``body_only`` whether it
    needs nothing but the body, ``moves_body`` whether it places the body (``root``: sit; the
    runtime rig doesn't yet)."""
    entries = []
    for name in motion_names(assets_dir):
        spec = load_motion(name, assets_dir)
        params = {k: (v.get("default") if isinstance(v, dict) else v) for k, v in (spec.get("params") or {}).items()}
        entries.append({
            "name": spec["name"],
            "category": spec.get("category", "action"),
            "summary": spec.get("summary", ""),
            "params": params,
            "roles": {r: dict(e or {}) for r, e in (spec.get("roles") or {}).items()},
            "props": spec.get("props") or {},
            "holds": spec.get("holds") or {},
            "body_only": body_only(spec),
            "moves_body": any("root" in (beat or {}) for beat in spec["beats"].values()),
            "demo": spec.get("demo") or [],
            "beats": list(spec["beats"]),
        })
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return sorted(entries, key=lambda e: (order.get(e["category"], len(order)), e["name"]))


def write_motions(generated: Path, assets_dir: Path | None = None) -> Path:
    """``motions.json`` in a runtime's generated directory: every motion by name, validated and
    with ``extends`` merged, ``{expr}`` left for the runtime to resolve when it starts one (from
    its params and the body's ``extras.geogen.rig.params``), as ``MotionRig.start`` does."""
    import json

    generated = Path(generated)
    generated.mkdir(parents=True, exist_ok=True)
    motions = {}
    for name in motion_names(assets_dir):
        spec = load_motion(name, assets_dir)
        motions[spec.get("name", name)] = spec
    path = generated / MOTIONS_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"format": MOTIONS_FORMAT, "version": MOTIONS_VERSION, "motions": motions,
                               "catalogue": motion_catalogue(assets_dir)},
                              indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def validate_motion(spec: dict, where: str = "motion") -> None:
    """Raise ``MotionError`` for unknown keys, unknown beats, and holds or interrupts that aren't
    one of the vocabulary."""
    known_keys(spec, MOTION_KEYS, f"{where}", MotionError)
    if spec.get("category", "action") not in CATEGORIES:
        raise MotionError(f"{where}: category '{spec['category']}' is not one of {CATEGORIES}")
    for i, entry in enumerate(spec.get("demo") or []):
        if not isinstance(entry, dict) or not isinstance(entry.get("start"), str):
            raise MotionError(f"{where}: demo[{i}]: expected {{at, start: <motion>, <role>: ...}}")
    for role in (spec.get("props") or {}):
        if role not in (spec.get("roles") or {}):
            raise MotionError(f"{where}: props.{role} is not one of its roles")
    for activity, effect in (spec.get("while") or {}).items():
        if activity not in COUPLINGS or not isinstance(effect, dict) or set(effect) - set(COUPLINGS[activity]):
            raise MotionError(f"{where}: while.{activity}: expected one of {COUPLINGS}")
        if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in effect.values()) or effect.get("speed", 1) == 0:
            raise MotionError(f"{where}: while.{activity}: shares are 0..1 (a speed above 0)")
    beats = spec.get("beats") or {}
    if not beats:
        raise MotionError(f"{where}: no beats")
    initial = spec.get("initial", next(iter(beats)))
    if initial not in beats:
        raise MotionError(f"{where}: initial beat '{initial}' is not one of {sorted(beats)}")
    for strength in (spec.get("holds") or {}):
        if strength not in (*STRENGTHS, "share"):
            raise MotionError(f"{where}: holds.{strength}: expected require, claim, prefer or share")
    for name, beat in beats.items():
        known_keys(beat or {}, BEAT_KEYS, f"{where}: beat '{name}'", MotionError)
        for key in ("then", "exit", "resume"):
            target = (beat or {}).get(key)
            if target is not None and target != "done" and target not in beats:
                raise MotionError(f"{where}: beat '{name}': {key} '{target}' is not a beat")
        loop = (beat or {}).get("loop")
        if loop is not None:
            if not isinstance(loop, dict) or set(loop) - {"to", "times"} or "to" not in loop:
                raise MotionError(f"{where}: beat '{name}': loop is {{to: <beat>, times: n}}")
            if loop["to"] not in beats:
                raise MotionError(f"{where}: beat '{name}': loop to '{loop['to']}' is not a beat")
        policy = (beat or {}).get("interrupt", "yield")
        if policy not in INTERRUPTS:
            raise MotionError(f"{where}: beat '{name}': interrupt '{policy}' is not one of {INTERRUPTS}")
        if (beat or {}).get("ease", "smooth") not in EASES:
            raise MotionError(f"{where}: beat '{name}': ease must be one of {EASES}")


# --- targets -------------------------------------------------------------------------------------

@dataclass
class Target:
    """A point a beat aims at: ``node``'s world transform times ``local`` (followed every frame),
    or a fixed world matrix when ``node`` is None."""

    node: SceneNode | None
    local: NDArray[np.float64]
    name: str = ""
    grip: str | None = None              # how a body touches it (the attachment's or affordance's)

    def world(self) -> NDArray[np.float64]:
        return self.local if self.node is None else self.node.world_transform() @ self.local


@dataclass
class Held:
    """A thing going with a hand (``attach``): ``rel`` is its transform in the hand's frame, or, when
    ``level``, in the body's heading with its translation the offset from the hand."""

    node: SceneNode
    end: str
    rel: NDArray[np.float64]
    level: bool = False


def _attachment_local(node: SceneNode, key: str) -> NDArray[np.float64]:
    """Attachment ``key`` in ``node``'s frame, as a contact: one with the default facing (toward
    the object's middle, which suits something hung there) faces the way the object does."""
    point = node.attachments[key]
    local = point.resolve(node.size).to_matrix() if node.size is not None else np.eye(4)
    if point.rotation is None and point.facing == "center":
        local[:3, :3] = np.eye(3)
    return local


def grip_points(node: SceneNode) -> dict:
    """``node``'s attachments that say how a body touches it (``grip:``), in order, as a runtime's
    motion rig binds roles to them (``extras.geogen.grips``): {name: {grip, at: [tx, ty, tz, qx,
    qy, qz, qw]}} in the node's own frame."""
    out = {}
    for key, point in node.attachments.items():
        if point.grip and node.size is not None:
            local = _attachment_local(node, key)
            out[key] = {"grip": point.grip,
                        "at": [round(float(v), 6) for v in (*local[:3, 3], *quat_from_matrix(local[:3, :3]))]}
    return out


def contact(node: SceneNode, grip: str | None = None, name: str | None = None) -> Target:
    """Where a body touches ``node``: the attachment ``name`` (or, when ``name`` is one of its
    interactions, the attachment that interaction's ``touch:`` names), else the first attachment
    with ``grip`` (``any``: the first a hand holds it by), else (for sit / lie / stand / use) the anchor of its first affordance of that
    type, else the node's own origin."""
    def attached(key: str) -> Target:
        point = node.attachments[key]
        local = np.linalg.inv(node.world_transform()) @ node.get_attachment(key).to_matrix()
        if point.rotation is None and point.facing == "center":
            # An attachment's default facing turns it toward the object's middle, which suits
            # something hung there but not a contact: a contact faces the way the object does.
            local[:3, :3] = np.eye(3)
        return Target(node, local, f"{node.name}.{key}", point.grip)

    if name is not None:
        touch = next((i.touch for i in node.interactions if i.name == name and i.touch), None)
        if name not in node.attachments and touch is not None:
            return attached(touch)
        if name not in node.attachments:
            raise MotionError(f"'{node.name}' has no attachment '{name}' (it has {sorted(node.attachments)})")
        return attached(name)
    if grip is not None:
        for key, point in node.attachments.items():
            if point.grip == grip or (grip == "any" and point.grip and point.grip not in NOT_HELD):
                return attached(key)
        for a in node.meta.get("affordances") or []:
            if a.get("type") == grip:
                local = Transform(translation=np.asarray(a["position"], dtype=float),
                                  rotation=np.array([0.0, math.radians(a["yaw"]), 0.0])).to_matrix()
                return Target(node, local, f"{node.name}[{grip}]", grip)
    return Target(node, np.eye(4), node.name)


# --- the body as a runtime sees it -----------------------------------------------------------------

def clip_pose(skeleton: Skeleton, clip, t: float) -> Pose:
    """A clip's joints at ``t`` as a ``Pose`` (rotations relative to each bone's rest, offsets)."""
    pose = Pose()
    for joint, track in clip.tracks.items():
        if joint not in skeleton.bones:
            continue
        rotation, translation = track.sample(t)
        rest = skeleton.rest_local(joint)
        if rotation is not None:
            pose.rotations[joint] = quat_from_matrix(rest[:3, :3].T @ matrix_from_quat(rotation))
        if translation is not None and skeleton[joint].parent is not None:
            delta = np.asarray(translation) - rest[:3, 3]
            if np.abs(delta).max() > 1e-6:
                pose.offsets[joint] = delta
    return pose


def pose_library(actor: SceneNode, skeleton: Skeleton) -> dict[str, Pose]:
    """The body's poses as its build left them (each ``pose_<name>`` clip: arms cleared, seats fitted)."""
    out = {clip.name[5:]: clip_pose(skeleton, clip, 0.0) for clip in actor.clips if clip.name.startswith("pose_")}
    out.setdefault("stand", Pose())
    return out


def rig_extras(actor: SceneNode) -> dict | None:
    """What a runtime's motion rig needs of a built humanoid (``extras.geogen.rig``): the bones
    with their rests (the rig poses relative to these, not to the engine's import), the
    skeleton's params (for ``{expr}`` in motions), channels, hand shapes and the pose library.
    Quaternions are [x, y, z, w]; rests are [tx, ty, tz, qx, qy, qz, qw] in the parent's frame."""
    s = (actor.meta.get("humanoid") or {}).get("rig")
    if not isinstance(s, Skeleton):
        return None

    def r(v, n=6):
        return [round(float(x), n) for x in v]

    bones = []
    for name in s.names:
        rest = s.rest_local(name)
        bones.append({"name": name, "parent": s[name].parent or "",
                      "rest": r([*rest[:3, 3], *quat_from_matrix(rest[:3, :3])])})
    poses = {}
    for name, pose in pose_library(actor, s).items():
        poses[name] = {"rotations": {b: r(q) for b, q in pose.rotations.items()
                                     if abs(float(q[3])) < 1 - 1e-9 or abs(q[:3]).max() > 1e-9},
                       "offsets": {b: r(o) for b, o in pose.offsets.items()}}
    return {"bones": bones,
            "params": {k: round(float(v), 6) for k, v in s.params.items()},
            "channels": {c: {"bones": list(e["bones"]), **({"end": e["end"]} if e.get("end") else {}),
                             **({"shared": True} if e.get("shared") else {})} for c, e in s.channels.items()},
            "hand_shapes": {n: {b: r(v, 4) for b, v in shape.items()} for n, shape in s.hand_shapes.items()},
            "poses": poses,
            "root_poses": root_poses_extras(actor.meta.get("poses") or {})}


def root_poses_extras(poses: dict) -> dict:
    """Where each posture puts the body against what it is at (a beat's ``root: {at, posture}``):
    the body asset's ``poses`` as ``extras.geogen.rig.root_poses`` (the loader sets them once
    the seated pose is fitted)."""
    return {n: {"offset": [round(float(v), 6) for v in p.get("offset", [0, 0, 0])],
                "rotation": [round(float(v), 6) for v in p.get("rotation", [0, 0, 0])]} for n, p in poses.items()}


def transition_data(actor: SceneNode, postures: set[str] | None = None) -> dict:
    """The body's transition clips (sit_down: stand -> sit) and pose correctives as plain data,
    what a conformance trace carries (a runtime reads the same from the body's exported
    animations and ``extras.geogen.clips``): ``transitions`` [{from, to, duration, tracks: {bone:
    {times, rotations?, translations?}}, root: the clip's root path, morphs: {"node:target":
    {times, values}}}] and ``pose_morphs`` {pose: {"node:target": weight}}; with ``postures``,
    only the transitions between those."""
    s = (actor.meta.get("humanoid") or {}).get("rig")

    def r(v, n=6):
        return [round(float(x), n) for x in v]

    transitions = []
    for clip in actor.clips:
        if not (clip.meta.get("from") and clip.meta.get("to")) or clip.loop:
            continue
        if postures is not None and not {clip.meta["from"], clip.meta["to"]} <= postures:
            continue
        tracks = {}
        for bone, track in clip.tracks.items():
            if bone not in s.bones:
                continue
            entry = {"times": r(track.times)}
            if track.rotations is not None:
                entry["rotations"] = [r(q, 7) for q in track.rotations]
            if track.translations is not None:
                entry["translations"] = [r(v) for v in track.translations]
            tracks[bone] = entry
        root = clip.meta.get("root") or {}
        transitions.append({
            "from": clip.meta["from"], "to": clip.meta["to"], "duration": round(float(clip.duration), 6),
            "tracks": tracks,
            **({"root": {"times": r(root["times"]), "positions": [r(v) for v in root["positions"]],
                         "rotations": [r(q, 7) for q in root["rotations"]]}} if root else {}),
            "morphs": {f"{node}:{k}": {"times": r(m.times), "values": r(v)}
                       for node, m in (clip.morphs or {}).items() for k, v in m.weights.items()}})
    pose_morphs = {}
    for clip in actor.clips:
        if clip.name.startswith("pose_"):
            pose_morphs[clip.name[5:]] = {f"{node}:{k}": round(w, 6) for node, track in (clip.morphs or {}).items()
                                          for k, w in track.sample(0.0).items()}
    return {"transitions": transitions, "pose_morphs": pose_morphs}


# --- easing --------------------------------------------------------------------------------------

def ease(kind: str, u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    if kind == "linear":
        return u
    if kind == "settle":                    # quick start, long soft landing
        return 1.0 - (1.0 - u) ** 4
    return u * u * u * (u * (u * 6 - 15) + 10)     # minimum jerk


def _bump(u: float, at: float) -> float:
    """0 -> 1 at ``at`` -> 0, smooth."""
    u = min(max(u, 0.0), 1.0)
    x = u / at if u < at else (1.0 - u) / max(1.0 - at, 1e-6)
    return ease("smooth", x)


def _yaw_of(m: NDArray[np.float64]) -> float:
    z = m[:3, 2]
    return math.atan2(z[0], z[2])


def _placement(pos: NDArray[np.float64], yaw: float) -> NDArray[np.float64]:
    return Transform(translation=np.asarray(pos, dtype=float), rotation=np.array([0.0, yaw, 0.0])).to_matrix()


def _angle(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    d = abs(float(np.dot(a, b)))
    return 2.0 * math.acos(min(1.0, d))


def _quat_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _quat_inv(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([-q[0], -q[1], -q[2], q[3]])


IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
HANDS = {"hand": "RightHand", "right": "RightHand", "left": "LeftHand",
         "right_hand": "RightHand", "left_hand": "LeftHand"}
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
LOOK_BONES = ("Neck", "Head")


# --- running motions -----------------------------------------------------------------------------

@dataclass
class Run:
    """One motion running on the body."""

    motion: dict
    bindings: dict[str, SceneNode | Target]
    order: int
    params: dict[str, Any] = field(default_factory=dict)
    beat: str = ""
    t: float = 0.0                       # seconds into the beat
    over: float = 0.0                    # its arrival time
    dwell: float = 0.0                   # then how long it stays
    pending_exit: bool = False           # interrupted with `finish`: exit when this beat ends
    done: bool = False
    start: dict = field(default_factory=dict)    # the body as the beat found it
    loops: dict = field(default_factory=dict)    # beat -> times its loop has gone round
    paused: str | None = None            # the beat an overlay took a channel in (time stands still)
    resume_to: str | None = None         # asked to yield while paused: its exit, not its resume
    origins: dict = field(default_factory=dict)  # role -> its thing's world transform at the start

    @property
    def name(self) -> str:
        return self.motion["name"]

    @property
    def spec(self) -> dict:
        return self.motion["beats"][self.beat] or {}

    def strength(self, channel: str) -> int:
        holds = self.motion.get("holds") or {}
        best = 0
        for kind, value in STRENGTHS.items():
            if channel in (holds.get(kind) or []):
                best = max(best, value)
        if channel in (holds.get("share") or {}):
            best = max(best, STRENGTHS["claim"])
        return best

    def share(self, channel: str) -> float:
        """This run's weight in time-slicing ``channel`` (``holds.share``), 0 when it doesn't share it."""
        return float(((self.motion.get("holds") or {}).get("share") or {}).get(channel) or 0.0)


class MotionRig:
    """Runs motions on one skeletal body (an instance of a humanoid asset placed in a scene)."""

    def __init__(self, actor: SceneNode, scene: SceneNode | None = None, fps: int = FPS,
                 assets_dir: Path | None = None) -> None:
        rig = (actor.meta.get("humanoid") or {}).get("rig")
        if not isinstance(rig, Skeleton):
            raise MotionError(f"'{actor.name}' is not a skeletal body built with its rig (meta.humanoid.rig)")
        self.actor, self.scene, self.skeleton = actor, scene, rig
        self.assets_dir = assets_dir
        self.dt = 1.0 / fps
        self.time = 0.0
        self.joints = {n.name: n for n in actor.iter_nodes() if n.name in rig.bones}
        self.channel_of = {b: c for c, entry in rig.channels.items() for b in entry["bones"]}
        self.fingers = {side: [b for b in rig.names if b.startswith(side) and any(f in b for f in FINGERS)]
                        for side in ("Left", "Right")}
        # Pose correctives (a skirt's drape) per pose: {pose: {node: {target: weight}}}.
        self.pose_morphs: dict[str, dict[str, dict[str, float]]] = {}
        self.poses = self._pose_library()
        self.morph_nodes = {n.name: n for n in actor.iter_nodes() if n.mesh is not None and n.mesh.morphs}
        self.root_poses = actor.meta.get("poses") or {}
        self.posture = "stand"
        self.root = actor.transform.to_matrix()
        # Standing targets (an approach point in front of a seat, a spot by a TV) stand on the floor
        # the person started on, whatever height the contact they are measured from is at.
        self.floor = float(self.root[1, 3])
        self.out = self.poses["stand"].copy()
        self.runs: list[Run] = []
        self.owners: dict[str, Run | None] = {c: None for c in rig.channels}
        self.offsets: dict[str, tuple[NDArray[np.float64], float, float]] = {}   # bone -> (quat, age, half-life)
        self.root_offset: tuple[NDArray[np.float64], float, float] | None = None
        self.attached: dict[str, Held] = {}
        self.falling: list[list] = []        # [node, speed]: let go of by a break
        self.trace: list[dict] = []
        self.samples: list[dict] = []
        self._order = 0
        self._changed: set[str] = set()     # channels whose source changed this step (inertialized)
        self.morphs = self._pose_morph_vector("stand")
        self.morph_offset: tuple[dict, float] | None = None
        self.prelook = {b: self.out.rotations.get(b, IDENTITY) for b in LOOK_BONES}
        self.look_key = None                # (run, what) the gaze is on
        self.look_t = 0.0
        self.look_point: NDArray[np.float64] | None = None
        self.look_weight = 0.0              # how far the current look has turned the head (0..1)
        self.look_fade: tuple | None = None   # (point, weight, t): a look that ended, easing off
        self.share_t: dict[str, float] = {}           # channel -> how long its sharer has had this turn
        self.share_looked: dict[tuple[str, int], float] = {}   # (channel, run order) -> its time with it
        self._apply()

    # The pose library: each pose_<name> clip (as the build left it: arms cleared, seats fitted).
    # Transitions: the body's baked clips between two postures (sit_down: stand -> sit), whose arms
    # the build cleared of the body and its clothes and whose drape correctives it keyed.
    def _pose_library(self) -> dict[str, Pose]:
        self.transitions = {(c.meta["from"], c.meta["to"]): c for c in self.actor.clips
                            if c.meta.get("from") and c.meta.get("to") and not c.loop}
        for clip in self.actor.clips:
            if clip.name.startswith("pose_"):
                morphs = getattr(clip, "morphs", None) or {}
                self.pose_morphs[clip.name[5:]] = {node: track.sample(0.0) for node, track in morphs.items()}
        return pose_library(self.actor, self.skeleton)

    def _clip_pose(self, clip, t: float) -> Pose:
        return clip_pose(self.skeleton, clip, t)

    def _transition(self, run: Run):
        """The body's baked transition clip this beat's posture change follows, if it has one."""
        to = run.spec.get("pose")
        return self.transitions.get((run.start.get("posture"), to)) if to and run.start.get("posture") != to else None

    def _pose_morph_vector(self, pose: str) -> dict[tuple[str, str], float]:
        """The drape correctives ``pose`` was baked with, for every morph target of the body."""
        known = self.pose_morphs.get(pose) or self.pose_morphs.get("stand") or {}
        return {(node, k): float((known.get(node) or {}).get(k, 0.0))
                for node, n in self.morph_nodes.items() for k in n.mesh.morphs}

    # --- requests ---

    def start(self, motion: str | dict, **bindings) -> Run:
        """Start ``motion`` with its roles bound: a SceneNode, a reference into the scene
        (``"chair"``, ``"tv.screen"``) or a ``Target``. ``duration=`` sets what ``after: duration``
        means."""
        spec = load_motion(motion, self.assets_dir) if not isinstance(motion, dict) else motion
        params = {k: (v.get("default") if isinstance(v, dict) else v) for k, v in (spec.get("params") or {}).items()}
        params.update({k: bindings.pop(k) for k in list(bindings) if k in params or k == "duration"})
        resolved = {}
        for role, entry in (spec.get("roles") or {}).items():
            value = bindings.pop(role, None)
            if value is None:
                if not (entry or {}).get("optional"):
                    raise MotionError(f"motion '{spec['name']}': role '{role}' is not bound")
                continue
            resolved[role] = self._bind(value, (entry or {}).get("grip"))
        if bindings:
            raise MotionError(f"motion '{spec['name']}': no roles or params {sorted(bindings)}")
        for holder in ("require", "claim", "prefer"):
            for channel in (spec.get("holds") or {}).get(holder) or []:
                if channel not in self.skeleton.channels:
                    raise MotionError(f"motion '{spec['name']}': this body has no channel '{channel}' "
                                      f"(it has {sorted(self.skeleton.channels)})")
        from .layout.expressions import resolve_value

        values = {**self.skeleton.params, **{k: v for k, v in params.items() if isinstance(v, (int, float))}}
        for k, v in params.items():            # a param of the body's size ("{0.2 * height}")
            if isinstance(v, str) and v.startswith("{"):
                values[k] = params[k] = resolve_value(v, values)
        spec = {**spec, "beats": resolve_value(spec["beats"], values)}
        spec = self._borrow_hand(spec, resolved)
        self._order += 1
        run = Run(spec, resolved, self._order, params)
        run.origins = {role: t.node.world_transform().copy() for role, t in resolved.items() if t.node is not None}
        self.runs.append(run)
        self._event("start", run)
        self._enter(run, spec.get("initial", next(iter(spec["beats"]))))
        return run

    def _borrow_hand(self, spec: dict, bound: dict[str, Target]) -> dict:
        """``spec`` for the hand that holds its ``held`` roles' things: as written, or mirrored when
        it is written for the other hand."""
        for role, entry in (spec.get("roles") or {}).items():
            if not (entry or {}).get("held") or role not in bound:
                continue
            held = next((h for h in self.attached.values() if h.node is bound[role].node), None)
            if held is None:
                raise MotionError(f"motion '{spec['name']}': role '{role}' is held, and no hand holds it")
            hand, other = (f"{side}_hand" for side in (("right", "left") if held.end == "RightHand" else ("left", "right")))
            holds = spec.get("holds") or {}
            claimed = {c for kind in STRENGTHS for c in holds.get(kind) or []}
            if other in claimed and hand not in claimed:
                return mirror_motion(spec)
        return spec

    def coupling(self, activity: str = "walk") -> dict[str, float]:
        """What the running motions' ``while`` make of ``activity``: each share, multiplied."""
        out = dict.fromkeys(COUPLINGS[activity], 1.0)
        for run in self.running():
            for key, share in ((run.motion.get("while") or {}).get(activity) or {}).items():
                out[key] *= float(share)
        return out

    def interrupt(self, run: Run | str, how: str = "yield") -> str:
        """Ask a running motion to stop: ``yield`` follows its current beat's policy, ``drop``
        stops it where it is whatever the beat says. Returns what happened."""
        run = self._find(run)
        if run is None or run.done:
            return "not running"
        policy = "drop" if how == "drop" else run.spec.get("interrupt", "yield")
        self._event("interrupt", run, policy=policy)
        if policy == "hold":
            return "held"
        if policy == "finish":
            run.pending_exit = True
            return "finishing"
        if policy == "drop":
            self._end(run)
            return "dropped"
        self._changed |= self._owned(run)
        exit_beat = run.spec.get("exit")
        if exit_beat in (None, "done"):
            self._end(run)
            return "ended"
        if run.paused is not None:
            run.paused = exit_beat               # out through its exit once the overlay gives the channel back
            run.resume_to = exit_beat
            return "exiting"
        self._enter(run, exit_beat)
        return "exiting"

    def break_(self, reaction: str | dict | None = None, **bindings) -> Run | None:
        """Everything stops at once (a shove, a fall): every motion drops with no exits, what the
        hands hold is let go of (to physics: here it drops to the floor), and the body inertializes
        into ``reaction`` (a motion: a stagger) if one is given."""
        self._event("break", None, reaction=reaction if isinstance(reaction, str) else None)
        for run in self.running():
            self._end(run)
        for role, held in list(self.attached.items()):
            del self.attached[role]
            self.falling.append([held.node, 0.0])
            self._event("release", None, role=role, to="physics")
        return self.start(reaction, **bindings) if reaction is not None else None

    def running(self, name: str | None = None) -> list[Run]:
        return [r for r in self.runs if not r.done and (name is None or r.name == name)]

    # --- the step ---

    def step(self) -> None:
        self.time += self.dt
        for run in list(self.runs):
            if not run.done:
                self._advance(run)
        self.runs = [r for r in self.runs if not r.done]
        self._arbitrate()
        pose, root, aims = self._compose()
        self.out, self.root = pose, root
        self._apply()
        self._sample(aims)

    def _advance(self, run: Run) -> None:
        if run.paused is not None:
            return
        run.t += self.dt
        guard = 0
        self._arrive(run)
        while not run.done and run.t >= run.over + run.dwell and guard < 16:
            guard += 1
            spill = run.t - (run.over + run.dwell)
            nxt = run.spec.get("exit") if run.pending_exit else self._next(run)
            run.pending_exit = False
            if nxt in (None, "done"):
                self._end(run, arrived=True)
                return
            self._enter(run, nxt)
            run.t = spill
            self._arrive(run)

    def _next(self, run: Run) -> str:
        """The beat after this one: back round its ``loop`` while repetitions are left, else ``then``."""
        loop = run.spec.get("loop")
        if loop:
            done = run.loops.get(run.beat, 0) + 1
            if done < int(loop.get("times", 1)):
                run.loops[run.beat] = done
                self._event("loop", run, to=loop["to"], round=done + 1)
                return loop["to"]
            run.loops[run.beat] = 0
        return run.spec.get("then", "done")

    def _arrive(self, run: Run) -> None:
        """A beat that has arrived fires its emit and takes or lets go of what it names, once."""
        if run.done or run.t < run.over or run.start.get("arrived"):
            return
        run.start["arrived"] = True
        spec = run.spec
        for role, how in (spec.get("attach") or {}).items():
            target = run.bindings.get(role)
            if target is None or target.node is None:
                raise MotionError(f"motion '{run.name}': attach: '{role}' is not a bound thing")
            how = how if isinstance(how, dict) else {"hand": how}
            end = HANDS.get(how.get("hand", "hand"), how.get("hand"))
            if end not in self.joints:
                raise MotionError(f"motion '{run.name}': attach: '{role}' to '{how.get('hand')}', which is not a hand")
            hand_world = self.root @ self.skeleton.fk(self.out)[end]
            thing = target.node.world_transform()
            if how.get("level"):
                # Kept as it is against the body's heading, carried along by the hand's position.
                heading = _placement(np.zeros(3), _yaw_of(self.root))
                rel = np.linalg.inv(heading) @ thing
                rel[:3, 3] = heading[:3, :3].T @ (thing[:3, 3] - hand_world[:3, 3])
            else:
                rel = np.linalg.inv(hand_world) @ thing
            # A thing already in a hand is re-gripped where it is, under the role that took it.
            key = next((k for k, h in self.attached.items() if h.node is target.node), role)
            self.attached[key] = Held(target.node, end, rel, bool(how.get("level")))
            self._event("attach", run, role=role, hand=end)
        release = spec.get("release")
        for role in ([release] if isinstance(release, str) else release or []):
            held = self.attached.pop(role, None)
            if held is None:
                continue
            self._event("release", run, role=role)
            # Whatever else holds it (a carry) has nothing left to do.
            for other in self.running():
                if other is not run and any(b.node is held.node for b in other.bindings.values()):
                    self._end(other)
        if spec.get("emit"):
            self._event("emit", run, emit=spec["emit"])
        if spec.get("pose") and "hips" in self._owned(run) and self.posture != spec["pose"]:
            self.posture = spec["pose"]           # the body is now in this pose (sit, stand)
            self._changed |= {c for c, r in self.owners.items() if r is None}
            self._event("posture", run, posture=self.posture)
    def _enter(self, run: Run, beat: str) -> None:
        run.beat, run.t = beat, 0.0
        spec = run.spec
        if spec.get(k := next((k for k in NOT_YET if k in spec), "")):
            raise MotionError(f"motion '{run.name}' beat '{beat}': '{k}' is not supported by the rig yet")
        start = self.out.copy()
        # The head turn is laid on afresh every frame, so a beat starts from the head before it.
        start.rotations.update({b: q for b, q in self.prelook.items()})
        ends = self._ends_world()
        held = [(h.node, ends[h.end] - h.node.world_transform()[:3, 3]) for h in self.attached.values()]
        fk = self.skeleton.fk(start)
        hands = {e: quat_from_matrix(fk[e][:3, :3] / np.linalg.norm(fk[e][:3, :3], axis=0))   # (hands are scaled)
                 for e in ("LeftHand", "RightHand") if e in fk}
        run.start = {"pose": start, "root": self.root.copy(), "ends": ends, "held": held, "hands": hands,
                     "feet": self._ends_world(("LeftFoot", "RightFoot")), "morphs": dict(self.morphs),
                     "posture": self.posture}
        for end, entry in self._ik_entries(run).items():
            if entry.get("holding") is not None:
                self._held_gap(run, end, entry["holding"])      # (raises when nothing holds it)
        run.over = self._over(run)
        after = spec.get("after", 0.0)
        run.dwell = float(run.params.get("duration", 3.0) if after == "duration" else after or 0.0)
        self._event("beat", run, beat=beat, over=round(run.over, 3), dwell=round(run.dwell, 3))

    def _over(self, run: Run) -> float:
        spec = run.spec
        if "over" in spec:
            return max(float(spec["over"]), self.dt)
        travel = 0.0
        for end, entry in self._ik_entries(run).items():
            target = self._ik_target(run, end, entry)
            if target is not None:
                travel = max(travel, float(np.linalg.norm(target - run.start["ends"][end])))
        if spec.get("root"):
            travel = max(travel, float(np.linalg.norm(self._root_target(run)[:3, 3] - self.root[:3, 3])))
        # (under TRAVEL_EPS is no travel: a body already where it is asked to be arrives at once)
        return max(MIN_OVER, travel / REACH_SPEED + 0.15) if travel > TRAVEL_EPS else (MIN_OVER if spec.get("pose") else 0.0)

    def _end(self, run: Run, arrived: bool = False) -> None:
        """Stop ``run``. One whose last beat ``arrived`` leaves the body where that beat put it
        (its last composed frame was a step short of arriving), not where it was a step before."""
        if not run.done:
            if arrived and run.spec.get("root") and self.owners.get("root") is run:
                self.root = self._root_target(run)
            self._changed |= self._owned(run)
            run.done = True
            self._event("end", run)

    def _find(self, run: Run | str) -> Run | None:
        if isinstance(run, Run):
            return run
        found = self.running(run)
        return found[-1] if found else None

    def _bind(self, value, grip: str | None) -> Target:
        if isinstance(value, Target):
            return value
        if isinstance(value, SceneNode):
            return contact(value, grip)
        if self.scene is None:
            raise MotionError(f"can't resolve {value!r} without a scene")
        from . import refs

        text = str(value)
        try:
            return contact(refs.resolve(self.scene, text), grip)
        except Exception:
            head, _, tail = text.rpartition(".")
            if not head:
                raise
            return contact(refs.resolve(self.scene, head), name=tail)

    def _ref(self, run: Run, ref) -> Target | NDArray[np.float64] | None:
        """A beat's reference: a role, ``role.attachment``, ``role@start`` (where the role's thing
        stood when the run started), a scene reference, or [x, y, z] in the body frame (a world
        point is returned as an array)."""
        if isinstance(ref, (list, tuple)):
            return (self.root @ np.r_[np.asarray(ref, dtype=float), 1.0])[:3]
        if not isinstance(ref, str):
            raise MotionError(f"motion '{run.name}': can't aim at {ref!r}")
        if ref.endswith("@start"):
            # Where a role's thing was when the run started (put a cup back where it stood).
            role = ref[: -len("@start")]
            if role not in run.origins:
                raise MotionError(f"motion '{run.name}': '{ref}': '{role}' is not a bound thing")
            return Target(None, run.origins[role].copy(), ref)
        if ref.startswith("self."):
            # A point on the body as it is now (the mouth is ``self.Head`` and an offset), in the
            # body's frame: an entry's offset is ahead / up / to the left of the body, not the bone.
            bone = ref[5:]
            if bone not in self.skeleton.bones:
                raise MotionError(f"motion '{run.name}': this body has no bone '{bone}'")
            m = _placement(np.zeros(3), _yaw_of(self.root))
            m[:3, 3] = (self.root @ self.skeleton.fk(self.out)[bone])[:3, 3]
            return Target(None, m, ref)
        if ref in run.bindings:
            return run.bindings[ref]
        head, _, tail = ref.partition(".")
        if head in run.bindings:
            bound = run.bindings[head]
            return contact(bound.node, name=tail) if bound.node is not None else None
        if head in (run.motion.get("roles") or {}):
            return None                                    # an optional role left unbound
        return self._bind(ref, None)

    # --- arbitration ---

    def _rank(self, run: Run, channel: str) -> tuple:
        """Who gets a channel: strength, then priority, then a claim over a share, then the later started."""
        return (run.strength(channel), run.motion.get("priority", 0), 0 if run.share(channel) else 1, run.order)

    def _arbitrate(self) -> None:
        for channel in self.skeleton.channels:
            best = None
            for run in self.runs:
                if run.strength(channel) and (best is None or self._rank(run, channel) > self._rank(best, channel)):
                    best = run
            best = self._share_turn(channel, best)
            was = self.owners.get(channel)
            if best is not was:
                self.share_t[channel] = 0.0
                if channel == "gaze" and self.look_key is not None:
                    # The head eases off the old look (and onto the new one) rather than springing.
                    self._fade_look()
                    self._changed.discard(channel)  # (an ended owner asked for it to be inertialized)
                else:
                    self._changed.add(channel)
                self._event("owner", best, channel=channel, was=was.name if was else None)
                # A motion that loses a channel it requires is asked to yield.
                if was is not None and not was.done and was.strength(channel) == STRENGTHS["require"]:
                    self.interrupt(was, "yield")
            self.owners[channel] = best
            if best is not None and best.share(channel):
                self.share_t[channel] = self.share_t.get(channel, 0.0) + self.dt
                key = (channel, best.order)
                self.share_looked[key] = self.share_looked.get(key, 0.0) + self.dt
        self._pause_and_resume()

    def _share_turn(self, channel: str, best: Run | None) -> Run | None:
        """Runs that share a channel (``holds.share: {gaze: weight}``) and rank equal take turns at
        it: the one that has it keeps it for ``SHARE_TURN`` x its weight, then it passes to the one
        that has had it least for its weight (eating and watching TV look at the plate and the
        screen by turns). A claim at the same strength outranks every share (a glance)."""
        if best is None or not best.share(channel):
            return best
        rank = self._rank(best, channel)[:2]
        sharers = [r for r in self.runs if not r.done and r.share(channel) and self._rank(r, channel)[:2] == rank]
        if len(sharers) < 2:
            return best
        was = self.owners.get(channel)
        if was in sharers and self.share_t.get(channel, 0.0) < SHARE_TURN * was.share(channel) - 1e-9:
            return was
        others = [r for r in sharers if r is not was]
        return min(others, key=lambda r: (self.share_looked.get((channel, r.order), 0.0) / r.share(channel), -r.order))

    def _pause_and_resume(self) -> None:
        """A run that loses a channel it claims (not a shared one: the gaze) to another run waits
        where it is, its beat's clock stopped; when it has them all back it picks up from the
        beat's ``resume`` (else that beat again, from where the body is now)."""
        for run in self.running():
            lost = {c for c, owner in self.owners.items()
                    if owner is not None and owner is not run and run.strength(c) >= STRENGTHS["claim"]
                    and not self.skeleton.channels[c].get("shared")}
            if lost and run.paused is None:
                run.paused = run.beat
                self._event("pause", run, channels=sorted(lost))
            elif not lost and run.paused is not None:
                beat = run.resume_to or run.spec.get("resume") or run.paused
                run.paused, run.resume_to = None, None
                self._event("resume", run, to=beat)
                self._enter(run, beat)

    def _owned(self, run: Run) -> set[str]:
        return {c for c, r in self.owners.items() if r is run}

    # --- composing the pose ---

    def _compose(self) -> tuple[Pose, NDArray[np.float64], dict]:
        s = self.skeleton
        pose = self.poses.get(self.posture, self.poses["stand"]).copy()
        root = self.root.copy()
        aims: dict[str, NDArray[np.float64]] = {}
        # Poses first, every owner over its own channels, eased from where its beat began.
        for run in self.runs:
            channels = self._owned(run)
            if not channels:
                continue
            spec, u = run.spec, ease(run.spec.get("ease", "smooth"), run.t / run.over if run.over else 1.0)
            target = self._beat_pose(run)
            # A posture change plays the body's transition clip (sit_down) on the channels it
            # holds, other than a hand it reaches with and the gaze: the build cleared its arms of
            # the body and clothes and keyed its drape against that whole movement, so it is taken
            # whole, not mixed with a blend of the rig's. What the body started with fades out
            # over the beat.
            clip, carried = self._transition(run), set()
            if clip is not None:
                reaching = {self.channel_of.get(e) for e, entry in self._ik_entries(run).items()
                            if self._ik_target(run, e, entry) is not None}
                at = min(run.t / run.over if run.over else 1.0, 1.0) * clip.duration
                now, first = self._clip_pose(clip, at), self._clip_pose(clip, 0.0)
                for channel in channels - reaching - {"gaze", "root"}:
                    for bone in s.channels[channel]["bones"]:
                        start = run.start["pose"].rotations.get(bone, IDENTITY)
                        offset = _quat_mul(start, _quat_inv(first.rotations.get(bone, IDENTITY)))
                        pose.rotations[bone] = _quat_mul(slerp(IDENTITY, offset, 1.0 - u),
                                                         now.rotations.get(bone, IDENTITY))
                        oa, o0, o1 = (p.offsets.get(bone) for p in (run.start["pose"], first, now))
                        if oa is not None or o1 is not None:
                            z = np.zeros(3)
                            pose.offsets[bone] = (z if o1 is None else o1) + \
                                ((z if oa is None else oa) - (z if o0 is None else o0)) * (1.0 - u)
                        carried.add(bone)
            for bone in s.names:
                if self.channel_of.get(bone) not in channels or bone in carried:
                    continue
                a = run.start["pose"].rotations.get(bone, IDENTITY)
                b = target.rotations.get(bone, IDENTITY)
                pose.rotations[bone] = slerp(a, b, u)
                oa, ob = run.start["pose"].offsets.get(bone), target.offsets.get(bone)
                if oa is not None or ob is not None:
                    oa = np.zeros(3) if oa is None else oa
                    ob = np.zeros(3) if ob is None else ob
                    pose.offsets[bone] = oa + (ob - oa) * u
            if "root" in channels and spec.get("root") and clip is not None and clip.meta.get("root"):
                # The clip's own root path, ending where the beat puts the body.
                from .generators.clips import root_at

                path = clip.meta["root"]
                goal = self._root_target(run) @ np.linalg.inv(root_at(path, clip.duration))
                now_root, first_root = goal @ root_at(path, at), goal @ root_at(path, 0.0)
                start = run.start["root"]
                turn = quat_from_matrix(start[:3, :3] @ first_root[:3, :3].T)
                root = np.eye(4)
                root[:3, :3] = matrix_from_quat(slerp(IDENTITY, turn, 1.0 - u)) @ now_root[:3, :3]
                root[:3, 3] = now_root[:3, 3] + (start[:3, 3] - first_root[:3, 3]) * (1.0 - u)
            elif "root" in channels and spec.get("root"):
                goal = self._root_target(run)
                start = run.start["root"]
                if spec.get("plant") == "feet":
                    # Over planted feet the body travels before it drops or rises (hips back, then
                    # down; forward over the feet, then up): the root leads the pose.
                    u = ease("smooth", (run.t / run.over if run.over else 1.0) / ROOT_LEAD)
                yaw = _yaw_of(start) + _wrap(_yaw_of(goal) - _yaw_of(start)) * u
                root = _placement(start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * u, yaw)
        # Drape correctives follow the body pose the hips owner is laying (a skirt seated or standing).
        morphs = self._pose_morph_vector(self.posture)
        hips = self.owners.get("hips")
        if hips is not None and "morphs" in hips.start:
            goal = self._pose_morph_vector(hips.spec.get("pose") or self.posture)
            u = ease(hips.spec.get("ease", "smooth"), hips.t / hips.over if hips.over else 1.0)
            clip = self._transition(hips)
            if clip is not None and clip.morphs:
                # Keyed through the transition by the build; the difference it started from fades.
                at = min(hips.t / hips.over if hips.over else 1.0, 1.0) * clip.duration
                now, first = ({(n, k): w for n, track in clip.morphs.items() for k, w in track.sample(x).items()}
                              for x in (at, 0.0))
                morphs = {k: now.get(k, w) + (hips.start["morphs"].get(k, 0.0) - first.get(k, 0.0)) * (1.0 - u)
                          for k, w in goal.items()}
            else:
                morphs = {k: hips.start["morphs"].get(k, 0.0) + (w - hips.start["morphs"].get(k, 0.0)) * u
                          for k, w in goal.items()}
        self._morph_goal = morphs
        # Owner changes are inertialized on the laid pose, before the solve: IK then lands on its
        # targets exactly while the rest of the body blends from where it was.
        self._inertialize(pose, root)
        # Then the solve, outside in: lean, planted feet, hands, gaze.
        inv_root = np.linalg.inv(root)
        for run in self.runs:
            channels = self._owned(run)
            spec = run.spec
            u_raw = run.t / run.over if run.over else 1.0
            u = ease(spec.get("ease", "smooth"), u_raw)
            lean = spec.get("lean")
            if lean and "spine" in channels and self._transition(run) is None:   # (a transition clip leans itself)
                by = float(lean.get("by", 0)) * _bump(u_raw, float(lean.get("at", 0.5)))
                extra = quat_from_matrix(Transform(rotation=np.radians([by, 0, 0])).to_matrix()[:3, :3])
                rest = s["Spine"].rotation
                pose.rotations["Spine"] = _quat_mul(pose.rotations.get("Spine", IDENTITY),
                                                    quat_from_matrix(rest.T @ matrix_from_quat(extra) @ rest))
            if spec.get("plant") == "feet" and "legs" in channels:
                for foot, world in run.start["feet"].items():
                    local = (inv_root @ np.r_[world, 1.0])[:3]
                    solve_limb(s, pose, foot, local, None, np.eye(3))
            laid = pose.copy()          # the spine as laid, before any reach bends it
            for end, entry in self._ik_entries(run).items():
                channel = self.channel_of.get(end)
                if channel not in channels:
                    continue
                goal = self._ik_target(run, end, entry)
                if goal is None:
                    continue
                entry = self._turn_entry(run, entry, root)
                if entry.get("tip"):
                    # The fingertip goes on the target, the hand turned as asked: the wrist stops short
                    # by the finger as this hand and its shape make it (a big hand stops further off).
                    full = _hand_turn(end, entry)
                    if full is None:
                        raise MotionError(f"motion '{run.name}': {end} tip: needs fingers / palm to turn the hand")
                    goal = goal - root[:3, :3] @ full @ self._tip(end, entry["tip"], pose)
                start = run.start["ends"][end]
                point = start + (goal - start) * u
                if entry.get("arc"):
                    # Up and over on the way (a hand coming from beside a table onto something on
                    # it): the path rises by `arc` a third of the way along, up before it goes over
                    # the edge, rather than sweeping through it. The arm is blended in from where it
                    # was by `u`, which scales what the solve asks for: asked for 6.75 (1 - u)^2 arc,
                    # the hand rises by about 6.75 u (1 - u)^2 arc (asked for no more than twice
                    # the arc: at the start the solve would be sent far out of reach).
                    lift = float(entry["arc"]) * min(6.75 * (1.0 - u) ** 2, 2.0)
                    point = point + np.array([0.0, lift, 0.0])
                aims[end] = goal
                if run.t >= run.over - self.dt - 1e-9:      # the beat's last composed frame
                    aims[f"{end}@arrived"] = goal
                if "spine" in channels and end in ("LeftHand", "RightHand"):
                    self._bend_to_reach(pose, laid, root, end, point, run.start, u,
                                        goal if entry.get("arc") else None)
                local = (inv_root @ np.r_[point, 1.0])[:3]
                # Solved, then blended in from the arm as the beat found it by the beat's progress:
                # it starts with its own elbow and ends exactly on the goal, rather than snapping to
                # the solver's elbow on the beat's first frame.
                # A hand turned by `fingers` / `palm` turns in the body's frame, from how the beat
                # found it to the turn it asks for: the same arc every frame, whatever the arm does
                # (blending its rotation under the forearm flipped where the arc nears half a turn).
                turn = _hand_turn(end, entry)
                if turn is not None:
                    rest = s[end].rotation
                    turn = matrix_from_quat(slerp(run.start["hands"][end], quat_from_matrix(turn @ rest), u)) @ rest.T
                solved = pose.copy()
                solve_limb(s, solved, end, local, entry.get("pole"), turn)
                for bone in (*LIMBS[end][:2], end):
                    pose.rotations[bone] = slerp(run.start["pose"].rotations.get(bone, IDENTITY), solved.rotations[bone], u)
                if turn is not None:            # under the arm as blended, not as solved
                    pose.rotations[end] = s.pose_quat(end, s.world(LIMBS[end][1], pose), turn @ s[end].rotation)
        self._hang_free_arms(pose)
        self.prelook = {b: pose.rotations.get(b, IDENTITY).copy() for b in LOOK_BONES}
        gaze = self.owners.get("gaze")
        look = gaze.spec.get("look") if gaze is not None else None
        target = None
        if look is not None:
            target = self._ref(gaze, look.get("at") if isinstance(look, dict) else look)
        # A look that ended eases off over LOOK_OUT, under whatever look comes next.
        if self.look_key is not None and (target is None or self.look_key is not gaze):
            self._fade_look()
        if self.look_fade is not None:
            faded, w0, t = self.look_fade
            t += self.dt
            if t >= LOOK_OUT:
                self.look_fade = None
            else:
                self.look_fade = (faded, w0, t)
                self._turn_head(pose, root, faded, w0 * (1.0 - ease("smooth", t / LOOK_OUT)))
        if target is None:
            return pose, root, aims
        point = target.world()[:3, 3] if isinstance(target, Target) else np.asarray(target, dtype=float)
        aims["gaze"] = point
        # One look runs on across beats: the turn eases in once, and a new point to look at is
        # followed (the head travels to it) instead of the turn starting again from straight ahead.
        if self.look_key is None:
            self.look_key, self.look_t, self.look_point = gaze, 0.0, point.copy()
        else:
            self.look_t += self.dt
            k = 1.0 - 0.5 ** (self.dt / LOOK_FOLLOW)
            self.look_point = self.look_point + (point - self.look_point) * k
        weight = float(look.get("weight", 1.0)) if isinstance(look, dict) else 1.0
        self.look_weight = weight * ease("smooth", self.look_t / LOOK_IN)
        self._turn_head(pose, root, self.look_point, self.look_weight)
        return pose, root, aims

    def _fade_look(self) -> None:
        """The current look ends: its turn eases off from where it is instead of the head springing
        back (an inertialized gaze snaps back at its short half-life)."""
        if self.look_key is not None and self.look_point is not None:
            self.look_fade = (self.look_point.copy(), self.look_weight, 0.0)
        self.look_key, self.look_point = None, None

    def _hang_free_arms(self, pose: Pose) -> None:
        """An arm nothing holds hangs as it would from an upright spine when the spine bends or
        leans (a reach, a lean): the upper arm swings HANG_SHARE of the way to the shoulder-to-hand
        line it has with the posture's spine, rather than swinging out behind with the chest (and
        not all the way: hanging plumb from a shoulder bent over a table puts the hand in its edge).
        A swing, not a solve, so with no bend it is no change at all."""
        if self.posture != "stand":
            return
        s, base = self.skeleton, self.poses.get(self.posture, self.poses["stand"])
        spine = s.channels["spine"]["bones"]
        if all(_angle(pose.rotations.get(b, IDENTITY), base.rotations.get(b, IDENTITY)) < 1e-4 for b in spine):
            return
        free = [side for side in ("Left", "Right") if self.owners.get(f"{side.lower()}_hand") is None]
        if not free:
            return
        upright = pose.copy()
        for b in spine:
            upright.rotations[b] = base.rotations.get(b, IDENTITY)
        was, now = s.fk(upright), s.fk(pose)
        for side in free:
            upper, end = f"{side}UpperArm", f"{side}Hand"
            swing = _swing(now[end][:3, 3] - now[upper][:3, 3], was[end][:3, 3] - was[upper][:3, 3])
            swing = matrix_from_quat(slerp(IDENTITY, quat_from_matrix(swing), HANG_SHARE))
            parent = s.world(s[upper].parent, pose)
            pose.rotations[upper] = s.pose_quat(upper, parent, swing @ now[upper][:3, :3])

    def _bend_to_reach(self, pose: Pose, beat: Pose, root: NDArray[np.float64], end: str,
                       point: NDArray[np.float64], start: dict, u: float,
                       lead: NDArray[np.float64] | None = None) -> None:
        """Bend forward at the spine, as little as will do, when the hand can't reach ``point``
        from where the shoulder is (a button low on a TV, a cup at the back of a table). The bend
        is measured from the spine as laid this frame (the beat's own bones eased in, before any
        reach), so beats that keep reaching bend continuously rather than adding up; one that stops reaching eases it out with its pose. What the arm
        already spanned when the beat began counts as in reach at its start (a hanging arm is nearly
        straight) and eases to REACH_SLACK by its end (``u``), so a reach starts without a bend and
        arrives with the elbow a little bent. With ``lead`` (the goal of a reach that arcs up and
        over) the bend the goal needs eases in with the beat as well: an arc's last stretch drops
        fast, and a bend that only starts when it leaves reach starts with a jolt."""
        s = self.skeleton
        shoulder = end.replace("Hand", "UpperArm")
        rest = s.fk()
        reach = REACH_SLACK * sum(float(np.linalg.norm(rest[a][:3, 3] - rest[b][:3, 3])) for a, b in
                                  ((shoulder, end.replace("Hand", "LowerArm")), (end.replace("Hand", "LowerArm"), end)))
        full = reach / REACH_SLACK
        spanned = float(np.linalg.norm(start["ends"][end] - (start["root"] @ s.world(shoulder, start["pose"]))[:3, 3]))
        reach = max(reach, reach + (min(spanned, full) - reach) * (1.0 - u))
        base = {b: beat.rotations.get(b, IDENTITY) for b in BEND_BONES}
        trial = pose.copy()

        def bend(degrees: float, at: NDArray[np.float64], within: float) -> bool:
            for bone in BEND_BONES:
                r = s[bone].rotation
                extra = Transform(rotation=np.radians([degrees / len(BEND_BONES), 0, 0])).to_matrix()[:3, :3]
                trial.rotations[bone] = _quat_mul(base[bone], quat_from_matrix(r.T @ extra @ r))
            return np.linalg.norm(at - (root @ s.world(shoulder, trial))[:3, 3]) <= within

        def least(at: NDArray[np.float64], within: float) -> float:
            # The least bend that reaches, found continuously (a stepped search makes the spine and
            # arm jitter as the hand travels).
            if bend(0.0, at, within):
                return 0.0
            lo, hi = 0.0, float(MAX_BEND)
            if bend(hi, at, within):
                for _ in range(14):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (lo, mid) if bend(mid, at, within) else (mid, hi)
            return hi

        degrees = least(point, reach)
        if lead is not None:
            degrees = max(degrees, least(lead, full * REACH_SLACK) * u)
        bend(degrees, point, reach)
        for bone in BEND_BONES:
            pose.rotations[bone] = trial.rotations[bone]

    def _beat_pose(self, run: Run) -> Pose:
        spec = run.spec
        name = spec.get("pose")
        base = self.poses.get(name) if name else None
        if name and base is None:
            raise MotionError(f"motion '{run.name}': this body has no pose '{name}' (it has {sorted(self.poses)})")
        pose = (base or self.poses.get(self.posture, self.poses["stand"])).copy()
        if spec.get("hips") is not None:
            pose.offsets["Hips"] = pose.offsets.get("Hips", np.zeros(3)) + np.asarray(spec["hips"], dtype=float)
        for bone, degrees in (spec.get("bones") or {}).items():
            extra = Transform(rotation=np.radians(np.asarray(degrees, dtype=float))).to_matrix()[:3, :3]
            rest = self.skeleton[bone].rotation
            pose.rotations[bone] = _quat_mul(pose.rotations.get(bone, IDENTITY),
                                             quat_from_matrix(rest.T @ extra @ rest))
        for end, entry in self._ik_entries(run).items():
            shape = self._hand_shape(run, end, entry)
            if shape is None:
                continue
            side = end[: -len("Hand")]
            for bone in self.fingers.get(side, ()):
                # A shape poses every finger: one it doesn't name is straight.
                pose.rotations[bone] = self.skeleton.body_quat(
                    bone, Transform(rotation=np.radians(shape.get(bone, [0.0, 0.0, 0.0]))).to_matrix()[:3, :3])
        return pose

    def _hand_shape(self, run: Run, end: str, entry: dict) -> dict[str, list[float]] | None:
        """The hand shape a hand's ``ik`` entry takes (``grip:``, else its target's grip), or None
        for an end that isn't a hand or a grip that isn't a shape (sit, look)."""
        if end not in ("LeftHand", "RightHand"):
            return None
        shapes = self.skeleton.hand_shapes
        grip = entry.get("grip")
        if grip is None:
            target = self._ref(run, entry.get("target"))
            grip = target.grip if isinstance(target, Target) else None
            return shapes.get(grip) if grip else None
        if grip in shapes:
            return shapes[grip]
        if grip in run.bindings:
            return shapes.get(run.bindings[grip].grip or "")
        raise MotionError(f"motion '{run.name}' beat '{run.beat}': {end} grip '{grip}' is neither a hand shape "
                          f"({sorted(shapes)}) nor one of its roles")

    def _ik_entries(self, run: Run) -> dict[str, dict]:
        from .core.skeleton import mirror_name

        spec = run.spec.get("ik") or {}
        entries = {k: dict(v) for k, v in spec.items() if k != "mirror"}
        if spec.get("mirror"):
            for name, entry in list(entries.items()):
                twin = mirror_name(name)
                if twin and twin not in entries:
                    twin_entry = dict(entry)
                    target = entry.get("target")
                    if isinstance(target, str) and target.endswith(".left"):
                        twin_entry["target"] = target[:-5] + ".right"
                    elif isinstance(target, (list, tuple)):           # a body-frame point: across the body
                        twin_entry["target"] = [-float(target[0]), *target[1:]]
                    for key in ("pole", "fingers", "palm"):
                        if key in entry:
                            twin_entry[key] = list(np.asarray(entry[key], dtype=float) * [-1, 1, 1])
                    entries[twin] = twin_entry
        return entries

    def _ik_target(self, run: Run, end: str, entry: dict) -> NDArray[np.float64] | None:
        target = self._ref(run, entry.get("target"))
        if target is None:
            return None
        if isinstance(target, Target):
            m = target.world()
            offset = np.asarray(entry.get("offset", [0, 0, 0]), dtype=float)
            target = (m @ np.r_[offset, 1.0])[:3]
        if entry.get("holding") is not None:
            # The thing in that hand goes onto the target: the hand goes where that puts it, by the
            # hand-to-thing offset as the beat found it.
            target = target + self._held_gap(run, end, entry["holding"])
        return target

    def _turn_entry(self, run: Run, entry: dict, root: NDArray[np.float64]) -> dict:
        """``entry`` with its ``fingers`` / ``palm`` in the body's frame: with ``frame: target``
        they are given in the target's frame (into a button's face, down onto a table's top), so a
        hand meets the thing the same way wherever the body stands."""
        if entry.get("frame", "body") == "body" or not ("fingers" in entry or "palm" in entry):
            return entry
        if entry["frame"] != "target":
            raise MotionError(f"motion '{run.name}': ik frame '{entry['frame']}': expected body or target")
        target = self._ref(run, entry.get("target"))
        if not isinstance(target, Target):
            return entry
        turn = root[:3, :3].T @ target.world()[:3, :3]
        return {**entry, **{k: list(turn @ np.asarray(entry[k], dtype=float)) for k in ("fingers", "palm") if k in entry}}

    def _tip(self, end: str, finger: str, pose: Pose) -> NDArray[np.float64]:
        """From the wrist to ``finger``'s tip (index, middle, ...) in the body's frame, the hand at
        its rest orientation with the fingers as ``pose`` shapes them."""
        from .generators.hand import TIP_LENGTH

        side = end[: -len("Hand")]
        name = finger.capitalize()
        if name not in TIP_LENGTH:
            raise MotionError(f"{end} tip '{finger}': expected one of {sorted(k.lower() for k in TIP_LENGTH)}")
        s = self.skeleton
        shaped = Pose({b: q for b, q in pose.rotations.items() if b.startswith(side + name)})
        fk = s.fk(shaped)
        distal = fk[f"{side}{name}Distal"][:3, 3]
        along = distal - fk[f"{side}{name}Intermediate"][:3, 3]
        along /= np.linalg.norm(along)
        scale = float(s.params.get("height", 1.7)) / 1.7 * float(s.params.get("hand_size", 1.0))
        return distal + along * TIP_LENGTH[name] * scale - fk[end][:3, 3]

    def _held_gap(self, run: Run, end: str, role: str) -> NDArray[np.float64]:
        """From the thing bound to ``role`` to the hand holding it, as the beat found them."""
        bound = run.bindings.get(role)
        for node, gap in run.start.get("held") or []:
            if bound is not None and node is bound.node:
                return gap
        raise MotionError(f"motion '{run.name}' beat '{run.beat}': {end} holding '{role}', which no hand holds")

    def _root_target(self, run: Run) -> NDArray[np.float64]:
        spec = run.spec["root"]
        target = self._ref(run, spec.get("at"))
        m = target.world() if isinstance(target, Target) else _placement(target, _yaw_of(self.root))
        m = _placement(m[:3, 3], _yaw_of(m))
        posture = spec.get("posture")
        if posture is not None:
            p = self.root_poses.get(posture) or {}
            local = Transform(translation=np.asarray(p.get("offset", [0, 0, 0]), dtype=float),
                              rotation=np.radians(np.asarray(p.get("rotation", [0, 0, 0]), dtype=float))).to_matrix()
            return m @ local
        m = m @ Transform(translation=np.asarray(spec.get("offset", [0, 0, 0]), dtype=float)).to_matrix()
        facing = spec.get("facing", "away")
        turn = 0.0 if facing == "away" else math.pi if facing == "toward" else math.radians(float(facing))
        return _placement([m[0, 3], self.floor, m[2, 3]], _yaw_of(m) + turn)

    def _turn_head(self, pose: Pose, root: NDArray[np.float64], point: NDArray[np.float64], weight: float) -> None:
        s = self.skeleton
        fk = s.fk(pose)
        head = (root @ fk["Head"])[:3, 3]
        local = np.linalg.inv(root)[:3, :3] @ (point - head)
        # Soft limits: the turn slows into them instead of stopping dead at a clamp.
        yaw = YAW_LIMIT * math.tanh(math.atan2(local[0], local[2]) / YAW_LIMIT)
        pitch = PITCH_LIMIT * math.tanh(-math.atan2(local[1], math.hypot(local[0], local[2])) / PITCH_LIMIT)
        # What the pose already does to the head's facing is kept; the turn is split neck / head.
        for bone, share in (("Neck", NECK_SHARE), ("Head", 1.0 - NECK_SHARE)):
            r = Transform(rotation=np.array([pitch * share * weight, yaw * share * weight, 0.0])).to_matrix()[:3, :3]
            rest = s[bone].rotation
            pose.rotations[bone] = _quat_mul(pose.rotations.get(bone, IDENTITY),
                                             quat_from_matrix(rest.T @ r @ rest))

    # --- blending ---

    def _inertialize(self, pose: Pose, root: NDArray[np.float64]) -> None:
        """Channels whose owner changed this step start from where the body was (as shown, solved):
        the difference to the newly laid pose decays over the channel's half-life instead of the
        pose jumping. Runs between laying and solving, so IK ends and planted feet stay exact."""
        for channel in self._changed:
            life = HALF_LIFE.get(channel, DEFAULT_HALF_LIFE)
            for bone in self.skeleton.channels[channel]["bones"]:
                old = self.out.rotations.get(bone, IDENTITY)      # as shown, offsets included
                new = pose.rotations.get(bone, IDENTITY)
                self.offsets[bone] = (_quat_mul(old, _quat_inv(new)), 0.0, life)
            if channel == "root":
                self.root_offset = (self.root @ np.linalg.inv(root), 0.0, life)
            if channel == "hips":
                self.morph_offset = ({k: self.morphs.get(k, 0.0) - w for k, w in self._morph_goal.items()}, 0.0)
        self._changed = set()
        morphs = dict(self._morph_goal)
        if self.morph_offset is not None:
            delta, age = self.morph_offset
            w = 0.5 ** (age / MORPH_HALF_LIFE)
            if w < 0.01:
                self.morph_offset = None
            else:
                morphs = {k: v + delta.get(k, 0.0) * w for k, v in morphs.items()}
                self.morph_offset = (delta, age + self.dt)
        self.morphs = morphs
        for bone, (offset, age, life) in list(self.offsets.items()):
            w = 0.5 ** (age / life)
            if w < 0.01:
                del self.offsets[bone]
                continue
            pose.rotations[bone] = _quat_mul(slerp(IDENTITY, offset, w), pose.rotations.get(bone, IDENTITY))
            self.offsets[bone] = (offset, age + self.dt, life)
        if self.root_offset is not None:
            offset, age, life = self.root_offset
            w = 0.5 ** (age / life)
            if w < 0.01:
                self.root_offset = None
            else:
                was = offset @ root
                yaw = _yaw_of(root) + _wrap(_yaw_of(was) - _yaw_of(root)) * w
                root[:] = _placement(root[:3, 3] + (was[:3, 3] - root[:3, 3]) * w, yaw)
                self.root_offset = (offset, age + self.dt, life)

    # --- output ---

    def _apply(self) -> None:
        s = self.skeleton
        for name, node in self.joints.items():
            node.transform = Transform.from_matrix(s.posed_local(name, self.out))
            node.transform.scale = np.ones(3)
        for (node, k), w in self.morphs.items():
            self.morph_nodes[node].morph_weights[k] = w
        parent = self.actor.parent.world_transform() if self.actor.parent is not None else np.eye(4)
        self.actor.transform = Transform.from_matrix(np.linalg.inv(parent) @ self.root)
        for entry in list(self.falling):
            node, speed = entry
            speed += FALL * self.dt
            world = node.world_transform()
            world[1, 3] = max(self.floor, world[1, 3] - speed * self.dt)
            up = node.parent.world_transform() if node.parent is not None else np.eye(4)
            node.transform = Transform.from_matrix(np.linalg.inv(up) @ world)
            entry[1] = speed
            if world[1, 3] <= self.floor:
                self.falling.remove(entry)
        if self.attached:
            fk = s.fk(self.out)
            for held in self.attached.values():
                node, hand = held.node, self.root @ fk[held.end]
                if held.level:
                    heading = _placement(np.zeros(3), _yaw_of(self.root))
                    world = heading @ held.rel
                    world[:3, 3] = hand[:3, 3] + heading[:3, :3] @ held.rel[:3, 3]
                else:
                    world = hand @ held.rel
                up = node.parent.world_transform() if node.parent is not None else np.eye(4)
                node.transform = Transform.from_matrix(np.linalg.inv(up) @ world)

    def _ends_world(self, ends=("LeftHand", "RightHand", "LeftFoot", "RightFoot")) -> dict[str, NDArray[np.float64]]:
        fk = self.skeleton.fk(self.out)
        return {e: (self.root @ fk[e])[:3, 3] for e in ends}

    def _sample(self, aims: dict) -> None:
        ends = self._ends_world(("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head"))
        self.samples.append({
            "t": round(self.time, 4),
            "rotations": {b: q.copy() for b, q in self.out.rotations.items()},
            "root": self.root.copy(),
            "ends": ends,
            "aims": {k: np.asarray(v).copy() for k, v in aims.items()},
            "owners": {c: (r.name if r else None) for c, r in self.owners.items()},
            "beats": {r.name: r.beat for r in self.runs},
            "planted": [f for r in self.runs if r.spec.get("plant") == "feet" and "legs" in self._owned(r)
                        for f in ("LeftFoot", "RightFoot")],
        })

    def _event(self, event: str, run: Run | None, **data) -> None:
        entry = {"t": round(self.time, 4), "event": event, "motion": run.name if run else None}
        if run is not None:
            entry["beat"] = run.beat or None
        entry.update(data)
        self.trace.append(entry)


def mirror_motion(spec: dict) -> dict:
    """``spec`` for the other side of the body: hands, bones, body-frame points and turns swap
    across the body. A point on a thing stays where it is."""
    from .core.skeleton import mirror_name

    def across(v) -> list:
        x = v[0]
        return [f"{{-({x.strip('{}')})}}" if isinstance(x, str) else -float(x), *v[1:]]

    def side(name: str) -> str:
        swap = {"hand": "left", "left": "right", "right": "left", "left_hand": "right_hand", "right_hand": "left_hand"}
        return swap.get(name) or mirror_name(name) or name

    beats = {}
    for name, beat in spec["beats"].items():
        beat = dict(beat or {})
        ik = {}
        for end, entry in (beat.get("ik") or {}).items():
            if end == "mirror":
                ik[end] = entry
                continue
            entry = dict(entry)
            target = entry.get("target")
            on_body = isinstance(target, (list, tuple)) or (isinstance(target, str) and target.startswith("self."))
            if isinstance(target, (list, tuple)):
                entry["target"] = across(target)
            for key in ("pole", "fingers", "palm", *(("offset",) if on_body else ())):
                if key in entry:
                    entry[key] = across(entry[key])
            ik[side(end)] = entry
        if ik:
            beat["ik"] = ik
        if beat.get("bones"):
            beat["bones"] = {side(b): [d[0], -float(d[1]), -float(d[2])] for b, d in beat["bones"].items()}
        if beat.get("hips") is not None:
            beat["hips"] = across(beat["hips"])
        if beat.get("attach"):
            beat["attach"] = {role: {**how, "hand": side(how.get("hand", "hand"))} if isinstance(how, dict) else side(how)
                              for role, how in beat["attach"].items()}
        beats[name] = beat
    holds = {kind: ({side(c): w for c, w in v.items()} if isinstance(v, dict) else [side(c) for c in v])
             for kind, v in (spec.get("holds") or {}).items()}
    return {**spec, "beats": beats, "holds": holds}


def _hand_turn(end: str, entry: dict) -> NDArray[np.float64] | None:
    """The body-frame turn from a hand's T-pose orientation (fingers out along the arm, palm down)
    to the one its ``ik`` entry asks for: ``fingers`` the way the fingers point, ``palm`` the way
    the palm faces (made square to the fingers). None when the entry names neither."""
    if end not in ("LeftHand", "RightHand") or ("fingers" not in entry and "palm" not in entry):
        return None
    side = 1.0 if end == "LeftHand" else -1.0
    rest_f, rest_p = np.array([side, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
    f = np.asarray(entry.get("fingers", rest_f), dtype=float)
    f = f / np.linalg.norm(f)
    p = np.asarray(entry.get("palm", rest_p), dtype=float)
    p = p - np.dot(p, f) * f
    if np.linalg.norm(p) < 1e-6:
        raise MotionError(f"{end}: palm {entry.get('palm')} lies along the fingers {entry.get('fingers')}")
    p = p / np.linalg.norm(p)
    target = np.column_stack([f, p, np.cross(f, p)])
    rest = np.column_stack([rest_f, rest_p, np.cross(rest_f, rest_p)])
    return target @ rest.T


def _swing(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """The shortest rotation (3x3) turning direction ``a`` onto ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis, c = np.cross(a, b), float(np.dot(a, b))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c) if c > -0.9999 else -np.eye(3)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
