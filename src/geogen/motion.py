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

Not yet (later geogen.103 tasks): ``when``/``until``/``do``/``bend``/``hips``/``while``/``layer``
and shared-gaze time slicing (103.6), ``resume`` after an overlay (103.5),
locomotion between places (the body slides onto ``root`` targets), and the Godot port (103.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import roots
from .core.ik import solve_limb
from .core.node import SceneNode
from .core.skeleton import Pose, Skeleton
from .core.transform import Transform, matrix_from_quat, quat_from_matrix, slerp

MOTION_KIND = "motion"
MOTIONS_DIR = "motions"

MOTION_KEYS = {
    "kind": "motion",
    "name": "the motion's name (defaults to the file's)",
    "roles": "{role: {grip, optional}}: the things it acts on, bound by whoever runs it",
    "params": "{name: {default}}: build-time values for {expr} in its beats",
    "holds": "{require: [channels], claim: [...], prefer: [...], share: {channel: weight}}",
    "posture": "{from, to}: its edge in the posture graph (stand -> sit)",
    "while": "{motion: {...}}: what it does to other motions while it runs (not yet)",
    "layer": "true: it outlives the action that started it (carry; not yet)",
    "initial": "the first beat (default: the first listed)",
    "beats": "{name: beat}: its named moments",
    "extends": "another motion whose keys this one starts from",
    "priority": "breaks ties between equal claims (default 0; reactions higher)",
}

BEAT_KEYS = {
    "pose": "a pose of the body (body.poses) laid over the channels the motion holds",
    "bones": "{Bone: [x, y, z] degrees}: body-frame rotations on top of the pose",
    "ik": "{End: {target: <ref> | [x, y, z], offset, pole, rest, grip}, mirror}: hands and feet onto points; "
          "a hand takes the shape `grip` names (a hand shape, or a role: its contact's grip), else its "
          "target's grip when that is a hand shape (skeleton hand_shapes: press, power, pinch, ...)",
    "root": "{at: <ref>, offset: [x, y, z], facing: away | toward | <deg>, posture: <pose>}: where the body stands",
    "look": "<ref> | {at: <ref>, weight}: where the head turns",
    "plant": "feet: the feet stay where they are on the floor (legs by IK)",
    "lean": "{by: deg, at: 0..1}: a forward lean of the spine that peaks at `at` of the beat and is gone by its end",
    "attach": "{role: hand}: the thing goes with that hand from here",
    "release": "<role>: let go of it where it is",
    "over": "seconds the beat takes to arrive (default: from the distance its targets travel)",
    "ease": "smooth (default) | settle | linear",
    "emit": "an event fired on arrival (the world changes in step with the body)",
    "then": "the next beat, or done",
    "after": "seconds to dwell once arrived, or `duration` (the runner's duration)",
    "interrupt": "yield (default: go to `exit`) | finish (this beat first) | hold (refuse) | drop (stop here)",
    "exit": "the beat to go to when asked to yield",
    "resume": "where to pick up after a passing interruption (not yet)",
    "when": "skip the beat unless this holds (not yet)",
    "until": "dwell until this holds (not yet)",
    "do": "a whole motion as one beat (not yet)",
    "bend": "auto: low targets blend toward reach_low (not yet)",
    "hips": "<ref>: the hips onto a point (not yet: use root.posture)",
}
NOT_YET = {"when", "until", "do", "bend", "hips", "resume"}
INTERRUPTS = ("yield", "finish", "hold", "drop")
STRENGTHS = {"prefer": 1, "claim": 2, "require": 3}
EASES = ("smooth", "settle", "linear", "spring")

FPS = 30
REACH_SPEED = 0.9          # m/s an effector or the body travels when a beat gives no `over`
MIN_OVER = 0.35            # s: the shortest arrival
HALF_LIFE = {"gaze": 0.08, "left_hand": 0.12, "right_hand": 0.12}   # s, inertialization per channel
DEFAULT_HALF_LIFE = 0.18
REACH_SLACK = 0.96         # of the arm's length: a hand reaches no further without bending
BEND_BONES = ("Spine", "Chest")
MAX_BEND = 75              # degrees, shared by BEND_BONES
ROOT_LEAD = 0.7            # over planted feet the root arrives by this fraction of the beat
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
    return spec


def validate_motion(spec: dict, where: str = "motion") -> None:
    """Raise ``MotionError`` for unknown keys, unknown beats, and holds or interrupts that aren't
    one of the vocabulary."""
    unknown = set(spec) - set(MOTION_KEYS)
    if unknown:
        raise MotionError(f"{where}: unknown keys {sorted(unknown)} (known: {sorted(MOTION_KEYS)})")
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
        unknown = set(beat or {}) - set(BEAT_KEYS)
        if unknown:
            raise MotionError(f"{where}: beat '{name}': unknown keys {sorted(unknown)} (known: {sorted(BEAT_KEYS)})")
        for key in ("then", "exit", "resume"):
            target = (beat or {}).get(key)
            if target is not None and target != "done" and target not in beats:
                raise MotionError(f"{where}: beat '{name}': {key} '{target}' is not a beat")
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


def contact(node: SceneNode, grip: str | None = None, name: str | None = None) -> Target:
    """Where a body touches ``node``: the attachment ``name`` (or, when ``name`` is one of its
    interactions, the attachment that interaction's ``touch:`` names), else the first attachment
    with ``grip``, else (for sit / lie / stand / use) the anchor of its first affordance of that
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
            if point.grip == grip:
                return attached(key)
        for a in node.meta.get("affordances") or []:
            if a.get("type") == grip:
                local = Transform(translation=np.asarray(a["position"], dtype=float),
                                  rotation=np.array([0.0, math.radians(a["yaw"]), 0.0])).to_matrix()
                return Target(node, local, f"{node.name}[{grip}]", grip)
    return Target(node, np.eye(4), node.name)


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
HANDS = {"hand": "RightHand", "right_hand": "RightHand", "left_hand": "LeftHand"}
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")


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
        self.poses = self._pose_library()
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
        self.attached: dict[str, tuple[SceneNode, str, NDArray[np.float64]]] = {}
        self.trace: list[dict] = []
        self.samples: list[dict] = []
        self._order = 0
        self._changed: set[str] = set()     # channels whose source changed this step (inertialized)
        self._apply()

    # The pose library: each pose_<name> clip (as the build left it: arms cleared, seats fitted).
    def _pose_library(self) -> dict[str, Pose]:
        s = self.skeleton
        out = {}
        for clip in self.actor.clips:
            if not clip.name.startswith("pose_"):
                continue
            pose = Pose()
            for joint, track in clip.tracks.items():
                if joint not in s.bones:
                    continue
                rotation, translation = track.sample(0.0)
                rest = s.rest_local(joint)
                if rotation is not None:
                    pose.rotations[joint] = quat_from_matrix(rest[:3, :3].T @ matrix_from_quat(rotation))
                if translation is not None and s[joint].parent is not None:
                    delta = np.asarray(translation) - rest[:3, 3]
                    if np.abs(delta).max() > 1e-6:
                        pose.offsets[joint] = delta
            out[clip.name[5:]] = pose
        out.setdefault("stand", Pose())
        return out

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
        spec = {**spec, "beats": resolve_value(spec["beats"], values)}
        self._order += 1
        run = Run(spec, resolved, self._order, params)
        self.runs.append(run)
        self._event("start", run)
        self._enter(run, spec.get("initial", next(iter(spec["beats"]))))
        return run

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
        self._enter(run, exit_beat)
        return "exiting"

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
        self._inertialize(pose, root)
        self.out, self.root = pose, root
        self._apply()
        self._sample(aims)

    def _advance(self, run: Run) -> None:
        run.t += self.dt
        guard = 0
        self._arrive(run)
        while not run.done and run.t >= run.over + run.dwell and guard < 16:
            guard += 1
            spill = run.t - (run.over + run.dwell)
            nxt = run.spec.get("exit") if run.pending_exit else run.spec.get("then", "done")
            run.pending_exit = False
            if nxt in (None, "done"):
                self._end(run)
                return
            self._enter(run, nxt)
            run.t = spill
            self._arrive(run)

    def _arrive(self, run: Run) -> None:
        """A beat that has arrived fires its emit and takes or lets go of what it names, once."""
        if run.done or run.t < run.over or run.start.get("arrived"):
            return
        run.start["arrived"] = True
        spec = run.spec
        for role, hand in (spec.get("attach") or {}).items():
            target = run.bindings.get(role)
            if target is None or target.node is None:
                raise MotionError(f"motion '{run.name}': attach: '{role}' is not a bound thing")
            end = HANDS.get(hand, hand)
            hand_world = self.root @ self.skeleton.fk(self.out)[end]
            self.attached[role] = (target.node, end, np.linalg.inv(hand_world) @ target.node.world_transform())
            self._event("attach", run, role=role, hand=end)
        release = spec.get("release")
        for role in ([release] if isinstance(release, str) else release or []):
            if self.attached.pop(role, None) is not None:
                self._event("release", run, role=role)
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
        run.start = {"pose": self.out.copy(), "root": self.root.copy(), "ends": self._ends_world(),
                     "feet": self._ends_world(("LeftFoot", "RightFoot"))}
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
        return max(MIN_OVER, travel / REACH_SPEED + 0.15) if travel else (MIN_OVER if spec.get("pose") else 0.0)

    def _end(self, run: Run) -> None:
        if not run.done:
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
        """A beat's reference: a role, ``role.attachment``, a scene reference, or [x, y, z] in the
        body frame (a world point is returned as an array)."""
        if isinstance(ref, (list, tuple)):
            return (self.root @ np.r_[np.asarray(ref, dtype=float), 1.0])[:3]
        if not isinstance(ref, str):
            raise MotionError(f"motion '{run.name}': can't aim at {ref!r}")
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

    def _arbitrate(self) -> None:
        for channel in self.skeleton.channels:
            best = None
            for run in self.runs:
                s = run.strength(channel)
                if s and (best is None or (s, run.motion.get("priority", 0), run.order)
                          > (best.strength(channel), best.motion.get("priority", 0), best.order)):
                    best = run
            was = self.owners.get(channel)
            if best is not was:
                self._changed.add(channel)
                self._event("owner", best, channel=channel, was=was.name if was else None)
                # A motion that loses a channel it requires is asked to yield.
                if was is not None and not was.done and was.strength(channel) == STRENGTHS["require"]:
                    self.interrupt(was, "yield")
            self.owners[channel] = best

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
            for bone in s.names:
                if self.channel_of.get(bone) not in channels:
                    continue
                a = run.start["pose"].rotations.get(bone, IDENTITY)
                b = target.rotations.get(bone, IDENTITY)
                pose.rotations[bone] = slerp(a, b, u)
                oa, ob = run.start["pose"].offsets.get(bone), target.offsets.get(bone)
                if oa is not None or ob is not None:
                    oa = np.zeros(3) if oa is None else oa
                    ob = np.zeros(3) if ob is None else ob
                    pose.offsets[bone] = oa + (ob - oa) * u
            if "root" in channels and spec.get("root"):
                goal = self._root_target(run)
                start = run.start["root"]
                if spec.get("plant") == "feet":
                    # Over planted feet the body travels before it drops or rises (hips back, then
                    # down; forward over the feet, then up): the root leads the pose.
                    u = ease("smooth", (run.t / run.over if run.over else 1.0) / ROOT_LEAD)
                yaw = _yaw_of(start) + _wrap(_yaw_of(goal) - _yaw_of(start)) * u
                root = _placement(start[:3, 3] + (goal[:3, 3] - start[:3, 3]) * u, yaw)
        # Then the solve, outside in: lean, planted feet, hands, gaze.
        inv_root = np.linalg.inv(root)
        for run in self.runs:
            channels = self._owned(run)
            spec = run.spec
            u_raw = run.t / run.over if run.over else 1.0
            u = ease(spec.get("ease", "smooth"), u_raw)
            lean = spec.get("lean")
            if lean and "spine" in channels:
                by = float(lean.get("by", 0)) * _bump(u_raw, float(lean.get("at", 0.5)))
                extra = quat_from_matrix(Transform(rotation=np.radians([by, 0, 0])).to_matrix()[:3, :3])
                rest = s["Spine"].rotation
                pose.rotations["Spine"] = _quat_mul(pose.rotations.get("Spine", IDENTITY),
                                                    quat_from_matrix(rest.T @ matrix_from_quat(extra) @ rest))
            if spec.get("plant") == "feet" and "legs" in channels:
                for foot, world in run.start["feet"].items():
                    local = (inv_root @ np.r_[world, 1.0])[:3]
                    solve_limb(s, pose, foot, local, None, np.eye(3))
            for end, entry in self._ik_entries(run).items():
                channel = self.channel_of.get(end)
                if channel not in channels:
                    continue
                goal = self._ik_target(run, end, entry)
                if goal is None:
                    continue
                start = run.start["ends"][end]
                point = start + (goal - start) * u
                aims[end] = goal
                if run.t >= run.over - self.dt - 1e-9:      # the beat's last composed frame
                    aims[f"{end}@arrived"] = goal
                if "spine" in channels and end in ("LeftHand", "RightHand"):
                    self._bend_to_reach(pose, self._beat_pose(run), root, end, point)
                local = (inv_root @ np.r_[point, 1.0])[:3]
                solve_limb(s, pose, end, local, entry.get("pole"))
        gaze = self.owners.get("gaze")
        if gaze is not None and gaze.spec.get("look") is not None:
            look = gaze.spec["look"]
            ref = look.get("at") if isinstance(look, dict) else look
            target = self._ref(gaze, ref)
            if target is not None:
                point = target.world()[:3, 3] if isinstance(target, Target) else target
                aims["gaze"] = point
                u = ease("smooth", gaze.t / max(gaze.over, 0.25))
                self._turn_head(pose, root, point, u)
        return pose, root, aims

    def _bend_to_reach(self, pose: Pose, beat: Pose, root: NDArray[np.float64], end: str,
                       point: NDArray[np.float64]) -> None:
        """Bend forward at the spine, as little as will do, when the hand can't reach ``point``
        from where the shoulder is (a button low on a TV, a cup at the back of a table). The bend
        is measured from the beat's own pose, so beats that keep reaching bend continuously
        rather than adding up; one that stops reaching eases it out with its pose."""
        s = self.skeleton
        shoulder = end.replace("Hand", "UpperArm")
        rest = s.fk()
        reach = REACH_SLACK * sum(float(np.linalg.norm(rest[a][:3, 3] - rest[b][:3, 3])) for a, b in
                                  ((shoulder, end.replace("Hand", "LowerArm")), (end.replace("Hand", "LowerArm"), end)))
        base = {b: beat.rotations.get(b, IDENTITY) for b in BEND_BONES}
        trial = pose.copy()
        for degrees in range(0, MAX_BEND + 1, 3):
            for bone in BEND_BONES:
                r = s[bone].rotation
                extra = Transform(rotation=np.radians([degrees / len(BEND_BONES), 0, 0])).to_matrix()[:3, :3]
                trial.rotations[bone] = _quat_mul(base[bone], quat_from_matrix(r.T @ extra @ r))
            if np.linalg.norm(point - (root @ s.world(shoulder, trial))[:3, 3]) <= reach:
                break
        for bone in BEND_BONES:
            pose.rotations[bone] = trial.rotations[bone]

    def _beat_pose(self, run: Run) -> Pose:
        spec = run.spec
        name = spec.get("pose")
        base = self.poses.get(name) if name else None
        if name and base is None:
            raise MotionError(f"motion '{run.name}': this body has no pose '{name}' (it has {sorted(self.poses)})")
        pose = (base or self.poses.get(self.posture, self.poses["stand"])).copy()
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
                    if isinstance(entry.get("target"), str) and entry["target"].endswith(".left"):
                        twin_entry["target"] = entry["target"][:-5] + ".right"
                    if "pole" in entry:
                        twin_entry["pole"] = list(np.asarray(entry["pole"], dtype=float) * [-1, 1, 1])
                    entries[twin] = twin_entry
        return entries

    def _ik_target(self, run: Run, end: str, entry: dict) -> NDArray[np.float64] | None:
        target = self._ref(run, entry.get("target"))
        if target is None:
            return None
        if isinstance(target, Target):
            m = target.world()
            offset = np.asarray(entry.get("offset", [0, 0, 0]), dtype=float)
            return (m @ np.r_[offset, 1.0])[:3]
        return target

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
        yaw = max(-YAW_LIMIT, min(YAW_LIMIT, math.atan2(local[0], local[2])))
        pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, -math.atan2(local[1], math.hypot(local[0], local[2]))))
        # What the pose already does to the head's facing is kept; the turn is split neck / head.
        for bone, share in (("Neck", NECK_SHARE), ("Head", 1.0 - NECK_SHARE)):
            r = Transform(rotation=np.array([pitch * share * weight, yaw * share * weight, 0.0])).to_matrix()[:3, :3]
            rest = s[bone].rotation
            pose.rotations[bone] = _quat_mul(pose.rotations.get(bone, IDENTITY),
                                             quat_from_matrix(rest.T @ r @ rest))

    # --- blending ---

    def _inertialize(self, pose: Pose, root: NDArray[np.float64]) -> None:
        """Channels whose owner changed this step start from where the body was: the difference
        decays over the channel's half-life instead of the pose jumping."""
        for channel in self._changed:
            life = HALF_LIFE.get(channel, DEFAULT_HALF_LIFE)
            for bone in self.skeleton.channels[channel]["bones"]:
                old = self.out.rotations.get(bone, IDENTITY)      # as shown, offsets included
                new = pose.rotations.get(bone, IDENTITY)
                self.offsets[bone] = (_quat_mul(old, _quat_inv(new)), 0.0, life)
            if channel == "root":
                self.root_offset = (self.root @ np.linalg.inv(root), 0.0, life)
        self._changed = set()
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
        parent = self.actor.parent.world_transform() if self.actor.parent is not None else np.eye(4)
        self.actor.transform = Transform.from_matrix(np.linalg.inv(parent) @ self.root)
        if self.attached:
            fk = s.fk(self.out)
            for node, end, rel in self.attached.values():
                up = node.parent.world_transform() if node.parent is not None else np.eye(4)
                node.transform = Transform.from_matrix(np.linalg.inv(up) @ self.root @ fk[end] @ rel)

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


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
