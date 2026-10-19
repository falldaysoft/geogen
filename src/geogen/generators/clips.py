"""Procedural skeletal clips for humanoids: walk, idle, wave, and static poses.

Clips are data in the body YAML (``body.clips``), generated against the
character's own skeleton so a tall and a short body both walk without foot
sliding::

    clips:
      walk: { generator: walk, speed: 1.2, arm_swing: 16, hip_sway: 0.02 }
      idle: { generator: idle, duration: 6 }

The walk is an in-place cycle (the root stays put; the runtime moves the
character). Feet follow ground-locked stance paths -- sliding back at exactly
``speed`` -- and swing arcs, placed with two-bone leg IK; the pelvis bobs,
sways and turns, the chest counter-rotates, arms swing opposite the legs and
the head stays level. The stride is chosen from the leg length so the IK can
reach it. The clip's ``meta.speed`` is recorded in extras so the runtime
scales playback to the NPC's actual velocity.

Transitions between poses (``generator: transition``: sit_down, stand_up, lie_down,
get_up) are keyed too, with the root's path between the two placements in ``meta.root``
for the runtime to follow.

Every clip keys every bone (absolute local rotations) plus the Hips
translation, sampled at ``fps`` and exported as glTF LINEAR channels.
"""

from __future__ import annotations

import numpy as np

from ..core import ik
from ..core.skeleton import Pose, Skeleton
from ..core.skin import Clip, Track
from ..core.transform import Transform, matrix_from_quat, quat_from_matrix, quats_from_matrices, slerp

WALK_DEFAULTS = {"speed": 1.2, "duty": 0.6, "knee_drop": 0.03, "bob": 0.026, "hip_sway": 0.02,
                 "pelvis_yaw": 6.0, "pelvis_list": 3.0, "chest_yaw": 8.0, "arm_swing": 16.0,
                 "lift": 0.07, "heel_raise": 34.0, "stance_width": 0.0, "fps": 30}
IDLE_DEFAULTS = {"duration": 6.0, "breathe": 1.5, "sway": 0.012, "glance": 12.0, "fps": 15}
WAVE_DEFAULTS = {"duration": 2.4, "waves": 3, "swing": 18.0, "raise": 15.0, "ease": 0.45, "fps": 20}
TRANSITION_DEFAULTS = {"from": None, "to": None, "duration": 1.4, "approach": [0.0, 0.0, 0.5],
                       "anchor_height": None, "plant": False, "lean": 0.0, "via": None, "via_at": 0.5,
                       "move": [0.0, 1.0], "turn": [0.0, 1.0], "fps": 20}
STANDING = "stand"          # the pose held at the approach point; every other pose is at the anchor


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def tracks_from_poses(skeleton: Skeleton, poses: list[Pose], times: np.ndarray) -> dict[str, Track]:
    """Every bone's absolute local rotation per pose (+ Hips translation) as clip tracks."""
    locals_ = skeleton.posed_locals(poses)                  # (poses, bones, 4, 4)
    rotations = quats_from_matrices(locals_[:, :, :3, :3])
    # Keep consecutive quaternions in one hemisphere so LINEAR (slerp) keys take the short way.
    for i in range(1, len(rotations)):
        flip = np.einsum("bi,bi->b", rotations[i], rotations[i - 1]) < 0
        rotations[i, flip] *= -1
    offset = {bone for p in poses for bone in p.offsets}
    tracks = {}
    for b, bone in enumerate(skeleton.names):
        translations = locals_[:, b, :3, 3].copy() if bone == "Hips" or bone in offset else None
        tracks[bone] = Track(times, rotations=rotations[:, b].copy(), translations=translations)
    return tracks


class _Legs:
    """One leg for the clip generators: IK to an ankle target with an explicit foot rotation."""

    def __init__(self, skeleton: Skeleton, side: str) -> None:
        self.skeleton = skeleton
        self.upper, self.lower = f"{side}UpperLeg", f"{side}LowerLeg"
        self.foot, self.toes = f"{side}Foot", f"{side}Toes"
        s = skeleton
        self.a = float(np.linalg.norm(s[self.lower].head - s[self.upper].head))
        self.b = float(np.linalg.norm(s[self.foot].head - s[self.lower].head))
        self.ankle_rest = s[self.foot].head.copy()
        self.ball = float(np.linalg.norm(s[self.toes].head - s[self.foot].head))

    def solve(self, pose: Pose, ankle: np.ndarray, foot_rotation: np.ndarray, toe_bend: float) -> bool:
        """Set the leg bones in ``pose`` so the ankle reaches ``ankle``; returns whether it did."""
        reached = ik.solve_limb(self.skeleton, pose, self.foot, ankle, end_rotation=foot_rotation)
        pose.rotations[self.toes] = self.skeleton.body_quat(self.toes, _rx(toe_bend))
        return reached


def walk_clip(skeleton: Skeleton, base: Pose, name: str = "walk", **params) -> Clip:
    """An in-place walk cycle at ``speed`` m/s over ``base`` (the standing pose's upper body)."""
    p = {**WALK_DEFAULTS, **params}
    unknown = set(params) - set(WALK_DEFAULTS)
    if unknown:
        raise ValueError(f"walk clip: unknown params {sorted(unknown)}")
    s = skeleton
    k = s.landmarks["head_top"][1] / 1.7
    legs = {side: _Legs(s, side) for side in ("Left", "Right")}
    leg = legs["Left"].a + legs["Left"].b
    hip_rest = s["LeftUpperLeg"].head[1]
    ankle_y = s["LeftFoot"].head[1]
    drop, bob = p["knee_drop"] * k, p["bob"] * k
    # Front reach at heel strike (pelvis lowest) sets the stance length and so the stride.
    reach_h = hip_rest - drop - bob - ankle_y
    front = np.sqrt(max((0.985 * leg) ** 2 - reach_h ** 2, 0.01))
    stance = 2.4 * front                                   # heel raise buys the extra back reach
    duty, speed = p["duty"], p["speed"]
    period = stance / (duty * speed)                       # the ground moves `stance` in `duty` of it
    frames = max(8, int(round(period * p["fps"])))
    times = np.linspace(0.0, period, frames + 1)
    arm_swing, heel_raise = np.radians(p["arm_swing"]), np.radians(p["heel_raise"])

    poses = []
    for t in times:
        phase = (t / period) % 1.0
        pose = base.copy()
        w = 2 * np.pi * phase
        # Pelvis: lowest at heel strikes (phase 0, 0.5), over the stance foot, turning with the legs.
        pose.offsets["Hips"] = np.array([p["hip_sway"] * k * np.cos(w - 2 * np.pi * duty / 2),
                                         -drop - bob * np.cos(2 * w),
                                         0.0])
        yaw = -np.radians(p["pelvis_yaw"]) * np.cos(w)
        roll = np.radians(p["pelvis_list"]) * np.cos(w - 2 * np.pi * duty / 2)
        pose.rotations["Hips"] = s.body_quat("Hips", _ry(yaw) @ _rz(roll))
        chest = np.radians(p["chest_yaw"]) * np.cos(w)
        pose.rotations["Chest"] = s.body_quat("Chest", _ry(-yaw * 0.6 + chest * 0.5))
        pose.rotations["UpperChest"] = s.body_quat("UpperChest", _ry(chest * 0.5) @ _rz(-roll * 0.5))
        pose.rotations["Neck"] = s.body_quat("Neck", _ry(-(chest - yaw * 0.6) * 0.8) @ _rz(roll * 0.3))
        # Arms swing opposite the legs (left arm forward as the right foot strikes, phase 0.5).
        for side, sign in (("Left", 1.0), ("Right", -1.0)):
            swing = sign * arm_swing * np.cos(w)                 # < 0: forward
            down = _rz(-sign * np.radians(74))
            pose.rotations[f"{side}UpperArm"] = s.body_quat(f"{side}UpperArm", _rx(swing) @ down)
            bend = np.radians(12 + 14 * max(0.0, -swing / max(arm_swing, 1e-6)))
            pose.rotations[f"{side}LowerArm"] = s.body_quat(f"{side}LowerArm", _ry(-sign * bend))
        # Legs: left heel strikes at phase 0, right at 0.5.
        for side, offset in (("Left", 0.0), ("Right", 0.5)):
            lp = (phase + offset) % 1.0
            rest = legs[side].ankle_rest
            x = rest[0] - np.sign(rest[0]) * p["stance_width"]
            if lp < duty:                                        # stance: ground-locked, heel then roll
                f = lp / duty
                z = front - stance * f
                pitch = -np.radians(8) * (1 - _smooth(f / 0.15))
                lift_heel = _smooth((f - 0.6) / 0.4)
                pitch += heel_raise * lift_heel
                # Heel raise pivots on the ball of the foot: the ankle rises and moves forward.
                y = ankle_y + legs[side].ball * np.sin(max(pitch, 0.0))
                z += legs[side].ball * (1 - np.cos(max(pitch, 0.0)))
                toe = -max(pitch, 0.0)
            else:                                                # swing: forward arc
                f = (lp - duty) / (1 - duty)
                z = front - stance + stance * _smooth(f)
                pitch = heel_raise * (1 - _smooth(f / 0.35)) - np.radians(8) * _smooth((f - 0.6) / 0.4)
                y = ankle_y + p["lift"] * k * np.sin(np.pi * f) + legs[side].ball * np.sin(max(pitch, 0.0)) * (1 - f)
                z += legs[side].ball * (1 - np.cos(max(pitch, 0.0))) * (1 - f)
                toe = -max(pitch, 0.0) * (1 - f)
            legs[side].solve(pose, np.array([x, y, z]), _ry(yaw * 0.5) @ _rx(pitch), toe)
        poses.append(pose)
    clip = Clip(name, tracks_from_poses(s, poses, times), loop=True)
    clip.meta = {"speed": round(float(speed), 4), "stride": round(float(speed * period), 4)}
    return clip


def idle_clip(skeleton: Skeleton, base: Pose, name: str = "idle", **params) -> Clip:
    """A standing idle loop: breathing, a slow weight shift over planted feet, a glance each way."""
    p = {**IDLE_DEFAULTS, **params}
    unknown = set(params) - set(IDLE_DEFAULTS)
    if unknown:
        raise ValueError(f"idle clip: unknown params {sorted(unknown)}")
    s = skeleton
    k = s.landmarks["head_top"][1] / 1.7
    legs = {side: _Legs(s, side) for side in ("Left", "Right")}
    duration = float(p["duration"])
    frames = max(8, int(round(duration * p["fps"])))
    times = np.linspace(0.0, duration, frames + 1)
    breaths = max(1, round(duration / 4.0))                   # whole breaths per loop
    poses = []
    for t in times:
        u = t / duration
        pose = base.copy()
        shift = np.sin(2 * np.pi * u)
        pose.offsets["Hips"] = np.array([p["sway"] * k * shift, -0.004 * k * abs(shift), 0.0])
        pose.rotations["Hips"] = s.body_quat("Hips", _rz(np.radians(1.5) * shift))
        breath = np.sin(2 * np.pi * breaths * u)
        pose.rotations["UpperChest"] = s.body_quat("UpperChest", _rx(-np.radians(p["breathe"]) * breath)
                                                   @ _rz(-np.radians(1.2) * shift))
        # Glance: look left in the first half, right in the second, eased in and out.
        glance = np.radians(p["glance"]) * (np.sin(np.pi * _smooth(min(u * 2, 1.0))) if u < 0.5
                                            else -np.sin(np.pi * _smooth(u * 2 - 1.0)))
        pose.rotations["Neck"] = s.body_quat("Neck", _ry(glance * 0.4))
        pose.rotations["Head"] = s.body_quat("Head", _ry(glance * 0.6) @ _rz(np.radians(1.0) * shift))
        for side in ("Left", "Right"):
            ankle = legs[side].ankle_rest.copy()
            legs[side].solve(pose, ankle, np.eye(3), 0.0)
        poses.append(pose)
    return Clip(name, tracks_from_poses(s, poses, times), loop=True)


def wave_clip(skeleton: Skeleton, base: Pose, name: str = "wave", **params) -> Clip:
    """A one-shot wave with the right hand: the upper arm lifts out to the side (``raise``
    degrees below horizontal), the forearm stands up and swings ``waves`` times by ``swing``
    degrees, then the arm eases back to ``base``. Not looped."""
    p = {**WAVE_DEFAULTS, **params}
    unknown = set(params) - set(WAVE_DEFAULTS)
    if unknown:
        raise ValueError(f"wave clip: unknown params {sorted(unknown)}")
    s = skeleton
    duration, ease = float(p["duration"]), float(p["ease"])
    frames = max(8, int(round(duration * p["fps"])))
    times = np.linspace(0.0, duration, frames + 1)
    legs = {side: _Legs(s, side) for side in ("Left", "Right")}
    up = {"RightUpperArm": _ry(np.radians(-25)) @ _rz(np.radians(p["raise"])),
          "RightShoulder": _rz(np.radians(-6))}
    poses = []
    for t in times:
        # 0 -> 1 over the ease in, 1 while waving, back to 0 over the ease out.
        w = _smooth(t / ease) * _smooth((duration - t) / ease)
        pose = base.copy()
        for bone, target in up.items():
            q0 = pose.rotations.get(bone, s.body_quat(bone, np.eye(3)))
            pose.rotations[bone] = slerp(q0, s.body_quat(bone, target), w)
        swing = np.radians(p["swing"]) * np.sin(2 * np.pi * p["waves"] * max(t - ease, 0.0)
                                                / max(duration - 2 * ease, 1e-6))
        pose.rotations["RightLowerArm"] = s.body_quat("RightLowerArm", _rz(-np.radians(105) * w + swing * w))
        pose.rotations["RightHand"] = s.body_quat("RightHand", _rz(swing * 0.4 * w))
        pose.rotations["Head"] = s.body_quat("Head", _rz(np.radians(-4) * w))
        for side in ("Left", "Right"):
            legs[side].solve(pose, legs[side].ankle_rest.copy(), np.eye(3), 0.0)
        poses.append(pose)
    return Clip(name, tracks_from_poses(s, poses, times), loop=False)


def _blend(a: Pose, b: Pose, t: float) -> Pose:
    """``a`` to ``b``: rotations slerped per bone, offsets lerped."""
    identity, zero = np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3)
    out = Pose(root=(1 - t) * a.root + t * b.root)
    for bone in {*a.rotations, *b.rotations}:
        out.rotations[bone] = slerp(a.rotations.get(bone, identity), b.rotations.get(bone, identity), t)
    for bone in {*a.offsets, *b.offsets}:
        out.offsets[bone] = (1 - t) * a.offsets.get(bone, zero) + t * b.offsets.get(bone, zero)
    return out


def _window(u: float, span) -> float:
    return _smooth((u - float(span[0])) / max(float(span[1]) - float(span[0]), 1e-6))


def root_at(root: dict, t: float) -> np.ndarray:
    """A transition clip's ``meta.root`` sampled at ``t``: the body's 4x4 transform in the frame
    of where the character ends up (the anchor; the approach point when it ends standing)."""
    times = np.asarray(root["times"])
    t = float(np.clip(t, times[0], times[-1]))
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
    f = (t - times[i]) / max(times[i + 1] - times[i], 1e-9)
    m = np.eye(4)
    m[:3, :3] = matrix_from_quat(slerp(root["rotations"][i], root["rotations"][i + 1], f))
    m[:3, 3] = (1 - f) * np.asarray(root["positions"][i]) + f * np.asarray(root["positions"][i + 1])
    return m


def transition_clip(skeleton: Skeleton, base: Pose, name: str = "transition", *, poses: dict | None = None,
                    roots: dict | None = None, **params) -> Clip:
    """A one-shot move from pose ``from`` to pose ``to`` (sitting down, standing up, lying down).

    The body's root moves too: standing, it is on the floor at ``approach`` (anchor frame: x
    right, z forward) ``anchor_height`` below the anchor; in any other pose it is where the
    asset's root ``poses:`` put it. The hips travel between the two (over the ``move`` share of
    the clip) while the body turns about them (``turn``), the skeleton blends from one pose to
    the other (through ``via`` at ``via_at``, if given) and the torso leans ``lean`` degrees
    forward mid-way. ``plant`` keeps both feet where they stand, by leg IK, sliding only as
    far as the two poses' foot positions differ; it also sets ``anchor_height`` from them.

    ``meta`` records ``from``, ``to`` and ``root`` (``times``, ``positions``, ``rotations``
    xyzw): the root's path in the frame of where the character ends up, which a runtime
    replays while the clip plays (see ``root_at``)."""
    p = {**TRANSITION_DEFAULTS, **params}
    unknown = set(params) - set(TRANSITION_DEFAULTS)
    if unknown:
        raise ValueError(f"transition clip '{name}': unknown params {sorted(unknown)}")
    poses, roots = poses or {}, roots or {}
    for key in ("from", "to", "via"):
        if p[key] is not None and p[key] not in poses:
            raise ValueError(f"transition clip '{name}': unknown pose '{p[key]}'. Poses: {sorted(poses)}")
    if p["from"] is None or p["to"] is None:
        raise ValueError(f"transition clip '{name}' needs 'from' and 'to' poses")
    s = skeleton
    ends = [Pose.from_spec(poses[p[key]], s) for key in ("from", "to")]
    via = Pose.from_spec(poses[p["via"]], s) if p["via"] is not None else None
    fks = [s.fk(pose) for pose in ends]
    feet = ("LeftFoot", "RightFoot") if p["plant"] else ()

    def placed(pose_name: str) -> np.ndarray:
        spec = roots.get(pose_name) or {}
        return Transform(translation=np.asarray(spec.get("offset", [0, 0, 0]), dtype=np.float64),
                         rotation=np.radians(np.asarray(spec.get("rotation", [0, 0, 0]), dtype=np.float64))).to_matrix()

    height = p["anchor_height"]
    if height is None:
        seated = [i for i, key in enumerate(("from", "to")) if p[key] != STANDING]
        if not feet or len(seated) != 1:
            raise ValueError(f"transition clip '{name}' needs 'anchor_height' (or 'plant' feet to take it from)")
        i = seated[0]
        there = placed(p[("from", "to")[i]])
        height = -float(np.mean([(there @ fks[i][f])[1, 3] - fks[1 - i][f][1, 3] for f in feet]))
    standing = np.eye(4)
    standing[:3, 3] = np.asarray(p["approach"], dtype=np.float64) - [0.0, height, 0.0]
    places = [standing if p[key] == STANDING else placed(p[key]) for key in ("from", "to")]
    quats = [quat_from_matrix(m[:3, :3]) for m in places]
    hips = [(m @ fk["Hips"])[:3, 3] for m, fk in zip(places, fks)]
    ankles = {f: [(m @ fk[f])[:3, 3] for m, fk in zip(places, fks)] for f in feet}
    legs = {f: _Legs(s, f.removesuffix("Foot")) for f in feet}

    duration = float(p["duration"])
    frames = max(8, int(round(duration * p["fps"])))
    times = np.linspace(0.0, duration, frames + 1)
    keyed, positions, rotations = [], [], []
    for t in times:
        u = t / duration
        if via is None:
            pose = _blend(ends[0], ends[1], _smooth(u))
        elif u < p["via_at"]:
            pose = _blend(ends[0], via, _smooth(u / p["via_at"]))
        else:
            pose = _blend(via, ends[1], _smooth((u - p["via_at"]) / (1 - p["via_at"])))
        lean = np.radians(p["lean"]) * np.sin(np.pi * u) / 2
        for bone in ("Spine", "Chest"):
            # A body-frame rotation on top of the blended one.
            q = pose.rotations.get(bone, np.array([0.0, 0.0, 0.0, 1.0]))
            pose.rotations[bone] = quat_from_matrix(matrix_from_quat(s.body_quat(bone, _rx(lean))) @ matrix_from_quat(q))
        q = slerp(quats[0], quats[1], _window(u, p["turn"]))
        turn = matrix_from_quat(q)
        g = _window(u, p["move"])
        position = (1 - g) * hips[0] + g * hips[1] - turn @ s.fk(pose)["Hips"][:3, 3]
        for foot, (a, b) in ankles.items():
            e = _smooth(u)
            legs[foot].solve(pose, turn.T @ ((1 - e) * a + e * b - position), turn.T, 0.0)
        keyed.append(pose)
        positions.append(position)
        rotations.append(q if not rotations or np.dot(q, rotations[-1]) >= 0 else -q)
    clip = Clip(name, tracks_from_poses(s, keyed, times), loop=False)
    end = places[1][:3, 3] if p["to"] == STANDING else np.zeros(3)     # where the character's node ends up
    clip.meta = {"from": p["from"], "to": p["to"],
                 "root": {"times": [round(float(t), 4) for t in times],
                          "positions": [[round(float(v), 4) for v in pos - end] for pos in positions],
                          "rotations": [[round(float(v), 5) for v in q] for q in rotations]}}
    return clip


GESTURE_DEFAULTS = {"pose": None, "duration": 2.0, "hands": {}, "bones": {}, "fps": 15}
GESTURE_HAND_KEYS = {"lift", "path", "cycles", "drift"}
GESTURE_BONE_KEYS = {"path", "cycles"}
GESTURE_SAMPLES = 4             # frames of a gesture clip the build's arm and drape fitting look at


def loop_path(keys, u: float, cycles: int = 1) -> np.ndarray:
    """A closed path through ``keys`` (evenly spaced points, any dimension) at ``u`` (0..1 of the
    loop), gone round ``cycles`` times: periodic Catmull-Rom, so it is smooth and loops seamlessly.
    No keys: zeros; one key: that point."""
    keys = np.asarray(keys, dtype=float)
    if keys.size == 0:
        return np.zeros(3)
    n = len(keys)
    if n == 1:
        return keys[0].copy()
    x = (u * cycles % 1.0) * n
    i = int(np.floor(x)) % n
    f = x - np.floor(x)
    p0, p1, p2, p3 = keys[(i - 1) % n], keys[i], keys[(i + 1) % n], keys[(i + 2) % n]
    return 0.5 * (2 * p1 + (p2 - p0) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f
                  + (3 * p1 - p0 - 3 * p2 + p3) * f ** 3)


def gesture_clip(skeleton: Skeleton, base: Pose, name: str = "gesture", *, poses: dict | None = None,
                 **params) -> Clip:
    """A looping piece of work over one of the body's IK poses (``pose``: ``work_counter``, standing
    at a worktop), all of it declared:

    - ``hands: {End: {lift, path, cycles, drift}}``: the IK end's target moves from the pose's
      by ``lift`` [x, y, z] m plus a closed ``path`` of [x, y, z] offsets (m; body frame: x the
      character's left, y up, z forward) gone round ``cycles`` times a loop, plus a ``drift`` path
      gone round once (a knife working across the board).
    - ``bones: {Bone: {path, cycles}}``: [x, y, z] degree offsets (body frame) on top of the pose:
      a nod, breathing.

    Paths are evenly spaced keys joined smoothly (``loop_path``). The pose's resting hands that
    ``hands`` doesn't move stay resting (the clip's ``rests``: a runtime puts them on the real top).
    """
    p = {**GESTURE_DEFAULTS, **params}
    unknown = set(params) - set(GESTURE_DEFAULTS)
    if unknown:
        raise ValueError(f"gesture clip '{name}': unknown params {sorted(unknown)}")
    spec = (poses or {}).get(p["pose"])
    if spec is None or not (spec.get("ik") or {}):
        raise ValueError(f"gesture clip '{name}': pose {p['pose']!r} is not one of the body's IK poses")
    ik_spec = {k: v for k, v in spec["ik"].items() if k != "mirror"}
    if spec["ik"].get("mirror"):
        from ..core.skeleton import mirror_name

        for end, entry in list(ik_spec.items()):
            twin = mirror_name(end)
            if twin and twin not in ik_spec:
                ik_spec[twin] = {**entry, "target": [-float(entry["target"][0]), *entry["target"][1:]]}
                if "pole" in entry:
                    ik_spec[twin]["pole"] = [-float(entry["pole"][0]), *entry["pole"][1:]]
    for end, h in (p["hands"] or {}).items():
        if end not in ik_spec or not isinstance(h, dict) or set(h) - GESTURE_HAND_KEYS:
            raise ValueError(f"gesture clip '{name}': hands.{end} is {{{', '.join(sorted(GESTURE_HAND_KEYS))}}} "
                             f"on one of the pose's IK ends {sorted(ik_spec)}")
    for bone, b in (p["bones"] or {}).items():
        if bone not in skeleton.bones or not isinstance(b, dict) or set(b) - GESTURE_BONE_KEYS:
            raise ValueError(f"gesture clip '{name}': bones.{bone} is {{path, cycles}} on a skeleton bone")
    hands = p["hands"] or {}
    duration = float(p["duration"])
    frames = max(8, int(round(duration * p["fps"])))
    times = np.linspace(0.0, duration, frames + 1)
    # The pose once; each frame re-solves only the hands that move (the rest of the pose holds).
    held = Pose.from_spec({**spec, "ik": {k: v for k, v in ik_spec.items() if k not in hands}}, skeleton)
    out = []
    for t in times:
        u = t / duration
        pose = held.copy()
        for end, h in hands.items():
            entry = ik_spec[end]
            offset = (np.asarray(h.get("lift", [0, 0, 0]), dtype=float)
                      + loop_path(h.get("path") or [], u, int(h.get("cycles", 1)))
                      + loop_path(h.get("drift") or [], u))
            ik.solve_limb(skeleton, pose, end, np.asarray(entry["target"], dtype=float) + offset, entry.get("pole"),
                          np.eye(3) if entry.get("flat") else None)
        for bone, b in (p["bones"] or {}).items():
            degrees = np.radians(loop_path(b.get("path") or [], u, int(b.get("cycles", 1))))
            extra = skeleton.body_quat(bone, Transform(rotation=degrees).to_matrix()[:3, :3])
            q = pose.rotations.get(bone, np.array([0.0, 0.0, 0.0, 1.0]))
            pose.rotations[bone] = quat_from_matrix(matrix_from_quat(q) @ matrix_from_quat(extra))
        out.append(pose)
    rests = sorted(end for end in Pose.rest_ends(spec) if end not in hands)
    # Only hands move, in front of the body, about one pose: the build's fitting needs few frames.
    return Clip(name, tracks_from_poses(skeleton, out, times), loop=True, meta={"rests": rests} if rests else {},
                samples=GESTURE_SAMPLES)


GENERATORS = {"walk": walk_clip, "idle": idle_clip, "wave": wave_clip, "transition": transition_clip,
              "gesture": gesture_clip}


def generate_clip(skeleton: Skeleton, base: Pose, name: str, spec: dict, poses: dict | None = None,
                  roots: dict | None = None) -> Clip:
    """``poses`` are the body's skeletal pose specs and ``roots`` its root placements per pose
    (the asset's top-level ``poses:``); transitions need both."""
    spec = dict(spec)
    kind = spec.pop("generator", name)
    if kind not in GENERATORS:
        raise ValueError(f"clip '{name}': unknown generator '{kind}'. Known: {sorted(GENERATORS)}")
    if kind == "transition":
        return transition_clip(skeleton, base, name=name, poses=poses, roots=roots, **spec)
    if kind == "gesture":
        return gesture_clip(skeleton, base, name=name, poses=poses, **spec)
    return GENERATORS[kind](skeleton, base, name=name, **spec)
