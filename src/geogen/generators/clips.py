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

Every clip keys every bone (absolute local rotations) plus the Hips
translation, sampled at ``fps`` and exported as glTF LINEAR channels.
"""

from __future__ import annotations

import numpy as np

from ..core import ik
from ..core.skeleton import Pose, Skeleton
from ..core.skin import Clip, Track
from ..core.transform import quat_from_matrix

WALK_DEFAULTS = {"speed": 1.2, "duty": 0.6, "knee_drop": 0.03, "bob": 0.026, "hip_sway": 0.02,
                 "pelvis_yaw": 6.0, "pelvis_list": 3.0, "chest_yaw": 8.0, "arm_swing": 16.0,
                 "lift": 0.07, "heel_raise": 34.0, "stance_width": 0.0, "fps": 30}
IDLE_DEFAULTS = {"duration": 6.0, "breathe": 1.5, "sway": 0.012, "glance": 12.0, "fps": 15}
WAVE_DEFAULTS = {"duration": 2.4, "waves": 3, "swing": 18.0, "raise": 15.0, "ease": 0.45, "fps": 20}


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
    tracks = {}
    for bone in skeleton.names:
        locals_ = [skeleton.posed_local(bone, p) for p in poses]
        rotations = np.array([quat_from_matrix(m[:3, :3]) for m in locals_])
        # Keep consecutive quaternions in one hemisphere so LINEAR (slerp) keys take the short way.
        for i in range(1, len(rotations)):
            if np.dot(rotations[i], rotations[i - 1]) < 0:
                rotations[i] = -rotations[i]
        translations = None
        if bone == "Hips" or any(bone in p.offsets for p in poses):
            translations = np.array([m[:3, 3] for m in locals_])
        tracks[bone] = Track(times, rotations=rotations, translations=translations)
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
            from ..core.transform import slerp

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


GENERATORS = {"walk": walk_clip, "idle": idle_clip, "wave": wave_clip}


def generate_clip(skeleton: Skeleton, base: Pose, name: str, spec: dict) -> Clip:
    spec = dict(spec)
    kind = spec.pop("generator", name)
    if kind not in GENERATORS:
        raise ValueError(f"clip '{name}': unknown generator '{kind}'. Known: {sorted(GENERATORS)}")
    return GENERATORS[kind](skeleton, base, name=name, **spec)
