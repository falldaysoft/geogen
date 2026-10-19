"""Inverse kinematics helpers: analytic two-bone IK and hinge-preserving bone aiming."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _norm(v: NDArray) -> float:
    x, y, z = np.asarray(v, dtype=np.float64).tolist()
    return math.sqrt(x * x + y * y + z * z)


def _unit(v: NDArray) -> NDArray:
    return v / max(_norm(v), 1e-12)


def _cross(a: NDArray, b: NDArray) -> NDArray:           # (np.cross is several times slower on 3-vectors)
    ax, ay, az = np.asarray(a, dtype=np.float64).tolist()
    bx, by, bz = np.asarray(b, dtype=np.float64).tolist()
    return np.array([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])


def two_bone(root: NDArray, upper: float, lower: float, target: NDArray, pole: NDArray
             ) -> tuple[NDArray, NDArray, bool]:
    """Place the middle joint of a two-bone limb reaching from ``root`` toward ``target``.

    The limb bends toward ``pole`` (a direction). Returns (middle joint, end point,
    reached); an out-of-reach target leaves the limb almost straight along the line
    to it (``reached`` False) and a too-close one folds it as far as the lengths allow.
    """
    delta = np.asarray(target, dtype=np.float64) - root
    dist = _norm(delta)
    u = _unit(delta) if dist > 1e-9 else np.array([0.0, -1.0, 0.0])
    lo, hi = abs(upper - lower) + 1e-6, upper + lower - 1e-6
    reach = min(max(dist, lo), hi)
    cos_a = min(max((upper ** 2 + reach ** 2 - lower ** 2) / (2 * upper * reach), -1.0), 1.0)
    pole = np.asarray(pole, dtype=np.float64)
    side = pole - float(pole @ u) * u
    if _norm(side) < 1e-9:           # pole along the limb: pick any perpendicular
        side = _cross(u, np.array([1.0, 0.0, 0.0])) if abs(u[0]) < 0.9 else _cross(u, np.array([0.0, 0.0, 1.0]))
    side = _unit(side)
    middle = root + upper * (cos_a * u + math.sqrt(1 - cos_a ** 2) * side)
    return middle, root + reach * u, bool(abs(reach - dist) < 1e-4)


def aim(rest_rotation: NDArray, rest_dir: NDArray, rest_normal: NDArray,
        direction: NDArray, normal: NDArray) -> NDArray:
    """World rotation for a bone whose rest ``rest_rotation`` points ``rest_dir`` with hinge axis
    ``rest_normal``, turned to point ``direction`` with hinge axis ``normal`` (roll kept)."""
    def frame(d, n):
        d = _unit(np.asarray(d, dtype=np.float64))
        n = np.asarray(n, dtype=np.float64)
        n = _unit(n - float(n @ d) * d)
        return np.column_stack([d, n, _cross(d, n)])

    return frame(direction, normal) @ frame(rest_dir, rest_normal).T @ rest_rotation


# Two-bone limbs by end bone: (upper, lower, rest direction, rest pole = where the middle joint
# bends toward in the T-pose rest).
LIMBS = {
    "LeftFoot": ("LeftUpperLeg", "LeftLowerLeg", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "RightFoot": ("RightUpperLeg", "RightLowerLeg", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "LeftHand": ("LeftUpperArm", "LeftLowerArm", (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "RightHand": ("RightUpperArm", "RightLowerArm", (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
}


def solve_limb(skeleton, pose, end: str, target, pole=None, end_rotation=None) -> bool:
    """Set a limb's two bones in ``pose`` (a ``core.skeleton.Pose``) so bone ``end``'s head reaches
    ``target`` (body frame), bending toward ``pole`` (default: the limb's rest pole). With
    ``end_rotation`` (3x3, body frame) the end bone gets that world rotation times its rest.
    Returns whether the target was reached.
    """
    upper, lower, rest_dir, rest_pole = LIMBS[end]
    pole = np.asarray(rest_pole if pole is None else pole, dtype=np.float64)
    rest_dir = np.asarray(rest_dir)
    rest_normal = _cross(np.asarray(rest_pole), rest_dir)
    s = skeleton
    a = _norm(s[lower].head - s[upper].head)
    b = _norm(s[end].head - s[lower].head)
    # Only the limb's chain is posed: its parent once, then each bone under the one before.
    parent = s.world(s[upper].parent, pose)
    root = (parent @ s.posed_local(upper, pose))[:3, 3]
    middle, tip, reached = two_bone(root, a, b, np.asarray(target, dtype=np.float64), pole)
    normal = _cross(pole, tip - root)
    up = aim(s[upper].rotation, rest_dir, rest_normal, middle - root, normal)
    pose.rotations[upper] = s.pose_quat(upper, parent, up)
    upper_world = parent @ s.posed_local(upper, pose)
    low = aim(s[lower].rotation, rest_dir, rest_normal, tip - middle, normal)
    pose.rotations[lower] = s.pose_quat(lower, upper_world, low)
    if end_rotation is not None:
        lower_world = upper_world @ s.posed_local(lower, pose)
        pose.rotations[end] = s.pose_quat(end, lower_world, np.asarray(end_rotation) @ s[end].rotation)
    return reached
