"""Inverse kinematics helpers: analytic two-bone IK and hinge-preserving bone aiming."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _unit(v: NDArray) -> NDArray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def two_bone(root: NDArray, upper: float, lower: float, target: NDArray, pole: NDArray
             ) -> tuple[NDArray, NDArray, bool]:
    """Place the middle joint of a two-bone limb reaching from ``root`` toward ``target``.

    The limb bends toward ``pole`` (a direction). Returns (middle joint, end point,
    reached); an out-of-reach target leaves the limb almost straight along the line
    to it (``reached`` False) and a too-close one folds it as far as the lengths allow.
    """
    delta = np.asarray(target, dtype=np.float64) - root
    dist = float(np.linalg.norm(delta))
    u = _unit(delta) if dist > 1e-9 else np.array([0.0, -1.0, 0.0])
    lo, hi = abs(upper - lower) + 1e-6, upper + lower - 1e-6
    reach = min(max(dist, lo), hi)
    cos_a = np.clip((upper ** 2 + reach ** 2 - lower ** 2) / (2 * upper * reach), -1.0, 1.0)
    side = np.asarray(pole, dtype=np.float64) - np.dot(pole, u) * u
    if np.linalg.norm(side) < 1e-9:           # pole along the limb: pick any perpendicular
        side = np.cross(u, [1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.cross(u, [0.0, 0.0, 1.0])
    side = _unit(side)
    middle = root + upper * (cos_a * u + np.sqrt(1 - cos_a ** 2) * side)
    return middle, root + reach * u, bool(abs(reach - dist) < 1e-4)


def aim(rest_rotation: NDArray, rest_dir: NDArray, rest_normal: NDArray,
        direction: NDArray, normal: NDArray) -> NDArray:
    """World rotation for a bone whose rest ``rest_rotation`` points ``rest_dir`` with hinge axis
    ``rest_normal``, turned to point ``direction`` with hinge axis ``normal`` (roll kept)."""
    def frame(d, n):
        d = _unit(np.asarray(d, dtype=np.float64))
        n = _unit(np.asarray(n, dtype=np.float64) - np.dot(n, d) * d)
        return np.column_stack([d, n, np.cross(d, n)])

    return frame(direction, normal) @ frame(rest_dir, rest_normal).T @ rest_rotation
