"""Sculpted hands: a signed-distance hand along the skeleton's hand bones, skinned finger by finger.

The left hand is a smooth union, in the rest pose, of a rounded palm (wrist to knuckles), four
fingers (tapered capsules along each finger's proximal, intermediate and distal bones to a tip),
a thumb along its three bones with the pad at its root, and the wrist running back into the
forearm. Fingers blend into each other only a little, so they read as fingers rather than a
mitten. It is polygonised like the head (``generators/head.py``) and each vertex is weighted to
the bones it lies nearest, blending across the knuckles, so a clip that curls the fingers curls
these. The right hand is its mirror image (``ringloft.mirror_mesh``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.mesh import Mesh
from ..core.skeleton import Skeleton
from .head import _capsule, _ellipsoid, _smin, build_head, sdf_normals

FINGERS = ("Index", "Middle", "Ring", "Little")
FINGER_RADIUS = {"Index": 0.0108, "Middle": 0.011, "Ring": 0.0105, "Little": 0.0092}   # at the knuckle, hand size 1
TIP_LENGTH = {"Index": 0.02, "Middle": 0.021, "Ring": 0.02, "Little": 0.016}
BLEND = 0.006                 # how far (m) across a knuckle the weights hand over


def _box(q: NDArray, half: NDArray, round_: float) -> NDArray:
    d = np.abs(q) - half
    return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(d.max(axis=-1), 0.0) - round_


class HandShape:
    """The signed-distance left hand for one skeleton and hand size (rest pose, body frame)."""

    def __init__(self, skeleton: Skeleton, size: float = 1.0, wrist: tuple[float, float] = (0.02, 0.028)) -> None:
        self.s = float(size)
        self.joints = {name: np.asarray(skeleton[name].head, dtype=np.float64) for name in skeleton.names
                       if name.startswith("Left") and any(k in name for k in ("Hand", "Thumb", *FINGERS, "LowerArm"))}
        j = self.joints
        self.wrist = j["LeftHand"]
        # Hand frame: along the hand, across it toward the thumb, and the back of the hand.
        along = j["LeftMiddleProximal"] - self.wrist
        self.along = along / np.linalg.norm(along)
        across = j["LeftIndexProximal"] - j["LeftRingProximal"]
        across -= across.dot(self.along) * self.along
        self.across = across / np.linalg.norm(across)
        self.back = np.cross(self.across, self.along)
        if self.back[1] < 0:                                          # the back of a hand faces up at rest
            self.back = -self.back
        self.wrist_radii = wrist
        # Bone segments for the shape and the weights: name -> (start, end).
        self.segments: dict[str, tuple[NDArray, NDArray]] = {}
        for finger in FINGERS:
            names = [f"Left{finger}{part}" for part in ("Proximal", "Intermediate", "Distal")]
            heads = [j[n] for n in names]
            direction = (heads[2] - heads[1]) / np.linalg.norm(heads[2] - heads[1])
            ends = heads[1:] + [heads[2] + direction * TIP_LENGTH[finger] * self.s]
            for n, a, b in zip(names, heads, ends):
                self.segments[n] = (a, b)
        thumb = [f"LeftThumb{part}" for part in ("Metacarpal", "Proximal", "Distal")]
        heads = [j[n] for n in thumb]
        direction = (heads[2] - heads[1]) / np.linalg.norm(heads[2] - heads[1])
        for n, a, b in zip(thumb, heads, heads[1:] + [heads[2] + direction * 0.022 * self.s]):
            self.segments[n] = (a, b)
        self.segments["LeftHand"] = (self.wrist, j["LeftMiddleProximal"])
        forearm = j["LeftLowerArm"]
        self.segments["LeftLowerArm"] = (self.wrist + (forearm - self.wrist) * 0.25, self.wrist)

    def _local(self, p: NDArray) -> NDArray:
        q = p - self.wrist
        return np.stack([q @ self.along, q @ self.back, q @ self.across], axis=-1)

    def __call__(self, points: NDArray) -> NDArray:
        points = np.asarray(points, dtype=np.float64)
        p = points.reshape(-1, 3)
        s = self.s
        q = self._local(p)
        knuckles = (self.joints["LeftMiddleProximal"] - self.wrist) @ self.along
        # Palm: a rounded slab from the wrist to the knuckles, a little thicker on the thumb side.
        span = (self.joints["LeftIndexProximal"] - self.joints["LeftLittleProximal"]) @ self.across
        centre = np.array([knuckles * 0.52, 0.0, (self.joints["LeftIndexProximal"] - self.wrist) @ self.across - span / 2])
        palm = _box(q - centre, np.array([knuckles * 0.42, 0.006 * s, span / 2 + 0.002 * s]), 0.009 * s)
        d = palm
        # Fingers: tapered capsules along their bones, barely blended into each other.
        fingers = None
        for finger in FINGERS:
            r0 = FINGER_RADIUS[finger] * s
            f = None
            for k, part in enumerate(("Proximal", "Intermediate", "Distal")):
                a, b = self.segments[f"Left{finger}{part}"]
                seg = _capsule(p, a, b, r0 * (1 - 0.08 * k), r0 * (1 - 0.08 * (k + 1)))
                f = seg if f is None else np.minimum(f, seg)
            fingers = f if fingers is None else _smin(fingers, f, 0.002 * s)
        d = _smin(d, fingers, 0.008 * s)
        # Thumb, and the pad at its root.
        thumb = None
        for k, part in enumerate(("Metacarpal", "Proximal", "Distal")):
            a, b = self.segments[f"LeftThumb{part}"]
            seg = _capsule(p, a, b, (0.013 - 0.0015 * k) * s, (0.0115 - 0.0015 * k) * s)
            thumb = seg if thumb is None else np.minimum(thumb, seg)
        d = _smin(d, thumb, 0.006 * s)
        pad_at = self.wrist + 0.45 * (self.joints["LeftThumbProximal"] - self.wrist)
        pad = _ellipsoid(q, self._local(pad_at[None])[0] + np.array([0, -0.004 * s, 0]),
                         np.array([0.024, 0.012, 0.014]) * s)
        d = _smin(d, pad, 0.01 * s)
        # The wrist, flattened like the forearm, running back into the arm.
        a, b = self.segments["LeftLowerArm"]
        ry, rz = self.wrist_radii
        wq = q.copy()
        wq[:, 1] *= rz / ry
        wa, wb = self._local(a[None])[0], self._local(b[None])[0] + np.array([0.012 * s, 0, 0])
        wrist = _capsule(wq, wa, wb, rz * 0.98, rz * 0.95)
        d = _smin(d, wrist, 0.012 * s)
        return d.reshape(points.shape[:-1])

    def bounds(self) -> tuple[NDArray, NDArray]:
        pts = np.array([x for seg in self.segments.values() for x in seg])
        return pts.min(axis=0) - 0.035 * self.s, pts.max(axis=0) + 0.035 * self.s

    def weights(self, vertices: NDArray, names: list[str]) -> tuple[NDArray, NDArray]:
        """Each vertex to the bones it lies nearest, blending over ``BLEND`` (at most 4 influences)."""
        from ..core.skin import normalize_weights

        bones = list(self.segments)
        dist = np.stack([_capsule(vertices, a, b, 0.0, 0.0) for a, b in self.segments.values()], axis=1)
        # The palm's own segment is the hand's: anything nearer it than a finger root is palm.
        near = dist.min(axis=1, keepdims=True)
        w = np.exp(-(dist - near) / BLEND)
        w[w < 0.02] = 0.0
        dense = np.zeros((len(vertices), len(names)))
        for k, bone in enumerate(bones):
            dense[:, names.index(bone)] += w[:, k]
        return normalize_weights(np.tile(np.arange(len(names)), (len(vertices), 1)), dense)


def build_hand(skeleton: Skeleton, size: float = 1.0, wrist: tuple[float, float] = (0.02, 0.028)) -> tuple[Mesh, HandShape]:
    """The sculpted left hand (rest pose, body frame), skinned, with the SDF's normals."""
    shape = HandShape(skeleton, size, wrist)
    lo, hi = shape.bounds()
    mesh = build_head(shape, np.zeros(3), step=0.003, tolerance=0.0012, bounds=(lo, hi))
    mesh.normals = sdf_normals(shape, mesh.vertices)
    mesh.joints, mesh.weights = shape.weights(mesh.vertices, skeleton.names)
    return mesh, shape
