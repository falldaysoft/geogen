"""Sweep a 2D profile along a 3D path (skirting, cornices, rails, pipes, curbs).

Frames are rotation-minimising (double-reflection method), started from an
``up`` hint so a horizontal path keeps the profile's y axis vertical. At
polyline corners the ring sits on the bisector plane and is stretched by
1/cos(half angle) across the bend, so mitred corners keep the profile's
thickness. Open paths get flat caps; closed paths distribute any leftover
frame twist so the seam matches.

Profile coordinates: x runs to the *right* of the direction of travel (seen
from above for a horizontal path with ``up = +Y``), y along ``up``. A
counter-clockwise profile gives outward-facing faces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core import meshops
from ..core.mesh import Mesh
from ..core.profile import Shape, arc_lengths, triangulate
from .base import MeshGenerator


def _unit(v: NDArray) -> NDArray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


@dataclass
class SweepGenerator(MeshGenerator):
    """Sweep ``profile`` along ``path``.

    Attributes:
        profile: Cross-section (outer loop used), metres
        path: (N, 3) points in metres
        closed: Join the last point back to the first (no caps)
        up: Hint for the profile's y axis at the start
        twist: Total twist along the path, degrees
        scale: Profile scale at the start and end (linear in between)
        crease_angle: Edges sharper than this shade hard
        center: Centre the mesh on its bounds (like other primitives); False
            keeps path coordinates as given
    """

    profile: Shape = field(default_factory=lambda: Shape(np.array([[-0.05, 0], [0.05, 0], [0.05, 0.1], [-0.05, 0.1]])))
    path: NDArray = field(default_factory=lambda: np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    closed: bool = False
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    twist: float = 0.0
    scale: tuple[float, float] = (1.0, 1.0)
    crease_angle: float = 40.0
    center: bool = True

    def _points(self) -> NDArray:
        pts = np.asarray(self.path, dtype=np.float64)
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
        pts = pts[keep]
        if self.closed and len(pts) > 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
            pts = pts[:-1]
        if len(pts) < 2:
            raise ValueError("sweep path needs at least two distinct points")
        return pts

    def _frames(self, pts: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Per-vertex (tangent, side, normal, miter stretch direction * factor)."""
        n = len(pts)
        if self.closed:
            d_in = _unit(pts - np.roll(pts, 1, axis=0))
            d_out = _unit(np.roll(pts, -1, axis=0) - pts)
        else:
            seg = _unit(np.diff(pts, axis=0))
            d_in = np.vstack([seg[:1], seg])
            d_out = np.vstack([seg, seg[-1:]])
        tangent = _unit(d_in + d_out)
        # Degenerate U-turns: fall back to the outgoing direction.
        bad = np.linalg.norm(d_in + d_out, axis=1) < 1e-6
        tangent[bad] = d_out[bad]

        up = np.asarray(self.up, dtype=np.float64)
        side0 = np.cross(tangent[0], up)  # right of travel
        if np.linalg.norm(side0) < 1e-6:
            side0 = np.cross(tangent[0], [1.0, 0.0, 0.0] if abs(tangent[0][0]) < 0.9 else [0.0, 0.0, 1.0])
        sides = np.zeros((n, 3))
        sides[0] = _unit(side0)
        # Double-reflection rotation-minimising frames.
        for i in range(n - 1):
            v1 = pts[i + 1] - pts[i]
            c1 = v1 @ v1
            r_l = sides[i] - (2 / c1) * (v1 @ sides[i]) * v1
            t_l = tangent[i] - (2 / c1) * (v1 @ tangent[i]) * v1
            v2 = tangent[i + 1] - t_l
            c2 = v2 @ v2
            sides[i + 1] = r_l - (2 / c2) * (v2 @ r_l) * v2 if c2 > 1e-12 else r_l
        sides = _unit(sides - (np.sum(sides * tangent, axis=1, keepdims=True)) * tangent)

        twist = np.radians(self.twist) * np.linspace(0.0, 1.0, n)
        if self.closed:
            # Carry the frame once more around to the start and remove the mismatch.
            v1 = pts[0] - pts[-1]
            c1 = v1 @ v1
            r_l = sides[-1] - (2 / c1) * (v1 @ sides[-1]) * v1
            t_l = tangent[-1] - (2 / c1) * (v1 @ tangent[-1]) * v1
            v2 = tangent[0] - t_l
            r_end = r_l - (2 / (v2 @ v2)) * (v2 @ r_l) * v2 if v2 @ v2 > 1e-12 else r_l
            mismatch = np.arctan2(np.cross(r_end, sides[0]) @ tangent[0], r_end @ sides[0])
            twist = twist + mismatch * np.arange(n) / n
        normals = np.cross(sides, tangent)  # profile y: "up" for a level path
        c, s = np.cos(twist)[:, None], np.sin(twist)[:, None]
        sides, normals = c * sides + s * normals, -s * sides + c * normals

        # Miter: stretch across the bend so the profile keeps its thickness.
        bend = d_out - d_in
        cos_half = np.clip(np.sum(tangent * d_out, axis=1), 0.2, 1.0)
        bend = bend - np.sum(bend * tangent, axis=1, keepdims=True) * tangent
        miter = _unit(bend) * (1.0 / cos_half - 1.0)[:, None]
        return tangent, sides, normals, miter

    def generate(self) -> Mesh:
        pts = self._points()
        n = len(pts)
        _, sides, normals, miter = self._frames(pts)
        loop = self.profile.outer
        m = len(loop)
        closed_loop = np.vstack([loop, loop[:1]])      # seam column for UVs
        v_coord = arc_lengths(loop, closed=True)
        scales = np.linspace(self.scale[0], self.scale[1], n)

        ring_pts = np.vstack([pts, pts[:1]]) if self.closed else pts
        rings = n + 1 if self.closed else n
        u_coord = arc_lengths(ring_pts, closed=False)

        verts, uvs = [], []
        for k in range(rings):
            i = k % n
            local = closed_loop * scales[i]
            offsets = local[:, :1] * sides[i] + local[:, 1:2] * normals[i]
            offsets = offsets + (offsets @ _unit(miter[i]))[:, None] * miter[i] if np.any(miter[i]) else offsets
            verts.append(pts[i] + offsets)
            uvs.append(np.column_stack([np.full(m + 1, u_coord[k]), v_coord]))
        verts = np.vstack(verts)
        uvs = np.vstack(uvs)

        stride = m + 1
        faces = []
        for k in range(rings - 1):
            base = k * stride
            j = np.arange(m)
            a, b = base + j, base + j + 1
            c_, d = b + stride, a + stride
            faces += [np.column_stack([a, c_, b]), np.column_stack([a, d, c_])]
        faces = [np.vstack(faces)]
        count = len(verts)

        if not self.closed:
            cap_pts, tris = triangulate(loop)
            extra_v, extra_uv = [], []
            for end, flip in ((0, False), (n - 1, True)):
                local = cap_pts * scales[end]
                ring = pts[end] + local[:, :1] * sides[end] + local[:, 1:2] * normals[end]
                extra_v.append(ring)
                extra_uv.append(local - local.min(axis=0))
                faces.append((tris[:, [0, 2, 1]] if flip else tris) + count)
                count += len(ring)
            verts = np.vstack([verts, *extra_v])
            uvs = np.vstack([uvs, *extra_uv])

        mesh = Mesh(verts, np.vstack(faces).astype(np.int64), uvs=uvs)
        if self.center:
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            move = np.eye(4)
            move[:3, 3] = -(lo + hi) / 2
            mesh = mesh.transform(move)
        return meshops.compute_normals(mesh, self.crease_angle)
