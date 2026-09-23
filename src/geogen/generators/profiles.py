"""Profile-based generators: extrude a 2D shape, or revolve (lathe) a profile.

Both produce closed, consistently wound meshes with metric UVs, and use a
crease angle to decide which edges shade smoothly (fillets, curved
profiles) and which stay hard (real corners).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..core import meshops
from ..core.mesh import Mesh
from ..core.profile import Shape, arc_lengths, dedupe, triangulate
from .base import MeshGenerator
from .primitives import CubeGenerator

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint
    from ..layout.surfaces import Surface

# (u, v, axis) world vectors for mapping profile (x, y) + extrusion depth t.
# Each frame is right-handed (u x v == axis) so CCW profiles stay CCW.
_AXIS_FRAMES = {
    "z": (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
    "y": (np.array([1.0, 0, 0]), np.array([0, 0, -1.0]), np.array([0, 1.0, 0])),
    "x": (np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])),
}


def _left_offset(loop: NDArray, distance: float) -> NDArray:
    """Move every vertex ``distance`` to the left of the loop direction (mitred).

    Material lies to the left of CCW outer loops and CW holes, so a positive
    distance always insets into the material. Vertex count is preserved,
    which lets bevel rings be stitched together directly.
    """
    if distance == 0:
        return loop.copy()
    prev_pts, next_pts = np.roll(loop, 1, axis=0), np.roll(loop, -1, axis=0)
    e_in = loop - prev_pts
    e_out = next_pts - loop
    e_in /= np.maximum(np.linalg.norm(e_in, axis=1, keepdims=True), 1e-12)
    e_out /= np.maximum(np.linalg.norm(e_out, axis=1, keepdims=True), 1e-12)
    n_in = np.column_stack([-e_in[:, 1], e_in[:, 0]])
    n_out = np.column_stack([-e_out[:, 1], e_out[:, 0]])
    miter = n_in + n_out
    miter /= np.maximum(np.linalg.norm(miter, axis=1, keepdims=True), 1e-12)
    cos_half = np.clip(np.einsum("ij,ij->i", miter, n_in), 0.25, 1.0)  # cap spikes at 4x
    return loop + miter * (distance / cos_half)[:, None]


def _levels(depth: float, bevel: float, segments: int) -> list[tuple[float, float]]:
    """(t, inset) pairs from the bottom cap edge to the top cap edge."""
    h = depth / 2
    if bevel <= 0:
        return [(-h, 0.0), (h, 0.0)]
    phis = np.linspace(0, np.pi / 2, segments + 1)
    bottom = [(-h + bevel * (1 - np.cos(p)), bevel * (1 - np.sin(p))) for p in phis]
    top = [(h - bevel * (1 - np.cos(p)), bevel * (1 - np.sin(p))) for p in phis[::-1]]
    return bottom + top


@dataclass
class ExtrudeGenerator(MeshGenerator):
    """Extrude a 2D :class:`Shape` (with holes) along an axis, centred on the origin.

    Attributes:
        shape: Profile region in metres
        depth: Extrusion length along ``axis``
        axis: "y" (profile lies in XZ, e.g. a tabletop), "z" (profile in XY,
            e.g. a wall panel) or "x"
        bevel: Radius of the rounded edge where caps meet the sides
        bevel_segments: Arc segments for the bevel (1 = chamfer)
        crease_angle: Edges sharper than this (degrees) shade hard
        caps: Whether to close the ends
    """

    shape: Shape = field(default_factory=lambda: Shape(np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])))
    depth: float = 1.0
    axis: str = "y"
    bevel: float = 0.0
    bevel_segments: int = 3
    crease_angle: float = 40.0
    caps: bool = True

    def _size(self) -> np.ndarray:
        u, v, a = _AXIS_FRAMES[self.axis]
        (x0, y0), (x1, y1) = self.shape.bounds
        return np.abs(u) * (x1 - x0) + np.abs(v) * (y1 - y0) + np.abs(a) * self.depth

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        return CubeGenerator().get_attachment_points(size)

    def get_surfaces(self, size: np.ndarray) -> dict[str, Surface]:
        return CubeGenerator().get_surfaces(size)

    def generate(self) -> Mesh:
        u_axis, v_axis, axis = _AXIS_FRAMES[self.axis]
        (x0, y0), (x1, y1) = self.shape.bounds
        max_bevel = 0.45 * min(self.depth, x1 - x0, y1 - y0)
        bevel = float(np.clip(self.bevel, 0.0, max_bevel))
        levels = _levels(self.depth, bevel, max(1, int(self.bevel_segments)))

        def to3d(p2: NDArray, t: float | NDArray) -> NDArray:
            t = np.broadcast_to(np.asarray(t, dtype=np.float64), (len(p2),))
            return p2[:, :1] * u_axis + p2[:, 1:2] * v_axis + t[:, None] * axis

        verts, uvs, faces = [], [], []
        count = 0

        # Side walls: one ring per level for every loop, plus a seam column for UVs.
        for loop in self.shape.loops:
            n = len(loop)
            rings = [_left_offset(loop, inset) for _, inset in levels]
            v_coord = 0.0
            prev_ring = None
            for (t, _), ring in zip(levels, rings):
                closed_ring = np.vstack([ring, ring[:1]])
                if prev_ring is not None:
                    v_coord += float(np.mean(np.linalg.norm(
                        np.column_stack([ring - prev_ring[0], np.full(n, t - prev_ring[1])]), axis=1)))
                verts.append(to3d(closed_ring, t))
                uvs.append(np.column_stack([arc_lengths(ring, closed=True), np.full(n + 1, v_coord)]))
                prev_ring = (ring, t)
            stride = n + 1
            for li in range(len(levels) - 1):
                base = count + li * stride
                i = np.arange(n)
                a, b = base + i, base + i + 1
                c, d = b + stride, a + stride
                faces += [np.column_stack([a, b, c]), np.column_stack([a, c, d])]
            count += stride * len(levels)

        # Caps: triangulate the inset outline at each end.
        if self.caps:
            for (t, inset), flip in ((levels[0], True), (levels[-1], False)):
                loops = [_left_offset(lp, inset) for lp in self.shape.loops]
                points, tris = triangulate(loops[0], loops[1:])
                verts.append(to3d(points, t))
                uvs.append(points - points.min(axis=0))
                tris = tris[:, [0, 2, 1]] if flip else tris
                faces.append(tris + count)
                count += len(points)

        mesh = Mesh(np.vstack(verts), np.vstack(faces).astype(np.int64), uvs=np.vstack(uvs))
        return meshops.compute_normals(mesh, self.crease_angle)


@dataclass
class LatheGenerator(MeshGenerator):
    """Revolve an open (radius, height) profile around the Y axis.

    Profile points with radius 0 close the surface at the axis (e.g. the
    top of a vase lid or the tip of a finial). Order the profile bottom to
    top on the *outside* of the object so faces point outward.

    Attributes:
        profile: (N, 2) array of (r, y) points in metres
        segments: Number of steps around the axis
        sweep: Revolution angle in degrees (360 = closed)
        cap_bottom / cap_top: Close open profile ends with a flat disc
        crease_angle: Profile corners sharper than this shade hard
    """

    profile: NDArray = field(default_factory=lambda: np.array([[0.5, -0.5], [0.5, 0.5]]))
    segments: int = 32
    sweep: float = 360.0
    cap_bottom: bool = True
    cap_top: bool = True
    crease_angle: float = 40.0

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        return CubeGenerator().get_attachment_points(size)

    def generate(self) -> Mesh:
        prof = dedupe(np.asarray(self.profile, dtype=np.float64), closed=False)
        prof[:, 0] = np.maximum(prof[:, 0], 0.0)
        full = abs(self.sweep - 360.0) < 1e-6
        n_seg = max(3, int(self.segments))
        theta = np.radians(np.linspace(0.0, self.sweep, n_seg + 1))  # seam duplicated for UVs

        r, y = prof[:, 0], prof[:, 1]
        # Grid (profile index i, angle index j); angle 0 at +Z, increasing toward +X.
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        verts = np.stack([r[:, None] * sin_t, np.broadcast_to(y[:, None], (len(r), len(theta))),
                          r[:, None] * cos_t], axis=-1).reshape(-1, 3)
        v_coord = arc_lengths(prof)
        uvs = np.stack([r[:, None] * theta, np.broadcast_to(v_coord[:, None], (len(r), len(theta)))],
                       axis=-1).reshape(-1, 2)

        cols = len(theta)
        i, j = np.meshgrid(np.arange(len(r) - 1), np.arange(n_seg), indexing="ij")
        a = (i * cols + j).ravel()
        b = a + 1
        c = a + cols + 1
        d = a + cols
        faces = [np.column_stack([a, b, c]), np.column_stack([a, c, d])]

        verts_list, uv_list = [verts], [uvs]
        count = len(verts)
        for idx, is_top in ((0, False), (len(r) - 1, True)):
            want = self.cap_top if is_top else self.cap_bottom
            if not want or r[idx] < 1e-9 or not full:
                continue
            ring = np.column_stack([r[idx] * sin_t[:-1], np.full(n_seg, y[idx]), r[idx] * cos_t[:-1]])
            center = np.array([[0.0, y[idx], 0.0]])
            verts_list.append(np.vstack([center, ring]))
            uv_list.append(np.vstack([[0.0, 0.0], ring[:, [0, 2]]]) + r[idx])
            k = np.arange(n_seg)
            tri = np.column_stack([np.zeros(n_seg, dtype=np.int64), 1 + k, 1 + (k + 1) % n_seg]) + count
            faces.append(tri if is_top else tri[:, [0, 2, 1]])
            count += n_seg + 1

        mesh = Mesh(np.vstack(verts_list), np.vstack(faces).astype(np.int64), uvs=np.vstack(uv_list))
        # Faces touching the axis collapse to lines; drop them before shading.
        _, area = meshops.face_normals(mesh.vertices, mesh.faces)
        mesh = Mesh(mesh.vertices, mesh.faces[area > 1e-14], uvs=mesh.uvs)
        return meshops.compute_normals(mesh, self.crease_angle)
