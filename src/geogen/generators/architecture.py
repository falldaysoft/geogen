"""Architectural generators: roofs and gable prisms.

Roofs are sized by the footprint they cover (the top of the walls) plus a
rise; overhang extends beyond that footprint. All pieces are closed solids
with metric UVs laid out in each face's plane, so shingle rows run parallel
to the eaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull

from ..core import csg, meshops, uvmap
from ..core.mesh import Mesh
from ..core.profile import Shape
from .base import MeshGenerator
from .primitives import CubeGenerator, CylinderGenerator
from .profiles import ExtrudeGenerator

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint
    from ..layout.surfaces import Surface

ROOF_STYLES = ("gable", "hip", "shed", "flat")


def _hull_mesh(points: np.ndarray) -> Mesh:
    """Closed convex hull with outward-facing triangles."""
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    center = points.mean(axis=0)
    v = points[faces]
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    flip = np.einsum("ij,ij->i", normals, v.mean(axis=1) - center) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return Mesh(points.astype(np.float64), faces.astype(np.int64))


def _finish(mesh: Mesh) -> Mesh:
    """Flat-shade (roof planes meet at hard edges) and lay out face-planar UVs."""
    mesh = meshops.compute_normals(mesh, crease_angle=20.0)
    return uvmap.face_planar_project(mesh)


@dataclass
class RoofGenerator(MeshGenerator):
    """Pitched or flat roof over a rectangular footprint, centred on the origin.

    The generated mesh's y=0 is the wall-top plane (``-height/2`` in the
    part frame so the part's bounding box is ``width x height x depth``).

    Attributes:
        width, depth: Footprint covered (the wall top), metres
        height: Rise from wall top to ridge (ignored for flat)
        style: gable | hip | shed | flat
        overhang: Eave overhang beyond the footprint on every side
        thickness: Roof slab thickness measured perpendicular to the slope
        ridge_axis: "x" (ridge runs along the width), "z", or "auto" (the
            longer side; for shed roofs the slope runs across the ridge axis)
        ridge_cap: Add a rounded cap along the ridge (gable/hip)
    """

    width: float = 8.0
    height: float = 2.0
    depth: float = 6.0
    style: str = "gable"
    overhang: float = 0.35
    thickness: float = 0.12
    ridge_axis: str = "auto"
    ridge_cap: bool = True

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        return CubeGenerator().get_attachment_points(size)

    def get_surfaces(self, size: np.ndarray) -> dict[str, Surface]:
        return CubeGenerator().get_surfaces(size)

    def generate(self) -> Mesh:
        if self.style not in ROOF_STYLES:
            raise ValueError(f"Unknown roof style '{self.style}'. Use one of {ROOF_STYLES}")
        axis = self.ridge_axis
        if axis == "auto":
            axis = "x" if self.width >= self.depth else "z"
        # Build with the ridge along X, then rotate if requested.
        w, d = (self.width, self.depth) if axis == "x" else (self.depth, self.width)
        builder = {"gable": self._gable, "hip": self._hip, "shed": self._shed, "flat": self._flat}[self.style]
        mesh = builder(w, d)
        # Part frame is centred: wall-top plane at -height/2.
        mesh.vertices[:, 1] -= self.height / 2 if self.style != "flat" else 0.0
        if axis == "z":
            rot = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.0]])
            mesh = mesh.transform(rot)
        return _finish(mesh)

    # -- styles -----------------------------------------------------------

    def _slope(self, run: float) -> tuple[float, float]:
        """(tan of pitch, vertical slab thickness) for a slope rising ``height`` over ``run``."""
        tan_a = self.height / max(run, 1e-6)
        return tan_a, self.thickness * np.sqrt(1 + tan_a**2)

    def _panel(self, run: float, length: float, tan_a: float, tv: float, flip: bool) -> Mesh:
        """One sloped slab from the ridge line (z=0) down over the eave on +z (or -z)."""
        oh = self.overhang
        eave_y = -oh * tan_a
        # Profile in the YZ plane as (-z, y) because ExtrudeGenerator's x-axis frame maps u -> -Z.
        z_pts = np.array([0.0, run + oh, run + oh, 0.0])
        y_pts = np.array([self.height, eave_y, eave_y + tv, self.height + tv])
        if flip:
            z_pts = -z_pts
        shape = Shape(np.column_stack([-z_pts, y_pts]))
        return ExtrudeGenerator(shape=shape, depth=length, axis="x", crease_angle=20.0).generate()

    def _ridge_cap(self, length: float, y: float) -> Mesh:
        r = self.thickness * 0.9
        cap = CylinderGenerator(radius=r, height=length, segments=16).generate()
        rot = np.array([[0, 1, 0, 0], [-1, 0, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.0]])  # Y axis -> X axis
        return cap.transform(rot)

    def _gable(self, w: float, d: float) -> Mesh:
        half = d / 2
        tan_a, tv = self._slope(half)
        length = w + 2 * self.overhang
        parts = [self._panel(half, length, tan_a, tv, False), self._panel(half, length, tan_a, tv, True)]
        if self.ridge_cap:
            parts.append(self._ridge_cap(length, self.height + tv * 0.85))
        return csg.union(*parts, crease_angle=20.0)

    def _shed(self, w: float, d: float) -> Mesh:
        # Single slope rising from the front (+z) eave to the back (-z) wall.
        tan_a, tv = self._slope(d)
        panel = self._panel(d, w + 2 * self.overhang, tan_a, tv, False)
        # Extend the high end past the back wall by the overhang, along the slope.
        v = panel.vertices
        high = v[:, 2] > -1e-9
        high &= np.isclose(v[:, 2], 0.0)
        v[high, 2] -= self.overhang
        v[high, 1] += self.overhang * tan_a
        v[:, 2] -= d / 2
        return panel

    def _hip(self, w: float, d: float) -> Mesh:
        oh = self.overhang
        half = d / 2
        tan_a, tv = self._slope(half)
        ex, ez = w / 2 + oh, d / 2 + oh
        eave_y = -oh * tan_a
        top_y = self.height + tv
        ridge_half = max(w / 2 - half, 0.0)
        pts = [[sx * ex, y, sz * ez] for sx in (-1, 1) for sz in (-1, 1) for y in (eave_y, eave_y + tv)]
        if ridge_half > 1e-6:
            pts += [[-ridge_half, top_y, 0.0], [ridge_half, top_y, 0.0]]
        else:
            pts += [[0.0, top_y, 0.0]]
        mesh = _hull_mesh(np.array(pts))
        if self.ridge_cap and ridge_half > 1e-6:
            mesh = csg.union(mesh, self._ridge_cap(2 * ridge_half, top_y - tv * 0.15), crease_angle=20.0)
        return mesh

    def _flat(self, w: float, d: float) -> Mesh:
        oh = self.overhang
        slab = CubeGenerator(w + 2 * oh, self.thickness, d + 2 * oh, bevel=min(0.02, self.thickness / 4)).generate()
        slab.vertices[:, 1] += self.thickness / 2 - self.height / 2
        return slab


@dataclass
class PrismGenerator(MeshGenerator):
    """Triangular prism filling a box: apex along the top, ridge along X.

    Used for gable-end walls under a pitched roof, and for ramps/wedges
    (``apex: back`` puts the high edge at the back instead of the centre).
    """

    width: float = 1.0
    height: float = 1.0
    depth: float = 1.0
    apex: str = "center"  # center | back | front

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        return CubeGenerator().get_attachment_points(size)

    def get_surfaces(self, size: np.ndarray) -> dict[str, Surface]:
        return CubeGenerator().get_surfaces(size)

    def generate(self) -> Mesh:
        hd, hh = self.depth / 2, self.height / 2
        apex_z = {"center": 0.0, "back": -hd, "front": hd}[self.apex]
        # Profile as (-z, y) for the x-axis extrusion frame.
        tri = np.array([[-hd, -hh], [hd, -hh], [-apex_z, hh]])
        mesh = ExtrudeGenerator(shape=Shape(tri), depth=self.width, axis="x", crease_angle=20.0).generate()
        return uvmap.box_project(mesh, origin=np.array([-self.width / 2, -hh, -hd]))
