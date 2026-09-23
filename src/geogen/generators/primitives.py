"""Primitive geometry generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..core.mesh import Mesh
from ..core.geometry import make_cap_faces, make_tube_faces, make_cone_side_faces
from .base import MeshGenerator

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint
    from ..layout.surfaces import Surface


@dataclass
class CubeGenerator(MeshGenerator):
    """Generates a cube/box mesh with optional edge beveling.

    Attributes:
        size_x: Width of the cube (X axis)
        size_y: Height of the cube (Y axis)
        size_z: Depth of the cube (Z axis)
        bevel: Edge rounding radius (0 = sharp edges, default 0.02)
        bevel_segments: Arc segments per 45 degrees of each rounded edge
    """

    size_x: float = 1.0
    size_y: float = 1.0
    size_z: float = 1.0
    bevel: float = 0.02
    bevel_segments: int = 2

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        """Generate attachment points at the center of each face."""
        from ..layout.attachments import AttachmentPoint

        half_x = size[0] / 2
        half_y = size[1] / 2
        half_z = size[2] / 2

        return {
            "top": AttachmentPoint(
                name="top", anchor="center",
                offset=np.array([0, half_y, 0]), facing="north",
            ),
            "bottom": AttachmentPoint(
                name="bottom", anchor="center",
                offset=np.array([0, -half_y, 0]), facing="north",
            ),
            "left": AttachmentPoint(
                name="left", anchor="center",
                offset=np.array([-half_x, 0, 0]), facing="west",
            ),
            "right": AttachmentPoint(
                name="right", anchor="center",
                offset=np.array([half_x, 0, 0]), facing="east",
            ),
            "front": AttachmentPoint(
                name="front", anchor="center",
                offset=np.array([0, 0, half_z]), facing="south",
            ),
            "back": AttachmentPoint(
                name="back", anchor="center",
                offset=np.array([0, 0, -half_z]), facing="north",
            ),
        }

    def get_surfaces(self, size: np.ndarray) -> dict[str, "Surface"]:
        """Expose the six faces of the cube as surfaces.

        For each wall-like face, u is horizontal (left → right when viewed
        from outside) and v is vertical (+Y is up). Normals point outward.
        For top/bottom, u follows +X and v follows the remaining horizontal.
        """
        from ..layout.surfaces import Surface

        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

        return {
            # Front (+Z). Viewed from +Z, right is +X.
            "front": Surface(
                name="front",
                origin=np.array([-hx, -hy, hz]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                v_axis=np.array([0.0, 1.0, 0.0]),
                normal=np.array([0.0, 0.0, 1.0]),
                u_extent=sx, v_extent=sy,
            ),
            # Back (-Z). Viewed from -Z, right is -X.
            "back": Surface(
                name="back",
                origin=np.array([hx, -hy, -hz]),
                u_axis=np.array([-1.0, 0.0, 0.0]),
                v_axis=np.array([0.0, 1.0, 0.0]),
                normal=np.array([0.0, 0.0, -1.0]),
                u_extent=sx, v_extent=sy,
            ),
            # Left (-X). Viewed from -X, right is -Z.
            "left": Surface(
                name="left",
                origin=np.array([-hx, -hy, hz]),
                u_axis=np.array([0.0, 0.0, -1.0]),
                v_axis=np.array([0.0, 1.0, 0.0]),
                normal=np.array([-1.0, 0.0, 0.0]),
                u_extent=sz, v_extent=sy,
            ),
            # Right (+X). Viewed from +X, right is +Z.
            "right": Surface(
                name="right",
                origin=np.array([hx, -hy, -hz]),
                u_axis=np.array([0.0, 0.0, 1.0]),
                v_axis=np.array([0.0, 1.0, 0.0]),
                normal=np.array([1.0, 0.0, 0.0]),
                u_extent=sz, v_extent=sy,
            ),
            # Top (+Y). u along +X, v along +Z (back → front).
            "top": Surface(
                name="top",
                origin=np.array([-hx, hy, -hz]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                v_axis=np.array([0.0, 0.0, 1.0]),
                normal=np.array([0.0, 1.0, 0.0]),
                u_extent=sx, v_extent=sz,
            ),
            # Bottom (-Y). u along +X, v along +Z.
            "bottom": Surface(
                name="bottom",
                origin=np.array([-hx, -hy, -hz]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                v_axis=np.array([0.0, 0.0, 1.0]),
                normal=np.array([0.0, -1.0, 0.0]),
                u_extent=sx, v_extent=sz,
            ),
        }

    def generate(self) -> Mesh:
        """Generate a box centred at the origin with rounded (beveled) edges.

        Each face is a grid projected onto an inner box of half-extents
        ``h - bevel``: every vertex is ``inner + bevel * dir`` where ``dir`` is
        the unit vector from the nearest inner-box point. Flat regions keep
        the face normal; edge and corner bands become circular arcs with
        analytic normals, so shading is smooth and seam-free.
        """
        half = np.array([self.size_x, self.size_y, self.size_z], dtype=np.float64) / 2
        # Clamp bevel to half the smallest dimension
        r = min(self.bevel, float(half.min()) * 0.5)
        if r <= 1e-4:
            r = 0.0
        return _rounded_box(half, r, max(1, int(self.bevel_segments)))


# (normal, u axis, v axis) per face, with u x v == normal so quads wind CCW.
_BOX_FACES = (
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),     # front (+Z)
    ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),   # back (-Z)
    ((1, 0, 0), (0, 0, -1), (0, 1, 0)),    # right (+X)
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),    # left (-X)
    ((0, 1, 0), (1, 0, 0), (0, 0, -1)),    # top (+Y)
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),    # bottom (-Y)
)


def _rounded_box_coords(h: float, r: float, segments: int) -> np.ndarray:
    """Face-plane sample positions along one axis of a rounded box face.

    A face covers 45 degrees of each adjoining edge arc; sampling at
    ``tan(theta)`` gives evenly spaced angles once projected onto the arc.
    """
    inner = h - r
    if r <= 0.0:
        return np.array([-h, h])
    theta = np.linspace(0.0, np.pi / 4, segments + 1)[1:]
    band = inner + r * np.tan(theta)
    coords = np.concatenate([-band[::-1], [-inner, inner], band])
    return np.unique(np.round(coords, 12))


def _arc_length(coord: np.ndarray, h: float, r: float) -> np.ndarray:
    """Metric UV coordinate for a face-plane coordinate, following the edge arc."""
    inner = h - r
    mag = np.abs(coord)
    over = np.maximum(mag - inner, 0.0)
    band = r * np.arctan(over / r) if r > 0 else over
    return np.sign(coord) * (np.minimum(mag, inner) + band) + h


def _rounded_box(half: np.ndarray, r: float, segments: int) -> Mesh:
    inner = half - r
    vertices, normals, uvs, faces = [], [], [], []
    offset = 0
    for n, u, v in _BOX_FACES:
        n, u, v = (np.array(a, dtype=np.float64) for a in (n, u, v))
        hn = float(np.abs(n) @ half)
        hu = float(np.abs(u) @ half)
        hv = float(np.abs(v) @ half)
        us = _rounded_box_coords(hu, r, segments)
        vs = _rounded_box_coords(hv, r, segments)
        a, b = np.meshgrid(us, vs, indexing="xy")  # (len(vs), len(us))
        p = n * hn + a.reshape(-1, 1) * u + b.reshape(-1, 1) * v
        core = np.clip(p, -inner, inner)
        d = p - core
        length = np.linalg.norm(d, axis=1, keepdims=True)
        dirs = np.where(length > 1e-12, d / np.maximum(length, 1e-12), n)
        pos = core + dirs * r if r > 0 else p

        vertices.append(pos)
        normals.append(dirs)
        uvs.append(np.column_stack([_arc_length(a.reshape(-1), hu, r), _arc_length(b.reshape(-1), hv, r)]))

        nu, nv = len(us), len(vs)
        idx = np.arange(nu * nv).reshape(nv, nu) + offset
        q00, q10 = idx[:-1, :-1].reshape(-1), idx[:-1, 1:].reshape(-1)
        q01, q11 = idx[1:, :-1].reshape(-1), idx[1:, 1:].reshape(-1)
        faces.append(np.column_stack([q00, q10, q11]))
        faces.append(np.column_stack([q00, q11, q01]))
        offset += nu * nv

    return Mesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces).astype(np.int64),
        normals=np.vstack(normals),
        uvs=np.vstack(uvs),
    )


@dataclass
class SphereGenerator(MeshGenerator):
    """Generates a UV sphere mesh.

    Attributes:
        radius: Radius of the sphere
        segments: Number of horizontal segments (longitude)
        rings: Number of vertical rings (latitude)
    """

    radius: float = 0.5
    segments: int = 32
    rings: int = 16

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        """Generate attachment points at poles and equator."""
        from ..layout.attachments import AttachmentPoint

        radius_x = size[0] / 2
        radius_y = size[1] / 2
        radius_z = size[2] / 2

        return {
            "top": AttachmentPoint(
                name="top", anchor="center",
                offset=np.array([0, radius_y, 0]), facing="north",
            ),
            "bottom": AttachmentPoint(
                name="bottom", anchor="center",
                offset=np.array([0, -radius_y, 0]), facing="north",
            ),
            "left": AttachmentPoint(
                name="left", anchor="center",
                offset=np.array([-radius_x, 0, 0]), facing="west",
            ),
            "right": AttachmentPoint(
                name="right", anchor="center",
                offset=np.array([radius_x, 0, 0]), facing="east",
            ),
            "front": AttachmentPoint(
                name="front", anchor="center",
                offset=np.array([0, 0, radius_z]), facing="south",
            ),
            "back": AttachmentPoint(
                name="back", anchor="center",
                offset=np.array([0, 0, -radius_z]), facing="north",
            ),
        }

    def generate(self) -> Mesh:
        """Generate a UV sphere mesh centered at the origin with UV coordinates."""
        vertices = []
        uvs = []
        faces = []

        # Top pole - need multiple vertices for different U values at seam
        for seg in range(self.segments):
            vertices.append([0.0, self.radius, 0.0])
            u = (seg + 0.5) / self.segments  # Center of each segment
            uvs.append([u, 1.0])

        # Middle rings
        for ring in range(1, self.rings):
            phi = np.pi * ring / self.rings
            v = 1.0 - ring / self.rings
            y = self.radius * np.cos(phi)
            ring_radius = self.radius * np.sin(phi)

            for seg in range(self.segments + 1):  # +1 for seam vertex
                u = seg / self.segments
                theta = 2 * np.pi * seg / self.segments
                x = ring_radius * np.cos(theta)
                z = ring_radius * np.sin(theta)
                vertices.append([x, y, z])
                uvs.append([u, v])

        # Bottom pole - need multiple vertices for different U values at seam
        for seg in range(self.segments):
            vertices.append([0.0, -self.radius, 0.0])
            u = (seg + 0.5) / self.segments
            uvs.append([u, 0.0])

        vertices = np.array(vertices, dtype=np.float64)
        uvs = np.array(uvs, dtype=np.float64)

        # Top cap triangles (normals point outward/upward)
        first_ring_start = self.segments  # After pole vertices
        for seg in range(self.segments):
            pole_idx = seg
            ring_idx = first_ring_start + seg
            ring_next = first_ring_start + seg + 1
            faces.append([pole_idx, ring_next, ring_idx])

        # Middle quads (as triangles, normals point outward)
        for ring in range(self.rings - 2):
            ring_start = self.segments + ring * (self.segments + 1)
            next_ring_start = ring_start + (self.segments + 1)

            for seg in range(self.segments):
                tl = ring_start + seg
                tr = ring_start + seg + 1
                bl = next_ring_start + seg
                br = next_ring_start + seg + 1
                faces.append([tl, br, bl])
                faces.append([tl, tr, br])

        # Bottom cap triangles (normals point outward/downward)
        last_ring_start = self.segments + (self.rings - 2) * (self.segments + 1)
        bottom_pole_start = last_ring_start + (self.segments + 1)
        for seg in range(self.segments):
            ring_idx = last_ring_start + seg
            ring_next = last_ring_start + seg + 1
            pole_idx = bottom_pole_start + seg
            faces.append([ring_idx, ring_next, pole_idx])

        faces = np.array(faces, dtype=np.int64)
        uvs *= [2 * np.pi * self.radius, np.pi * self.radius]  # metres

        return Mesh(vertices=vertices, faces=faces, uvs=uvs)


@dataclass
class EllipsoidGenerator(SphereGenerator):
    """A sphere stretched to fill an arbitrary (x, y, z) box.

    Unlike ``sphere`` (which uses the smallest size component), every axis
    is honoured. Normals are left for the crease-angle pass so they stay
    correct after the non-uniform stretch.
    """

    size_x: float = 1.0
    size_y: float = 1.0
    size_z: float = 1.0

    def generate(self) -> Mesh:
        self.radius = 0.5
        unit = super().generate()
        scale = np.array([self.size_x, self.size_y, self.size_z])
        # Rescale metric UVs by the mean equatorial / vertical stretch.
        uvs = unit.uvs * [(self.size_x + self.size_z) / 2, self.size_y]
        return Mesh(unit.vertices * scale, unit.faces, uvs=uvs)


@dataclass
class CylinderGenerator(MeshGenerator):
    """Generates a cylinder mesh.

    Attributes:
        radius: Radius of the cylinder
        height: Height of the cylinder
        segments: Number of segments around the circumference
    """

    radius: float = 0.5
    height: float = 1.0
    segments: int = 32

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        """Generate attachment points at top, bottom, and radial mid-height."""
        from ..layout.attachments import AttachmentPoint

        radius = min(size[0], size[2]) / 2
        half_y = size[1] / 2

        return {
            "top": AttachmentPoint(
                name="top", anchor="center",
                offset=np.array([0, half_y, 0]), facing="north",
            ),
            "bottom": AttachmentPoint(
                name="bottom", anchor="center",
                offset=np.array([0, -half_y, 0]), facing="north",
            ),
            "left": AttachmentPoint(
                name="left", anchor="center",
                offset=np.array([-radius, 0, 0]), facing="west",
            ),
            "right": AttachmentPoint(
                name="right", anchor="center",
                offset=np.array([radius, 0, 0]), facing="east",
            ),
            "front": AttachmentPoint(
                name="front", anchor="center",
                offset=np.array([0, 0, radius]), facing="south",
            ),
            "back": AttachmentPoint(
                name="back", anchor="center",
                offset=np.array([0, 0, -radius]), facing="north",
            ),
        }

    def generate(self) -> Mesh:
        """Generate a cylinder mesh centered at the origin with UV coordinates."""
        vertices = []
        uvs = []
        faces = []

        half_height = self.height / 2
        circumference = 2 * np.pi * self.radius

        # === Top cap vertices ===
        # Center vertex
        top_center = len(vertices)
        vertices.append([0.0, half_height, 0.0])
        uvs.append([self.radius, self.radius])

        # Top cap ring (for cap faces)
        top_cap_ring_start = len(vertices)
        for seg in range(self.segments):
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, half_height, z])
            # Radial UV for cap
            u = self.radius * (1 + np.cos(theta))
            v = self.radius * (1 + np.sin(theta))
            uvs.append([u, v])

        # === Side vertices (separate for different UVs) ===
        # Top ring for sides
        side_top_start = len(vertices)
        for seg in range(self.segments + 1):  # +1 for seam
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, half_height, z])
            uvs.append([seg / self.segments * circumference, self.height])

        # Bottom ring for sides
        side_bottom_start = len(vertices)
        for seg in range(self.segments + 1):  # +1 for seam
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, -half_height, z])
            uvs.append([seg / self.segments * circumference, 0.0])

        # === Bottom cap vertices ===
        # Bottom cap ring
        bottom_cap_ring_start = len(vertices)
        for seg in range(self.segments):
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, -half_height, z])
            # Radial UV for cap (flipped for bottom view)
            u = self.radius * (1 + np.cos(theta))
            v = self.radius * (1 - np.sin(theta))
            uvs.append([u, v])

        # Center vertex
        bottom_center = len(vertices)
        vertices.append([0.0, -half_height, 0.0])
        uvs.append([self.radius, self.radius])

        vertices = np.array(vertices, dtype=np.float64)
        uvs = np.array(uvs, dtype=np.float64)

        # Top cap faces (CCW when viewed from +Y means normal points +Y)
        for seg in range(self.segments):
            next_seg = (seg + 1) % self.segments
            # Winding: center -> next_seg -> seg (reversed to get outward normal)
            faces.append([top_center, top_cap_ring_start + next_seg, top_cap_ring_start + seg])

        # Side faces (normals point radially outward)
        for seg in range(self.segments):
            tl = side_top_start + seg
            tr = side_top_start + seg + 1
            bl = side_bottom_start + seg
            br = side_bottom_start + seg + 1
            faces.append([tl, br, bl])
            faces.append([tl, tr, br])

        # Bottom cap faces (CCW when viewed from -Y means normal points -Y)
        for seg in range(self.segments):
            next_seg = (seg + 1) % self.segments
            # Winding: center -> seg -> next_seg (for outward normal pointing -Y)
            faces.append([bottom_center, bottom_cap_ring_start + seg, bottom_cap_ring_start + next_seg])

        faces = np.array(faces, dtype=np.int64)

        return Mesh(vertices=vertices, faces=faces, uvs=uvs)


@dataclass
class ConeGenerator(MeshGenerator):
    """Generates a cone mesh.

    Attributes:
        radius: Radius of the cone base
        height: Height of the cone
        segments: Number of segments around the circumference
    """

    radius: float = 0.5
    height: float = 1.0
    segments: int = 32

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        """Generate attachment points at top, bottom, and radial mid-height."""
        from ..layout.attachments import AttachmentPoint

        radius = min(size[0], size[2]) / 2
        half_y = size[1] / 2

        return {
            "top": AttachmentPoint(
                name="top", anchor="center",
                offset=np.array([0, half_y, 0]), facing="north",
            ),
            "bottom": AttachmentPoint(
                name="bottom", anchor="center",
                offset=np.array([0, -half_y, 0]), facing="north",
            ),
            "left": AttachmentPoint(
                name="left", anchor="center",
                offset=np.array([-radius, 0, 0]), facing="west",
            ),
            "right": AttachmentPoint(
                name="right", anchor="center",
                offset=np.array([radius, 0, 0]), facing="east",
            ),
            "front": AttachmentPoint(
                name="front", anchor="center",
                offset=np.array([0, 0, radius]), facing="south",
            ),
            "back": AttachmentPoint(
                name="back", anchor="center",
                offset=np.array([0, 0, -radius]), facing="north",
            ),
        }

    def generate(self) -> Mesh:
        """Generate a cone mesh centered at the origin, with UV coordinates."""
        vertices = []
        uvs = []
        faces = []

        half_height = self.height / 2
        circumference = 2 * np.pi * self.radius
        slant = float(np.hypot(self.radius, self.height))

        # === Side vertices ===
        # Apex vertices (one per segment for proper UV seam)
        apex_start = 0
        for seg in range(self.segments + 1):
            vertices.append([0.0, half_height, 0.0])
            uvs.append([(seg + 0.5) / self.segments * circumference, slant])

        # Base ring for sides
        side_base_start = len(vertices)
        for seg in range(self.segments + 1):  # +1 for seam
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, -half_height, z])
            uvs.append([seg / self.segments * circumference, 0.0])

        # === Base cap vertices ===
        base_cap_ring_start = len(vertices)
        for seg in range(self.segments):
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, -half_height, z])
            # Radial UV for cap
            u = self.radius * (1 + np.cos(theta))
            v = self.radius * (1 - np.sin(theta))
            uvs.append([u, v])

        # Base center vertex
        base_center = len(vertices)
        vertices.append([0.0, -half_height, 0.0])
        uvs.append([self.radius, self.radius])

        vertices = np.array(vertices, dtype=np.float64)
        uvs = np.array(uvs, dtype=np.float64)

        # Side faces (triangles from apex to base ring, normals point outward)
        for seg in range(self.segments):
            apex_idx = apex_start + seg
            base_left = side_base_start + seg
            base_right = side_base_start + seg + 1
            faces.append([apex_idx, base_right, base_left])

        # Base cap faces (CCW when viewed from -Y means normal points -Y)
        for seg in range(self.segments):
            next_seg = (seg + 1) % self.segments
            # Winding: center -> seg -> next_seg (for outward normal pointing -Y)
            faces.append([base_center, base_cap_ring_start + seg, base_cap_ring_start + next_seg])

        faces = np.array(faces, dtype=np.int64)

        return Mesh(vertices=vertices, faces=faces, uvs=uvs)


@dataclass
class PlaneGenerator(MeshGenerator):
    """Generates a plane mesh (flat rectangular surface).

    Attributes:
        size_x: Width of the plane (X axis)
        size_z: Depth of the plane (Z axis)
        subdivisions_x: Number of subdivisions along X (for terrain)
        subdivisions_z: Number of subdivisions along Z (for terrain)
    """

    size_x: float = 1.0
    size_z: float = 1.0
    subdivisions_x: int = 1
    subdivisions_z: int = 1

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        """Generate attachment points at edges and center of the plane."""
        from ..layout.attachments import AttachmentPoint

        half_x = size[0] / 2
        half_z = size[2] / 2

        return {
            "center": AttachmentPoint(
                name="center", anchor="center",
                offset=np.array([0, 0, 0]), facing="north",
            ),
            "left": AttachmentPoint(
                name="left", anchor="center",
                offset=np.array([-half_x, 0, 0]), facing="west",
            ),
            "right": AttachmentPoint(
                name="right", anchor="center",
                offset=np.array([half_x, 0, 0]), facing="east",
            ),
            "front": AttachmentPoint(
                name="front", anchor="center",
                offset=np.array([0, 0, half_z]), facing="south",
            ),
            "back": AttachmentPoint(
                name="back", anchor="center",
                offset=np.array([0, 0, -half_z]), facing="north",
            ),
        }

    def get_surfaces(self, size: np.ndarray) -> dict[str, "Surface"]:
        """Expose the plane's top face as a surface.

        u along +X, v along +Z (back → front), normal +Y.
        """
        from ..layout.surfaces import Surface

        hx = size[0] / 2
        hz = size[2] / 2
        return {
            "top": Surface(
                name="top",
                origin=np.array([-hx, 0.0, -hz]),
                u_axis=np.array([1.0, 0.0, 0.0]),
                v_axis=np.array([0.0, 0.0, 1.0]),
                normal=np.array([0.0, 1.0, 0.0]),
                u_extent=float(size[0]),
                v_extent=float(size[2]),
            ),
        }

    def generate(self) -> Mesh:
        """Generate a plane mesh at Y=0 with UV coordinates."""
        hx = self.size_x / 2
        hz = self.size_z / 2

        vertices = []
        uvs = []
        faces = []

        # Generate grid of vertices
        for z_idx in range(self.subdivisions_z + 1):
            for x_idx in range(self.subdivisions_x + 1):
                # Position
                x = -hx + (x_idx / self.subdivisions_x) * self.size_x
                z = -hz + (z_idx / self.subdivisions_z) * self.size_z
                vertices.append([x, 0.0, z])

                # UVs
                uvs.append([x + hx, z + hz])  # metres

        # Generate faces (two triangles per quad)
        for z_idx in range(self.subdivisions_z):
            for x_idx in range(self.subdivisions_x):
                # Vertex indices for this quad
                tl = z_idx * (self.subdivisions_x + 1) + x_idx
                tr = tl + 1
                bl = (z_idx + 1) * (self.subdivisions_x + 1) + x_idx
                br = bl + 1

                # Two triangles, CCW winding (normal points +Y)
                faces.append([tl, bl, tr])
                faces.append([tr, bl, br])

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        )
