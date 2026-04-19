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
        bevel: Bevel/chamfer size (0 = sharp edges, default 0.02)
    """

    size_x: float = 1.0
    size_y: float = 1.0
    size_z: float = 1.0
    bevel: float = 0.02

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
        """Generate a cube mesh centered at the origin with UV coordinates."""
        hx, hy, hz = self.size_x / 2, self.size_y / 2, self.size_z / 2

        # Clamp bevel to half the smallest dimension
        max_bevel = min(hx, hy, hz) * 0.5
        b = min(self.bevel, max_bevel)

        if b <= 0.0001:
            return self._generate_sharp(hx, hy, hz)
        return self._generate_beveled(hx, hy, hz, b)

    def _generate_sharp(self, hx: float, hy: float, hz: float) -> Mesh:
        """Generate a sharp-edged cube (no bevel)."""
        vertices = []
        uvs = []
        faces = []

        face_defs = [
            ([-hx, -hy, -hz], [-hx, +hy, -hz], [+hx, +hy, -hz], [+hx, -hy, -hz]),
            ([+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz], [-hx, -hy, +hz]),
            ([-hx, -hy, +hz], [-hx, +hy, +hz], [-hx, +hy, -hz], [-hx, -hy, -hz]),
            ([+hx, -hy, -hz], [+hx, +hy, -hz], [+hx, +hy, +hz], [+hx, -hy, +hz]),
            ([-hx, -hy, +hz], [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, -hy, +hz]),
            ([-hx, +hy, -hz], [-hx, +hy, +hz], [+hx, +hy, +hz], [+hx, +hy, -hz]),
        ]

        uv_corners = [[0, 0], [0, 1], [1, 1], [1, 0]]

        for face_idx, corners in enumerate(face_defs):
            base_idx = face_idx * 4
            for corner, uv in zip(corners, uv_corners):
                vertices.append(corner)
                uvs.append(uv)
            faces.append([base_idx, base_idx + 1, base_idx + 2])
            faces.append([base_idx, base_idx + 2, base_idx + 3])

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
        )

    def _generate_beveled(self, hx: float, hy: float, hz: float, b: float) -> Mesh:
        """Generate a beveled cube with chamfered edges."""
        vertices = []
        uvs = []
        faces = []

        # Inset amounts for each axis
        bx = b
        by = b
        bz = b

        # Inner extents (face centers are inset by bevel)
        ix = hx - bx  # inner x half
        iy = hy - by  # inner y half
        iz = hz - bz  # inner z half

        def add_vert(pos, uv):
            idx = len(vertices)
            vertices.append(pos)
            uvs.append(uv)
            return idx

        def add_quad(a, b_, c, d):
            faces.append([a, b_, c])
            faces.append([a, c, d])

        # === 6 main faces (inset by bevel) ===

        # Back face (-Z): normal points -Z
        v0 = add_vert([-ix, -iy, -hz], [0, 0])
        v1 = add_vert([-ix, +iy, -hz], [0, 1])
        v2 = add_vert([+ix, +iy, -hz], [1, 1])
        v3 = add_vert([+ix, -iy, -hz], [1, 0])
        add_quad(v0, v1, v2, v3)

        # Front face (+Z)
        v4 = add_vert([+ix, -iy, +hz], [0, 0])
        v5 = add_vert([+ix, +iy, +hz], [0, 1])
        v6 = add_vert([-ix, +iy, +hz], [1, 1])
        v7 = add_vert([-ix, -iy, +hz], [1, 0])
        add_quad(v4, v5, v6, v7)

        # Left face (-X)
        v8 = add_vert([-hx, -iy, +iz], [0, 0])
        v9 = add_vert([-hx, +iy, +iz], [0, 1])
        v10 = add_vert([-hx, +iy, -iz], [1, 1])
        v11 = add_vert([-hx, -iy, -iz], [1, 0])
        add_quad(v8, v9, v10, v11)

        # Right face (+X)
        v12 = add_vert([+hx, -iy, -iz], [0, 0])
        v13 = add_vert([+hx, +iy, -iz], [0, 1])
        v14 = add_vert([+hx, +iy, +iz], [1, 1])
        v15 = add_vert([+hx, -iy, +iz], [1, 0])
        add_quad(v12, v13, v14, v15)

        # Bottom face (-Y)
        v16 = add_vert([-ix, -hy, +iz], [0, 0])
        v17 = add_vert([-ix, -hy, -iz], [0, 1])
        v18 = add_vert([+ix, -hy, -iz], [1, 1])
        v19 = add_vert([+ix, -hy, +iz], [1, 0])
        add_quad(v16, v17, v18, v19)

        # Top face (+Y)
        v20 = add_vert([-ix, +hy, -iz], [0, 0])
        v21 = add_vert([-ix, +hy, +iz], [0, 1])
        v22 = add_vert([+ix, +hy, +iz], [1, 1])
        v23 = add_vert([+ix, +hy, -iz], [1, 0])
        add_quad(v20, v21, v22, v23)

        # === 12 edge bevels ===
        # Each edge connects two face corners with a quad strip

        # 4 edges along X (top/bottom × front/back)
        # Top-front edge: connects top face front-left/right to front face top-left/right
        e0 = add_vert([-ix, +hy, +iz], [0, 1])
        e1 = add_vert([+ix, +hy, +iz], [1, 1])
        e2 = add_vert([+ix, +iy, +hz], [1, 0])
        e3 = add_vert([-ix, +iy, +hz], [0, 0])
        add_quad(e0, e1, e2, e3)

        # Top-back edge
        e4 = add_vert([+ix, +hy, -iz], [0, 1])
        e5 = add_vert([-ix, +hy, -iz], [1, 1])
        e6 = add_vert([-ix, +iy, -hz], [1, 0])
        e7 = add_vert([+ix, +iy, -hz], [0, 0])
        add_quad(e4, e5, e6, e7)

        # Bottom-front edge
        e8 = add_vert([+ix, -hy, +iz], [0, 1])
        e9 = add_vert([-ix, -hy, +iz], [1, 1])
        e10 = add_vert([-ix, -iy, +hz], [1, 0])
        e11 = add_vert([+ix, -iy, +hz], [0, 0])
        add_quad(e8, e9, e10, e11)

        # Bottom-back edge
        e12 = add_vert([-ix, -hy, -iz], [0, 1])
        e13 = add_vert([+ix, -hy, -iz], [1, 1])
        e14 = add_vert([+ix, -iy, -hz], [1, 0])
        e15 = add_vert([-ix, -iy, -hz], [0, 0])
        add_quad(e12, e13, e14, e15)

        # 4 edges along Y (left/right × front/back)
        # Left-front edge
        e16 = add_vert([-ix, -iy, +hz], [0, 0])
        e17 = add_vert([-ix, +iy, +hz], [0, 1])
        e18 = add_vert([-hx, +iy, +iz], [1, 1])
        e19 = add_vert([-hx, -iy, +iz], [1, 0])
        add_quad(e16, e17, e18, e19)

        # Left-back edge
        e20 = add_vert([-hx, -iy, -iz], [0, 0])
        e21 = add_vert([-hx, +iy, -iz], [0, 1])
        e22 = add_vert([-ix, +iy, -hz], [1, 1])
        e23 = add_vert([-ix, -iy, -hz], [1, 0])
        add_quad(e20, e21, e22, e23)

        # Right-front edge
        e24 = add_vert([+hx, -iy, +iz], [0, 0])
        e25 = add_vert([+hx, +iy, +iz], [0, 1])
        e26 = add_vert([+ix, +iy, +hz], [1, 1])
        e27 = add_vert([+ix, -iy, +hz], [1, 0])
        add_quad(e27, e26, e25, e24)

        # Right-back edge
        e28 = add_vert([+ix, -iy, -hz], [0, 0])
        e29 = add_vert([+ix, +iy, -hz], [0, 1])
        e30 = add_vert([+hx, +iy, -iz], [1, 1])
        e31 = add_vert([+hx, -iy, -iz], [1, 0])
        add_quad(e28, e29, e30, e31)

        # 4 edges along Z (left/right × top/bottom)
        # Top-left edge
        e32 = add_vert([-ix, +hy, -iz], [0, 0])
        e33 = add_vert([-ix, +hy, +iz], [0, 1])
        e34 = add_vert([-hx, +iy, +iz], [1, 1])
        e35 = add_vert([-hx, +iy, -iz], [1, 0])
        add_quad(e32, e33, e34, e35)

        # Top-right edge
        e36 = add_vert([+ix, +hy, +iz], [0, 0])
        e37 = add_vert([+ix, +hy, -iz], [0, 1])
        e38 = add_vert([+hx, +iy, -iz], [1, 1])
        e39 = add_vert([+hx, +iy, +iz], [1, 0])
        add_quad(e36, e37, e38, e39)

        # Bottom-left edge
        e40 = add_vert([-hx, -iy, +iz], [0, 0])
        e41 = add_vert([-hx, -iy, -iz], [0, 1])
        e42 = add_vert([-ix, -hy, -iz], [1, 1])
        e43 = add_vert([-ix, -hy, +iz], [1, 0])
        add_quad(e40, e41, e42, e43)

        # Bottom-right edge
        e44 = add_vert([+hx, -iy, -iz], [0, 0])
        e45 = add_vert([+hx, -iy, +iz], [0, 1])
        e46 = add_vert([+ix, -hy, +iz], [1, 1])
        e47 = add_vert([+ix, -hy, -iz], [1, 0])
        add_quad(e44, e45, e46, e47)

        # === 8 corner triangles ===
        # Each corner connects 3 edges meeting at that corner

        # Top-left-front (+Y, -X, +Z)
        c0 = add_vert([-ix, +hy, +iz], [0, 1])
        c1 = add_vert([-hx, +iy, +iz], [0, 0])
        c2 = add_vert([-ix, +iy, +hz], [1, 0])
        faces.append([c0, c1, c2])

        # Top-right-front (+Y, +X, +Z)
        c3 = add_vert([+ix, +hy, +iz], [0, 1])
        c4 = add_vert([+ix, +iy, +hz], [0, 0])
        c5 = add_vert([+hx, +iy, +iz], [1, 0])
        faces.append([c3, c4, c5])

        # Top-left-back (+Y, -X, -Z)
        c6 = add_vert([-ix, +hy, -iz], [0, 1])
        c7 = add_vert([-ix, +iy, -hz], [0, 0])
        c8 = add_vert([-hx, +iy, -iz], [1, 0])
        faces.append([c6, c7, c8])

        # Top-right-back (+Y, +X, -Z)
        c9 = add_vert([+ix, +hy, -iz], [0, 1])
        c10 = add_vert([+hx, +iy, -iz], [0, 0])
        c11 = add_vert([+ix, +iy, -hz], [1, 0])
        faces.append([c9, c10, c11])

        # Bottom-left-front (-Y, -X, +Z)
        c12 = add_vert([-ix, -hy, +iz], [0, 1])
        c13 = add_vert([-ix, -iy, +hz], [0, 0])
        c14 = add_vert([-hx, -iy, +iz], [1, 0])
        faces.append([c12, c13, c14])

        # Bottom-right-front (-Y, +X, +Z)
        c15 = add_vert([+ix, -hy, +iz], [0, 1])
        c16 = add_vert([+hx, -iy, +iz], [0, 0])
        c17 = add_vert([+ix, -iy, +hz], [1, 0])
        faces.append([c15, c16, c17])

        # Bottom-left-back (-Y, -X, -Z)
        c18 = add_vert([-ix, -hy, -iz], [0, 1])
        c19 = add_vert([-hx, -iy, -iz], [0, 0])
        c20 = add_vert([-ix, -iy, -hz], [1, 0])
        faces.append([c18, c19, c20])

        # Bottom-right-back (-Y, +X, -Z)
        c21 = add_vert([+ix, -hy, -iz], [0, 1])
        c22 = add_vert([+ix, -iy, -hz], [0, 0])
        c23 = add_vert([+hx, -iy, -iz], [1, 0])
        faces.append([c21, c22, c23])

        return Mesh(
            vertices=np.array(vertices, dtype=np.float64),
            faces=np.array(faces, dtype=np.int64),
            uvs=np.array(uvs, dtype=np.float64),
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

        return Mesh(vertices=vertices, faces=faces, uvs=uvs)


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

        # === Top cap vertices ===
        # Center vertex
        top_center = len(vertices)
        vertices.append([0.0, half_height, 0.0])
        uvs.append([0.5, 0.5])

        # Top cap ring (for cap faces)
        top_cap_ring_start = len(vertices)
        for seg in range(self.segments):
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, half_height, z])
            # Radial UV for cap
            u = 0.5 + 0.5 * np.cos(theta)
            v = 0.5 + 0.5 * np.sin(theta)
            uvs.append([u, v])

        # === Side vertices (separate for different UVs) ===
        # Top ring for sides
        side_top_start = len(vertices)
        for seg in range(self.segments + 1):  # +1 for seam
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, half_height, z])
            uvs.append([seg / self.segments, 1.0])

        # Bottom ring for sides
        side_bottom_start = len(vertices)
        for seg in range(self.segments + 1):  # +1 for seam
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, -half_height, z])
            uvs.append([seg / self.segments, 0.0])

        # === Bottom cap vertices ===
        # Bottom cap ring
        bottom_cap_ring_start = len(vertices)
        for seg in range(self.segments):
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, -half_height, z])
            # Radial UV for cap (flipped for bottom view)
            u = 0.5 + 0.5 * np.cos(theta)
            v = 0.5 - 0.5 * np.sin(theta)
            uvs.append([u, v])

        # Center vertex
        bottom_center = len(vertices)
        vertices.append([0.0, -half_height, 0.0])
        uvs.append([0.5, 0.5])

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

        # === Side vertices ===
        # Apex vertices (one per segment for proper UV seam)
        apex_start = 0
        for seg in range(self.segments + 1):
            vertices.append([0.0, half_height, 0.0])
            uvs.append([(seg + 0.5) / self.segments, 1.0])

        # Base ring for sides
        side_base_start = len(vertices)
        for seg in range(self.segments + 1):  # +1 for seam
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, -half_height, z])
            uvs.append([seg / self.segments, 0.0])

        # === Base cap vertices ===
        base_cap_ring_start = len(vertices)
        for seg in range(self.segments):
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, -half_height, z])
            # Radial UV for cap
            u = 0.5 + 0.5 * np.cos(theta)
            v = 0.5 - 0.5 * np.sin(theta)
            uvs.append([u, v])

        # Base center vertex
        base_center = len(vertices)
        vertices.append([0.0, -half_height, 0.0])
        uvs.append([0.5, 0.5])

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
                u = x_idx / self.subdivisions_x
                v = z_idx / self.subdivisions_z
                uvs.append([u, v])

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
