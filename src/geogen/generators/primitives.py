"""Primitive geometry generators."""

from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh
from ..core.geometry import make_cap_faces, make_tube_faces, make_cone_side_faces
from .base import MeshGenerator


@dataclass
class CubeGenerator(MeshGenerator):
    """Generates a cube/box mesh.

    Attributes:
        size_x: Width of the cube (X axis)
        size_y: Height of the cube (Y axis)
        size_z: Depth of the cube (Z axis)
    """

    size_x: float = 1.0
    size_y: float = 1.0
    size_z: float = 1.0

    def generate(self) -> Mesh:
        """Generate a cube mesh centered at the origin with UV coordinates."""
        hx, hy, hz = self.size_x / 2, self.size_y / 2, self.size_z / 2

        # 24 vertices (4 per face) for proper UV mapping
        # Each face needs its own vertices so UVs can be independent
        vertices = []
        uvs = []
        faces = []

        # Face definitions: (normal_axis, sign, corners in CCW order when viewed from outside)
        # Each corner is defined by which of the 3 axes are positive (+1) or negative (-1)
        face_defs = [
            # Back face (-Z)
            ([-hx, -hy, -hz], [-hx, +hy, -hz], [+hx, +hy, -hz], [+hx, -hy, -hz]),
            # Front face (+Z)
            ([+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz], [-hx, -hy, +hz]),
            # Left face (-X)
            ([-hx, -hy, +hz], [-hx, +hy, +hz], [-hx, +hy, -hz], [-hx, -hy, -hz]),
            # Right face (+X)
            ([+hx, -hy, -hz], [+hx, +hy, -hz], [+hx, +hy, +hz], [+hx, -hy, +hz]),
            # Bottom face (-Y)
            ([-hx, -hy, +hz], [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, -hy, +hz]),
            # Top face (+Y)
            ([-hx, +hy, -hz], [-hx, +hy, +hz], [+hx, +hy, +hz], [+hx, +hy, -hz]),
        ]

        # UV corners for each face (CCW from bottom-left)
        uv_corners = [[0, 0], [0, 1], [1, 1], [1, 0]]

        for face_idx, corners in enumerate(face_defs):
            base_idx = face_idx * 4
            for corner, uv in zip(corners, uv_corners):
                vertices.append(corner)
                uvs.append(uv)
            # Two triangles per face (CCW winding)
            faces.append([base_idx, base_idx + 1, base_idx + 2])
            faces.append([base_idx, base_idx + 2, base_idx + 3])

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
