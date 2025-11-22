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
        """Generate a cube mesh centered at the origin."""
        hx, hy, hz = self.size_x / 2, self.size_y / 2, self.size_z / 2

        # 8 vertices of a cube
        vertices = np.array([
            [-hx, -hy, -hz],  # 0: back-bottom-left
            [+hx, -hy, -hz],  # 1: back-bottom-right
            [+hx, +hy, -hz],  # 2: back-top-right
            [-hx, +hy, -hz],  # 3: back-top-left
            [-hx, -hy, +hz],  # 4: front-bottom-left
            [+hx, -hy, +hz],  # 5: front-bottom-right
            [+hx, +hy, +hz],  # 6: front-top-right
            [-hx, +hy, +hz],  # 7: front-top-left
        ], dtype=np.float64)

        # 12 triangles (2 per face), CCW winding when viewed from outside
        faces = np.array([
            # Back face (-Z): normal points -Z, viewed from -Z vertices go CCW
            [0, 2, 1], [0, 3, 2],
            # Front face (+Z): normal points +Z, viewed from +Z vertices go CCW
            [4, 5, 6], [4, 6, 7],
            # Left face (-X): normal points -X
            [0, 4, 7], [0, 7, 3],
            # Right face (+X): normal points +X
            [1, 2, 6], [1, 6, 5],
            # Bottom face (-Y): normal points -Y
            [0, 1, 5], [0, 5, 4],
            # Top face (+Y): normal points +Y
            [3, 7, 6], [3, 6, 2],
        ], dtype=np.int64)

        return Mesh(vertices=vertices, faces=faces)


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
        """Generate a UV sphere mesh centered at the origin."""
        vertices = []
        faces = []

        # Top pole
        vertices.append([0.0, self.radius, 0.0])

        # Middle rings
        for ring in range(1, self.rings):
            phi = np.pi * ring / self.rings
            y = self.radius * np.cos(phi)
            ring_radius = self.radius * np.sin(phi)

            for seg in range(self.segments):
                theta = 2 * np.pi * seg / self.segments
                x = ring_radius * np.cos(theta)
                z = ring_radius * np.sin(theta)
                vertices.append([x, y, z])

        # Bottom pole
        vertices.append([0.0, -self.radius, 0.0])

        vertices = np.array(vertices, dtype=np.float64)

        # Top cap triangles - use helper
        top_pole = 0
        first_ring = list(range(1, 1 + self.segments))
        top_cap = make_cap_faces(
            top_pole, first_ring,
            normal_direction=np.array([0.0, 1.0, 0.0]),
            vertices=vertices
        )
        faces.extend(top_cap)

        # Middle quads (as triangles) - use tube helper for each ring pair
        for ring in range(self.rings - 2):
            ring_start = 1 + ring * self.segments
            next_ring_start = ring_start + self.segments

            top_ring = list(range(ring_start, ring_start + self.segments))
            bottom_ring = list(range(next_ring_start, next_ring_start + self.segments))

            tube_faces = make_tube_faces(top_ring, bottom_ring, vertices)
            faces.extend(tube_faces)

        # Bottom cap triangles - use helper
        bottom_pole = len(vertices) - 1
        last_ring_start = 1 + (self.rings - 2) * self.segments
        last_ring = list(range(last_ring_start, last_ring_start + self.segments))
        bottom_cap = make_cap_faces(
            bottom_pole, last_ring,
            normal_direction=np.array([0.0, -1.0, 0.0]),
            vertices=vertices
        )
        faces.extend(bottom_cap)

        faces = np.array(faces, dtype=np.int64)

        return Mesh(vertices=vertices, faces=faces)


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
        """Generate a cylinder mesh centered at the origin."""
        vertices = []

        half_height = self.height / 2

        # Top center vertex (index 0)
        top_center = 0
        vertices.append([0.0, half_height, 0.0])

        # Top ring vertices (indices 1 to segments)
        top_ring_start = 1
        for seg in range(self.segments):
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, half_height, z])

        # Bottom ring vertices (indices segments+1 to 2*segments)
        bottom_ring_start = 1 + self.segments
        for seg in range(self.segments):
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, -half_height, z])

        # Bottom center vertex
        bottom_center = 1 + 2 * self.segments
        vertices.append([0.0, -half_height, 0.0])

        vertices = np.array(vertices, dtype=np.float64)

        # Build index lists
        top_ring = list(range(top_ring_start, top_ring_start + self.segments))
        bottom_ring = list(range(bottom_ring_start, bottom_ring_start + self.segments))

        # Top cap - normal points +Y
        top_cap = make_cap_faces(
            top_center, top_ring,
            normal_direction=np.array([0.0, 1.0, 0.0]),
            vertices=vertices
        )

        # Side faces - normals point radially outward
        side_faces = make_tube_faces(top_ring, bottom_ring, vertices)

        # Bottom cap - normal points -Y
        bottom_cap = make_cap_faces(
            bottom_center, bottom_ring,
            normal_direction=np.array([0.0, -1.0, 0.0]),
            vertices=vertices
        )

        faces = top_cap + side_faces + bottom_cap
        faces = np.array(faces, dtype=np.int64)

        return Mesh(vertices=vertices, faces=faces)


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
        """Generate a cone mesh with base at y=0 and apex at y=height."""
        vertices = []

        # Apex vertex (index 0)
        apex = 0
        vertices.append([0.0, self.height, 0.0])

        # Base ring vertices (indices 1 to segments)
        base_ring_start = 1
        for seg in range(self.segments):
            theta = 2 * np.pi * seg / self.segments
            x = self.radius * np.cos(theta)
            z = self.radius * np.sin(theta)
            vertices.append([x, 0.0, z])

        # Base center vertex
        base_center = 1 + self.segments
        vertices.append([0.0, 0.0, 0.0])

        vertices = np.array(vertices, dtype=np.float64)

        # Build index list
        base_ring = list(range(base_ring_start, base_ring_start + self.segments))

        # Side faces - normals point outward from cone surface
        side_faces = make_cone_side_faces(apex, base_ring, vertices)

        # Base cap - normal points -Y (downward)
        base_cap = make_cap_faces(
            base_center, base_ring,
            normal_direction=np.array([0.0, -1.0, 0.0]),
            vertices=vertices
        )

        faces = side_faces + base_cap
        faces = np.array(faces, dtype=np.int64)

        return Mesh(vertices=vertices, faces=faces)
