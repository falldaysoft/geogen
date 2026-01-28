"""Terrain generator using heightmaps."""

from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..textures.noise import fractal_noise
from .base import MeshGenerator
from .primitives import PlaneGenerator


@dataclass
class TerrainGenerator(MeshGenerator):
    """Generates terrain mesh with procedural heightmap.

    Creates a subdivided plane and applies a heightmap generated from
    fractal noise to create hills, valleys, and natural-looking terrain.

    Attributes:
        size_x: Width of the terrain (X axis)
        size_z: Depth of the terrain (Z axis)
        subdivisions_x: Number of subdivisions along X (more = smoother)
        subdivisions_z: Number of subdivisions along Z (more = smoother)
        height_scale: Maximum height variation (peak to valley)
        octaves: Number of noise octaves (more = more detail)
        scale: Base scale of the noise (higher = larger features)
        seed: Random seed for reproducibility
    """

    size_x: float = 10.0
    size_z: float = 10.0
    subdivisions_x: int = 50
    subdivisions_z: int = 50
    height_scale: float = 2.0
    octaves: int = 4
    scale: float = 3.0
    seed: int | None = None

    def generate(self) -> Mesh:
        """Generate a terrain mesh with heightmap-displaced vertices."""
        # Generate base plane
        plane_gen = PlaneGenerator(
            size_x=self.size_x,
            size_z=self.size_z,
            subdivisions_x=self.subdivisions_x,
            subdivisions_z=self.subdivisions_z,
        )
        mesh = plane_gen.generate()

        # Generate heightmap using fractal noise
        heightmap = fractal_noise(
            width=self.subdivisions_x + 1,
            height=self.subdivisions_z + 1,
            octaves=self.octaves,
            scale=self.scale,
            seed=self.seed,
        )

        # Heightmap is in range [-1, 1], normalize to [0, 1]
        heightmap = (heightmap + 1.0) / 2.0

        # Apply heightmap to vertices
        vertices = mesh.vertices.copy()
        for z_idx in range(self.subdivisions_z + 1):
            for x_idx in range(self.subdivisions_x + 1):
                vert_idx = z_idx * (self.subdivisions_x + 1) + x_idx
                height = heightmap[z_idx, x_idx] * self.height_scale
                vertices[vert_idx, 1] = height

        # Create new mesh with displaced vertices
        return Mesh(
            vertices=vertices,
            faces=mesh.faces,
            uvs=mesh.uvs,
        )
