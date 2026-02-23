"""Roof shingle texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import NoiseTextureGenerator, NoiseLayer


@dataclass
class RoofTextureGenerator(NoiseTextureGenerator):
    """Generates procedural roof shingle textures.

    Creates a repeated rectangular shingle pattern with color variation
    and staggered rows for a realistic shingled roof appearance.

    Attributes:
        color_base: Base shingle color as (R, G, B) tuple, 0-255
        color_dark: Dark shingle variation as (R, G, B) tuple
        shingles_x: Number of shingles across the texture
        shingles_y: Number of shingle rows down the texture
        color_variation: Amount of per-shingle color variation (0-1)
        edge_width: Width of mortar/shadow lines between shingles (0-1)
    """

    color_base: tuple[int, int, int] = (90, 85, 80)
    color_dark: tuple[int, int, int] = (55, 50, 48)
    shingles_x: int = 8
    shingles_y: int = 16
    color_variation: float = 0.3
    edge_width: float = 0.08

    def __post_init__(self) -> None:
        self.color_a = self.color_dark
        self.color_b = self.color_base
        self.color_shift_strength = 0.0
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="variation", octaves=3, scale=6.0,
                       seed_offset=0, weight=0.0),
            NoiseLayer(name="surface", octaves=4, persistence=0.6,
                       scale=20.0, seed_offset=100, weight=0.0),
        ]

    def _compute_pattern(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        xv, yv = np.meshgrid(x, y)

        # Scale to shingle grid
        sx = xv * self.shingles_x
        sy = yv * self.shingles_y

        # Stagger every other row
        row = np.floor(sy).astype(int)
        offset = (row % 2) * 0.5
        sx_staggered = sx + offset

        # Fractional position within each shingle
        fx = sx_staggered - np.floor(sx_staggered)
        fy = sy - np.floor(sy)

        # Edge detection: shadow at bottom and sides of each shingle
        edge_x = np.minimum(fx, 1.0 - fx)
        edge_y = fy  # Shadow at top of each shingle (fy near 0)

        edge_mask = np.ones_like(fx)
        edge_mask = np.where(edge_x < self.edge_width, 0.3, edge_mask)
        edge_mask = np.where(edge_y < self.edge_width * 0.5, 0.4, edge_mask)

        # Per-shingle color variation using noise sampled at shingle centers
        shingle_noise = (layers["variation"] + 1.0) / 2.0
        surface_detail = (layers["surface"] + 1.0) / 2.0

        # Combine: base pattern with edge darkening and surface detail
        pattern = edge_mask * 0.7 + surface_detail * 0.2 + shingle_noise * 0.1

        return np.clip(pattern, 0, 1)

    def generate_normal_map(self) -> Image.Image | None:
        layers = self._generate_noise_layers()
        pattern = self._compute_pattern(layers)
        return self._height_to_normal(pattern, strength=1.5)

    def generate_roughness_map(self) -> Image.Image | None:
        layers = self._generate_noise_layers()
        surface = (layers["surface"] + 1.0) / 2.0
        return self._create_roughness_from_variation(0.8, surface, variation_strength=0.15)

    def generate_ao_map(self) -> Image.Image | None:
        """Generate AO map — shingle edges/overlaps accumulate shadow."""
        layers = self._generate_noise_layers()
        pattern = self._compute_pattern(layers)
        # Edge areas (low pattern values from edge_mask) are more occluded
        # pattern already has edge darkening built in
        ao = 0.5 + pattern * 0.5
        ao = np.clip(ao, 0.0, 1.0)
        return Image.fromarray((ao * 255).astype(np.uint8), mode='L')
