"""Brick texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import NoiseTextureGenerator, NoiseLayer


@dataclass
class BrickTextureGenerator(NoiseTextureGenerator):
    """Generates procedural brick wall textures.

    Creates a red/brown brick pattern with mortar lines and per-brick
    color variation for a realistic masonry appearance.

    Attributes:
        color_base: Base brick color as (R, G, B) tuple, 0-255
        color_dark: Dark brick variation as (R, G, B) tuple
        mortar_color: Mortar/grout color as (R, G, B) tuple
        bricks_x: Number of bricks across the texture
        bricks_y: Number of brick rows
        mortar_width: Width of mortar lines (0-1 fraction of brick)
        color_variation: Per-brick color variation strength
    """

    color_base: tuple[int, int, int] = (160, 75, 55)
    color_dark: tuple[int, int, int] = (120, 55, 40)
    mortar_color: tuple[int, int, int] = (200, 195, 185)
    bricks_x: int = 6
    bricks_y: int = 12
    mortar_width: float = 0.06
    color_variation: float = 0.3

    def __post_init__(self) -> None:
        self.color_a = self.color_dark
        self.color_b = self.color_base
        self.color_shift_strength = 0.0
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="brick_color", octaves=3, scale=8.0,
                       seed_offset=0, weight=0.0),
            NoiseLayer(name="surface", octaves=4, persistence=0.6,
                       scale=20.0, seed_offset=100, weight=0.0),
        ]

    def _compute_pattern(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        xv, yv = np.meshgrid(x, y)

        # Scale to brick grid
        bx = xv * self.bricks_x
        by = yv * self.bricks_y

        # Stagger every other row (standard brick bond)
        row = np.floor(by).astype(int)
        offset = (row % 2) * 0.5
        bx_staggered = bx + offset

        # Fractional position within each brick
        fx = bx_staggered - np.floor(bx_staggered)
        fy = by - np.floor(by)

        # Mortar detection
        is_mortar_x = (fx < self.mortar_width) | (fx > (1.0 - self.mortar_width))
        is_mortar_y = (fy < self.mortar_width * 0.7)
        is_mortar = is_mortar_x | is_mortar_y

        # Brick surface: noise-based color variation
        brick_noise = (layers["brick_color"] + 1.0) / 2.0
        surface_detail = (layers["surface"] + 1.0) / 2.0

        pattern = brick_noise * 0.6 + surface_detail * 0.4

        # Mortar is represented as high pattern value (maps toward color_b / lighter)
        # We'll override mortar pixels in generate()
        return np.clip(pattern, 0, 1), is_mortar

    def generate(self) -> Image.Image:
        """Generate brick texture with mortar lines."""
        layers = self._generate_noise_layers()
        pattern, is_mortar = self._compute_pattern(layers)

        # Map brick areas between dark and base colors
        dark = np.array(self.color_dark, dtype=np.float64)
        base = np.array(self.color_base, dtype=np.float64)
        mortar = np.array(self.mortar_color, dtype=np.float64)

        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for i in range(3):
            brick_color = dark[i] + (base[i] - dark[i]) * pattern
            rgb[:, :, i] = np.where(is_mortar, mortar[i], brick_color)

        # Add slight surface noise to mortar too
        surface = (layers["surface"] + 1.0) / 2.0
        mortar_noise = (surface - 0.5) * 10
        for i in range(3):
            rgb[:, :, i] = np.where(
                is_mortar,
                np.clip(rgb[:, :, i] + mortar_noise, 0, 255),
                rgb[:, :, i]
            )

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')

    def generate_normal_map(self) -> Image.Image | None:
        layers = self._generate_noise_layers()
        pattern, is_mortar = self._compute_pattern(layers)
        # Mortar is recessed, bricks are raised
        height = np.where(is_mortar, 0.2, 0.8)
        surface = (layers["surface"] + 1.0) / 2.0
        height = height + surface * 0.1
        return self._height_to_normal(height, strength=1.5)

    def generate_roughness_map(self) -> Image.Image | None:
        layers = self._generate_noise_layers()
        _, is_mortar = self._compute_pattern(layers)
        surface = (layers["surface"] + 1.0) / 2.0
        # Mortar is rougher than brick
        roughness = np.where(is_mortar, 0.9, 0.75)
        roughness = roughness + (surface - 0.5) * 0.1
        roughness = np.clip(roughness, 0.0, 1.0)
        return Image.fromarray((roughness * 255).astype(np.uint8), mode='L')

    def generate_ao_map(self) -> Image.Image | None:
        """Generate AO map — mortar joints are recessed and darker."""
        layers = self._generate_noise_layers()
        _, is_mortar = self._compute_pattern(layers)
        # Mortar is recessed -> occluded (darker AO)
        ao = np.where(is_mortar, 0.6, 1.0)
        # Soften edges slightly with surface noise
        surface = (layers["surface"] + 1.0) / 2.0
        ao = ao - (1.0 - surface) * 0.05
        ao = np.clip(ao, 0.0, 1.0)
        return Image.fromarray((ao * 255).astype(np.uint8), mode='L')
