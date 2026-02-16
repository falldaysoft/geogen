"""Grass texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import NoiseTextureGenerator, NoiseLayer
from .noise import fractal_noise


@dataclass
class GrassTextureGenerator(NoiseTextureGenerator):
    """Generates procedural grass textures.

    Creates realistic grass patterns using layered noise for:
    - Base color variation (different shades of green)
    - Blade structure (small-scale detail)
    - Clumping patterns (groups of grass)

    Attributes:
        color_base: Base grass color as (R, G, B) tuple, 0-255
        color_variation: Secondary grass color for variation
        blade_scale: Scale of individual grass blade detail
        clump_scale: Scale of grass clumping pattern
        variation_strength: Amount of color variation (0-1)
    """

    color_base: tuple[int, int, int] = (60, 110, 40)
    color_variation: tuple[int, int, int] = (80, 140, 50)
    blade_scale: float = 30.0
    clump_scale: float = 5.0
    variation_strength: float = 0.6

    def __post_init__(self) -> None:
        self.color_a = self.color_base
        self.color_b = self.color_variation
        self.color_shift_strength = 15.0
        self.color_shift_scale = 2.0
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="clump", octaves=3, persistence=0.5,
                       scale=self.clump_scale, seed_offset=0, weight=0.5),
            NoiseLayer(name="blade", octaves=5, persistence=0.6,
                       scale=self.blade_scale, seed_offset=100, weight=0.3),
            NoiseLayer(name="streak", octaves=2, persistence=0.4,
                       scale=15.0, seed_offset=200, weight=0.2),
        ]

    def _compute_pattern(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        clump_value = (layers["clump"] + 1.0) / 2.0
        blade_value = (layers["blade"] + 1.0) / 2.0
        streak_value = (layers["streak"] + 1.0) / 2.0

        pattern = clump_value * 0.5 + blade_value * 0.3 + streak_value * 0.2
        return np.clip(pattern, 0, 1)

    def generate(self) -> Image.Image:
        """Generate a grass texture."""
        layers = self._generate_noise_layers()
        pattern = self._compute_pattern(layers)

        base = np.array(self.color_base, dtype=np.float64)
        variation = np.array(self.color_variation, dtype=np.float64)

        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for i in range(3):
            rgb[:, :, i] = base[i] + (variation[i] - base[i]) * pattern * self.variation_strength

        # Apply color shift
        rgb = self._apply_color_shift(rgb, layers)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')

    def _apply_color_shift(
        self, rgb: NDArray[np.float64], layers: dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        color_shift = fractal_noise(
            self.width, self.height,
            octaves=2, scale=2.0,
            seed=(self.seed + 300) if self.seed else 300,
        ) * 15
        for i in range(3):
            rgb[:, :, i] = np.clip(rgb[:, :, i] + color_shift, 0, 255)
        return rgb

    def generate_normal_map(self) -> Image.Image | None:
        """Generate a normal map for grass (subtle bumps)."""
        layers = self._generate_noise_layers()
        height = (layers["blade"] + 1.0) / 2.0
        return self._height_to_normal(height, strength=0.3)

    def generate_roughness_map(self) -> Image.Image | None:
        """Generate a roughness map for grass (fairly rough surface)."""
        variation = fractal_noise(
            self.width, self.height,
            octaves=3, scale=self.blade_scale * 0.5,
            seed=(self.seed + 400) if self.seed else 400,
        )
        variation = (variation + 1.0) / 2.0
        return self._create_roughness_from_variation(0.8, variation, variation_strength=0.15)
