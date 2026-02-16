"""Dirt/ground texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import NoiseTextureGenerator, NoiseLayer
from .noise import fractal_noise


@dataclass
class DirtTextureGenerator(NoiseTextureGenerator):
    """Generates procedural dirt/soil textures.

    Creates realistic ground patterns using layered noise for:
    - Soil clumps and particles
    - Pebbles and small rocks
    - Moisture/color variation

    Attributes:
        color_base: Base dirt color as (R, G, B) tuple, 0-255
        color_dark: Darker soil areas as (R, G, B) tuple
        color_light: Lighter/drier areas as (R, G, B) tuple
        particle_scale: Scale of soil particles/clumps
        pebble_scale: Scale of pebbles and small rocks
        color_variation: Amount of color variation (0-1)
    """

    color_base: tuple[int, int, int] = (100, 75, 50)
    color_dark: tuple[int, int, int] = (70, 50, 35)
    color_light: tuple[int, int, int] = (130, 100, 70)
    particle_scale: float = 20.0
    pebble_scale: float = 8.0
    color_variation: float = 0.4

    def __post_init__(self) -> None:
        self.color_a = self.color_dark
        self.color_b = self.color_light
        self.color_shift_scale = 3.0
        self.color_shift_strength = self.color_variation * 20
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="particle", octaves=6, persistence=0.65,
                       scale=self.particle_scale, seed_offset=0, weight=0.5),
            NoiseLayer(name="pebble", octaves=3, persistence=0.5,
                       scale=self.pebble_scale, seed_offset=100, weight=0.3),
            NoiseLayer(name="moisture", octaves=2, persistence=0.4,
                       scale=4.0, seed_offset=200, weight=0.2),
        ]

    def _compute_pattern(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        particle_value = (layers["particle"] + 1.0) / 2.0
        pebble_value = (layers["pebble"] + 1.0) / 2.0
        pebble_value = np.power(pebble_value, 1.5)
        moisture_value = (layers["moisture"] + 1.0) / 2.0

        pattern = particle_value * 0.5 + pebble_value * 0.3 + moisture_value * 0.2
        return np.clip(pattern, 0, 1)

    def _apply_color_shift(
        self, rgb: NDArray[np.float64], layers: dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        color_shift = fractal_noise(
            self.width, self.height,
            octaves=2, scale=3.0,
            seed=(self.seed + 300) if self.seed else 300,
        )
        color_shift = color_shift * self.color_variation * 20
        for i in range(3):
            rgb[:, :, i] = np.clip(rgb[:, :, i] + color_shift, 0, 255)
        return rgb

    def generate_normal_map(self) -> Image.Image | None:
        """Generate a normal map for dirt (bumpy surface)."""
        layers = self._generate_noise_layers()
        particle_value = (layers["particle"] + 1.0) / 2.0
        pebble_value = (layers["pebble"] + 1.0) / 2.0
        pebble_value = np.power(pebble_value, 1.5)
        height = particle_value * 0.4 + pebble_value * 0.6
        return self._height_to_normal(height, strength=0.8)

    def generate_roughness_map(self) -> Image.Image | None:
        """Generate a roughness map for dirt (rough with some variation)."""
        layers = self._generate_noise_layers()
        pebble_value = (layers["pebble"] + 1.0) / 2.0
        variation = 1.0 - pebble_value * 0.15
        return self._create_roughness_from_variation(0.75, variation, variation_strength=0.1)
