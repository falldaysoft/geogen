"""Dirt/ground texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .base import TextureGenerator
from .noise import fractal_noise


@dataclass
class DirtTextureGenerator(TextureGenerator):
    """Generates procedural dirt/soil textures.

    Creates realistic ground patterns using layered noise for:
    - Soil clumps and particles
    - Pebbles and small rocks
    - Moisture/color variation

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        seed: Random seed for reproducibility
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

    def generate(self) -> Image.Image:
        """Generate a dirt texture."""
        # Soil particle structure (fine granularity)
        particle_noise = fractal_noise(
            self.width, self.height,
            octaves=6,
            persistence=0.65,
            scale=self.particle_scale,
            seed=self.seed,
        )

        # Pebbles and small rocks (coarser features)
        pebble_noise = fractal_noise(
            self.width, self.height,
            octaves=3,
            persistence=0.5,
            scale=self.pebble_scale,
            seed=(self.seed + 100) if self.seed else 100,
        )

        # Moisture/color variation (patches of wet/dry soil)
        moisture_noise = fractal_noise(
            self.width, self.height,
            octaves=2,
            persistence=0.4,
            scale=4.0,
            seed=(self.seed + 200) if self.seed else 200,
        )

        # Combine noise layers
        # Particles give fine texture
        particle_value = (particle_noise + 1.0) / 2.0  # Normalize to [0, 1]

        # Pebbles create local bright spots
        pebble_value = (pebble_noise + 1.0) / 2.0
        pebble_value = np.power(pebble_value, 1.5)  # Make pebbles stand out

        # Moisture affects overall brightness
        moisture_value = (moisture_noise + 1.0) / 2.0

        # Combine for dirt pattern
        dirt_pattern = (
            particle_value * 0.5 +
            pebble_value * 0.3 +
            moisture_value * 0.2
        )

        # Convert to RGB
        base = np.array(self.color_base, dtype=np.float64)
        dark = np.array(self.color_dark, dtype=np.float64)
        light = np.array(self.color_light, dtype=np.float64)

        # Interpolate between dark (moist/shadow) and light (dry/peaks)
        dirt_pattern = np.clip(dirt_pattern, 0, 1)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)

        for i in range(3):
            rgb[:, :, i] = dark[i] + (light[i] - dark[i]) * dirt_pattern

        # Add random color variation (organic matter, minerals)
        color_shift = fractal_noise(
            self.width, self.height,
            octaves=2,
            scale=3.0,
            seed=(self.seed + 300) if self.seed else 300,
        )
        color_shift = color_shift * self.color_variation * 20

        for i in range(3):
            rgb[:, :, i] = np.clip(rgb[:, :, i] + color_shift, 0, 255)

        # Convert to uint8
        rgb_uint8 = rgb.astype(np.uint8)

        return Image.fromarray(rgb_uint8, mode='RGB')

    def generate_normal_map(self) -> Image.Image | None:
        """Generate a normal map for dirt (bumpy surface)."""
        # Combine particles and pebbles for height
        particle_noise = fractal_noise(
            self.width, self.height,
            octaves=6,
            persistence=0.65,
            scale=self.particle_scale,
            seed=self.seed,
        )

        pebble_noise = fractal_noise(
            self.width, self.height,
            octaves=3,
            persistence=0.5,
            scale=self.pebble_scale,
            seed=(self.seed + 100) if self.seed else 100,
        )

        # Normalize
        particle_value = (particle_noise + 1.0) / 2.0
        pebble_value = (pebble_noise + 1.0) / 2.0
        pebble_value = np.power(pebble_value, 1.5)

        # Combine: pebbles stick out more
        height = particle_value * 0.4 + pebble_value * 0.6

        return self._height_to_normal(height, strength=0.8)

    def generate_roughness_map(self) -> Image.Image | None:
        """Generate a roughness map for dirt (rough with some variation)."""
        # Dirt is fairly rough, smoother where pebbles are
        base_roughness = 0.75

        # Pebbles are slightly smoother
        pebble_noise = fractal_noise(
            self.width, self.height,
            octaves=3,
            scale=self.pebble_scale,
            seed=(self.seed + 100) if self.seed else 100,
        )
        pebble_value = (pebble_noise + 1.0) / 2.0

        # Pebbles reduce roughness slightly
        variation = 1.0 - pebble_value * 0.15

        return self._create_roughness_from_variation(
            base_roughness,
            variation,
            variation_strength=0.1,
        )
