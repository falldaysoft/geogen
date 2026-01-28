"""Grass texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .base import TextureGenerator
from .noise import fractal_noise


@dataclass
class GrassTextureGenerator(TextureGenerator):
    """Generates procedural grass textures.

    Creates realistic grass patterns using layered noise for:
    - Base color variation (different shades of green)
    - Blade structure (small-scale detail)
    - Clumping patterns (groups of grass)

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        seed: Random seed for reproducibility
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

    def generate(self) -> Image.Image:
        """Generate a grass texture."""
        # Create coordinate grids
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        xv, yv = np.meshgrid(x, y)

        # Generate clumping pattern (large-scale variation)
        clump_noise = fractal_noise(
            self.width, self.height,
            octaves=3,
            persistence=0.5,
            scale=self.clump_scale,
            seed=self.seed,
        )

        # Generate blade detail (fine-scale texture)
        blade_noise = fractal_noise(
            self.width, self.height,
            octaves=5,
            persistence=0.6,
            scale=self.blade_scale,
            seed=(self.seed + 100) if self.seed else 100,
        )

        # Generate directional streaks (simulating grass blades orientation)
        streak_noise = fractal_noise(
            self.width, self.height,
            octaves=2,
            persistence=0.4,
            scale=15.0,
            seed=(self.seed + 200) if self.seed else 200,
        )

        # Combine noise layers
        # Clump pattern affects overall brightness
        clump_value = (clump_noise + 1.0) / 2.0  # Normalize to [0, 1]

        # Blade detail adds fine texture
        blade_value = (blade_noise + 1.0) / 2.0

        # Streak adds directional variation
        streak_value = (streak_noise + 1.0) / 2.0

        # Combine: clumps for variation, blades for detail, streaks for direction
        grass_pattern = (
            clump_value * 0.5 +
            blade_value * 0.3 +
            streak_value * 0.2
        )

        # Convert to RGB
        base = np.array(self.color_base, dtype=np.float64)
        variation = np.array(self.color_variation, dtype=np.float64)

        # Interpolate between base and variation based on pattern
        grass_pattern = np.clip(grass_pattern, 0, 1)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)

        for i in range(3):
            rgb[:, :, i] = base[i] + (variation[i] - base[i]) * grass_pattern * self.variation_strength

        # Add subtle random color shifts for realism
        color_shift = fractal_noise(
            self.width, self.height,
            octaves=2,
            scale=2.0,
            seed=(self.seed + 300) if self.seed else 300,
        ) * 15

        for i in range(3):
            rgb[:, :, i] = np.clip(rgb[:, :, i] + color_shift, 0, 255)

        # Convert to uint8
        rgb_uint8 = rgb.astype(np.uint8)

        return Image.fromarray(rgb_uint8, mode='RGB')

    def generate_normal_map(self) -> Image.Image | None:
        """Generate a normal map for grass (subtle bumps)."""
        # Use blade noise as height for normal generation
        blade_noise = fractal_noise(
            self.width, self.height,
            octaves=5,
            persistence=0.6,
            scale=self.blade_scale,
            seed=(self.seed + 100) if self.seed else 100,
        )

        height = (blade_noise + 1.0) / 2.0  # Normalize to [0, 1]
        return self._height_to_normal(height, strength=0.3)

    def generate_roughness_map(self) -> Image.Image | None:
        """Generate a roughness map for grass (fairly rough surface)."""
        # Grass is generally rough with some variation
        base_roughness = 0.8

        # Add variation based on blade pattern
        variation = fractal_noise(
            self.width, self.height,
            octaves=3,
            scale=self.blade_scale * 0.5,
            seed=(self.seed + 400) if self.seed else 400,
        )
        variation = (variation + 1.0) / 2.0  # Normalize to [0, 1]

        return self._create_roughness_from_variation(
            base_roughness,
            variation,
            variation_strength=0.15,
        )
