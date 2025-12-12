"""Wall texture generators for plaster, drywall, and painted surfaces."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .base import TextureGenerator
from .noise import fractal_noise


@dataclass
class PlasterTextureGenerator(TextureGenerator):
    """Generates procedural plaster/stucco wall textures.

    Creates subtle textured wall surfaces with:
    - Fine surface texture variation
    - Subtle color variation
    - Optional aging/wear patterns

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        seed: Random seed for reproducibility
        base_color: Base wall color as (R, G, B) tuple, 0-255
        texture_scale: Scale of surface texture (higher = finer)
        texture_strength: How pronounced the texture is (0-1)
        color_variation: Amount of color variation (0-1)
    """

    base_color: tuple[int, int, int] = (240, 235, 230)  # Off-white
    texture_scale: float = 20.0
    texture_strength: float = 0.15
    color_variation: float = 0.05

    def generate(self) -> Image.Image:
        """Generate a plaster wall texture."""
        # Base texture noise
        texture_noise = fractal_noise(
            self.width, self.height,
            octaves=4,
            persistence=0.5,
            scale=self.texture_scale,
            seed=self.seed,
        )

        # Fine detail noise
        fine_noise = fractal_noise(
            self.width, self.height,
            octaves=3,
            persistence=0.6,
            scale=self.texture_scale * 2,
            seed=(self.seed + 100) if self.seed else 100,
        )

        # Combine for surface detail
        surface = texture_noise * 0.7 + fine_noise * 0.3

        # Color variation noise
        color_noise = fractal_noise(
            self.width, self.height,
            octaves=2,
            scale=5.0,
            seed=(self.seed + 200) if self.seed else 200,
        )

        # Create RGB image
        base = np.array(self.base_color, dtype=np.float64)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)

        for i in range(3):
            rgb[:, :, i] = base[i]

        # Apply surface texture as brightness variation
        brightness = surface * self.texture_strength * 50
        for i in range(3):
            rgb[:, :, i] = rgb[:, :, i] + brightness

        # Apply subtle color variation
        color_var = color_noise * self.color_variation * 20
        for i in range(3):
            rgb[:, :, i] = rgb[:, :, i] + color_var

        # Clamp to valid range
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return Image.fromarray(rgb, mode='RGB')


@dataclass
class PaintedWallTextureGenerator(TextureGenerator):
    """Generates painted wall textures with subtle roller/brush marks.

    Creates smooth painted surfaces with:
    - Very subtle texture from paint application
    - Minor color variation

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        seed: Random seed for reproducibility
        base_color: Wall paint color as (R, G, B) tuple, 0-255
        sheen: Surface sheen level (0=matte, 1=glossy)
    """

    base_color: tuple[int, int, int] = (245, 240, 235)  # Warm white
    sheen: float = 0.2

    def generate(self) -> Image.Image:
        """Generate a painted wall texture."""
        # Very subtle roller texture
        roller_noise = fractal_noise(
            self.width, self.height,
            octaves=2,
            persistence=0.4,
            scale=30.0,
            seed=self.seed,
        )

        # Vertical streaking (common in painted walls)
        streak_noise = fractal_noise(
            self.width, self.height,
            octaves=2,
            scale=50.0,
            seed=(self.seed + 100) if self.seed else 100,
        )

        # Create RGB image
        base = np.array(self.base_color, dtype=np.float64)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)

        for i in range(3):
            rgb[:, :, i] = base[i]

        # Very subtle texture
        brightness = (roller_noise * 0.6 + streak_noise * 0.4) * 8
        for i in range(3):
            rgb[:, :, i] = rgb[:, :, i] + brightness

        # Clamp
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return Image.fromarray(rgb, mode='RGB')
