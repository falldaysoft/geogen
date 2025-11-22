"""Wood texture generator using procedural noise."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .base import TextureGenerator
from .noise import perlin_noise, fractal_noise


@dataclass
class WoodTextureGenerator(TextureGenerator):
    """Generates procedural wood grain textures.

    Creates realistic wood patterns using layered noise for:
    - Ring structure (annual growth rings)
    - Grain variation (natural wood grain waviness)
    - Color variation (subtle color changes)

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        seed: Random seed for reproducibility
        color_light: Light wood color as (R, G, B) tuple, 0-255
        color_dark: Dark wood color (grain lines) as (R, G, B) tuple
        ring_scale: Scale of wood rings (higher = more rings)
        ring_count: Approximate number of visible rings
        grain_scale: Scale of grain distortion noise
        grain_strength: How much the grain distorts the rings
        color_variation: Amount of random color variation (0-1)
    """

    color_light: tuple[int, int, int] = (210, 170, 120)
    color_dark: tuple[int, int, int] = (140, 90, 50)
    ring_scale: float = 12.0
    ring_count: float = 8.0
    grain_scale: float = 4.0
    grain_strength: float = 0.3
    color_variation: float = 0.15

    def generate(self) -> Image.Image:
        """Generate a wood grain texture."""
        # Create coordinate grids
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        xv, yv = np.meshgrid(x, y)

        # Center coordinates for ring pattern
        cx, cy = 0.5, 0.5
        dx = xv - cx
        dy = yv - cy

        # Generate grain distortion noise
        grain_noise_x = fractal_noise(
            self.width, self.height,
            octaves=3,
            scale=self.grain_scale,
            seed=self.seed,
        )
        grain_noise_y = fractal_noise(
            self.width, self.height,
            octaves=3,
            scale=self.grain_scale,
            seed=(self.seed + 100) if self.seed else 100,
        )

        # Apply grain distortion
        dx_distorted = dx + grain_noise_x * self.grain_strength
        dy_distorted = dy + grain_noise_y * self.grain_strength

        # Calculate distance from center (for rings)
        dist = np.sqrt(dx_distorted**2 + dy_distorted**2)

        # Create ring pattern
        ring_value = np.sin(dist * self.ring_count * np.pi * 2) * 0.5 + 0.5

        # Add fine grain detail
        fine_grain = fractal_noise(
            self.width, self.height,
            octaves=4,
            persistence=0.6,
            scale=self.ring_scale,
            seed=(self.seed + 200) if self.seed else 200,
        )
        # Stretch fine grain along Y axis for wood grain look
        fine_grain_stretched = fractal_noise(
            self.width, self.height,
            octaves=4,
            persistence=0.6,
            scale=self.ring_scale * 0.25,  # Stretched in one direction
            seed=(self.seed + 300) if self.seed else 300,
        )

        # Combine ring pattern with fine grain
        wood_pattern = ring_value * 0.7 + (fine_grain * 0.5 + 0.5) * 0.2 + (fine_grain_stretched * 0.5 + 0.5) * 0.1

        # Add color variation noise
        color_noise = fractal_noise(
            self.width, self.height,
            octaves=2,
            scale=2.0,
            seed=(self.seed + 400) if self.seed else 400,
        )

        # Convert to RGB
        light = np.array(self.color_light, dtype=np.float64)
        dark = np.array(self.color_dark, dtype=np.float64)

        # Interpolate between light and dark based on wood pattern
        wood_pattern = np.clip(wood_pattern, 0, 1)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for i in range(3):
            rgb[:, :, i] = light[i] + (dark[i] - light[i]) * (1 - wood_pattern)

        # Apply color variation
        color_var = color_noise * self.color_variation * 30  # Scale to reasonable color shift
        for i in range(3):
            rgb[:, :, i] = np.clip(rgb[:, :, i] + color_var, 0, 255)

        # Convert to uint8
        rgb_uint8 = rgb.astype(np.uint8)

        return Image.fromarray(rgb_uint8, mode='RGB')
