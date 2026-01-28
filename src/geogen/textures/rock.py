"""Rock texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .base import TextureGenerator
from .noise import fractal_noise, perlin_noise


@dataclass
class RockTextureGenerator(TextureGenerator):
    """Generates procedural rock/stone textures.

    Creates realistic rock patterns using layered noise for:
    - Large cracks and fissures
    - Surface roughness and pitting
    - Color variation (weathering, minerals)

    Attributes:
        width: Texture width in pixels
        height: Texture height in pixels
        seed: Random seed for reproducibility
        color_base: Base rock color as (R, G, B) tuple, 0-255
        color_dark: Dark areas (cracks, shadows) as (R, G, B) tuple
        color_light: Light areas (highlights) as (R, G, B) tuple
        crack_scale: Scale of large cracks
        roughness_scale: Scale of surface roughness
        color_variation: Amount of color variation (0-1)
    """

    color_base: tuple[int, int, int] = (120, 115, 110)
    color_dark: tuple[int, int, int] = (70, 68, 65)
    color_light: tuple[int, int, int] = (160, 155, 145)
    crack_scale: float = 4.0
    roughness_scale: float = 15.0
    color_variation: float = 0.5

    def generate(self) -> Image.Image:
        """Generate a rock texture."""
        # Large-scale cracks and structure
        crack_noise = fractal_noise(
            self.width, self.height,
            octaves=4,
            persistence=0.6,
            scale=self.crack_scale,
            seed=self.seed,
        )

        # Fine surface roughness (pitting, granularity)
        roughness_noise = fractal_noise(
            self.width, self.height,
            octaves=6,
            persistence=0.7,
            scale=self.roughness_scale,
            seed=(self.seed + 100) if self.seed else 100,
        )

        # Color variation (weathering, minerals, lichens)
        color_noise = fractal_noise(
            self.width, self.height,
            octaves=3,
            persistence=0.5,
            scale=3.0,
            seed=(self.seed + 200) if self.seed else 200,
        )

        # Combine noise layers
        # Cracks create dark lines
        crack_value = (crack_noise + 1.0) / 2.0  # Normalize to [0, 1]

        # Sharp crack definition
        crack_value = np.power(crack_value, 0.6)  # Enhance contrast

        # Roughness adds texture
        roughness_value = (roughness_noise + 1.0) / 2.0

        # Combine for base pattern
        rock_pattern = crack_value * 0.6 + roughness_value * 0.4

        # Convert to RGB
        base = np.array(self.color_base, dtype=np.float64)
        dark = np.array(self.color_dark, dtype=np.float64)
        light = np.array(self.color_light, dtype=np.float64)

        # Interpolate between dark and light based on pattern
        rock_pattern = np.clip(rock_pattern, 0, 1)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)

        # Use pattern to blend between dark (cracks) and light (peaks)
        for i in range(3):
            rgb[:, :, i] = dark[i] + (light[i] - dark[i]) * rock_pattern

        # Add color variation (weathering, minerals)
        color_shift = (color_noise + 1.0) / 2.0
        color_shift = (color_shift - 0.5) * self.color_variation * 40

        for i in range(3):
            rgb[:, :, i] = np.clip(rgb[:, :, i] + color_shift, 0, 255)

        # Convert to uint8
        rgb_uint8 = rgb.astype(np.uint8)

        return Image.fromarray(rgb_uint8, mode='RGB')

    def generate_normal_map(self) -> Image.Image | None:
        """Generate a normal map for rock (pronounced bumps and cracks)."""
        # Combine crack and roughness for height
        crack_noise = fractal_noise(
            self.width, self.height,
            octaves=4,
            persistence=0.6,
            scale=self.crack_scale,
            seed=self.seed,
        )

        roughness_noise = fractal_noise(
            self.width, self.height,
            octaves=6,
            persistence=0.7,
            scale=self.roughness_scale,
            seed=(self.seed + 100) if self.seed else 100,
        )

        # Cracks are deep indentations
        crack_value = (crack_noise + 1.0) / 2.0
        crack_value = np.power(crack_value, 0.6)

        # Roughness adds texture
        roughness_value = (roughness_noise + 1.0) / 2.0

        # Combine with emphasis on cracks
        height = crack_value * 0.7 + roughness_value * 0.3

        return self._height_to_normal(height, strength=1.2)

    def generate_roughness_map(self) -> Image.Image | None:
        """Generate a roughness map for rock (very rough with variation)."""
        # Rock is generally very rough
        base_roughness = 0.85

        # Cracks are slightly smoother (polished by water/wind)
        crack_noise = fractal_noise(
            self.width, self.height,
            octaves=4,
            scale=self.crack_scale,
            seed=self.seed,
        )
        crack_value = (crack_noise + 1.0) / 2.0

        # Invert: cracks = smoother
        variation = 1.0 - crack_value * 0.3

        return self._create_roughness_from_variation(
            base_roughness,
            variation,
            variation_strength=0.2,
        )

    def generate_ao_map(self) -> Image.Image | None:
        """Generate an ambient occlusion map (cracks are darker)."""
        # Use crack pattern for AO
        crack_noise = fractal_noise(
            self.width, self.height,
            octaves=4,
            persistence=0.6,
            scale=self.crack_scale,
            seed=self.seed,
        )

        # Normalize and invert (cracks = dark = low AO)
        ao = (crack_noise + 1.0) / 2.0
        ao = np.power(ao, 0.5)  # Soften the effect

        # Convert to image
        ao = np.clip(ao * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(ao, mode='L')
