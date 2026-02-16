"""Wall texture generators for plaster, drywall, and painted surfaces."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import TextureGenerator, NoiseTextureGenerator, NoiseLayer
from .noise import fractal_noise


@dataclass
class PlasterTextureGenerator(NoiseTextureGenerator):
    """Generates procedural plaster/stucco wall textures.

    Creates subtle textured wall surfaces with:
    - Fine surface texture variation
    - Subtle color variation
    - Optional aging/wear patterns

    Attributes:
        base_color: Base wall color as (R, G, B) tuple, 0-255
        texture_scale: Scale of surface texture (higher = finer)
        texture_strength: How pronounced the texture is (0-1)
        color_variation: Amount of color variation (0-1)
    """

    base_color: tuple[int, int, int] = (240, 235, 230)  # Off-white
    texture_scale: float = 20.0
    texture_strength: float = 0.15
    color_variation: float = 0.05

    def __post_init__(self) -> None:
        # Plaster uses base_color as both color endpoints (brightness variation only)
        self.color_a = self.base_color
        self.color_b = self.base_color
        self.color_shift_scale = 5.0
        self.color_shift_strength = self.color_variation * 20
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="texture", octaves=4, persistence=0.5,
                       scale=self.texture_scale, seed_offset=0, weight=0.7),
            NoiseLayer(name="fine", octaves=3, persistence=0.6,
                       scale=self.texture_scale * 2, seed_offset=100, weight=0.3),
        ]

    def _compute_pattern(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        surface = layers["texture"] * 0.7 + layers["fine"] * 0.3
        # Pattern maps to brightness shift, centered at 0.5
        return np.clip(surface * self.texture_strength + 0.5, 0, 1)

    def generate(self) -> Image.Image:
        """Generate a plaster wall texture."""
        layers = self._generate_noise_layers()
        pattern = self._compute_pattern(layers)

        # Start with base color
        base = np.array(self.base_color, dtype=np.float64)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for i in range(3):
            rgb[:, :, i] = base[i]

        # Apply surface texture as brightness variation
        brightness = (pattern - 0.5) * 2 * self.texture_strength * 50
        for i in range(3):
            rgb[:, :, i] = rgb[:, :, i] + brightness

        # Apply color shift
        rgb = self._apply_color_shift(rgb, layers)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')

    def generate_normal_map(self) -> Image.Image:
        """Generate normal map from plaster surface texture."""
        layers = self._generate_noise_layers()
        height = (layers["texture"] * 0.7 + layers["fine"] * 0.3) * 0.5 + 0.5
        return self._height_to_normal(height, strength=0.3)


@dataclass
class PaintedWallTextureGenerator(NoiseTextureGenerator):
    """Generates painted wall textures with subtle roller/brush marks.

    Creates smooth painted surfaces with:
    - Very subtle texture from paint application
    - Minor color variation

    Attributes:
        base_color: Wall paint color as (R, G, B) tuple, 0-255
        sheen: Surface sheen level (0=matte, 1=glossy)
    """

    base_color: tuple[int, int, int] = (245, 240, 235)  # Warm white
    sheen: float = 0.2

    def __post_init__(self) -> None:
        self.color_a = self.base_color
        self.color_b = self.base_color
        self.color_shift_strength = 0.0
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="roller", octaves=2, persistence=0.4,
                       scale=30.0, seed_offset=0, weight=0.6),
            NoiseLayer(name="streak", octaves=2, persistence=0.5,
                       scale=50.0, seed_offset=100, weight=0.4),
        ]

    def generate(self) -> Image.Image:
        """Generate a painted wall texture."""
        layers = self._generate_noise_layers()

        base = np.array(self.base_color, dtype=np.float64)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for i in range(3):
            rgb[:, :, i] = base[i]

        # Very subtle texture
        brightness = (layers["roller"] * 0.6 + layers["streak"] * 0.4) * 8
        for i in range(3):
            rgb[:, :, i] = rgb[:, :, i] + brightness

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')
