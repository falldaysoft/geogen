"""Wood texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import NoiseTextureGenerator, NoiseLayer


@dataclass
class WoodTextureGenerator(NoiseTextureGenerator):
    """Generates procedural wood grain textures.

    Creates realistic wood patterns using layered noise for:
    - Ring structure (annual growth rings)
    - Grain variation (natural wood grain waviness)
    - Color variation (subtle color changes)

    Attributes:
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

    def __post_init__(self) -> None:
        self.color_a = self.color_dark
        self.color_b = self.color_light
        self.color_shift_strength = 0.0  # handled in custom _apply_color_shift
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="grain_x", octaves=3, scale=self.grain_scale,
                       seed_offset=0, weight=0.0),
            NoiseLayer(name="grain_y", octaves=3, scale=self.grain_scale,
                       seed_offset=100, weight=0.0),
            NoiseLayer(name="fine", octaves=4, persistence=0.6,
                       scale=self.ring_scale, seed_offset=200, weight=0.0),
            NoiseLayer(name="fine_stretched", octaves=4, persistence=0.6,
                       scale=self.ring_scale * 0.25, seed_offset=300, weight=0.0),
            NoiseLayer(name="color", octaves=2, scale=2.0,
                       seed_offset=400, weight=0.0),
        ]

    def _compute_pattern(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        # Create coordinate grids
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        xv, yv = np.meshgrid(x, y)

        # Center coordinates for ring pattern
        dx = xv - 0.5
        dy = yv - 0.5

        # Apply grain distortion
        dx_distorted = dx + layers["grain_x"] * self.grain_strength
        dy_distorted = dy + layers["grain_y"] * self.grain_strength

        # Calculate distance from center (for rings)
        dist = np.sqrt(dx_distorted**2 + dy_distorted**2)

        # Create ring pattern
        ring_value = np.sin(dist * self.ring_count * np.pi * 2) * 0.5 + 0.5

        # Combine ring pattern with fine grain
        fine_grain = (layers["fine"] * 0.5 + 0.5)
        fine_stretched = (layers["fine_stretched"] * 0.5 + 0.5)
        wood_pattern = ring_value * 0.7 + fine_grain * 0.2 + fine_stretched * 0.1

        return np.clip(wood_pattern, 0, 1)

    def generate(self) -> Image.Image:
        """Generate a wood grain texture."""
        layers = self._generate_noise_layers()
        wood_pattern = self._compute_pattern(layers)

        # Interpolate between light and dark based on wood pattern
        light = np.array(self.color_light, dtype=np.float64)
        dark = np.array(self.color_dark, dtype=np.float64)
        rgb = np.zeros((self.height, self.width, 3), dtype=np.float64)
        for i in range(3):
            rgb[:, :, i] = light[i] + (dark[i] - light[i]) * (1 - wood_pattern)

        # Apply color shift
        rgb = self._apply_color_shift(rgb, layers)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')

    def _apply_color_shift(
        self, rgb: NDArray[np.float64], layers: dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        color_var = layers["color"] * self.color_variation * 30
        for i in range(3):
            rgb[:, :, i] = np.clip(rgb[:, :, i] + color_var, 0, 255)
        return rgb
