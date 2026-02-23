"""Wood texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import NoiseTextureGenerator, NoiseLayer


@dataclass
class WoodTextureGenerator(NoiseTextureGenerator):
    """Generates procedural wood grain textures.

    Creates realistic lengthwise plank grain using layered noise for:
    - Parallel grain lines running along the plank
    - Subtle waviness and variation in grain direction
    - Color variation between early/late wood

    Attributes:
        color_light: Light wood color as (R, G, B) tuple, 0-255
        color_dark: Dark wood color (grain lines) as (R, G, B) tuple
        ring_scale: Density of grain lines (higher = more lines)
        ring_count: Not used directly; kept for YAML compat
        grain_scale: Scale of grain waviness noise
        grain_strength: How much the grain lines wobble
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
            NoiseLayer(name="grain_warp", octaves=3, scale=self.grain_scale,
                       seed_offset=0, weight=0.0),
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
        xv, _ = np.meshgrid(x, y)

        # Lengthwise plank grain: lines run vertically (along Y axis)
        # Use X coordinate as the base for grain lines, with noise warp
        grain_coord = xv + layers["grain_warp"] * self.grain_strength * 0.3

        # Create parallel grain lines using sine waves at multiple frequencies
        grain_primary = np.sin(grain_coord * self.ring_count * np.pi * 2) * 0.5 + 0.5
        grain_secondary = np.sin(grain_coord * self.ring_count * np.pi * 4.7 + 1.3) * 0.5 + 0.5

        # Combine: primary grain dominates, secondary adds detail
        grain_value = grain_primary * 0.7 + grain_secondary * 0.3

        # Reduce contrast compared to the old radial pattern
        grain_value = 0.3 + grain_value * 0.5

        # Add subtle fine detail stretched along the grain direction
        fine_stretched = (layers["fine_stretched"] * 0.5 + 0.5)
        wood_pattern = grain_value * 0.85 + fine_stretched * 0.15

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

    def generate_roughness_map(self) -> Image.Image | None:
        """Generate roughness map based on wood grain pattern.

        Grain valleys are smoother, ridges are rougher.
        """
        layers = self._generate_noise_layers()
        wood_pattern = self._compute_pattern(layers)
        # Invert: grain valleys (low pattern) -> smoother (lower roughness)
        # Ridges (high pattern) -> rougher (higher roughness)
        variation = wood_pattern
        return self._create_roughness_from_variation(0.7, variation, variation_strength=0.25)

    def generate_ao_map(self) -> Image.Image | None:
        """Generate AO map with subtle darkening in grain valleys."""
        layers = self._generate_noise_layers()
        wood_pattern = self._compute_pattern(layers)
        # Grain valleys (low values) are slightly occluded
        ao = 0.85 + wood_pattern * 0.15
        ao = np.clip(ao, 0.0, 1.0)
        return Image.fromarray((ao * 255).astype(np.uint8), mode='L')

    def _apply_color_shift(
        self, rgb: NDArray[np.float64], layers: dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        color_var = layers["color"] * self.color_variation * 20
        for i in range(3):
            rgb[:, :, i] = np.clip(rgb[:, :, i] + color_var, 0, 255)
        return rgb
