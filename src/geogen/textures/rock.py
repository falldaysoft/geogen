"""Rock texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import NoiseLayer, NoiseTextureGenerator, shifted, unit


@dataclass
class RockTextureGenerator(NoiseTextureGenerator):
    """Generates procedural rock/stone textures.

    Creates realistic rock patterns using layered noise for:
    - Large cracks and fissures
    - Surface roughness and pitting
    - Color variation (weathering, minerals)

    Attributes:
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

    def __post_init__(self) -> None:
        self.color_a = self.color_dark
        self.color_b = self.color_light
        self.color_shift_strength = 0.0  # handled in custom _apply_color_shift
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="crack", octaves=4, persistence=0.6,
                       scale=self.crack_scale, seed_offset=0, weight=0.6),
            NoiseLayer(name="roughness", octaves=6, persistence=0.7,
                       scale=self.roughness_scale, seed_offset=100, weight=0.4),
            NoiseLayer(name="color", octaves=3, persistence=0.5,
                       scale=3.0, seed_offset=200, weight=0.0),
        ]

    def _compute_pattern(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        crack_value = unit(layers["crack"])
        crack_value = np.power(crack_value, 0.6)  # Enhance contrast
        roughness_value = unit(layers["roughness"])
        pattern = crack_value * 0.6 + roughness_value * 0.4
        return np.clip(pattern, 0, 1)

    def _apply_color_shift(
        self, rgb: NDArray[np.float64], layers: dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        color_shift = unit(layers["color"])
        color_shift = (color_shift - 0.5) * self.color_variation * 40
        return shifted(rgb, color_shift)

    normal_strength = 1.2

    def _height(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Height for the normal map: rock (pronounced bumps and cracks)."""
        crack_value = unit(layers["crack"])
        crack_value = np.power(crack_value, 0.6)
        roughness_value = unit(layers["roughness"])
        height = crack_value * 0.7 + roughness_value * 0.3
        return height

    def generate_roughness_map(self) -> Image.Image | None:
        """Generate a roughness map for rock (very rough with variation)."""
        layers = self._generate_noise_layers()
        crack_value = unit(layers["crack"])
        variation = 1.0 - crack_value * 0.3
        return self._create_roughness_from_variation(0.85, variation, variation_strength=0.2)

    def generate_ao_map(self) -> Image.Image | None:
        """Generate an ambient occlusion map (cracks are darker)."""
        layers = self._generate_noise_layers()
        ao = unit(layers["crack"])
        ao = np.power(ao, 0.5)
        ao = np.clip(ao * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(ao, mode='L')
