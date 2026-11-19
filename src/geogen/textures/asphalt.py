"""Asphalt texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import NoiseLayer, NoiseTextureGenerator, unit


@dataclass
class AsphaltTextureGenerator(NoiseTextureGenerator):
    """Generates procedural asphalt/tarmac textures.

    Creates a dark aggregate surface with noise for a realistic
    road or parking lot appearance.

    Attributes:
        color_base: Base asphalt color as (R, G, B) tuple, 0-255
        color_dark: Dark variation as (R, G, B) tuple
        aggregate_scale: Scale of the aggregate/gravel pattern
        surface_roughness: Overall surface roughness variation
    """

    color_base: tuple[int, int, int] = (65, 65, 68)
    color_dark: tuple[int, int, int] = (35, 35, 38)
    aggregate_scale: float = 25.0
    surface_roughness: float = 0.5

    def __post_init__(self) -> None:
        self.color_a = self.color_dark
        self.color_b = self.color_base
        self.color_shift_strength = 0.0
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="aggregate", octaves=5, persistence=0.7,
                       scale=self.aggregate_scale, seed_offset=0, weight=0.6),
            NoiseLayer(name="large_variation", octaves=3, persistence=0.5,
                       scale=3.0, seed_offset=100, weight=0.3),
            NoiseLayer(name="fine", octaves=6, persistence=0.6,
                       scale=40.0, seed_offset=200, weight=0.1),
        ]

    normal_strength = 0.6

    def _height(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        aggregate = unit(layers["aggregate"])
        fine = unit(layers["fine"])
        height = aggregate * 0.6 + fine * 0.4
        return height

    def generate_roughness_map(self) -> Image.Image | None:
        layers = self._generate_noise_layers()
        aggregate = unit(layers["aggregate"])
        return self._create_roughness_from_variation(0.9, aggregate, variation_strength=0.1)
