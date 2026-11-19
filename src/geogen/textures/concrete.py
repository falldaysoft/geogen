"""Concrete texture generator using procedural noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import NoiseLayer, NoiseTextureGenerator, unit


@dataclass
class ConcreteTextureGenerator(NoiseTextureGenerator):
    """Generates procedural concrete/cement textures.

    Creates a light gray surface with subtle variation for sidewalks,
    foundations, and other concrete surfaces.

    Attributes:
        color_base: Base concrete color as (R, G, B) tuple, 0-255
        color_dark: Dark variation as (R, G, B) tuple
        surface_scale: Scale of surface variation
        pitting_scale: Scale of small pits/pores
    """

    color_base: tuple[int, int, int] = (190, 188, 183)
    color_dark: tuple[int, int, int] = (155, 152, 148)
    surface_scale: float = 8.0
    pitting_scale: float = 30.0

    def __post_init__(self) -> None:
        self.color_a = self.color_dark
        self.color_b = self.color_base
        self.color_shift_strength = 0.0
        super().__post_init__()

    def _get_noise_layers(self) -> list[NoiseLayer]:
        return [
            NoiseLayer(name="surface", octaves=4, persistence=0.5,
                       scale=self.surface_scale, seed_offset=0, weight=0.6),
            NoiseLayer(name="pitting", octaves=5, persistence=0.6,
                       scale=self.pitting_scale, seed_offset=100, weight=0.3),
            NoiseLayer(name="color", octaves=2, scale=3.0,
                       seed_offset=200, weight=0.1),
        ]

    normal_strength = 0.4

    def _height(self, layers: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        surface = unit(layers["surface"])
        pitting = unit(layers["pitting"])
        height = surface * 0.5 + pitting * 0.5
        return height

    def generate_roughness_map(self) -> Image.Image | None:
        layers = self._generate_noise_layers()
        surface = unit(layers["surface"])
        return self._create_roughness_from_variation(0.85, surface, variation_strength=0.1)
