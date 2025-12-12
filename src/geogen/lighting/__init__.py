"""Lighting classes for scene illumination."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import numpy as np


class LightType(IntEnum):
    """Light type enum matching shader values."""

    DIRECTIONAL = 0
    POINT = 1


@dataclass
class Light:
    """Base light class."""

    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0

    @property
    def light_type(self) -> LightType:
        raise NotImplementedError


@dataclass
class DirectionalLight(Light):
    """Sun-like directional light.

    The direction points FROM the light source (like sun rays).
    """

    direction: tuple[float, float, float] = (0.0, -1.0, 0.0)

    @property
    def light_type(self) -> LightType:
        return LightType.DIRECTIONAL

    @property
    def position_or_direction(self) -> tuple[float, float, float]:
        """Return direction for shader uniform."""
        return self.direction


@dataclass
class PointLight(Light):
    """Omni-directional point light."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def light_type(self) -> LightType:
        return LightType.POINT

    @property
    def position_or_direction(self) -> tuple[float, float, float]:
        """Return position for shader uniform."""
        return self.position


@dataclass
class SceneLighting:
    """Lighting configuration for a scene."""

    ambient_color: tuple[float, float, float] = (0.15, 0.15, 0.18)
    lights: list[Light] = field(default_factory=list)

    @classmethod
    def default(cls) -> SceneLighting:
        """Create default lighting setup."""
        return cls(
            ambient_color=(0.2, 0.2, 0.22),
            lights=[
                DirectionalLight(
                    direction=(0.5, -0.7, 0.3),
                    color=(1.0, 0.98, 0.95),
                    intensity=0.8,
                ),
            ],
        )

    @classmethod
    def room_lighting(cls) -> SceneLighting:
        """Create lighting setup optimized for indoor room scenes."""
        return cls(
            ambient_color=(0.25, 0.25, 0.28),  # Higher ambient for indoor
            lights=[
                # Main ceiling light - increased intensity for PBR
                PointLight(
                    position=(0.0, 2.0, 0.0),
                    color=(1.0, 0.95, 0.9),
                    intensity=8.0,  # Much higher for PBR
                ),
                # Fill light from "window" direction
                DirectionalLight(
                    direction=(0.6, -0.4, -0.7),
                    color=(0.8, 0.85, 1.0),
                    intensity=2.0,  # Higher for PBR
                ),
                # Subtle back fill
                DirectionalLight(
                    direction=(-0.5, -0.3, 0.5),
                    color=(0.9, 0.85, 0.8),
                    intensity=1.0,
                ),
            ],
        )

    def get_shader_data(self) -> dict:
        """Get lighting data formatted for shader uniforms."""
        max_lights = 4
        light_count = min(len(self.lights), max_lights)

        types = [0] * max_lights
        positions = [(0.0, 0.0, 0.0)] * max_lights
        colors = [(0.0, 0.0, 0.0)] * max_lights
        intensities = [0.0] * max_lights

        for i, light in enumerate(self.lights[:max_lights]):
            types[i] = int(light.light_type)
            positions[i] = light.position_or_direction
            colors[i] = light.color
            intensities[i] = light.intensity

        return {
            "uAmbientColor": self.ambient_color,
            "uLightCount": light_count,
            "uLightTypes": types,
            "uLightPositions": positions,
            "uLightColors": colors,
            "uLightIntensities": intensities,
        }


__all__ = [
    "Light",
    "LightType",
    "DirectionalLight",
    "PointLight",
    "SceneLighting",
]
