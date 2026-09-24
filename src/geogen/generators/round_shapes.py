"""Rounded primitives built on the lathe: torus, capsule and bevelled cylinder.

All are closed, centred on the origin, fill their (x, y, z) size box on the
horizontal axes' smaller extent, and use metric UVs from the lathe. They
reuse the cylinder's attachment points (top/bottom/left/right/front/back).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh
from .base import MeshGenerator
from .primitives import CylinderGenerator
from .profiles import LatheGenerator


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int) -> np.ndarray:
    t = np.radians(np.linspace(a0, a1, n))
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


@dataclass
class TorusGenerator(MeshGenerator):
    """A ring: ``tube`` is the tube radius; the ring fills the x/z size.

    Attributes:
        size_x, size_y, size_z: Box to fill; the tube radius defaults to size_y / 2
        tube: Tube radius (0 = size_y / 2)
        segments: Steps around the ring; tube_segments: steps around the tube
    """

    size_x: float = 1.0
    size_y: float = 0.2
    size_z: float = 1.0
    tube: float = 0.0
    segments: int = 48
    tube_segments: int = 20

    def get_attachment_points(self, size):
        return CylinderGenerator().get_attachment_points(size)

    def generate(self) -> Mesh:
        tube = self.tube or self.size_y / 2
        outer = min(self.size_x, self.size_z) / 2
        major = outer - tube
        if major <= 0:
            raise ValueError("torus: tube radius must be smaller than half the ring's width")
        # Closed circle profile in (r, y), revolved; start/end at the inner equator.
        t = np.radians(np.linspace(180.0, -180.0, self.tube_segments + 1))
        profile = np.c_[major + tube * np.cos(t), tube * np.sin(t)]
        return LatheGenerator(profile=profile, segments=self.segments, cap_bottom=False, cap_top=False,
                              crease_angle=80.0).generate()


@dataclass
class CapsuleGenerator(MeshGenerator):
    """A cylinder with hemispherical ends filling the size box (radius = min(x, z) / 2)."""

    size_x: float = 0.5
    size_y: float = 1.5
    size_z: float = 0.5
    segments: int = 32
    cap_segments: int = 10

    def get_attachment_points(self, size):
        return CylinderGenerator().get_attachment_points(size)

    def generate(self) -> Mesh:
        r = min(self.size_x, self.size_z) / 2
        h = self.size_y
        r = min(r, h / 2)
        bottom = _arc(0.0, -h / 2 + r, r, -90.0, 0.0, self.cap_segments)
        top = _arc(0.0, h / 2 - r, r, 0.0, 90.0, self.cap_segments)
        profile = np.vstack([bottom, top[1:] if h / 2 - r <= -h / 2 + r + 1e-9 else top])
        profile[0, 0] = profile[-1, 0] = 0.0
        return LatheGenerator(profile=profile, segments=self.segments, crease_angle=80.0).generate()


@dataclass
class BevelledCylinderGenerator(CylinderGenerator):
    """Cylinder whose rims are rounded with radius ``bevel``."""

    bevel: float = 0.02
    bevel_segments: int = 4

    def generate(self) -> Mesh:
        r, h = self.radius, self.height
        b = float(np.clip(self.bevel, 0.0, min(r, h / 2) * 0.99))
        if b <= 0:
            return super().generate()
        n = self.bevel_segments + 1
        profile = np.vstack([
            [[0.0, -h / 2]],
            _arc(r - b, -h / 2 + b, b, -90.0, 0.0, n),
            _arc(r - b, h / 2 - b, b, 0.0, 90.0, n),
            [[0.0, h / 2]],
        ])
        return LatheGenerator(profile=profile, segments=self.segments, crease_angle=40.0).generate()

