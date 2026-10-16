"""Architectural panel cladding and painted hazard stripes.

``panel`` is large flat panels with thin dark seams: lab and sci-fi interiors
(white wall panels, dark metal floor tiles). Each panel gets its own tone and a
little grime that collects towards its lower edge, so a wall of them doesn't
read as one printed sheet. ``stripes`` is diagonal two-colour hazard striping.

Both are built on a [0, 1) lattice with integer repeat counts, so they tile
under metric UVs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .base import TextureGenerator
from .base import unit_grid as _unit_grid
from .interior import _to_gray, _to_image
from .noise import fractal_noise


@dataclass
class PanelTextureGenerator(TextureGenerator):
    """Flat cladding panels with seams, per-panel tone and grime.

    One texture repeat holds ``panels_x`` x ``panels_y`` panels, so a panel is
    the material's ``tile_size`` divided by those counts.

    Attributes:
        panel_color: Panel face colour as (R, G, B), 0-255
        seam_color: Seam (gap) colour as (R, G, B), 0-255
        panels_x: Panels across one texture repeat (integer)
        panels_y: Panels down one texture repeat (integer)
        layout: 'grid' or 'offset' (alternate rows shifted half a panel)
        seam_width: Seam width as a fraction of a panel's height
        color_variation: Per-panel brightness variation (0-1)
        tint_variation: Per-panel warm/cool shift (0-1)
        grime: Dirt over the face, heavier towards each panel's lower edge (0-1)
        edge_wear: Darkening along each panel's edges (0-1)
        bolts: Fasteners inset at each panel's corners (0 or 1)
        pillow: How much each face bulges between its seams in the normal map (0-1)
    """

    panel_color: tuple[int, int, int] = (226, 226, 222)
    seam_color: tuple[int, int, int] = (60, 62, 64)
    panels_x: int = 2
    panels_y: int = 1
    layout: str = "grid"
    seam_width: float = 0.012
    color_variation: float = 0.03
    tint_variation: float = 0.02
    grime: float = 0.3
    edge_wear: float = 0.3
    bolts: int = 0
    pillow: float = 0.2

    def _layout(self):
        """(edge distance, tile-space fx, fy, panel index), edge in panel-height units."""
        xv, yv = _unit_grid(self.width, self.height)
        nx, ny = max(1, int(self.panels_x)), max(1, int(self.panels_y))
        gy = yv * ny
        row = np.floor(gy).astype(np.int64)
        shift = (row % 2) * 0.5 if self.layout == "offset" and ny % 2 == 0 else 0.0
        gx = xv * nx + shift
        col = np.floor(gx).astype(np.int64)
        fx, fy = gx - col, gy - row
        aspect = ny / nx            # a panel's width in units of its height
        dx = np.minimum(fx, 1 - fx) * aspect
        dy = np.minimum(fy, 1 - fy)
        edge = np.minimum(dx, dy)
        index = (row % ny) * nx + (col % nx)
        return edge, fx * aspect, fy, index, aspect

    def _pixel(self) -> float:
        """One pixel in panel-height units."""
        return max(1, int(self.panels_y)) / self.height

    def _seam(self, edge: NDArray[np.float64]) -> NDArray[np.float64]:
        """1 in the seam, 0 on the face, with a pixel or two of chamfer."""
        half = self.seam_width / 2
        bevel = max(half * 0.8, 1.5 * self._pixel())
        return np.clip(1.0 - (edge - half) / bevel, 0, 1)

    def _bolts(self, fx, fy, aspect) -> NDArray[np.float64]:
        if not self.bolts:
            return np.zeros_like(fx)
        inset = 0.06
        radius = max(0.012, 2.0 * self._pixel())
        out = np.zeros_like(fx)
        for cx in (inset, aspect - inset):
            for cy in (inset, 1 - inset):
                d = np.hypot(fx - cx, fy - cy)
                out = np.maximum(out, np.clip(1.0 - (d - radius) / (0.6 * radius), 0, 1))
        return out

    def _grime(self, fy, index) -> NDArray[np.float64]:
        """0..1 dirt: mottled noise plus streaks gathering at each panel's foot."""
        if self.grime <= 0:
            return np.zeros_like(fy)
        mottle = fractal_noise(self.width, self.height, octaves=5, scale=8.0,
                               seed=self._seed_for(41)) * 0.5 + 0.5
        streak = fractal_noise(self.width, 4, octaves=3, scale=16.0,
                               seed=self._seed_for(43))
        streak = np.repeat(streak[:1], self.height, axis=0) * 0.5 + 0.5
        foot = np.clip((fy - 0.55) / 0.45, 0, 1) ** 2      # image y runs down the panel
        rng = np.random.default_rng(self._seed_for(47))
        dirty = 0.4 + rng.random(int(index.max()) + 1) * 0.6
        g = (np.clip(mottle - 0.45, 0, 1) * 1.4 + foot * streak * 0.8) * dirty[index]
        return np.clip(g * self.grime, 0, 1)

    def _height(self):
        edge, fx, fy, index, aspect = self._layout()
        seam = self._seam(edge)
        pillow = np.clip(edge / 0.25, 0, 1) ** 0.5 * self.pillow * 0.3
        height = 1.0 - seam * 0.7 + pillow - self._bolts(fx, fy, aspect) * 0.25
        return height, edge, fx, fy, index, aspect, seam

    def generate(self) -> Image.Image:
        _, edge, fx, fy, index, aspect, seam = self._height()
        rng = np.random.default_rng(self._seed_for(3))
        count = int(index.max()) + 1
        tone = 1.0 + (rng.random(count) - 0.5) * 2 * self.color_variation
        tint = (rng.random(count) - 0.5) * 2 * self.tint_variation
        warm = np.stack([1 + tint, np.ones(count), 1 - tint], axis=-1)

        face = np.array(self.panel_color, dtype=np.float64)[None, None, :]
        face = face * tone[index][..., None] * warm[index]
        fine = fractal_noise(self.width, self.height, octaves=4, scale=32.0,
                             seed=self._seed_for(31))
        face = face * (1.0 + fine[..., None] * 0.015)
        wear = np.clip(1.0 - edge / 0.04, 0, 1) * self.edge_wear * 0.12
        face = face * (1.0 - wear[..., None])
        face = face * (1.0 - self._grime(fy, index)[..., None] * 0.22)
        bolts = self._bolts(fx, fy, aspect)
        face = face * (1.0 - bolts[..., None] * 0.35)

        seam_rgb = np.array(self.seam_color, dtype=np.float64)[None, None, :]
        rgb = face * (1 - seam[..., None]) + seam_rgb * seam[..., None]
        return _to_image(rgb)

    def generate_normal_map(self) -> Image.Image:
        height = self._height()[0]
        return self._height_to_normal(height, strength=2.5)

    def generate_roughness_map(self) -> Image.Image:
        _, edge, fx, fy, index, aspect, seam = self._height()
        rng = np.random.default_rng(self._seed_for(5))
        per = 0.5 + (rng.random(int(index.max()) + 1) - 0.5) * 0.15
        rough = per[index] + self._grime(fy, index) * 0.25 + seam * 0.4
        return _to_gray(rough)

    def generate_ao_map(self) -> Image.Image:
        _, edge, fx, fy, index, aspect, seam = self._height()
        return _to_gray(1.0 - seam * 0.5 - self._bolts(fx, fy, aspect) * 0.2)


@dataclass
class StripeTextureGenerator(TextureGenerator):
    """Diagonal two-colour stripes (hazard paint), lightly scuffed.

    Attributes:
        color_a: First stripe colour as (R, G, B), 0-255
        color_b: Second stripe colour as (R, G, B), 0-255
        stripes: Stripe pairs across one texture repeat (integer)
        direction: 1 for '/' stripes, -1 for '\\'
        wear: Scuffing that lets the darker colour show through (0-1)
    """

    color_a: tuple[int, int, int] = (230, 180, 30)
    color_b: tuple[int, int, int] = (30, 30, 32)
    stripes: int = 4
    direction: int = 1
    wear: float = 0.25

    def _band(self) -> NDArray[np.float64]:
        xv, yv = _unit_grid(self.width, self.height)
        s = (xv + (1 if self.direction >= 0 else -1) * yv) * max(1, int(self.stripes))
        f = s - np.floor(s)
        px = max(1, int(self.stripes)) * 2.0 / self.width
        return np.clip((np.abs(f - 0.5) - 0.25) / px + 0.5, 0, 1)   # 1 = colour a

    def generate(self) -> Image.Image:
        band = self._band()
        scuff = fractal_noise(self.width, self.height, octaves=5, scale=16.0,
                              seed=self._seed_for(7)) * 0.5 + 0.5
        band = band * (1 - np.clip((scuff - (1 - self.wear * 0.5)) * 6, 0, 1))
        a = np.array(self.color_a, dtype=np.float64)[None, None, :]
        b = np.array(self.color_b, dtype=np.float64)[None, None, :]
        rgb = b + (a - b) * band[..., None]
        return _to_image(rgb * (0.95 + scuff[..., None] * 0.08))

    def generate_normal_map(self) -> Image.Image:
        return self._height_to_normal(self._band() * 0.2, strength=1.0)
