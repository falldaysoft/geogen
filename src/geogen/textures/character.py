"""Character textures: skin, a painted face decal, and hair strands.

Skin and face share one base (``SkinTextureGenerator``) so the face decal's
edge disappears into the body. Both are near-neutral: per-character skin tone
and hair colour come from vertex colours multiplying the albedo, so a crowd
shares a few materials.

The face is the one texture that doesn't tile: it covers the face region of
the head once (planar UVs over a face box, 0-1), painted in normalized face
coordinates (u across, v up from under the chin). Features are drawn large
and dark enough to read at ~5 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .base import TextureGenerator
from .noise import fractal_noise


def _norm(a: np.ndarray) -> np.ndarray:
    return (a - a.min()) / max(float(np.ptp(a)), 1e-9)


@dataclass
class SkinTextureGenerator(TextureGenerator):
    """Warm, faintly mottled skin (tiling)."""

    base_color: tuple[int, int, int] = (236, 204, 184)
    variation: float = 0.035

    def base(self) -> np.ndarray:
        noise = _norm(fractal_noise(self.width, self.height, octaves=3, scale=6.0,
                                    seed=self.seed if self.seed is not None else 11)) - 0.5
        rgb = np.asarray(self.base_color, dtype=np.float64)[None, None, :] / 255.0
        img = rgb * (1.0 + self.variation * noise[..., None] * np.array([1.0, 0.8, 0.8]))
        return np.clip(img, 0, 1)

    def generate(self) -> Image.Image:
        return Image.fromarray((self.base() * 255).astype(np.uint8))

    def generate_normal_map(self):
        return None


def _ellipse(u, v, cu, cv, ru, rv, soft=0.012):
    """Anti-aliased filled ellipse coverage (0..1)."""
    d = np.sqrt(((u - cu) / ru) ** 2 + ((v - cv) / rv) ** 2)
    return np.clip((1.0 - d) / (soft / min(ru, rv)) + 0.5, 0.0, 1.0)


def _stroke(u, v, points, width, soft=0.01):
    """Anti-aliased polyline coverage, ``width`` in u units (tapered by an optional 3rd coordinate)."""
    cover = np.zeros_like(u)
    for (u0, v0, *w0), (u1, v1, *w1) in zip(points[:-1], points[1:]):
        du, dv = u1 - u0, v1 - v0
        t = np.clip(((u - u0) * du + (v - v0) * dv) / max(du * du + dv * dv, 1e-12), 0, 1)
        dist = np.hypot(u - (u0 + t * du), v - (v0 + t * dv))
        half = width / 2 * ((w0[0] if w0 else 1.0) * (1 - t) + (w1[0] if w1 else 1.0) * t)
        cover = np.maximum(cover, np.clip((half - dist) / soft + 0.5, 0, 1))
    return cover


@dataclass
class FaceTextureGenerator(SkinTextureGenerator):
    """A stylized face painted over the skin base (not tiling: one face per texture).

    Layout (u across the face box, v up from under the chin): eyes at ``eye_v``,
    ``eye_spacing`` apart, brows above, nose shading, mouth at ``mouth_v``.
    """

    eye_color: tuple[int, int, int] = (84, 60, 40)
    eye_v: float = 0.61
    eye_spacing: float = 0.40
    eye_size: float = 1.0
    brow_color: tuple[int, int, int] = (70, 50, 36)
    brow_thickness: float = 1.0
    brow_arch: float = 1.0
    lashes: float = 0.0              # 0 .. 1: heavier upper lid line and lash flick
    lip_color: tuple[int, int, int] = (176, 96, 92)
    lip_strength: float = 0.5        # 0 = skin-coloured lips, 1 = full lip colour
    mouth_v: float = 0.24
    mouth_width: float = 1.0
    blush: float = 0.0
    freckles: float = 0.0

    def generate(self) -> Image.Image:
        h, w = self.height, self.width
        img = self.base()
        rows, cols = np.mgrid[0:h, 0:w]
        u = (cols + 0.5) / w
        v = 1.0 - (rows + 0.5) / h
        skin = img.copy()

        def paint(cover, color, alpha=1.0):
            nonlocal img
            c = np.asarray(color, dtype=np.float64)[None, None, :] / 255.0 if len(color) == 3 else color
            a = (cover * alpha)[..., None]
            img = img * (1 - a) + c * a

        es = self.eye_size
        if self.blush > 0:
            for side in (-1, 1):
                d = np.hypot((u - (0.5 + side * 0.25)) / 0.11, (v - (self.eye_v - 0.17)) / 0.07)
                paint(np.clip(1 - d, 0, 1) ** 1.5, (214, 120, 120), 0.35 * self.blush)
        if self.freckles > 0:
            rng = np.random.default_rng(self.seed or 5)
            for _ in range(int(40 * self.freckles)):
                side = rng.choice([-1, 1])
                cu = 0.5 + side * rng.uniform(0.08, 0.3)
                cv = self.eye_v - rng.uniform(0.07, 0.2)
                paint(_ellipse(u, v, cu, cv, 0.008, 0.008, 0.006), (150, 96, 70), 0.45)
        for side in (-1, 1):
            cu = 0.5 + side * self.eye_spacing / 2
            cv = self.eye_v
            # Sclera, iris, pupil, highlight.
            paint(_ellipse(u, v, cu, cv, 0.085 * es, 0.047 * es), (244, 240, 234))
            iris = _ellipse(u, v, cu, cv - 0.002, 0.047 * es, 0.047 * es) * _ellipse(u, v, cu, cv, 0.085 * es, 0.047 * es)
            paint(iris, self.eye_color)
            paint(_ellipse(u, v, cu, cv - 0.002, 0.022 * es, 0.022 * es), (18, 14, 12))
            paint(_ellipse(u, v, cu + side * -0.015 * es, cv + 0.015 * es, 0.01 * es, 0.01 * es), (255, 255, 255), 0.9)
            # Upper lid line (thicker with lashes, flicking out), faint lower lid.
            lid = [(cu - side * 0.09 * es, cv - 0.004, 0.6), (cu, cv + 0.048 * es, 1.0),
                   (cu + side * 0.09 * es, cv + 0.002, 0.7)]
            if self.lashes > 0:
                lid.append((cu + side * (0.09 + 0.03 * self.lashes) * es, cv + 0.02 * es * self.lashes, 0.2))
            paint(_stroke(u, v, lid, (0.012 + 0.014 * self.lashes) * es), (30, 22, 20))
            paint(_stroke(u, v, [(cu - side * 0.07 * es, cv - 0.03 * es), (cu + side * 0.07 * es, cv - 0.03 * es)],
                          0.006), (120, 80, 70), 0.35)
            # Brow: an arch over the eye, thick at the inner end.
            bt = self.brow_thickness
            arch = 0.025 * self.brow_arch
            brow = [(cu - side * 0.1, self.eye_v + 0.085, 1.0), (cu + side * 0.01, self.eye_v + 0.1 + arch, 0.9),
                    (cu + side * 0.11, self.eye_v + 0.08, 0.35)]
            paint(_stroke(u, v, brow, 0.022 * bt), self.brow_color, 0.95)
        # Nose: soft shading under the tip and two nostrils.
        nose_v = (self.eye_v + self.mouth_v) / 2 - 0.02
        paint(_ellipse(u, v, 0.5, nose_v - 0.02, 0.07, 0.03, 0.04), (150, 100, 80), 0.18)
        for side in (-1, 1):
            paint(_ellipse(u, v, 0.5 + side * 0.03, nose_v - 0.012, 0.016, 0.009, 0.01), (90, 55, 45), 0.6)
        # Mouth: lips tinted toward lip_color, a darker line between them.
        mw = 0.11 * self.mouth_width
        mv = self.mouth_v
        lips = np.maximum(_ellipse(u, v, 0.5, mv + 0.014, mw, 0.022), _ellipse(u, v, 0.5, mv - 0.016, mw * 0.85, 0.026))
        lip = np.asarray(self.lip_color, dtype=np.float64) / 255.0
        target = skin * (1 - self.lip_strength) + lip[None, None, :] * self.lip_strength
        a = lips[..., None] * 0.9
        img = img * (1 - a) + target * a
        paint(_stroke(u, v, [(0.5 - mw, mv + 0.004, 0.5), (0.5, mv - 0.002, 1.0), (0.5 + mw, mv + 0.004, 0.5)], 0.008),
              (96, 44, 44), 0.8)
        return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


@dataclass
class HairTextureGenerator(TextureGenerator):
    """Strands along v (tiling): light neutral, tinted per character by vertex colour."""

    base_color: tuple[int, int, int] = (228, 222, 214)
    contrast: float = 0.35

    def generate(self) -> Image.Image:
        rng = np.random.default_rng(self.seed if self.seed is not None else 3)
        cols = np.arange(self.width)
        strands = np.zeros(self.width)
        for freq, amp in ((3, 0.5), (11, 0.3), (37, 0.2), (97, 0.15)):
            phase = rng.uniform(0, 2 * np.pi, 3)
            for p in phase:
                strands += amp * np.sin(2 * np.pi * freq * cols / self.width + p) / 3
        strands = _norm(strands)
        along = _norm(fractal_noise(self.width, self.height, octaves=2, scale=3.0, seed=self.seed or 3))
        value = 1.0 - self.contrast * (0.7 * strands[None, :] + 0.3 * along)
        rgb = np.asarray(self.base_color, dtype=np.float64) / 255.0
        return Image.fromarray((np.clip(value[..., None] * rgb, 0, 1) * 255).astype(np.uint8))
