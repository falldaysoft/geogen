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
    eye_v: float = 0.6
    eye_spacing: float = 0.42
    eye_size: float = 1.1
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
                d = np.hypot((u - (0.5 + side * 0.26)) / 0.11, (v - (self.eye_v - 0.19)) / 0.065)
                paint(np.clip(1 - d, 0, 1) ** 1.6, (222, 128, 124), 0.32 * self.blush)
        if self.freckles > 0:
            rng = np.random.default_rng(self.seed or 5)
            for _ in range(int(40 * self.freckles)):
                side = rng.choice([-1, 1])
                cu = 0.5 + side * rng.uniform(0.07, 0.3)
                cv = self.eye_v - rng.uniform(0.08, 0.2)
                paint(_ellipse(u, v, cu, cv, 0.007, 0.007, 0.006), (150, 96, 70), 0.4)
        for side in (-1, 1):
            cu = 0.5 + side * self.eye_spacing / 2
            cv = self.eye_v
            # Soft socket shadow above the eye.
            d = np.hypot((u - cu) / (0.12 * es), (v - (cv + 0.025 * es)) / (0.06 * es))
            paint(np.clip(1 - d, 0, 1) ** 1.2, (150, 104, 92), 0.18)
            # Almond eye: sclera under an arched upper lid, a large iris, pupil, two highlights.
            eye = _ellipse(u, v, cu, cv, 0.078 * es, 0.05 * es) * np.clip((v - (cv - 0.052 * es)) * 40, 0, 1)
            paint(eye, (246, 243, 238))
            iris = _ellipse(u, v, cu, cv - 0.004, 0.044 * es, 0.047 * es) * eye
            paint(iris, self.eye_color)
            ring = np.clip(_ellipse(u, v, cu, cv - 0.004, 0.044 * es, 0.047 * es)
                           - _ellipse(u, v, cu, cv - 0.004, 0.036 * es, 0.039 * es), 0, 1) * eye
            paint(ring, (40, 30, 26), 0.45)
            paint(_ellipse(u, v, cu, cv - 0.004, 0.02 * es, 0.021 * es) * eye, (16, 12, 12))
            paint(_ellipse(u, v, cu - side * 0.014 * es, cv + 0.014 * es, 0.009 * es, 0.009 * es), (255, 255, 255), 0.95)
            paint(_ellipse(u, v, cu + side * 0.012 * es, cv - 0.018 * es, 0.005 * es, 0.005 * es), (255, 255, 255), 0.6)
            # Upper lid: a smooth arc, heavier toward the outer corner; lashes flick out.
            arc = [(cu - side * 0.082 * es, cv - 0.006, 0.5)]
            for f in np.linspace(-0.8, 0.8, 7):
                arc.append((cu + side * f * 0.08 * es, cv + 0.05 * es * np.cos(f * np.pi / 2) ** 0.8, 0.7 + 0.3 * (f + 1) / 2))
            arc.append((cu + side * 0.084 * es, cv, 0.8))
            if self.lashes > 0:
                arc.append((cu + side * (0.084 + 0.028 * self.lashes) * es, cv + 0.018 * es * self.lashes, 0.25))
            paint(_stroke(u, v, arc, (0.01 + 0.012 * self.lashes) * es, soft=0.008), (38, 26, 24))
            # Faint lower lid and crease.
            paint(_stroke(u, v, [(cu - side * 0.06 * es, cv - 0.04 * es), (cu, cv - 0.05 * es),
                                 (cu + side * 0.07 * es, cv - 0.036 * es)], 0.006, soft=0.008), (120, 80, 72), 0.3)
            paint(_stroke(u, v, [(cu - side * 0.06 * es, cv + 0.068 * es), (cu, cv + 0.078 * es),
                                 (cu + side * 0.07 * es, cv + 0.066 * es)], 0.006, soft=0.01), (130, 90, 80), 0.25)
            # Brow: a gentle arc, thick at the inner end, tapering outward.
            bt, arch = self.brow_thickness, self.brow_arch
            base = self.eye_v + 0.105
            brow = []
            for f in np.linspace(0.0, 1.0, 7):
                x = cu + side * (-0.085 + 0.2 * f)
                y = base + 0.018 * arch * np.sin(np.pi * min(f / 0.7, 1.0) * 0.9) - 0.012 * f * (1 - arch * 0.3)
                brow.append((x, y, 1.0 - 0.65 * f))
            paint(_stroke(u, v, brow, 0.024 * bt, soft=0.01), self.brow_color, 0.92)
        # Nose: a soft shadow under the tip and two small nostrils.
        nose_v = (self.eye_v + self.mouth_v) / 2 - 0.03
        paint(_ellipse(u, v, 0.5, nose_v - 0.018, 0.065, 0.025, 0.04), (160, 108, 90), 0.16)
        for side in (-1, 1):
            paint(_ellipse(u, v, 0.5 + side * 0.028, nose_v - 0.01, 0.013, 0.008, 0.01), (110, 66, 56), 0.5)
        # Mouth: upper and lower lip toward lip_color, the corners lifted in a small smile.
        mw = 0.1 * self.mouth_width
        mv = self.mouth_v
        upper = _ellipse(u, v, 0.5, mv + 0.012, mw, 0.02) * np.clip((v - (mv + 0.001)) * 60, 0, 1)
        lower = _ellipse(u, v, 0.5, mv - 0.012, mw * 0.82, 0.028) * np.clip(((mv + 0.001) - v) * 60, 0, 1)
        lips = np.maximum(upper * 0.9, lower)
        lip = np.asarray(self.lip_color, dtype=np.float64) / 255.0
        target = skin * (1 - self.lip_strength) + lip[None, None, :] * self.lip_strength
        a = lips[..., None] * 0.92
        img = img * (1 - a) + target * a
        paint(_ellipse(u, v, 0.5, mv - 0.022, mw * 0.4, 0.008, 0.01), (255, 255, 255), 0.12 * self.lip_strength)
        paint(_stroke(u, v, [(0.5 - mw * 1.05, mv + 0.008, 0.4), (0.5 - mw * 0.5, mv - 0.001, 0.9), (0.5, mv, 1.0),
                             (0.5 + mw * 0.5, mv - 0.001, 0.9), (0.5 + mw * 1.05, mv + 0.008, 0.4)], 0.007, soft=0.008),
              (104, 48, 48), 0.75)
        return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


@dataclass
class HairTextureGenerator(TextureGenerator):
    """Strands along v (tiling): light neutral, tinted per character by vertex colour."""

    base_color: tuple[int, int, int] = (228, 222, 214)
    contrast: float = 0.18

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
