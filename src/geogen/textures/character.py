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


def _curve_band(x, y, xs, top, bottom, soft):
    """Coverage between two curves ``top(x)`` and ``bottom(x)`` for ``|x| < xs`` (soft edges)."""
    inside = np.minimum(top - y, y - bottom)
    ends = xs - np.abs(x)
    return np.clip(np.minimum(inside, ends) / soft + 0.5, 0.0, 1.0)


class _Canvas:
    """An RGB image (0..1) in face units: x across from the centre line and y up from under the
    chin, both in face widths (so circles are round whatever the face box's ``aspect``)."""

    def __init__(self, img: np.ndarray, aspect: float) -> None:
        h, w = img.shape[:2]
        rows, cols = np.mgrid[0:h, 0:w]
        self.img, self.aspect = img, aspect
        self.x = (cols + 0.5) / w - 0.5
        self.y = (1.0 - (rows + 0.5) / h) * aspect
        self.px = 1.0 / w

    def paint(self, cover, color, alpha=1.0) -> None:
        """Lay ``color`` (0-255 RGB, or an image of 0..1 RGB) over ``cover``."""
        c = np.asarray(color, dtype=np.float64)
        c = c[None, None, :] / 255.0 if c.ndim == 1 else c
        a = np.clip(cover * alpha, 0, 1)[..., None]
        self.img = self.img * (1 - a) + c * a

    def shade(self, cover, factor: float) -> None:
        """Darken (``factor`` < 1) the image under ``cover``."""
        a = np.clip(cover, 0, 1)[..., None]
        self.img = self.img * (1 - a + a * factor)

    def stroke(self, points, width: float, soft: float | None = None) -> np.ndarray:
        """``_stroke`` coverage of a polyline given in face units (``width`` across, in face widths)."""
        uv = [(px + 0.5, py / self.aspect, *rest) for px, py, *rest in points]
        return _stroke(self.x + 0.5, self.y / self.aspect, uv, width, soft or 1.5 * self.px)

    def stroke_paint(self, points, width: float, color, alpha: float = 1.0) -> None:
        """Paint a short stroke, working only in the window round it (brow hairs, lash flicks)."""
        h, w = self.img.shape[:2]
        pts = np.array([pt[:2] for pt in points])
        pad = width + 3 * self.px
        c0 = max(int((pts[:, 0].min() - pad + 0.5) * w), 0)
        c1 = min(int((pts[:, 0].max() + pad + 0.5) * w) + 2, w)
        r0 = max(int((1 - (pts[:, 1].max() + pad) / self.aspect) * h), 0)
        r1 = min(int((1 - (pts[:, 1].min() - pad) / self.aspect) * h) + 2, h)
        if c0 >= c1 or r0 >= r1:
            return
        win = (slice(r0, r1), slice(c0, c1))
        uv = [(px + 0.5, py / self.aspect, *rest) for px, py, *rest in points]
        cover = _stroke(self.x[win] + 0.5, self.y[win] / self.aspect, uv, width, 1.2 * self.px)
        a = np.clip(cover * alpha, 0, 1)[..., None]
        self.img[win] = self.img[win] * (1 - a) + np.asarray(color, dtype=np.float64) / 255.0 * a

    def blob(self, cx, cy, rx, ry) -> np.ndarray:
        """A soft cone falling from 1 at the centre to 0 at the ellipse's edge."""
        return np.clip(1 - np.hypot((self.x - cx) / rx, (self.y - cy) / ry), 0, 1)

    def disc(self, cx, cy, r) -> np.ndarray:
        """An anti-aliased filled circle."""
        return np.clip((r - np.hypot(self.x - cx, self.y - cy)) / (1.2 * self.px) + 0.5, 0, 1)


@dataclass
class FaceTextureGenerator(SkinTextureGenerator):
    """A stylized face painted over the skin base (not tiling: one face per texture).

    Layout: u across the face box, v up from under the chin. Features are placed by ``eye_v``,
    ``eye_spacing`` (of the box's width) and ``mouth_v`` (``generators/head.py`` puts the sculpted
    sockets, nose and lips there too) and drawn in true proportions: ``aspect`` is the box's
    height over its width. The sculpt supplies the shapes of the nose, brow and lips, so the paint
    is the eyes, brows, lip colour and what age, blush, freckles and stubble add.
    """

    eye_color: tuple[int, int, int] = (84, 60, 40)
    eye_v: float = 0.6
    eye_spacing: float = 0.42
    eye_size: float = 1.1
    aspect: float = 1.35
    brow_color: tuple[int, int, int] = (70, 50, 36)
    brow_thickness: float = 1.0
    brow_arch: float = 1.0
    lashes: float = 0.0              # 0 .. 1: heavier upper lid line and lash flicks
    lip_color: tuple[int, int, int] = (176, 96, 92)
    lip_strength: float = 0.5        # 0 = skin-coloured lips, 1 = full lip colour
    mouth_v: float = 0.24
    mouth_width: float = 1.0
    smile: float = 0.4               # mouth corners lifted
    blush: float = 0.0
    freckles: float = 0.0
    age: float = 0.0                 # 0 .. 1: smile lines, crow's feet, under-eye and forehead lines
    stubble: float = 0.0             # 0 .. 1: a shadow over the beard area

    def generate(self) -> Image.Image:
        c = _Canvas(self.base(), self.aspect)
        skin = c.img.copy()
        rng = np.random.default_rng(self.seed if self.seed is not None else 5)
        ey, my = self.eye_v * self.aspect, self.mouth_v * self.aspect
        nose_y = (ey + my) / 2 - 0.02
        if self.blush > 0:
            for side in (-1, 1):
                c.paint(c.blob(side * 0.25, ey - 0.2, 0.13, 0.08) ** 1.5, (222, 128, 124), 0.3 * self.blush)
            c.paint(c.blob(0, nose_y + 0.03, 0.05, 0.05) ** 1.5, (222, 128, 124), 0.12 * self.blush)
        if self.stubble > 0:
            self._stubble(c, my, rng)
        if self.freckles > 0:
            for _ in range(int(60 * self.freckles)):
                side = rng.choice([-1, 1])
                r = rng.uniform(0.004, 0.008)
                c.paint(c.blob(side * rng.uniform(0.03, 0.32), ey - rng.uniform(0.06, 0.24), r, r) ** 0.6,
                        (150, 96, 70), rng.uniform(0.25, 0.45))
        if self.age > 0:
            self._age(c, ey, my, nose_y)
        for side in (-1, 1):
            self._eye(c, side * self.eye_spacing / 2, ey, side)
            self._brow(c, side * self.eye_spacing / 2, ey, side, rng)
        self._mouth(c, my, skin)
        return Image.fromarray((np.clip(c.img, 0, 1) * 255).astype(np.uint8))

    def _stubble(self, c: _Canvas, my: float, rng) -> None:
        """A grainy shadow over the jaw, chin and upper lip."""
        jaw = np.clip((my + 0.09 - c.y) / 0.08, 0, 1) * np.clip(1 - np.abs(c.x) / 0.48, 0, 1)
        lip = c.blob(0, my + 0.045, 0.12, 0.028) ** 0.4
        grain = rng.random(c.x.shape) * 0.6 + 0.4
        c.paint(np.maximum(jaw, lip) * grain, (66, 54, 48), 0.32 * self.stubble)

    def _age(self, c: _Canvas, ey: float, my: float, nose_y: float) -> None:
        a, es = self.age, self.eye_size
        for side in (-1, 1):
            # Smile lines from the nose wings round the corners of the mouth.
            # (a bow outward from the nose's wing, ending past the corner of the mouth)
            fold = [(side * (0.08 + 0.05 * f + 0.035 * np.sin(np.pi * f)), nose_y - (nose_y - my + 0.03) * f,
                     0.3 + 0.7 * np.sin(np.pi * f)) for f in np.linspace(0, 1, 9)]
            c.paint(c.stroke(fold, 0.014, 0.014), (150, 100, 86), 0.22 * a)
            # Crow's feet at the outer corner, a line under the eye.
            corner = side * (self.eye_spacing / 2 + 0.08 * es)
            for k, dy in enumerate((-0.016, 0.0, 0.018)):
                c.paint(c.stroke([(corner, ey + dy * 0.4, 0.4), (corner + side * 0.05, ey + dy * 1.8, 1.0)], 0.004),
                        (140, 92, 80), (0.35 - 0.06 * k) * a)
            cx = side * self.eye_spacing / 2
            bag = [(cx + side * f * 0.06 * es, ey - (0.05 + 0.016 * (1 - f * f)) * es, 0.3 + 0.7 * (1 - f * f))
                   for f in np.linspace(-0.9, 1.0, 9)]
            c.paint(c.stroke(bag, 0.006), (150, 100, 88), 0.3 * a)
        for k in range(3):
            fy = ey + 0.21 + 0.035 * k
            c.paint(c.stroke([(-0.16, fy - 0.006, 0.3), (0, fy, 1.0), (0.16, fy - 0.006, 0.3)], 0.004), (160, 112, 98),
                    0.25 * a * (1 - 0.25 * k))

    def _eye(self, c: _Canvas, cx: float, cy: float, side: int) -> None:
        es, x, y, px = self.eye_size, c.x, c.y, c.px
        half_w, up, down = 0.084 * es, 0.044 * es, 0.03 * es
        tilt = 0.008 * es                                          # the outer corner a little higher

        def upper(f):            # the upper lid peaks a little inward of centre
            return cy + tilt * f + up * (1 - f * f) ** 0.75 * (1 - 0.15 * f)

        def lower(f):            # the lower lid's low point a little outward
            return cy + tilt * f - down * (1 - f * f) ** 0.9 * (1 + 0.1 * f)

        t = np.clip((x - cx) * side / half_w, -1, 1)               # + toward the outer corner
        top, bottom = upper(t), lower(t)
        opening = _curve_band((x - cx) * side, y, half_w, top, bottom, 1.2 * px)
        # The lid up to the crease, a little shadowed.
        lid = _curve_band((x - cx) * side, y, half_w * 1.1, top + 0.026 * es, top - 0.004, 0.012) * (1 - opening)
        c.shade(lid * 0.6, 0.87)
        # Sclera: off-white, shadowed under the upper lid and into the corners.
        c.paint(opening, (240, 236, 230))
        c.paint(opening * (np.clip(1 - (top - y) / (0.022 * es), 0, 1) * 0.35 + np.abs(t) ** 3 * 0.25),
                (170, 150, 140), 0.6)
        # Iris, partly under the upper lid: a dark limbal ring, lighter toward a dark pupil, radial fibres.
        ir, iy = 0.037 * es, cy + 0.004 * es
        d = np.hypot(x - cx, y - iy) / ir
        iris = np.clip((1 - d) * ir / (1.2 * px), 0, 1) * opening
        angle = np.arctan2(y - iy, x - cx)
        fibres = 0.5 + 0.5 * np.sin(angle * 23 + 3 * np.sin(angle * 7))
        col = np.asarray(self.eye_color, dtype=np.float64) / 255.0
        light = np.clip(col * 1.5 + 0.08, 0, 1)
        mix = (np.clip(1 - d, 0, 1) * 0.7 + 0.15 * fibres)[..., None]
        rgb = col * (1 - mix) + light * mix
        rgb = rgb * (1 - 0.55 * np.clip((d - 0.72) / 0.28, 0, 1))[..., None]
        rgb = rgb * (1 - 0.35 * np.clip(1 - (top - y) / (0.02 * es), 0, 1))[..., None]
        c.paint(iris, rgb)
        c.paint(c.disc(cx, iy, 0.4 * ir) * opening, (14, 10, 10))
        # Catch-lights, on the same side in both eyes (one light).
        c.paint(c.disc(cx - 0.36 * ir, iy + 0.38 * ir, 0.2 * ir) * opening, (255, 255, 255), 0.95)
        c.paint(c.disc(cx + 0.32 * ir, iy - 0.36 * ir, 0.09 * ir) * opening, (255, 255, 255), 0.55)
        # The inner corner: a touch of pink.
        c.paint(c.blob(cx - side * half_w * 0.9, lower(-0.9) + 0.006, 0.012 * es, 0.01 * es) * opening,
                (214, 140, 136), 0.8)
        # The upper lid line, heavier toward the outer corner; lashes flick out past it.
        lash = self.lashes
        line = [(cx + side * f * half_w, upper(f) + 0.003, 0.55 + 0.45 * (f + 1) / 2) for f in np.linspace(-1, 1, 13)]
        if lash > 0:
            line.append((cx + side * (half_w + 0.02 * lash), upper(1.0) + 0.012 * lash, 0.25))
        c.paint(c.stroke(line, (0.01 + 0.008 * lash) * es), (34, 24, 22))
        for f in (0.35, 0.6, 0.85) if lash > 0 else ():
            bx, by = cx + side * f * half_w, upper(f)
            c.stroke_paint([(bx, by, 1.0), (bx + side * 0.012 * lash, by + 0.013 * lash, 0.2)], 0.004 * es,
                           (34, 24, 22), 0.8 * lash)
        # The lid crease above, a fainter lower lid line.
        crease = [(cx + side * f * half_w * 0.95, upper(f) + 0.017 * es * (1 - f * f) ** 0.6, 0.3 + 0.7 * (1 - f * f))
                  for f in np.linspace(-0.9, 0.9, 9)]
        c.paint(c.stroke(crease, 0.004 * es), (120, 82, 74), 0.4)
        bottom_line = [(cx + side * f * half_w, lower(f) - 0.002, 0.3 + 0.7 * (f + 1) / 2) for f in np.linspace(-0.85, 1, 9)]
        c.paint(c.stroke(bottom_line, 0.004 * es), (110, 72, 66), 0.45 + 0.2 * lash)

    def _brow(self, c: _Canvas, cx: float, cy: float, side: int, rng) -> None:
        """Short hair strokes over a soft fill: a thick inner end, tapering outward."""
        bt, arch = self.brow_thickness, self.brow_arch
        inner, outer = cx - side * 0.07, cx + side * 0.1

        def along(f):
            by = (cy + 0.088 + 0.022 * arch * np.sin(np.pi * min(f / 0.68, 1.0) * 0.85)
                  - 0.02 * max(f - 0.68, 0) / 0.32)
            return inner + (outer - inner) * f, by

        c.paint(c.stroke([(*along(f), 1.0 - 0.6 * f) for f in np.linspace(0, 1, 9)], 0.024 * bt, 3 * c.px),
                self.brow_color, 0.6)
        for _ in range(int(110 * bt)):
            f = rng.uniform(0, 1)
            bx, by = along(f)
            by += rng.uniform(-0.5, 0.5) * 0.024 * bt * (1.0 - 0.6 * f)
            # The inner hairs grow up, the rest along the brow toward its outer end.
            ang = np.radians(80 if f < 0.15 else 20 - 25 * f)
            length = 0.014 + 0.008 * rng.random()
            tip = (bx + side * np.cos(ang) * length, by + np.sin(ang) * length, 0.3)
            c.stroke_paint([(bx, by, 1.0), tip], 0.004, self.brow_color, 0.75)

    def _mouth(self, c: _Canvas, my: float, skin: np.ndarray) -> None:
        """Lip colour over the sculpted lips: a cupid's bow on top, a fuller lower lip, soft corners."""
        x, y, px = c.x, c.y, c.px
        mw = 0.115 * self.mouth_width
        lift = 0.01 * self.smile
        t = np.clip(x / mw, -1, 1)
        line = my + lift * t * t
        top = (line + 0.03 * (1 - t * t) ** 0.6 + 0.006 * np.exp(-((np.abs(x) - 0.032) / 0.022) ** 2)
               - 0.008 * np.exp(-(x / 0.015) ** 2))
        bottom = line - 0.04 * (1 - np.clip(x / (0.85 * mw), -1, 1) ** 2) ** 0.55
        upper = _curve_band(x, y, mw, top, line, 1.5 * px)
        lower = _curve_band(x, y, 0.88 * mw, line, bottom, 1.5 * px)
        lip = np.asarray(self.lip_color, dtype=np.float64) / 255.0
        target = skin * (1 - self.lip_strength) + lip[None, None, :] * self.lip_strength
        c.paint(upper * 0.85, target * 0.94)
        c.paint(lower * 0.9, target * 1.04)
        c.paint(c.blob(0, my - 0.02, 0.04, 0.008), (255, 255, 255), 0.18 * self.lip_strength)
        line_pts = [(f * mw, my + lift * min(f * f, 1.1) - 0.001, 0.35 + 0.65 * (1 - min(abs(f), 1) ** 2))
                    for f in np.linspace(-1.05, 1.05, 15)]
        c.paint(c.stroke(line_pts, 0.006), (104, 50, 50), 0.7)


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
