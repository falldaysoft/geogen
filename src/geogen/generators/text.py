"""Extruded 3D lettering (``primitive: text``).

Glyph outlines come from a TrueType font (``assets/fonts/``, OFL): quadratic
curves are flattened, each glyph's contours become :class:`Shape` regions
(outer contours unioned, counters subtracted, so overlapping contours and the
holes in A B O 8 & come out right) and every region is extruded with
:class:`ExtrudeGenerator`. Text reads along +X, up +Y, and faces +Z (the
extrusion runs along Z); the mesh is centred on its bounds, so parts without a
``size`` take their size from the laid-out text.

YAML::

    sign:
      primitive: text
      text: "Hotel\\nReception"   # newlines split lines; "{label}" with a `type: text` param
      font: sans                # sans | serif | a .ttf path under assets/fonts/
      height: 0.25              # cap height, metres
      depth: 0.03               # extrusion along Z
      align: center             # left | center | right (lines within the block)
      line_spacing: 1.0         # multiple of the font's line height
      tracking: 0.0             # extra letter spacing, em
      bevel: 0.0                # rounded cap edges (metres)
      material: brass
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ..core.mesh import Mesh
from ..core.profile import Shape
from .base import MeshGenerator
from .primitives import CubeGenerator
from .profiles import ExtrudeGenerator

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint
    from ..layout.surfaces import Surface

FONTS_DIR = Path(__file__).parents[3] / "assets" / "fonts"
FONTS = {"sans": "SourceSans3-Semibold.ttf", "serif": "SourceSerif4-Semibold.ttf"}
ALIGNS = ("left", "center", "right")


def font_path(font: str) -> Path:
    """The file for ``font``: a name in FONTS or a file under assets/fonts/."""
    path = FONTS_DIR / FONTS.get(font, font)
    if not path.exists():
        raise ValueError(f"Unknown font '{font}': use {sorted(FONTS)} or a file in {FONTS_DIR}")
    return path


class _Font:
    """A loaded font: metrics in font units and glyph regions, cached per glyph."""

    def __init__(self, path: Path, detail: float):
        from fontTools.ttLib import TTFont

        self.tt = TTFont(str(path))
        self.glyphs = self.tt.getGlyphSet()
        self.cmap = self.tt.getBestCmap()
        self.units = self.tt["head"].unitsPerEm
        hhea = self.tt["hhea"]
        self.line_height = hhea.ascent - hhea.descent + hhea.lineGap
        # Cap height as drawn (an H's top): OS/2 sCapHeight is only the design value.
        cap = self._bounds(self.cmap[ord("H")])[3] if ord("H") in self.cmap else 0
        self.cap_height = cap or getattr(self.tt["OS/2"], "sCapHeight", 0) or 0.7 * self.units
        self.detail = detail
        self._shapes: dict[str, list[Shape]] = {}

    def glyph_name(self, char: str) -> str | None:
        return self.cmap.get(ord(char))

    def advance(self, glyph: str) -> float:
        return float(self.tt["hmtx"][glyph][0])

    def _bounds(self, glyph: str):
        from fontTools.pens.boundsPen import BoundsPen

        pen = BoundsPen(self.glyphs)
        self.glyphs[glyph].draw(pen)
        return pen.bounds or (0, 0, 0, 0)

    def shapes(self, glyph: str) -> list[Shape]:
        """The glyph's filled regions in font units (baseline at y = 0)."""
        if glyph not in self._shapes:
            pen = _FlattenPen(self.glyphs, step=self.units * 0.035 / max(self.detail, 0.1))
            self.glyphs[glyph].draw(pen)
            self._shapes[glyph] = _regions(pen.contours)
        return self._shapes[glyph]


@lru_cache(maxsize=8)
def _load_font(path: str, detail: float) -> _Font:
    return _Font(Path(path), detail)


class _FlattenPen:
    """A fontTools pen collecting contours as point lists, curves flattened into ``step``-long pieces."""

    def __init__(self, glyph_set, step: float):
        from fontTools.pens.basePen import BasePen

        outer = self

        class Pen(BasePen):
            def _moveTo(self, pt):
                outer._current = [pt]

            def _lineTo(self, pt):
                outer._current.append(pt)

            def _curveToOne(self, p1, p2, p3):
                p0 = outer._current[-1]
                n = outer._segments(p0, p1, p2, p3)
                t = np.linspace(0, 1, n + 1)[1:, None]
                a, b, c, d = (np.array(p, dtype=float) for p in (p0, p1, p2, p3))
                pts = (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d
                outer._current.extend(map(tuple, pts))

            def _qCurveToOne(self, p1, p2):
                p0 = outer._current[-1]
                n = outer._segments(p0, p1, p2)
                t = np.linspace(0, 1, n + 1)[1:, None]
                a, b, c = (np.array(p, dtype=float) for p in (p0, p1, p2))
                pts = (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c
                outer._current.extend(map(tuple, pts))

            def _closePath(self):
                outer._finish()

            def _endPath(self):
                outer._finish()

        self.contours: list[np.ndarray] = []
        self._current: list = []
        self.step = step
        self._pen = Pen(glyph_set)

    def _segments(self, *points) -> int:
        pts = np.array(points, dtype=float)
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        return int(np.clip(np.ceil(length / self.step), 2, 16))

    def _finish(self):
        if len(self._current) >= 3:
            self.contours.append(np.array(self._current, dtype=float))
        self._current = []

    # The glyph draws into the inner BasePen.
    def __getattr__(self, name):
        return getattr(self._pen, name)


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _regions(contours: list[np.ndarray]) -> list[Shape]:
    """Filled regions from a glyph's contours: outers (the winding of the largest contour)
    unioned, counters (the other winding) cut out."""
    if not contours:
        return []
    areas = [_signed_area(c) for c in contours]
    outer_sign = np.sign(areas[int(np.argmax(np.abs(areas)))])
    outers = [Polygon(c).buffer(0) for c, a in zip(contours, areas) if np.sign(a) == outer_sign]
    holes = [Polygon(c).buffer(0) for c, a in zip(contours, areas) if np.sign(a) != outer_sign]
    region = unary_union(outers)
    if holes:
        region = region.difference(unary_union(holes))
    polys = list(region.geoms) if isinstance(region, MultiPolygon) else [region]
    shapes = []
    for poly in polys:
        if poly.is_empty or poly.area <= 0:
            continue
        poly = poly.simplify(1.0, preserve_topology=True)     # font units: drops near-collinear points
        shapes.append(Shape(np.array(poly.exterior.coords)[:-1],
                            [np.array(r.coords)[:-1] for r in poly.interiors]))
    return shapes


def layout(text: str, font: str = "sans", height: float = 0.2, align: str = "center",
           line_spacing: float = 1.0, tracking: float = 0.0, detail: float = 1.0) -> list[Shape]:
    """Lay ``text`` out in metres: [Shape] with the first line's baseline at y = 0 and x
    from the block's left edge (lines aligned within the block's width)."""
    if align not in ALIGNS:
        raise ValueError(f"text align must be one of {ALIGNS}, got {align!r}")
    f = _load_font(str(font_path(font)), float(detail))
    scale = height / f.cap_height
    lines = []
    for line in str(text).split("\n"):
        pen_x, placed = 0.0, []
        for char in line:
            glyph = f.glyph_name(char) or f.glyph_name(" ")
            if glyph is None:
                continue
            placed.append((glyph, pen_x))
            pen_x += f.advance(glyph) + tracking * f.units
        width = pen_x - (tracking * f.units if placed else 0.0)
        lines.append((placed, width))
    block = max((w for _, w in lines), default=0.0)
    shapes = []
    for row, (placed, width) in enumerate(lines):
        shift = {"left": 0.0, "center": (block - width) / 2, "right": block - width}[align]
        base_y = -row * f.line_height * line_spacing
        for glyph, x in placed:
            for shape in f.shapes(glyph):
                s = shape.translated(x + shift, base_y)
                shapes.append(Shape(s.outer * scale, [h * scale for h in s.holes]))
    return shapes


@dataclass
class TextGenerator(MeshGenerator):
    """Extruded lettering; see the module docstring."""

    text: str = "Text"
    font: str = "sans"
    height: float = 0.2
    depth: float = 0.02
    align: str = "center"
    line_spacing: float = 1.0
    tracking: float = 0.0
    bevel: float = 0.0
    detail: float = 1.0

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        return CubeGenerator().get_attachment_points(size)

    def get_surfaces(self, size: np.ndarray) -> dict[str, Surface]:
        return CubeGenerator().get_surfaces(size)

    def generate(self) -> Mesh:
        shapes = layout(self.text, self.font, self.height, self.align, self.line_spacing, self.tracking,
                        self.detail)
        if not shapes:
            raise ValueError(f"text {self.text!r} has no visible glyphs")
        meshes = [ExtrudeGenerator(shape=s, depth=self.depth, axis="z", bevel=self.bevel,
                                   bevel_segments=2).generate() for s in shapes]
        mesh = Mesh.merge(meshes)
        centre = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2
        return mesh.transform(np.array([[1, 0, 0, -centre[0]], [0, 1, 0, -centre[1]],
                                        [0, 0, 1, -centre[2]], [0, 0, 0, 1]], dtype=np.float64))
