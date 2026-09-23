"""2D profiles and shapes: the input to extrude, lathe and sweep generators.

A *loop* is an (N, 2) float array describing a closed polygon without a
repeated end point. A :class:`Shape` is an outer loop plus optional hole
loops; orientation is normalised so the outer loop is counter-clockwise and
holes are clockwise. An open *polyline* (used for lathe profiles) is an
(N, 2) array whose ends are not joined.

Curved inputs (arcs, fillets, splines) are sampled into dense polylines;
generators decide hard vs smooth shading afterwards with a crease angle, so
profiles never need per-vertex smoothing flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

Loop = NDArray[np.float64]


# --------------------------------------------------------------------------
# Basic loop utilities


def signed_area(loop: Loop) -> float:
    """Signed area (positive for counter-clockwise loops)."""
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def ensure_ccw(loop: Loop) -> Loop:
    return loop if signed_area(loop) >= 0 else loop[::-1].copy()


def ensure_cw(loop: Loop) -> Loop:
    return loop if signed_area(loop) <= 0 else loop[::-1].copy()


def dedupe(points: NDArray, closed: bool, tol: float = 1e-9) -> NDArray:
    """Drop consecutive duplicate points (and the closing duplicate for loops)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > tol
    pts = pts[keep]
    if closed and len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= tol:
        pts = pts[:-1]
    return pts


def perimeter(loop: Loop, closed: bool = True) -> float:
    pts = np.vstack([loop, loop[:1]]) if closed else loop
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def arc_lengths(points: NDArray, closed: bool = False) -> NDArray:
    """Cumulative distance along a polyline (length N, or N+1 if closed)."""
    pts = np.vstack([points, points[:1]]) if closed else points
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


# --------------------------------------------------------------------------
# Primitive loops


def circle(radius: float, segments: int = 32, center=(0.0, 0.0)) -> Loop:
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    return np.column_stack([np.cos(t) * radius + center[0], np.sin(t) * radius + center[1]])


def ellipse(rx: float, ry: float, segments: int = 32, center=(0.0, 0.0)) -> Loop:
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    return np.column_stack([np.cos(t) * rx + center[0], np.sin(t) * ry + center[1]])


def regular_polygon(radius: float, sides: int, rotation: float = 0.0) -> Loop:
    t = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.radians(rotation)
    return np.column_stack([np.cos(t) * radius, np.sin(t) * radius])


def rect(width: float, height: float, radius: float = 0.0, segments: int = 6, center=(0.0, 0.0)) -> Loop:
    """Axis-aligned rectangle centred on ``center``, with optional rounded corners."""
    hw, hh = width / 2, height / 2
    loop = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]) + np.asarray(center)
    return fillet(loop, radius, segments) if radius > 0 else loop


def arc(center, radius: float, start_deg: float, end_deg: float, segments: int = 16) -> NDArray:
    """Open polyline along a circular arc (inclusive of both ends)."""
    t = np.radians(np.linspace(start_deg, end_deg, segments + 1))
    return np.column_stack([center[0] + np.cos(t) * radius, center[1] + np.sin(t) * radius])


def bezier(p0, p1, p2, p3, segments: int = 16) -> NDArray:
    """Open polyline along a cubic Bézier curve (inclusive of both ends)."""
    t = np.linspace(0, 1, segments + 1)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def catmull_rom(points: NDArray, samples: int = 8, closed: bool = False) -> NDArray:
    """Smooth centripetal Catmull-Rom spline through ``points``.

    Useful for turned-wood and vase profiles: give a handful of control
    points and get a smooth curve that passes exactly through each one.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return pts.copy()
    if closed:
        ext = np.vstack([pts[-1:], pts, pts[:2]])
    else:
        ext = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    n_seg = len(pts) if closed else len(pts) - 1
    for i in range(n_seg):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0) ** 0.5, 1e-9)
        t2 = t1 + max(np.linalg.norm(p2 - p1) ** 0.5, 1e-9)
        t3 = t2 + max(np.linalg.norm(p3 - p2) ** 0.5, 1e-9)
        last = (i == n_seg - 1) and not closed
        for t in np.linspace(t1, t2, samples + 1)[: None if last else -1]:
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.array(out)


# --------------------------------------------------------------------------
# Loop operations


def fillet(loop: Loop, radius: float | Sequence[float], segments: int = 6) -> Loop:
    """Round every corner of a closed loop with a circular arc.

    ``radius`` may be a scalar or one value per corner (0 keeps the corner
    sharp). Radii are clamped so adjacent fillets never overlap.
    """
    pts = np.asarray(loop, dtype=np.float64)
    n = len(pts)
    radii = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n,)).copy()
    prev_pts, next_pts = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
    len_prev = np.linalg.norm(pts - prev_pts, axis=1)
    len_next = np.linalg.norm(next_pts - pts, axis=1)

    out = []
    for i in range(n):
        p, a, b = pts[i], prev_pts[i], next_pts[i]
        r = radii[i]
        d1 = (a - p) / max(len_prev[i], 1e-12)
        d2 = (b - p) / max(len_next[i], 1e-12)
        cos_theta = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
        theta = np.arccos(cos_theta)  # interior angle between the two edges
        if r <= 0 or theta < 1e-6 or abs(np.pi - theta) < 1e-6:
            out.append(p)
            continue
        tangent_dist = r / np.tan(theta / 2)
        # Never consume more than half of either adjacent edge.
        limit = 0.5 * min(len_prev[i], len_next[i])
        if tangent_dist > limit:
            tangent_dist = limit
            r = tangent_dist * np.tan(theta / 2)
        start = p + d1 * tangent_dist
        end = p + d2 * tangent_dist
        bisector = d1 + d2
        bisector /= np.linalg.norm(bisector)
        center = p + bisector * (r / np.sin(theta / 2))
        a0 = np.arctan2(*(start - center)[::-1])
        a1 = np.arctan2(*(end - center)[::-1])
        sweep = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        for t in np.linspace(0, 1, segments + 1):
            ang = a0 + sweep * t
            out.append(center + r * np.array([np.cos(ang), np.sin(ang)]))
    return dedupe(np.array(out), closed=True)


def offset(loop: Loop, distance: float, join: str = "miter", resolution: int = 8) -> list[Loop]:
    """Grow (positive) or shrink (negative) a closed loop.

    Uses shapely's robust buffer, so self-intersections are resolved and a
    shrink may split into several loops (or vanish, returning []).
    """
    from shapely.geometry import Polygon

    join_style = {"round": "round", "miter": "mitre", "bevel": "bevel"}[join]
    geom = Polygon(loop).buffer(distance, join_style=join_style, quad_segs=resolution, mitre_limit=5.0)
    polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    return [ensure_ccw(dedupe(np.asarray(p.exterior.coords), closed=True)) for p in polys if not p.is_empty]


def resample(points: NDArray, spacing: float, closed: bool = False) -> NDArray:
    """Resample a polyline so points are at most ``spacing`` apart, keeping corners."""
    pts = np.asarray(points, dtype=np.float64)
    seq = np.vstack([pts, pts[:1]]) if closed else pts
    out = [seq[0]]
    for a, b in zip(seq[:-1], seq[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + (b - a) * t)
    out = np.array(out)
    return out[:-1] if closed else out


# --------------------------------------------------------------------------
# Shapes


@dataclass
class Shape:
    """A planar region: outer boundary plus holes."""

    outer: Loop
    holes: list[Loop] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.outer = ensure_ccw(dedupe(self.outer, closed=True))
        self.holes = [ensure_cw(dedupe(h, closed=True)) for h in self.holes]

    @property
    def loops(self) -> list[Loop]:
        return [self.outer, *self.holes]

    @property
    def bounds(self) -> NDArray:
        return np.array([self.outer.min(axis=0), self.outer.max(axis=0)])

    @property
    def area(self) -> float:
        return signed_area(self.outer) + sum(signed_area(h) for h in self.holes)

    def triangulate(self) -> tuple[NDArray, NDArray]:
        """Return (points (N,2), triangles (M,3)) with CCW triangles."""
        return triangulate(self.outer, self.holes)

    def translated(self, dx: float, dy: float) -> Shape:
        d = np.array([dx, dy])
        return Shape(self.outer + d, [h + d for h in self.holes])

    def offset(self, distance: float, join: str = "miter") -> list[Shape]:
        """Offset the whole region (outer grows, holes shrink for positive distance)."""
        from shapely.geometry import Polygon

        join_style = {"round": "round", "miter": "mitre", "bevel": "bevel"}[join]
        geom = Polygon(self.outer, self.holes).buffer(distance, join_style=join_style, mitre_limit=5.0)
        polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        return [
            Shape(np.asarray(p.exterior.coords), [np.asarray(i.coords) for i in p.interiors])
            for p in polys if not p.is_empty
        ]

    def difference(self, other: Shape) -> list[Shape]:
        """Subtract ``other`` from this shape (e.g. cut a window out of a wall)."""
        from shapely.geometry import Polygon

        geom = Polygon(self.outer, self.holes).difference(Polygon(other.outer, other.holes))
        polys = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        return [
            Shape(np.asarray(p.exterior.coords), [np.asarray(i.coords) for i in p.interiors])
            for p in polys if not p.is_empty and p.area > 1e-12
        ]


def triangulate(outer: Loop, holes: Sequence[Loop] = ()) -> tuple[NDArray, NDArray]:
    """Triangulate a polygon with holes (earcut). Triangles are CCW."""
    import mapbox_earcut as earcut

    rings = [ensure_ccw(np.asarray(outer, dtype=np.float64))] + [
        ensure_cw(np.asarray(h, dtype=np.float64)) for h in holes
    ]
    points = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tris = earcut.triangulate_float64(points, ends).reshape(-1, 3).astype(np.int64)
    # earcut returns clockwise triangles in a y-up frame; flip to CCW.
    if len(tris):
        a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        flip = cross < 0
        tris[flip] = tris[flip][:, [0, 2, 1]]
    return points, tris


# --------------------------------------------------------------------------
# YAML construction


def loop_from_spec(spec: Any, segments: int = 32) -> Loop:
    """Build a closed loop from a YAML-friendly spec.

    Forms::

        [[x, y], ...]                       # explicit polygon
        {polygon: [[x, y], ...], fillet: r} # polygon with rounded corners
        {rect: [w, h], radius: r, center: [x, y]}
        {circle: r, segments: n, center: [x, y]}
        {ellipse: [rx, ry]}
        {ngon: r, sides: n}
        {spline: [[x, y], ...], samples: n} # smooth closed curve through points
    """
    if isinstance(spec, (list, tuple)):
        return dedupe(np.asarray(spec, dtype=np.float64), closed=True)
    if not isinstance(spec, dict):
        raise ValueError(f"Cannot build a profile from {spec!r}")
    segs = int(spec.get("segments", segments))
    center = tuple(spec.get("center", (0.0, 0.0)))
    if "rect" in spec:
        w, h = spec["rect"]
        return rect(float(w), float(h), float(spec.get("radius", 0.0)), max(2, segs // 4), center)
    if "circle" in spec:
        return circle(float(spec["circle"]), segs, center)
    if "ellipse" in spec:
        rx, ry = spec["ellipse"]
        return ellipse(float(rx), float(ry), segs, center)
    if "ngon" in spec:
        return regular_polygon(float(spec["ngon"]), int(spec.get("sides", 6)), float(spec.get("rotation", 0.0)))
    if "polygon" in spec:
        loop = dedupe(np.asarray(spec["polygon"], dtype=np.float64), closed=True)
        r = spec.get("fillet", 0.0)
        return fillet(loop, r, int(spec.get("fillet_segments", 6))) if np.any(np.asarray(r) > 0) else loop
    if "spline" in spec:
        return dedupe(catmull_rom(np.asarray(spec["spline"]), int(spec.get("samples", 8)), closed=True), True)
    raise ValueError(f"Unknown profile spec keys: {sorted(spec)}")


def shape_from_spec(spec: Any) -> Shape:
    """Build a :class:`Shape` from ``{outer: <loop spec>, holes: [<loop spec>, ...]}`` or a loop spec."""
    if isinstance(spec, dict) and "outer" in spec:
        return Shape(loop_from_spec(spec["outer"]), [loop_from_spec(h) for h in spec.get("holes", [])])
    return Shape(loop_from_spec(spec))


def polyline_from_spec(spec: Any) -> NDArray:
    """Build an open polyline (lathe/sweep profile).

    Forms: ``[[x, y], ...]`` (straight segments) or
    ``{spline: [[x, y], ...], samples: n}`` (smooth curve through the points),
    or ``{segments: [<form>, <form>, ...]}`` to chain several pieces.
    """
    if isinstance(spec, (list, tuple)):
        return dedupe(np.asarray(spec, dtype=np.float64), closed=False)
    if isinstance(spec, dict):
        if "spline" in spec:
            return dedupe(catmull_rom(np.asarray(spec["spline"]), int(spec.get("samples", 8))), False)
        if "segments" in spec:
            parts = [polyline_from_spec(s) for s in spec["segments"]]
            return dedupe(np.vstack(parts), closed=False)
        if "arc" in spec:
            a = spec["arc"]
            return arc(a["center"], float(a["radius"]), float(a["start"]), float(a["end"]), int(a.get("segments", 12)))
    raise ValueError(f"Cannot build a polyline from {spec!r}")
