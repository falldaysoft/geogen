"""Terrain (``primitive: terrain``): a heightfield solid with an island falloff.

The height at (x, z) is fractal gradient noise shaped by a falloff mask: it
rises from the sea floor (``sea_level - depth``) through the beach to a
``height`` metres above the sea in the middle. Flattened ``pads`` give
buildings level ground. The mesh is a closed solid (the heightfield on top,
vertical skirts round the edge, a flat bottom), so it validates; faces are
grouped by material: ``sand`` near the water line, ``rock`` where it's
steep, ``grass`` elsewhere, and ``path`` along ``paths`` (faces within half
the width of the polylines, so a path follows the ground). UVs are metric
(box-projected). Like ``paths``,
the part keeps the asset frame: (0, 0) is the middle of the terrain and y is
absolute.

YAML::

    ground:
      primitive: terrain
      size: [140, 140]          # x, z extent (m)
      resolution: 1.0           # grid cell (m)
      height: 14                # peak above sea level
      sea_level: 0.0
      depth: 4                  # sea floor below sea level
      seed: 3
      scale: 40                 # noise feature size (m)
      octaves: 4
      falloff: { radius: 0.42, edge: 0.18, power: 1.6 }   # fractions of the smaller side
      pads: [{ center: [20, -10], radius: 6, blend: 5 }]  # level ground (height: taken at the centre)
      materials: { sand: sand, grass: grass, rock: rock, path: gravel }
      paths: [{ points: [[0, -50], [10, -20], [0, 0]], width: 2.2 }]   # or points: {spline: [...]}
      shore: { depth: 0.5, height: 1.5, openings: [[4, -60, 3]] }   # invisible wall; gaps [x, z, r]
      walkable: true

``meta.terrain`` keeps these parameters, so scatter (``on_terrain:``) can
sample heights and slopes; ``height_at`` / ``slope_at`` evaluate them. ``shore``
adds a collider-only ``shore_barrier-colonly`` child along the ``coastline``
(from half a metre under it to ``height`` above sea level), so the player
can't wade out to sea.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core import meshops, uvmap
from ..core.mesh import Mesh
from ..core.subdiv import fractal_noise3
from .base import MeshGenerator, box_attachment_points, box_surfaces

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint
    from ..layout.surfaces import Surface

SAND, GRASS, ROCK, PATH = 0, 1, 2, 3
MATERIAL_SLOTS = ("sand", "grass", "rock", "path")
SAND_ABOVE_SEA = 1.2        # m: faces this close above the water line are beach
ROCK_SLOPE = 32.0           # degrees: steeper faces are bare rock


@dataclass
class TerrainGenerator(MeshGenerator):
    """An island heightfield solid; see the module docstring."""

    size: tuple[float, float] = (100.0, 100.0)
    resolution: float = 1.0
    height: float = 12.0
    sea_level: float = 0.0
    depth: float = 4.0
    seed: int = 0
    scale: float = 40.0
    octaves: int = 4
    falloff: dict[str, float] = field(default_factory=dict)
    pads: list[dict[str, Any]] = field(default_factory=list)
    island: bool = True
    paths: list[dict[str, Any]] = field(default_factory=list)

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        return box_attachment_points(size)

    def get_surfaces(self, size: np.ndarray) -> dict[str, Surface]:
        return box_surfaces(size)

    def params(self) -> dict[str, Any]:
        """What ``meta.terrain`` stores (enough to rebuild the height function)."""
        return {"size": [float(v) for v in self.size], "resolution": self.resolution, "height": self.height,
                "sea_level": self.sea_level, "depth": self.depth, "seed": self.seed, "scale": self.scale,
                "octaves": self.octaves, "falloff": dict(self.falloff), "pads": [dict(p) for p in self.pads],
                "island": self.island, "paths": [dict(p) for p in self.paths]}

    # --- the height function ------------------------------------------------------------------

    def _raw(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        pts = np.column_stack([x / self.scale, z / self.scale, np.full(len(x), 0.5)])
        n = fractal_noise3(pts, self.octaves, self.seed) * 0.5 + 0.5
        if not self.island:             # rolling ground: noise alone, 0..height above sea level
            return self.sea_level + self.height * np.clip(n, 0.0, 1.0)
        half = min(self.size) / 2
        radius = float(self.falloff.get("radius", 0.42)) * 2 * half
        edge = float(self.falloff.get("edge", 0.18)) * 2 * half
        power = float(self.falloff.get("power", 1.6))
        # A wobbly coastline: the falloff radius varies with a low-frequency noise.
        wobble = fractal_noise3(pts * 0.6 + 41.0, 2, self.seed + 101)
        dist = np.hypot(x, z) / (radius * (1.0 + 0.18 * wobble))
        mask = np.clip(1.0 - (dist - 1.0) * radius / max(edge, 1e-6), 0.0, 1.0)
        mask = np.where(dist < 1.0, 1.0, mask)
        inland = np.clip(1.0 - np.hypot(x, z) / (radius * 1.1), 0.0, 1.0) ** power
        # Hills rise from a beach just above the water; the noise fades out toward the shore
        # so beaches come out smooth and gentle.
        hill = self.height * np.clip(0.03 + 0.97 * inland + 0.35 * (n - 0.5) * np.sqrt(inland), 0.0, 1.2)
        land = self.sea_level + 0.4 + hill
        floor = self.sea_level - self.depth
        return floor + (land - floor) * (mask * mask * (3 - 2 * mask))

    def height_at(self, x, z) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        h = self._raw(x, z)
        for pad in self.pads:
            cx, cz = (float(v) for v in pad["center"])
            level = float(pad["height"]) if "height" in pad else float(self._raw(np.array([cx]), np.array([cz]))[0])
            r, blend = float(pad.get("radius", 5.0)), float(pad.get("blend", 4.0))
            d = np.hypot(x - cx, z - cz)
            w = np.clip(1.0 - (d - r) / max(blend, 1e-6), 0.0, 1.0)
            w = w * w * (3 - 2 * w)
            h = h + (level - h) * w
        return h

    def slope_at(self, x, z, eps: float = 0.5) -> np.ndarray:
        """Slope in degrees (finite differences of height_at)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        dx = (self.height_at(x + eps, z) - self.height_at(x - eps, z)) / (2 * eps)
        dz = (self.height_at(x, z + eps) - self.height_at(x, z - eps)) / (2 * eps)
        return np.degrees(np.arctan(np.hypot(dx, dz)))

    def surface_at(self, x, z) -> np.ndarray:
        """The height of the generated mesh at (x, z): ``height_at`` sampled on the grid and
        interpolated over its triangles (in hollows it sits a few centimetres above
        ``height_at``; ``_cut`` only splits faces, so the surface is unchanged)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        sx, sz = (float(v) for v in self.size)
        nx = max(2, int(round(sx / self.resolution)))
        nz = max(2, int(round(sz / self.resolution)))
        rx, rz = sx / nx, sz / nz
        fx = np.clip((x + sx / 2) / rx, 0.0, nx - 1e-9)
        fz = np.clip((z + sz / 2) / rz, 0.0, nz - 1e-9)
        i, j = np.floor(fx), np.floor(fz)
        u, v = fx - i, fz - j
        x0, z0 = i * rx - sx / 2, j * rz - sz / 2
        ha = self.height_at(x0, z0)
        hb = self.height_at(x0 + rx, z0)
        hc = self.height_at(x0 + rx, z0 + rz)
        hd = self.height_at(x0, z0 + rz)
        # The cell's two faces, as generate() splits it: a-d-c (v > u) and a-c-b.
        return np.where(v > u, ha + v * (hd - ha) + u * (hc - hd), ha + u * (hb - ha) + v * (hc - hb))

    def path_mesh(self, lift: float = 0.05, thickness: float = 0.065, step: float = 1 / 3) -> Mesh | None:
        """Smooth-edged ribbons along ``paths``, draped ``lift`` above the terrain's mesh (each
        path a centimetre higher than the last, so they don't z-fight where they meet). A
        closed slab per path, a grid of vertices ``step`` apart along and across it: a third
        of the terrain's grid, so the ground's creases bulge at most a couple of centimetres
        up between them (straight across a 2 m path they came through it). The terrain's own
        path faces sit hidden underneath. The thickness isn't a whole number of centimetres,
        so no underside lands on the ground level (a pad's floor or a foundation's underside
        would z-fight with it)."""
        from shapely.geometry import LineString

        from .paths import path_points

        slabs = []
        for k, spec in enumerate(self.paths):
            line = LineString(path_points(spec["points"]))
            width = float(spec.get("width", 2.0))
            s = np.linspace(0.0, line.length, max(2, int(np.ceil(line.length / step)) + 1))
            c = np.array([line.interpolate(v).coords[0] for v in s])
            t = np.gradient(c, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            normal = np.column_stack([t[:, 1], -t[:, 0]])
            across = np.linspace(0.5, -0.5, max(2, int(np.ceil(width / step)) + 1)) * width   # left to right
            xz = c[:, None, :] + normal[:, None, :] * across[None, :, None]                    # (rows, cols, 2)
            flat = xz.reshape(-1, 2)
            top_y = self.surface_at(flat[:, 0], flat[:, 1]) + lift + 0.01 * k
            rows, cols = xz.shape[:2]
            top = np.column_stack([flat[:, 0], top_y, flat[:, 1]])
            bottom = top - [0.0, thickness, 0.0]
            verts = np.vstack([top, bottom])
            n = rows * cols
            idx = np.arange(n).reshape(rows, cols)
            a, b, cc, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
            quads = [np.column_stack([a.ravel(), b.ravel(), cc.ravel(), d.ravel()])]            # top
            quads.append(quads[0][:, ::-1] + n)                                                  # bottom
            # The rim: left and right sides, then the two ends.
            for edge in (idx[:, 0], idx[::-1, -1], idx[0, ::-1], idx[-1, :]):
                e0, e1 = edge[:-1], edge[1:]
                quads.append(np.column_stack([e0, e1, e1 + n, e0 + n]))
            q = np.vstack(quads)
            faces = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int64)
            slab = Mesh(verts, faces)
            if slab.to_trimesh().volume < 0:
                slab = Mesh(verts, faces[:, [0, 2, 1]])
            slabs.append(slab)
        if not slabs:
            return None
        return meshops.compute_normals(uvmap.box_project(Mesh.merge(slabs)), 50.0)

    def on_path(self, xz: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Which (x, z) points lie on one of ``paths`` (within half its width, times ``scale``)."""
        from shapely import contains_xy
        from shapely.geometry import LineString
        from shapely.ops import unary_union

        from .paths import path_points

        region = unary_union([LineString(path_points(p["points"])).buffer(float(p.get("width", 2.0)) / 2 * scale)
                              for p in self.paths])
        return np.asarray(contains_xy(region, xz[:, 0], xz[:, 1]), dtype=bool)

    def coastline(self, level: float, rays: int = 240, step: float = 0.25) -> np.ndarray:
        """The outermost contour where the ground drops below ``level``, as (N, 2) x, z points
        marched out along rays from the middle (an island is star-shaped enough for this)."""
        reach = float(np.hypot(*self.size)) / 2
        r = np.arange(0.0, reach, step)
        angles = np.linspace(0.0, 2 * np.pi, rays, endpoint=False)
        x = np.outer(np.cos(angles), r)
        z = np.outer(np.sin(angles), r)
        inside = (np.abs(x) <= self.size[0] / 2) & (np.abs(z) <= self.size[1] / 2)
        above = (self.height_at(x.ravel(), z.ravel()).reshape(x.shape) > level) & inside
        last = np.array([np.flatnonzero(row)[-1] if row.any() else 0 for row in above])
        edge = r[np.minimum(last + 1, len(r) - 1)]
        return np.column_stack([np.cos(angles) * edge, np.sin(angles) * edge])

    # --- the mesh -----------------------------------------------------------------------------

    def generate(self) -> Mesh:
        sx, sz = (float(v) for v in self.size)
        nx = max(2, int(round(sx / self.resolution)))
        nz = max(2, int(round(sz / self.resolution)))
        xs = np.linspace(-sx / 2, sx / 2, nx + 1)
        zs = np.linspace(-sz / 2, sz / 2, nz + 1)
        gx, gz = np.meshgrid(xs, zs, indexing="ij")
        gy = self.height_at(gx.ravel(), gz.ravel()).reshape(gx.shape)
        bottom = float(gy.min()) - 1.0

        top = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        idx = np.arange((nx + 1) * (nz + 1)).reshape(nx + 1, nz + 1)
        a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        # CCW seen from above (+y): a(x0,z0) -> d(x0,z1) -> c(x1,z1), then a -> c -> b.
        faces = [np.column_stack([a.ravel(), d.ravel(), c.ravel()]),
                 np.column_stack([a.ravel(), c.ravel(), b.ravel()])]
        verts = [top]
        count = len(top)
        # Skirts: each edge ring of the top, down to the bottom plane.
        ring = np.r_[idx[:, 0], idx[-1, 1:], idx[-2::-1, -1], idx[0, -2:0:-1]]   # CCW round the edge
        low = top[ring].copy()
        low[:, 1] = bottom
        verts.append(low)
        k = np.arange(len(ring))
        k1 = (k + 1) % len(ring)
        up0, up1 = ring[k], ring[k1]
        dn0, dn1 = count + k, count + k1
        faces += [np.column_stack([up0, dn1, dn0]), np.column_stack([up0, up1, dn1])]
        # Bottom: a fan from the middle (faces down).
        centre = count + len(ring)
        verts.append(np.array([[0.0, bottom, 0.0]]))
        faces.append(np.column_stack([np.full(len(ring), centre), dn0, dn1]))
        v = np.vstack(verts)
        f = np.vstack(faces).astype(np.int64)
        # Material groups by height and slope, with their boundaries following the beach's height
        # contour and the rock's slope contour: the triangles that straddle one are split along it,
        # so the lines are smooth curves rather than steps of whole grid faces.
        if self.island:
            v, f = _cut(v, f, v[:, 1] - (self.sea_level + SAND_ABOVE_SEA), margin=0.01)
        # Rock where the ground is steep over a few metres (per-face slope alone leaves lone
        # square tiles of rock on the grid).
        rock = self.slope_at(v[:, 0], v[:, 2], eps=1.5) - ROCK_SLOPE
        v, f = _cut(v, f, rock, margin=0.2)
        rock = self.slope_at(v[:, 0], v[:, 2], eps=1.5) - ROCK_SLOPE
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        cy = tri[:, :, 1].mean(axis=1)
        groups = np.full(len(tri), GRASS, dtype=np.int64)
        if self.island:
            groups[cy < self.sea_level + SAND_ABOVE_SEA] = SAND
        if self.paths:
            on_path = self.on_path(tri[:, :, [0, 2]].mean(axis=1), 0.6)     # hidden under path_mesh()
            groups[on_path & (n[:, 1] > 0.05) & (cy > self.sea_level)] = PATH
        groups[(rock[f].mean(axis=1) > 0) & (n[:, 1] > 0.05)] = ROCK
        groups[n[:, 1] <= 0.05] = SAND          # skirts and bottom (under water / out of sight)
        mesh = Mesh(v, f, face_materials=groups, materials=[None] * len(MATERIAL_SLOTS))
        # Project the whole top from above (steep faces would otherwise flip axis in speckles).
        directions = np.where((n[:, 1] > 0.05)[:, None], [0.0, 1.0, 0.0], n)
        mesh = uvmap.box_project(mesh, directions=directions)
        return meshops.compute_normals(mesh, 50.0)


def _cut(v: np.ndarray, f: np.ndarray, field: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Split the triangles of (v, f) that ``field`` (one value a vertex) changes sign across,
    along its zero line: a new vertex where it crosses each such edge, shared by both faces
    on the edge (the mesh stays closed), interpolated along it (the surface is unchanged).
    Each split face becomes three, all on one side. Values within ``margin`` of zero are
    pushed out to it, so no new vertex lands on (or a sliver's width from) an old one."""
    field = np.where(np.abs(field) < margin, np.where(field < 0, -margin, margin), field)
    pos = field > 0
    sides = pos[f]
    mixed = np.flatnonzero(sides.any(axis=1) & ~sides.all(axis=1))
    if not len(mixed):
        return v, f
    new_verts: list[np.ndarray] = []
    made: dict[tuple[int, int], int] = {}

    def crossing(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in made:
            t = field[a] / (field[a] - field[b])
            made[key] = len(v) + len(new_verts)
            new_verts.append(v[a] + (v[b] - v[a]) * t)
        return made[key]

    out = []
    for face in f[mixed]:
        s = pos[face]
        lone = int(np.flatnonzero(s == (s.sum() == 1))[0])           # the one positive, or the one not
        p, q, r = (int(face[(lone + k) % 3]) for k in range(3))       # p alone on its side, winding kept
        m1, m2 = crossing(p, q), crossing(r, p)
        out += [[p, m1, m2], [m1, q, r], [m1, r, m2]]
    keep = np.ones(len(f), dtype=bool)
    keep[mixed] = False
    return np.vstack([v, np.array(new_verts)]), np.vstack([f[keep], np.array(out, dtype=np.int64)])


def terrain_from_meta(meta: dict[str, Any]) -> TerrainGenerator:
    """Rebuild a terrain's height function from ``meta.terrain``."""
    return TerrainGenerator(size=tuple(meta["size"]), resolution=float(meta["resolution"]),
                            height=float(meta["height"]), sea_level=float(meta["sea_level"]),
                            depth=float(meta["depth"]), seed=int(meta["seed"]), scale=float(meta["scale"]),
                            octaves=int(meta["octaves"]), falloff=dict(meta.get("falloff") or {}),
                            pads=list(meta.get("pads") or []), island=bool(meta.get("island", True)),
                            paths=list(meta.get("paths") or []))


def add_shore(node, terrain: TerrainGenerator, spec: dict[str, Any]) -> None:
    """A collider-only wall along the terrain's coastline at ``spec.depth`` below sea level."""
    from ..core.node import SceneNode
    from .fence import _box

    level = terrain.sea_level - float(spec.get("depth", 0.5))
    top = terrain.sea_level + float(spec.get("height", 1.5))
    bottom = level - 0.5
    pts = terrain.coastline(level)
    openings = [(np.array(o[:2], dtype=np.float64), float(o[2])) for o in spec.get("openings") or []]
    boxes = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-4 or any(np.linalg.norm((a + b) / 2 - c) < r for c, r in openings):
            continue
        mid = (a + b) / 2
        boxes.append(_box(np.array([mid[0], (top + bottom) / 2, mid[1]]),
                          np.array([0.1, top - bottom, length + 0.1]), float(np.arctan2(d[0], d[1]))))
    wall = SceneNode("shore_barrier-colonly", mesh=Mesh.merge(boxes))
    wall.meta.update({"type": "collider", "shape": "mesh", "collider": "none"})
    node.add_child(wall)
