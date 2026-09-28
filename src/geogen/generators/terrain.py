"""Terrain (``primitive: terrain``): a heightfield solid with an island falloff.

The height at (x, z) is fractal gradient noise shaped by a falloff mask: it
rises from the sea floor (``sea_level - depth``) through the beach to a
``height`` metres above the sea in the middle. Flattened ``pads`` give
buildings level ground. The mesh is a closed solid (the heightfield on top,
vertical skirts round the edge, a flat bottom), so it validates; faces are
grouped by material: ``sand`` near the water line, ``rock`` where it's
steep, ``grass`` elsewhere. UVs are metric (box-projected). Like ``paths``,
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
      materials: { sand: sand, grass: grass, rock: rock }
      shore: { depth: 0.5, height: 1.5 }   # invisible wall where the water gets this deep
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
from .base import MeshGenerator
from .primitives import CubeGenerator

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint
    from ..layout.surfaces import Surface

SAND, GRASS, ROCK = 0, 1, 2
MATERIAL_SLOTS = ("sand", "grass", "rock")
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

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        return CubeGenerator().get_attachment_points(size)

    def get_surfaces(self, size: np.ndarray) -> dict[str, Surface]:
        return CubeGenerator().get_surfaces(size)

    def params(self) -> dict[str, Any]:
        """What ``meta.terrain`` stores (enough to rebuild the height function)."""
        return {"size": [float(v) for v in self.size], "resolution": self.resolution, "height": self.height,
                "sea_level": self.sea_level, "depth": self.depth, "seed": self.seed, "scale": self.scale,
                "octaves": self.octaves, "falloff": dict(self.falloff), "pads": [dict(p) for p in self.pads],
                "island": self.island}

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
        hill = self.height * np.clip(0.25 + 0.75 * inland + 0.35 * (n - 0.5), 0.0, 1.2)
        land = self.sea_level + 0.4 + hill           # the beach sits just above the water
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
        # Material groups by height and slope.
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        slope = np.degrees(np.arccos(np.clip(n[:, 1], -1.0, 1.0)))
        cy = tri[:, :, 1].mean(axis=1)
        groups = np.full(len(tri), GRASS, dtype=np.int64)
        if self.island:
            groups[cy < self.sea_level + SAND_ABOVE_SEA] = SAND
        groups[(slope > ROCK_SLOPE) & (n[:, 1] > 0.05)] = ROCK
        groups[n[:, 1] <= 0.05] = SAND          # skirts and bottom (under water / out of sight)
        mesh = Mesh(v, f, face_materials=groups, materials=[None, None, None])
        # Project the whole top from above (steep faces would otherwise flip axis in speckles).
        directions = np.where((n[:, 1] > 0.05)[:, None], [0.0, 1.0, 0.0], n)
        mesh = uvmap.box_project(mesh, directions=directions)
        return meshops.compute_normals(mesh, 50.0)


def terrain_from_meta(meta: dict[str, Any]) -> TerrainGenerator:
    """Rebuild a terrain's height function from ``meta.terrain``."""
    return TerrainGenerator(size=tuple(meta["size"]), resolution=float(meta["resolution"]),
                            height=float(meta["height"]), sea_level=float(meta["sea_level"]),
                            depth=float(meta["depth"]), seed=int(meta["seed"]), scale=float(meta["scale"]),
                            octaves=int(meta["octaves"]), falloff=dict(meta.get("falloff") or {}),
                            pads=list(meta.get("pads") or []), island=bool(meta.get("island", True)))


def add_shore(node, terrain: TerrainGenerator, spec: dict[str, Any]) -> None:
    """A collider-only wall along the terrain's coastline at ``spec.depth`` below sea level."""
    from ..core.node import SceneNode
    from .fence import _box

    level = terrain.sea_level - float(spec.get("depth", 0.5))
    top = terrain.sea_level + float(spec.get("height", 1.5))
    bottom = level - 0.5
    pts = terrain.coastline(level)
    boxes = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            continue
        mid = (a + b) / 2
        boxes.append(_box(np.array([mid[0], (top + bottom) / 2, mid[1]]),
                          np.array([0.1, top - bottom, length + 0.1]), float(np.arctan2(d[0], d[1]))))
    wall = SceneNode("shore_barrier-colonly", mesh=Mesh.merge(boxes))
    wall.meta.update({"type": "collider", "shape": "mesh", "collider": "none"})
    node.add_child(wall)
