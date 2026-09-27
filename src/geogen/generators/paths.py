"""Footpath networks (``primitive: paths``): widened centre lines unioned into one slab.

Each path is a polyline of [x, z] points in the asset frame (``{spline: [...]}``
smooths it through its points); every path is buffered to ``width`` (round
ends and joins), the lot are unioned in 2D (shapely), so crossings and loops
come out as one clean region (loops leave lawns as holes), and the region is
extruded ``thickness`` up from y = 0. Unlike most primitives the part keeps
the asset frame: the points are where the paths go.

YAML::

    paths:
      primitive: paths
      width: 2.0
      thickness: 0.04
      paths:
        - [[-20, 0], [20, 0]]
        - { spline: [[0, -18], [6, -6], [0, 0], [-6, 8], [0, 18]], samples: 8 }
      material: gravel
      walkable: true
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import LineString, MultiPolygon
from shapely.ops import unary_union

from ..core.mesh import Mesh
from ..core.profile import Shape, catmull_rom
from .base import MeshGenerator
from .primitives import CubeGenerator
from .profiles import ExtrudeGenerator

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint
    from ..layout.surfaces import Surface


def path_points(spec: Any) -> np.ndarray:
    """[[x, z], ...] from a list of points or ``{spline: [...], samples: n}``."""
    if isinstance(spec, dict):
        if "spline" not in spec:
            raise ValueError(f"path must be [[x, z], ...] or {{spline: [...]}}, got keys {sorted(spec)}")
        pts = np.asarray(spec["spline"], dtype=np.float64)
        return catmull_rom(pts, int(spec.get("samples", 8)), closed=bool(spec.get("closed", False)))
    pts = np.asarray(spec, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError(f"path needs at least two [x, z] points, got {spec!r}")
    return pts


def path_region(paths: list[Any], width: float, resolution: int = 8):
    """The union of the paths buffered to ``width`` (a shapely Polygon / MultiPolygon)."""
    lines = [LineString(path_points(p)) for p in paths]
    return unary_union([line.buffer(width / 2, resolution=resolution) for line in lines])


@dataclass
class PathsGenerator(MeshGenerator):
    """A path network slab; see the module docstring."""

    paths: list[Any] = field(default_factory=list)
    width: float = 1.8
    thickness: float = 0.04
    bevel: float = 0.01
    detail: float = 1.0

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        return CubeGenerator().get_attachment_points(size)

    def get_surfaces(self, size: np.ndarray) -> dict[str, Surface]:
        return CubeGenerator().get_surfaces(size)

    def generate(self) -> Mesh:
        if not self.paths:
            raise ValueError("paths primitive needs at least one path")
        region = path_region(self.paths, self.width, max(2, int(round(6 * self.detail))))
        region = region.simplify(0.01, preserve_topology=True)
        polys = list(region.geoms) if isinstance(region, MultiPolygon) else [region]
        meshes = []
        for poly in polys:
            # Profile (x, y) -> world (x, -z) for axis "y": flip z into the profile's y.
            outer = np.array(poly.exterior.coords)[:-1] * [1, -1]
            holes = [np.array(r.coords)[:-1] * [1, -1] for r in poly.interiors]
            mesh = ExtrudeGenerator(shape=Shape(outer, holes), depth=self.thickness, axis="y",
                                    bevel=min(self.bevel, self.thickness * 0.4), bevel_segments=1).generate()
            meshes.append(mesh.transform(np.array([[1, 0, 0, 0], [0, 1, 0, self.thickness / 2],
                                                   [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)))
        return Mesh.merge(meshes)
