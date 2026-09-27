"""Railings and fences along a path (``primitive: fence``).

Posts, two rails and upright pickets (square bars) follow a polyline of
[x, z] points in the asset frame (like ``primitive: paths``, the part keeps
the asset frame). ``gaps`` leave openings for gates: [[from, to], ...] in
metres along the path. Every piece is a box; the result is one mesh.

YAML::

    railings:
      primitive: fence
      path: [[-30, -20], [30, -20], [30, 20], [-30, 20]]
      closed: true
      height: 1.2
      gaps: [[27, 31]]            # the gateway
      material: lamp_iron
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.mesh import Mesh
from .base import MeshGenerator
from .paths import path_points
from .primitives import CubeGenerator

if TYPE_CHECKING:
    from ..layout.attachments import AttachmentPoint
    from ..layout.surfaces import Surface


def _box(center: np.ndarray, size: np.ndarray, yaw: float) -> Mesh:
    mesh = CubeGenerator(bevel=0.0).generate()
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.array([[c * size[0], 0, s * size[2], center[0]],
                  [0, size[1], 0, center[1]],
                  [-s * size[0], 0, c * size[2], center[2]],
                  [0, 0, 0, 1]], dtype=np.float64)
    return mesh.transform(m)


@dataclass
class FenceGenerator(MeshGenerator):
    """Posts, rails and pickets along a path; see the module docstring."""

    path: Any = field(default_factory=lambda: [[-1.0, 0.0], [1.0, 0.0]])
    closed: bool = False
    height: float = 1.1
    post: float = 0.08          # post section (m)
    post_spacing: float = 2.5
    picket: float = 0.02        # picket section
    spacing: float = 0.12       # picket centres
    rail: float = 0.035         # rail section
    gaps: list[list[float]] = field(default_factory=list)

    def get_attachment_points(self, size: np.ndarray) -> dict[str, AttachmentPoint]:
        return CubeGenerator().get_attachment_points(size)

    def get_surfaces(self, size: np.ndarray) -> dict[str, Surface]:
        return CubeGenerator().get_surfaces(size)

    def _open(self, s: float) -> bool:
        return any(a <= s <= b for a, b in self.gaps)

    def generate(self) -> Mesh:
        pts = path_points(self.path)
        if self.closed:
            pts = np.vstack([pts, pts[:1]])
        boxes: list[Mesh] = []
        s0 = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            length = float(np.linalg.norm(d))
            if length < 1e-6:
                continue
            u = d / length
            yaw = float(np.arctan2(u[0], u[1]))    # box z along the segment
            # Rails: the segment minus the gaps, as runs [t0, t1] in metres from a.
            cuts = sorted([(max(g0 - s0, 0.0), min(g1 - s0, length)) for g0, g1 in self.gaps
                           if g1 > s0 and g0 < s0 + length])
            runs, t = [], 0.0
            for c0, c1 in cuts:
                if c0 > t:
                    runs.append((t, c0))
                t = max(t, c1)
            if t < length:
                runs.append((t, length))
            for t0, t1 in runs:
                if t1 - t0 < 0.05:
                    continue
                mid = a + u * (t0 + t1) / 2
                for y in (0.12, self.height - 0.06):
                    boxes.append(_box(np.array([mid[0], y, mid[1]]), np.array([self.rail, self.rail, t1 - t0]), yaw))
                # Posts at both ends of the run and every post_spacing between.
                n = max(1, int(np.ceil((t1 - t0) / self.post_spacing)))
                for i in range(n + 1):
                    p = a + u * (t0 + (t1 - t0) * i / n)
                    boxes.append(_box(np.array([p[0], (self.height + 0.08) / 2, p[1]]),
                                      np.array([self.post, self.height + 0.08, self.post]), yaw))
                # Pickets between the rails.
                count = int((t1 - t0) / self.spacing)
                for i in range(1, count):
                    p = a + u * (t0 + i * (t1 - t0) / count)
                    boxes.append(_box(np.array([p[0], self.height / 2, p[1]]),
                                      np.array([self.picket, self.height - 0.02, self.picket]), yaw))
            s0 += length
        if not boxes:
            raise ValueError("fence has nothing to build (all gaps?)")
        from ..core import uvmap

        return uvmap.box_project(Mesh.merge(boxes))     # metric UVs (the boxes were scaled unit cubes)
