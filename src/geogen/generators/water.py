"""Water surfaces (``primitive: water``): ponds, lakes and the sea.

A flat sheet (a thin slab, so it stays a closed mesh) from a 2D ``shape`` in
the part's XZ plane (``{ellipse: [...]}``, ``{rect: [...]}``, or ``{outer,
holes}`` for a sea around an island). The node gets ``meta.water`` (exported
as extras.geogen.water) so the Godot runtime swaps its material for an
animated water shader (waves, depth tint, shoreline foam); offline renderers
show the part's material (``water`` by default). Water has no collider.

``barrier: <height>`` adds an invisible wall (a collider-only child,
``shore_barrier-colonly``) along every edge of the shape, from half a metre
below the surface to ``height`` above it, so the player can't walk into deep
water (or off the edge of the sea).
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.profile import Shape
from .fence import _box


def barrier_mesh(shape: Shape, top: float, bottom: float = -0.5, thickness: float = 0.1) -> Mesh:
    """Walls along every loop of ``shape`` (profile (x, y) = world (x, -z), axis-y extrude frame)."""
    boxes = []
    for loop in shape.loops:
        pts = np.column_stack([loop[:, 0], -loop[:, 1]])      # profile y -> world -z
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            d = b - a
            length = float(np.linalg.norm(d))
            if length < 1e-4:
                continue
            mid = (a + b) / 2
            yaw = float(np.arctan2(d[0], d[1]))
            boxes.append(_box(np.array([mid[0], (top + bottom) / 2, mid[1]]),
                              np.array([thickness, top - bottom, length + thickness]), yaw))
    return Mesh.merge(boxes)


def water_meta(config: dict) -> dict:
    """``meta.water`` for a water part: what the runtime's shader needs."""
    meta = {}
    for key in ("shallow_color", "deep_color"):
        if key in config:
            meta[key] = [float(v) for v in config[key]]
    for key in ("depth_scale", "wave_scale", "wave_speed", "foam"):
        if key in config:
            meta[key] = float(config[key])
    return meta


def add_barrier(node: SceneNode, shape: Shape, height: float, thickness: float) -> None:
    """Give a water node its collider-only shore wall (see the module docstring)."""
    mesh = barrier_mesh(shape, top=height, bottom=-0.5 - thickness / 2)
    wall = SceneNode("shore_barrier-colonly", mesh=mesh)
    wall.meta.update({"type": "collider", "shape": "mesh", "collider": "none"})
    node.add_child(wall)
