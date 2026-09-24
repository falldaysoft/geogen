"""Doors for openings in walls of a known thickness (floor plans).

:class:`DoorGenerator` fills a ``width`` x ``height`` opening in a wall
``wall`` metres thick. Local frame: origin at the bottom-centre of the
opening on the wall's centre plane, X along the wall, +Z toward the side the
leaf swings into.

    door
    ├── lining             jambs + head lining the reveal (full wall depth)
    ├── architrave_front   casing on the +Z face
    ├── architrave_back    casing on the -Z face
    └── leaf_pivot         at the hinge axis (joint metadata for articulation)
        └── leaf           flush leaf, hinge edge on the pivot
            └── handles    lever handles on both faces

``style: archway`` keeps the lining and architraves but has no leaf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.transform import Transform
from .base import CompositeGenerator

STYLES = ("door", "archway")


def _box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, bevel: float = 0.0) -> Mesh:
    from .primitives import CubeGenerator

    mesh = CubeGenerator(size_x=x1 - x0, size_y=y1 - y0, size_z=z1 - z0, bevel=bevel).generate()
    move = np.eye(4)
    move[:3, 3] = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
    return mesh.transform(move)


def _u_frame(outer_w: float, outer_h: float, inner_w: float, inner_h: float, depth: float,
             z_center: float, bevel: float = 0.0) -> Mesh:
    """An upside-down U (two jambs and a head) standing on y = 0, in the XY plane."""
    from ..core.profile import Shape
    from .profiles import ExtrudeGenerator

    ow, iw = outer_w / 2, inner_w / 2
    outline = np.array([[-ow, 0], [-iw, 0], [-iw, inner_h], [iw, inner_h], [iw, 0], [ow, 0],
                        [ow, outer_h], [-ow, outer_h]])
    mesh = ExtrudeGenerator(shape=Shape(outline), depth=depth, axis="z", bevel=bevel).generate()
    move = np.eye(4)
    move[2, 3] = z_center
    return mesh.transform(move)


@dataclass
class DoorGenerator(CompositeGenerator):
    """Door (or archway) sized to an opening in a wall.

    Attributes:
        width, height: The opening cut in the wall
        wall: Wall thickness (the lining spans it)
        style: 'door' (lining, architraves, leaf) or 'archway' (no leaf)
        hinge: 'left' or 'right' as seen from the swing (+Z) side
        open_deg: Leaf angle, 0 = closed, swinging into +Z
        lining: Jamb lining thickness
        casing: Architrave width
        leaf_thickness: Door leaf thickness
        frame_material, leaf_material, handle_material: Material names
    """

    width: float = 0.9
    height: float = 2.1
    wall: float = 0.12
    style: str = "door"
    hinge: str = "left"
    open_deg: float = 0.0
    lining: float = 0.025
    casing: float = 0.07
    casing_depth: float = 0.018
    leaf_thickness: float = 0.04
    frame_material: str = "trim_white"
    leaf_material: str = "wood"
    handle_material: str = "chrome"
    name: str = "door"

    @property
    def leaf_width(self) -> float:
        return self.width - 2 * self.lining - 0.006

    @property
    def leaf_height(self) -> float:
        return self.height - self.lining - 0.012

    @property
    def hinge_x(self) -> float:
        """Hinge axis x (local), on the lining face."""
        edge = self.width / 2 - self.lining - 0.003
        return -edge if self.hinge == "left" else edge

    def swing(self) -> dict:
        """Swing clearance in the door's frame: hinge point, radius, sweep angles (deg, XZ plane)."""
        # Angle 0 = +X, 90 = +Z. Closed leaf points from the hinge across the opening.
        closed = 0.0 if self.hinge == "left" else 180.0
        return {
            "hinge": [self.hinge_x, 0.0, self.wall / 2],
            "radius": self.leaf_width,
            "from_deg": closed,
            "to_deg": 90.0,
        }

    def generate(self, material_loader=None) -> SceneNode:
        from ..core import uvmap
        from ..materials.loader import MaterialLoader

        if self.style not in STYLES:
            raise ValueError(f"door style must be one of {STYLES}, got {self.style!r}")
        if self.hinge not in ("left", "right"):
            raise ValueError(f"door hinge must be 'left' or 'right', got {self.hinge!r}")
        loader = material_loader or MaterialLoader()
        frame_mat = loader.load(self.frame_material)

        def part(name: str, mesh: Mesh, material) -> SceneNode:
            mesh = uvmap.box_project(mesh)
            mesh.material = material
            return SceneNode(name=name, mesh=mesh)

        w, h, t = self.width, self.height, self.wall
        root = SceneNode(name=self.name)
        root.size = np.array([w + 2 * self.casing, h + self.casing, t + 2 * self.casing_depth])

        root.add_child(part("lining", _u_frame(w, h, w - 2 * self.lining, h - self.lining, t, 0.0),
                            frame_mat))
        outer_w, outer_h = w + 2 * self.casing, h + self.casing
        zc = t / 2 + self.casing_depth / 2
        root.add_child(part("architrave_front",
                            _u_frame(outer_w, outer_h, w, h, self.casing_depth, zc, bevel=0.004), frame_mat))
        root.add_child(part("architrave_back",
                            _u_frame(outer_w, outer_h, w, h, self.casing_depth, -zc, bevel=0.004), frame_mat))

        if self.style == "door":
            root.add_child(self._leaf(loader, part))
        return root

    def _leaf(self, loader, part) -> SceneNode:
        lw, lh, lt = self.leaf_width, self.leaf_height, self.leaf_thickness
        sign = 1.0 if self.hinge == "left" else -1.0  # leaf extends toward +x from a left hinge

        pivot = SceneNode(
            name="leaf_pivot",
            transform=Transform(
                translation=np.array([self.hinge_x, 0.01, self.wall / 2]),
                rotation=np.array([0.0, -sign * np.radians(self.open_deg), 0.0]),
            ),
        )
        # Joint metadata for articulation (geogen-3cc.18): rotation about +Y,
        # positive opening angle swings the leaf into +Z.
        pivot.meta["joint"] = {"type": "hinge", "axis": [0.0, -sign, 0.0], "min_deg": 0.0, "max_deg": 95.0}
        pivot.tags = ["joint.hinge"]

        x0, x1 = (0.0, lw) if sign > 0 else (-lw, 0.0)
        leaf = part("leaf", _box(x0, x1, 0.0, lh, -lt, 0.0, bevel=0.004), loader.load(self.leaf_material))
        leaf.tags = ["door.leaf"]
        pivot.add_child(leaf)

        # Lever handles 1 m up, 6 cm in from the latch edge, on both faces.
        latch_x = sign * (lw - 0.06)
        y = 1.0
        handle_mat = loader.load(self.handle_material)
        meshes = []
        for face in (1.0, -1.0):
            z_face = 0.0 if face > 0 else -lt
            rose = _box(latch_x - 0.025, latch_x + 0.025, y - 0.025, y + 0.025,
                        *sorted((z_face, z_face + face * 0.012)), bevel=0.004)
            stem = _box(latch_x - 0.009, latch_x + 0.009, y - 0.009, y + 0.009,
                        *sorted((z_face, z_face + face * 0.06)), bevel=0.003)
            lever_x = sorted((latch_x, latch_x - sign * 0.13))
            lever = _box(lever_x[0], lever_x[1], y - 0.01, y + 0.01,
                         *sorted((z_face + face * 0.045, z_face + face * 0.065)), bevel=0.004)
            meshes += [rose, stem, lever]
        from ..core import csg

        handles = part("handles", csg.union(*meshes), handle_mat)
        handles.meta["collider"] = "none"
        leaf.add_child(handles)
        return pivot
