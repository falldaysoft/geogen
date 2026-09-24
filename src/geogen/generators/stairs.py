"""Stairs: straight, L-shaped (quarter landing) and spiral, with handrails.

The flight body is one watertight solid (stepped side profile extruded across
the width; flights and landings unioned), so it doubles as a trimesh
collider the player can climb (risers stay below ``max_riser``, well under
the player's step height). Handrails are swept round bars on posts, in a
child node with their own material.

Frame: the stairs climb toward -Z (the bottom step faces +Z), centred on the
origin like other primitives; ``rise`` is the total height climbed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import csg, meshops, uvmap
from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.profile import Shape, circle
from .base import MeshGenerator

STYLES = ("straight", "l", "spiral")


def _move(mesh: Mesh, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> Mesh:
    m = np.eye(4)
    if yaw_deg:
        t = np.radians(yaw_deg)
        m[:3, :3] = [[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]
    m[:3, 3] = [x, y, z]
    return mesh.transform(m)


def _box(x0, x1, y0, y1, z0, z1) -> Mesh:
    from .primitives import CubeGenerator

    mesh = CubeGenerator(size_x=x1 - x0, size_y=y1 - y0, size_z=z1 - z0, bevel=0).generate()
    return _move(mesh, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)


@dataclass
class StairsGenerator(MeshGenerator):
    """Parametric stairs.

    Attributes:
        style: 'straight', 'l' (two flights, quarter landing) or 'spiral'
        rise: Total height climbed (m)
        width: Clear flight width (m); for spiral, the outer radius is width + pole
        max_riser: Largest allowed riser; the step count follows from it
        tread: Going per step (m) (spiral: measured at the walking line)
        turn: 'left' or 'right' (L-shaped and spiral direction)
        landing_at: Fraction of the steps in the first flight (L-shaped)
        waist: Slab thickness under the flight
        railing: 'both', 'left', 'right', 'outer' or 'none'
        rail_height: Handrail height above the step nosings
        railing_material: Material for rails and posts
    """

    style: str = "straight"
    rise: float = 3.0
    width: float = 1.0
    max_riser: float = 0.18
    tread: float = 0.28
    turn: str = "left"
    landing_at: float = 0.5
    waist: float = 0.15
    railing: str = "both"
    rail_height: float = 0.9
    railing_material: str = "metal"

    @property
    def steps(self) -> int:
        return max(1, int(np.ceil(self.rise / self.max_riser - 1e-9)))

    @property
    def riser(self) -> float:
        return self.rise / self.steps

    def generate(self) -> Mesh:
        if self.style not in STYLES:
            raise ValueError(f"stairs style must be one of {STYLES}, got {self.style!r}")
        body = {"straight": self._straight, "l": self._l_shaped, "spiral": self._spiral}[self.style]()
        lo, hi = body.vertices.min(axis=0), body.vertices.max(axis=0)
        # Centre on the origin like other primitives; railings use the same shift.
        self._offset = -(lo + hi) / 2
        body = _move(body, *self._offset)
        return meshops.compute_normals(uvmap.box_project(body), 30.0)

    # ------------------------------------------------------------ flights

    def _flight(self, steps: int, y0: float = 0.0, top_tread: bool = True) -> Mesh:
        """Straight flight climbing -Z from z = 0: ``steps`` risers of ``riser``.

        Its top step is at y0 + steps * riser; with ``top_tread`` the last step
        gets a full tread, otherwise the flight ends at the top riser.
        """
        from .profiles import ExtrudeGenerator

        r, t = self.riser, self.tread
        treads = steps if top_tread else steps - 1
        length = treads * t
        pts = [(0.0, 0.0)]
        for i in range(steps):
            pts.append((i * t, (i + 1) * r))
            pts.append((min((i + 1) * t, length), (i + 1) * r))
        pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
        top = steps * r
        pitch = np.arctan2(steps * r, max(length, 1e-6))
        back_drop = min(top, self.waist / max(np.cos(pitch), 0.2))
        pts.append((length, top - back_drop))
        foot = min(length, max(t, (top - back_drop) / max(np.tan(pitch), 1e-6)))
        pts.append((length - foot if top - back_drop > 1e-6 else length, 0.0))
        loop = np.array(pts)
        loop = loop[np.r_[True, np.linalg.norm(np.diff(loop, axis=0), axis=1) > 1e-9]]
        # Extrude along X: profile (p, y) with world z = -p.
        mesh = ExtrudeGenerator(shape=Shape(loop), depth=self.width, axis="x", crease_angle=30.0).generate()
        return _move(mesh, y=y0)

    def _straight(self) -> Mesh:
        return self._flight(self.steps)

    def _l_shaped(self) -> Mesh:
        n1 = max(1, min(self.steps - 1, int(round(self.steps * self.landing_at))))
        n2 = self.steps - n1
        w, r, t = self.width, self.riser, self.tread
        # Flight 1 climbs -Z from z = 0 and arrives at the landing's front edge.
        f1 = self._flight(n1)  # its top tread merges into the landing
        z_land = -(n1 - 1) * t
        h1 = n1 * r
        landing = _box(-w / 2, w / 2, max(0.0, h1 - self.waist), h1, z_land - w, z_land)
        sign = 1.0 if self.turn == "left" else -1.0
        # Flight 2 leaves the landing sideways (left = -X), climbing from h1.
        f2 = self._flight(n2, y0=h1)
        f2 = _move(f2, yaw_deg=sign * 90.0)
        f2 = _move(f2, x=-sign * w / 2, z=z_land - w / 2)
        return csg.union(f1, landing, f2, crease_angle=30.0)

    def _spiral(self) -> Mesh:
        from .primitives import CylinderGenerator
        from .profiles import ExtrudeGenerator

        pole = 0.09
        r_out = pole + self.width
        walk = pole + self.width * 0.6
        step_angle = self.tread / walk
        sign = 1.0 if self.turn == "left" else -1.0
        parts = [_move(CylinderGenerator(radius=pole, height=self.rise + 1.0).generate(), y=(self.rise + 1.0) / 2)]
        overlap = 0.25  # each tread overlaps the next a little (walkable, no gaps)
        for i in range(self.steps):
            a0 = sign * i * step_angle
            a1 = sign * (i + 1 + overlap) * step_angle
            angles = np.linspace(a0, a1, 8)
            outer = np.c_[np.sin(angles) * r_out, np.cos(angles) * r_out]
            inner = np.c_[np.sin(angles[::-1]) * pole * 0.5, np.cos(angles[::-1]) * pole * 0.5]
            sector = Shape(np.vstack([outer, inner]))
            thickness = 0.05
            wedge = ExtrudeGenerator(shape=sector, depth=thickness, axis="y", crease_angle=30.0).generate()
            # axis y maps profile (u, v) to (X, -Z); flip v so angles run as intended.
            wedge = wedge.transform(np.diag([1.0, 1.0, -1.0, 1.0]))
            wedge = Mesh(wedge.vertices, wedge.faces[:, ::-1], uvs=wedge.uvs)
            parts.append(_move(wedge, y=(i + 1) * self.riser - thickness / 2))
        return csg.union(*parts, crease_angle=30.0)

    # ------------------------------------------------------------ railings

    def railing_paths(self) -> list[np.ndarray]:
        """Handrail centre lines (before centring), one per rail."""
        h, r, t, w = self.rail_height, self.riser, self.tread, self.width
        inset = 0.05
        sides = {"both": (-1, 1), "left": (-1,), "right": (1,), "outer": (-1, 1), "none": ()}[self.railing]
        paths = []
        if self.style == "straight":
            for s in sides:
                x = s * (w / 2 - inset)
                paths.append(np.array([[x, r + h, -t / 2], [x, self.steps * r + h, -(self.steps - 0.5) * t]]))
        elif self.style == "l":
            n1 = max(1, min(self.steps - 1, int(round(self.steps * self.landing_at))))
            n2 = self.steps - n1
            z_land = -(n1 - 1) * t
            h1 = n1 * r
            sign = 1.0 if self.turn == "left" else -1.0
            outer_x = sign * (w / 2 - inset)       # the side away from the turn
            inner_x = -outer_x
            # Outer rail: up flight 1, round the landing's two outer edges, up flight 2.
            if self.railing in ("both", "outer") or (self.railing == ("right" if sign > 0 else "left")):
                paths.append(np.array([
                    [outer_x, r + h, -t / 2],
                    [outer_x, h1 + h, z_land],
                    [outer_x, h1 + h, z_land - w + inset],
                    [-sign * (w / 2), h1 + h, z_land - w + inset],
                    [-sign * (w / 2 + (n2 - 0.5) * t), h1 + n2 * r + h, z_land - w + inset],
                ]))
            if self.railing == "both":
                paths.append(np.array([[inner_x, r + h, -t / 2], [inner_x, h1 + h, z_land]]))
                paths.append(np.array([[-sign * (w / 2), h1 + r + h, z_land - inset],
                                       [-sign * (w / 2 + (n2 - 0.5) * t), h1 + n2 * r + h, z_land - inset]]))
        else:  # spiral: outer helix
            if self.railing != "none":
                pole = 0.09
                rr = pole + w - inset
                step_angle = t / (pole + w * 0.6)
                sign = 1.0 if self.turn == "left" else -1.0
                k = np.linspace(0.5, self.steps + 0.5, self.steps * 4 + 1)
                a = sign * k * step_angle
                paths.append(np.c_[np.sin(a) * rr, k * r + h, np.cos(a) * rr])
        return paths

    def railing_mesh(self) -> Mesh | None:
        from .primitives import CylinderGenerator
        from .sweep import SweepGenerator

        meshes = []
        for path in self.railing_paths():
            meshes.append(SweepGenerator(profile=Shape(circle(0.022, 12)), path=path, center=False).generate())
            # Posts: at the ends and about every metre, down to the nosing.
            lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            for s in np.linspace(0, lengths[-1], max(2, int(lengths[-1] // 1.0) + 2)):
                p = np.array([np.interp(s, lengths, path[:, k]) for k in range(3)])
                height = self.rail_height - 0.02
                post = CylinderGenerator(radius=0.015, height=height, segments=10).generate()
                meshes.append(_move(post, p[0], p[1] - height / 2 - 0.01, p[2]))
        if not meshes:
            return None
        return Mesh.merge(meshes).transform(np.array([[1, 0, 0, self._offset[0]], [0, 1, 0, self._offset[1]],
                                                      [0, 0, 1, self._offset[2]], [0, 0, 0, 1.0]]))

    def to_node(self, name: str | None = None) -> SceneNode:
        from ..materials.loader import MaterialLoader

        node = SceneNode(name=name or "stairs", mesh=self.generate())
        node.tags = ["architecture.stairs"]
        node.meta["walkable"] = True
        node.meta["collider"] = "mesh"
        node.meta["stairs"] = {"style": self.style, "rise": self.rise, "steps": self.steps,
                               "riser": round(self.riser, 4), "tread": self.tread}
        rails = self.railing_mesh()
        if rails is not None:
            rails.material = MaterialLoader().load(self.railing_material)
            rail_node = SceneNode(name=f"{node.name}_railing", mesh=uvmap.box_project(rails))
            rail_node.meta["collider"] = "none"
            node.add_child(rail_node)
        return node
