"""Building façades generated from the floor plans' exterior openings.

``facade: {style: brick_hotel | stucco | modern, balconies: false}`` under
``building:`` dresses the outside of every storey:

- cladding: the style's material on the structural walls (``exterior`` plan
  material; rooms keep their interior linings);
- windows: a frame (with glazing bars on wide windows) and glass set back
  in each exterior window opening, plus a sill and lintel where the style
  has them; wide low windows on the ground floor read as storefronts;
- string courses at every floor line and a cornice under the parapet;
- a canopy over each exterior door on the ground floor;
- optional Juliet balcony rails on upper-floor guest room windows.

Façade parts are merged per material and storey under a ``facade`` node
(``facade_<n>`` per storey, ``meta.storey``; few meshes, no colliders
except canopies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.profile import Shape

WINDOW_RECESS = 0.1     # glass set back from the outer wall face
FRAME = 0.06            # frame member width


@dataclass(frozen=True)
class FacadeStyle:
    cladding: str
    trim: str               # sills, lintels, string courses, cornice
    frame: str
    sills: bool
    lintels: bool
    band_height: float
    cornice: bool


STYLES = {
    "brick_hotel": FacadeStyle(cladding="brick", trim="concrete", frame="trim_white", sills=True, lintels=True,
                               band_height=0.25, cornice=True),
    "stucco": FacadeStyle(cladding="wall_plaster", trim="paint_white", frame="plastic_black", sills=True,
                          lintels=False, band_height=0.18, cornice=True),
    "modern": FacadeStyle(cladding="concrete", trim="metal", frame="metal", sills=False, lintels=False,
                          band_height=0.12, cornice=False),
}

_INWARD = {"north": np.array([0.0, 0.0, -1.0]), "south": np.array([0.0, 0.0, 1.0]),
           "east": np.array([-1.0, 0.0, 0.0]), "west": np.array([1.0, 0.0, 0.0])}


def _box(lo, hi) -> Mesh:
    from .primitives import CubeGenerator

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    size = hi - lo
    mesh = CubeGenerator(size_x=size[0], size_y=size[1], size_z=size[2], bevel=0).generate()
    m = np.eye(4)
    m[:3, 3] = (lo + hi) / 2
    return mesh.transform(m)


def _oriented(mesh: Mesh, origin: np.ndarray, along: np.ndarray, out: np.ndarray) -> Mesh:
    """Map a mesh built in (x = along the wall, y = up, z = out of the wall) into place."""
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = along, [0.0, 1.0, 0.0], out, origin
    return mesh.transform(m)


def _window_parts(width: float, height: float, style: FacadeStyle, storefront: bool) -> dict[str, list[Mesh]]:
    """Window pieces in a local frame: x along the wall, y up from the sill, z out (0 = outer face)."""
    from .profiles import ExtrudeGenerator

    parts: dict[str, list[Mesh]] = {"frame": [], "glass": [], "trim": []}
    z_frame = -WINDOW_RECESS
    outline = Shape(np.array([[-width / 2, 0], [width / 2, 0], [width / 2, height], [-width / 2, height]]),
                    [np.array([[-width / 2 + FRAME, FRAME], [-width / 2 + FRAME, height - FRAME],
                               [width / 2 - FRAME, height - FRAME], [width / 2 - FRAME, FRAME]])])
    frame = ExtrudeGenerator(shape=outline, depth=0.07, axis="z", crease_angle=30.0).generate()
    m = np.eye(4)
    m[2, 3] = z_frame
    parts["frame"].append(frame.transform(m))
    # Glazing bars: one per ~1.2 m of width (storefronts get slimmer, more frequent mullions).
    bays = max(1, int(round(width / (1.5 if storefront else 1.2))))
    for i in range(1, bays):
        x = -width / 2 + i * width / bays
        parts["frame"].append(_box([x - 0.02, FRAME, z_frame - 0.03], [x + 0.02, height - FRAME, z_frame + 0.03]))
    if not storefront and height > 1.2:
        y = height * 0.65
        parts["frame"].append(_box([-width / 2 + FRAME, y - 0.02, z_frame - 0.03],
                                   [width / 2 - FRAME, y + 0.02, z_frame + 0.03]))
    parts["glass"].append(_box([-width / 2 + FRAME, FRAME, z_frame - 0.006], [width / 2 - FRAME, height - FRAME,
                                                                               z_frame + 0.006]))
    if style.sills and not storefront:
        parts["trim"].append(_box([-width / 2 - 0.06, -0.06, -WINDOW_RECESS - 0.02], [width / 2 + 0.06, 0.0, 0.06]))
    if style.lintels:
        parts["trim"].append(_box([-width / 2 - 0.1, height, -0.02], [width / 2 + 0.1, height + 0.16, 0.015]))
    return parts


def build_facade(building: SceneNode, storeys: list[tuple[SceneNode, Any]], spec: dict[str, Any],
                 material_loader) -> SceneNode:
    """Façade node for a building (see module docstring)."""
    from ..core import uvmap
    from .sweep import SweepGenerator

    style_name = spec.get("style", "brick_hotel")
    if style_name not in STYLES:
        raise ValueError(f"facade style must be one of {sorted(STYLES)}, got {style_name!r}")
    style = STYLES[style_name]
    # Façade pieces per storey (index -> group -> meshes), so storeys can be hidden together.
    per_storey: dict[int, dict[str, list[Mesh]]] = {}

    def bucket(index: int) -> dict[str, list[Mesh]]:
        return per_storey.setdefault(index, {"frame": [], "glass": [], "trim": [], "canopy": [], "rail": []})

    to_building = np.linalg.inv(building.world_transform())

    for index, (storey, plan) in enumerate(storeys):
        groups = bucket(index)
        thickness = plan.exterior_wall
        for room in storey.iter_nodes():
            if not isinstance(room.meta.get("room"), dict) or room.meta.get("type") == "room_volume":
                continue
            m = to_building @ room.world_transform()
            centre = m[:3, 3]
            half = np.asarray(room.size, dtype=float) / 2
            for o in room.meta.get("openings", []):
                side = o["side"]
                if f"{side}_exterior" not in room.surfaces:
                    continue
                inward = _INWARD[side]
                out = -inward
                along = np.array([-out[2], 0.0, out[0]])  # left -> right seen from outside
                mid = (o["lo"] + o["hi"]) / 2
                axis = 0 if side in ("north", "south") else 2
                face = centre.copy()
                face[axis] += mid
                other = 2 - axis
                face[other] += out[other] * (half[other] + thickness)
                width = o["hi"] - o["lo"]
                if o["kind"] == "window":
                    storefront = index == 0 and o["sill"] < 0.6
                    origin = face + np.array([0.0, o["sill"], 0.0])
                    for group, meshes in _window_parts(width, o["top"] - o["sill"], style, storefront).items():
                        groups[group] += [_oriented(mesh, origin, along, out) for mesh in meshes]
                    if spec.get("balconies") and index > 0 and room.meta["room"]["type"] == "hotel_bedroom":
                        groups["rail"] += [_oriented(mesh, face + np.array([0.0, o["sill"], 0.0]), along, out)
                                           for mesh in _juliet_rail(width)]
                elif o["kind"] == "door" and index == 0:
                    y = o["top"] + 0.35
                    canopy = _box([-width / 2 - 0.6, y, 0.0], [width / 2 + 0.6, y + 0.15, 1.6])
                    groups["canopy"].append(_oriented(canopy, face, along, out))

    # String courses at each floor line, cornice at the top.
    first = storeys[0][1]
    x0, z0, x1, z1 = first.bounds
    t = first.exterior_wall
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    hx, hz = (x1 - x0 + t) / 2, (z1 - z0 + t) / 2
    ring = np.array([[-hx, 0, -hz], [-hx, 0, hz], [hx, 0, hz], [hx, 0, -hz]])  # right of travel = outward
    band = Shape(np.array([[0.0, 0.0], [0.04, 0.0], [0.04, style.band_height], [0.0, style.band_height]]))
    for index, (storey, plan) in enumerate(storeys[1:], start=1):
        y = float((to_building @ storey.world_transform())[1, 3]) - plan.floor_thickness - style.band_height / 2
        path = ring + np.array([0.0, y, 0.0])
        bucket(index)["trim"].append(SweepGenerator(profile=band, path=path, closed=True, center=False).generate())
    if style.cornice:
        top_storey, top_plan = storeys[-1]
        y = float((to_building @ top_storey.world_transform())[1, 3]) + top_plan.wall_height - 0.25
        cornice = Shape(np.array([[0.0, 0.0], [0.05, 0.0], [0.05, 0.08], [0.14, 0.16], [0.14, 0.25], [0.0, 0.25]]))
        bucket(len(storeys) - 1)["trim"].append(SweepGenerator(profile=cornice, path=ring + np.array([0.0, y, 0.0]),
                                                                closed=True, center=False).generate())

    node = SceneNode(name="facade", tags=["facade", f"facade.{style_name}"])
    materials = {"frame": style.frame, "glass": "glass", "trim": style.trim, "canopy": style.trim,
                 "rail": "metal"}
    for index in sorted(per_storey):
        level = SceneNode(name=f"facade_{index}", tags=["facade.storey"])
        level.meta["storey"] = {"index": index}
        for group, meshes in per_storey[index].items():
            if not meshes:
                continue
            mesh = uvmap.box_project(Mesh.merge(meshes))
            mesh.material = material_loader.load(materials[group])
            part = SceneNode(name=f"facade_{group}", mesh=mesh)
            part.meta["collider"] = "box" if group == "canopy" else "none"
            level.add_child(part)
        if level.children:
            node.add_child(level)
    node.meta["facade"] = {"style": style_name}
    return node


def _juliet_rail(width: float) -> list[Mesh]:
    """A low rail across the window opening, just outside the frame (local window frame)."""
    rails = []
    z = 0.06
    for y in (0.25, 0.45):
        rails.append(_box([-width / 2 - 0.05, y, z], [width / 2 + 0.05, y + 0.03, z + 0.03]))
    for x in np.linspace(-width / 2, width / 2, max(3, int(width / 0.12))):
        rails.append(_box([x - 0.008, 0.0, z + 0.008], [x + 0.008, 0.48, z + 0.024]))
    return rails
