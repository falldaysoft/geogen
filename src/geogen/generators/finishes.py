"""Interior finishes for floor-plan rooms.

Per room (in the room's frame: origin at the clear-rectangle centre, floor
at y = 0):

- ``lining``: a 1 cm wall finish in the room's wall material, cut by the
  room's doors and windows, so each room can have its own paint or tiles;
- ``skirting``: a board swept around the perimeter, broken at doorways;
- ``cornice``: a cove swept around the ceiling line;
- ``light``: a pendant at the ceiling centre carrying ``meta.light`` (the
  Godot runtime turns it into an OmniLight3D);
- ``switch_*``: a light switch 1.1 m up beside each door's latch side
  (``meta.switch`` names the light, for interaction components).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.profile import Shape, rect
from ..core.transform import Transform

LINING = 0.01          # wall finish thickness
CASING = 0.07          # door architrave width (skirting stops at it)

SKIRTING_PROFILE = np.array([[0.0, 0.0], [0.014, 0.0], [0.014, 0.085], [0.008, 0.1], [0.0, 0.1]])
CORNICE_PROFILE = np.array([[0.0, -0.08], [0.01, -0.08], [0.02, -0.058], [0.035, -0.035], [0.055, -0.018],
                            [0.075, -0.01], [0.08, 0.0], [0.0, 0.0]])

DEFAULT_OPTIONS = {"lining": True, "skirting": True, "cornice": True, "light": True, "switches": True}


def _part(name: str, mesh: Mesh, material, tags: list[str] | None = None, collider: str | None = None) -> SceneNode:
    from ..core import uvmap

    if mesh.uvs is None:
        mesh = uvmap.box_project(mesh)
    mesh.material = material
    node = SceneNode(name=name, mesh=mesh, tags=list(tags or []))
    if collider:
        node.meta["collider"] = collider
    return node


def _perimeter(hx: float, hz: float):
    """Perimeter walked with the room on the right, as (corners, side order, length)."""
    corners = np.array([[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]])
    return corners, ("south", "east", "north", "west"), 4 * hx + 4 * hz


def _s_of(side: str, along: float, hx: float, hz: float) -> float:
    """Arc length along the perimeter for a point ``along`` a wall."""
    if side == "south":
        return along + hx
    if side == "east":
        return 2 * hx + along + hz
    if side == "north":
        return 2 * hx + 2 * hz + (hx - along)
    return 4 * hx + 2 * hz + (hz - along)


def _point_at(s: float, hx: float, hz: float) -> np.ndarray:
    corners, _, total = _perimeter(hx, hz)
    s = s % total
    lengths = [2 * hx, 2 * hz, 2 * hx, 2 * hz]
    for i, length in enumerate(lengths):
        if s <= length + 1e-9:
            a, b = corners[i], corners[(i + 1) % 4]
            return a + (b - a) * (s / length)
        s -= length
    return corners[0].copy()


def _skirting_paths(hx: float, hz: float, gaps: list[tuple[float, float]]) -> tuple[list[np.ndarray], bool]:
    """Open paths around the perimeter avoiding ``gaps`` (s intervals); or one closed loop."""
    corners, _, total = _perimeter(hx, hz)
    if not gaps:
        return [corners], True
    corner_s = [0.0, 2 * hx, 2 * hx + 2 * hz, 4 * hx + 2 * hz]
    gaps = sorted((max(0.0, a), min(total, b)) for a, b in gaps)
    runs = []
    for (a0, a1), (b0, _) in zip(gaps, gaps[1:] + [(gaps[0][0] + total, 0)]):
        if b0 - a1 > 0.02:
            runs.append((a1, b0))
    paths = []
    for s0, s1 in runs:
        pts = [_point_at(s0, hx, hz)]
        for cs in corner_s + [c + total for c in corner_s]:
            if s0 + 1e-6 < cs < s1 - 1e-6:
                pts.append(_point_at(cs, hx, hz))
        pts.append(_point_at(s1, hx, hz))
        paths.append(np.array(pts))
    return paths, False


def _to3(path2: np.ndarray, y: float) -> np.ndarray:
    return np.column_stack([path2[:, 0], np.full(len(path2), y), path2[:, 1]])


def add_room_finishes(room: SceneNode, clear_height: float, materials: dict[str, str], cutters: list[Mesh],
                      options: dict[str, bool], loader) -> None:
    """Add lining, skirting, cornice, light and switches under ``room``."""
    from ..core import csg
    from .profiles import ExtrudeGenerator
    from .sweep import SweepGenerator

    opts = {**DEFAULT_OPTIONS, **(options or {})}
    hx, hz = float(room.size[0]) / 2, float(room.size[2]) / 2
    # Furnishing keeps floor items off the skirting and hangs wall items on the lining.
    room.meta["wall_inset"] = {"floor": (LINING if opts["lining"] else 0.0) + (0.014 if opts["skirting"] else 0.0),
                               "mount": LINING if opts["lining"] else 0.0}
    openings = room.meta.get("openings", [])
    doors = [o for o in openings if o["kind"] == "door"]

    if opts["lining"]:
        shape = Shape(rect(2 * hx, 2 * hz), [rect(2 * hx - 2 * LINING, 2 * hz - 2 * LINING)])
        mesh = ExtrudeGenerator(shape=shape, depth=clear_height, axis="y", crease_angle=30.0).generate()
        move = np.eye(4)
        move[1, 3] = clear_height / 2
        mesh = mesh.transform(move)
        if cutters:
            mesh = csg.difference(mesh, *cutters, crease_angle=30.0)
        room.add_child(_part("lining", mesh, loader.load(materials["walls"]), ["wall.lining"], collider="none"))

    ix, iz = hx - LINING, hz - LINING  # skirting and cornice sit on the lining face
    trim = loader.load(materials.get("trim", "trim_white"))
    if opts["skirting"]:
        gaps = []
        for d in doors:
            # _s_of runs backwards along the north and west walls.
            a, b = sorted((_s_of(d["side"], d["lo"], ix, iz), _s_of(d["side"], d["hi"], ix, iz)))
            gaps.append((a - CASING, b + CASING))
        paths, closed = _skirting_paths(ix, iz, gaps)
        meshes = [SweepGenerator(profile=Shape(SKIRTING_PROFILE), path=_to3(p, 0.0), closed=closed,
                                 center=False, crease_angle=30.0).generate() for p in paths]
        if meshes:
            room.add_child(_part("skirting", Mesh.merge(meshes), trim, ["trim.skirting"], collider="none"))
    if opts["cornice"]:
        corners, _, _ = _perimeter(ix, iz)
        mesh = SweepGenerator(profile=Shape(CORNICE_PROFILE), path=_to3(corners, clear_height), closed=True,
                              center=False, crease_angle=30.0).generate()
        room.add_child(_part("cornice", mesh, loader.load(materials.get("ceiling", "ceiling_white")),
                             ["trim.cornice"], collider="none"))

    light_name = f"{room.name}_light"
    if opts["light"]:
        room.add_child(_pendant(light_name, clear_height, loader))
    if opts["switches"]:
        for i, d in enumerate(doors):
            switch = _switch(f"{room.name}_switch_{i + 1}", d, hx, hz, room.meta.get("door_swings", []), loader)
            if switch is not None:
                switch.meta["switch"] = {"light": light_name}
                room.add_child(switch)


def _pendant(name: str, ceiling: float, loader) -> SceneNode:
    from .primitives import CylinderGenerator
    from .profiles import LatheGenerator

    node = SceneNode(name=name, transform=Transform(translation=np.array([0.0, ceiling, 0.0])),
                     tags=["light.ceiling"])
    node.meta["light"] = {"type": "omni", "color": [1.0, 0.93, 0.82], "energy": 1.2, "range": 6.0,
                          "offset": [0.0, -0.47, 0.0]}  # just below the shade
    chrome = loader.load("chrome")

    def at(mesh: Mesh, y: float) -> Mesh:
        move = np.eye(4)
        move[1, 3] = y
        return mesh.transform(move)

    canopy = at(CylinderGenerator(radius=0.06, height=0.025).generate(), -0.0125)
    stem = at(CylinderGenerator(radius=0.006, height=0.4).generate(), -0.225)
    shade = LatheGenerator(profile=np.array([[0.02, 0.0], [0.2, 0.0], [0.17, 0.18], [0.02, 0.18]]),
                           segments=40).generate()
    shade = at(shade, -0.425)
    node.add_child(_part("canopy", Mesh.merge([canopy, stem]), chrome, collider="none"))
    node.add_child(_part("shade", shade, loader.load("lamp_shade"), collider="none"))
    return node


def _switch(name: str, door: dict[str, Any], hx: float, hz: float, swings: list[dict], loader) -> SceneNode | None:
    """Switch plate beside the door's latch side (away from the hinge), 1.1 m up."""
    from .primitives import CubeGenerator

    side = door["side"]
    along_axis = 0 if side in ("north", "south") else 1
    half = hx if along_axis == 0 else hz
    lo, hi = door["lo"], door["hi"]
    hinge_along = None
    for s in swings:
        h = np.array([s["hinge"][0], s["hinge"][2]])
        if lo - 0.1 <= h[along_axis] <= hi + 0.1:
            hinge_along = h[along_axis]
    candidates = [hi + CASING + 0.08, lo - CASING - 0.08]
    if hinge_along is not None and abs(hinge_along - hi) < abs(hinge_along - lo):
        candidates.reverse()  # hinge at the hi end: latch is at lo
    along = next((c for c in candidates if abs(c) + 0.05 < half), None)
    if along is None:
        return None
    inward = {"north": (0.0, -1.0), "south": (0.0, 1.0), "east": (-1.0, 0.0), "west": (1.0, 0.0)}[side]
    wall = {"north": (along, hz), "south": (along, -hz), "east": (hx, along), "west": (-hx, along)}[side]
    face = np.array(wall) + np.array(inward) * LINING
    yaw = float(np.arctan2(inward[0], inward[1]))
    node = SceneNode(name=name, transform=Transform(translation=np.array([face[0], 1.1, face[1]]),
                                                    rotation=np.array([0.0, yaw, 0.0])), tags=["switch.light"])
    plate = CubeGenerator(size_x=0.085, size_y=0.085, size_z=0.01, bevel=0.002).generate()
    rocker = CubeGenerator(size_x=0.03, size_y=0.05, size_z=0.008, bevel=0.002).generate()
    move = np.eye(4)
    move[2, 3] = 0.005
    plate = plate.transform(move)
    move[2, 3] = 0.013
    rocker = rocker.transform(move)
    node.add_child(_part("plate", plate, loader.load("paint_white"), collider="none"))
    rocker_node = _part("rocker", rocker, loader.load("paint_white"), collider="none")
    node.add_child(rocker_node)
    node.interactions = [_switch_interaction(rocker_node)]
    return node


def _switch_interaction(rocker: SceneNode):
    """On/off: the rocker tips 12 degrees; runtimes toggle meta.switch.light on arrival."""
    from ..layout.interactions import Interaction, Motion, State

    return Interaction(
        name="switch",
        states={"on": State(next="off", prompt="Lights off", emit="lights_on"),
                "off": State(next="on", prompt="Lights on", emit="lights_off")},
        motions=[Motion([rocker], "rotate", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.01]),
                        {"on": 0.0, "off": 12.0})],
        targets=[rocker], initial="on", duration=0.15,
    )
