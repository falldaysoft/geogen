"""Multi-storey buildings: stacked floor plans, stair cores and a roof.

YAML (an asset with ``building:`` instead of ``parts:``)::

    building:
      storeys:
        - floorplan: { generate: hotel_lobby, length: 30, depth: 15, wall_height: 3.6 }
        - floorplan: { generate: hotel_corridor, length: 30, depth: 15 }
          repeat: 3                # typical floors
          furnish: true            # furnish rooms from their archetypes
      roof: { parapet: 1.0, material: concrete }

Each storey is a ``storey_<n>`` node (tags ``storey``, ``storey.<n>``; meta
``storey: {index, elevation, height}``) holding its floor plan, raised so its
floor slab sits on the walls below. Layouts that take a ``floor`` param get
the storey number. Every room of type ``stair`` gets a straight flight up to
the room above (same position on the next storey), and the floor above and
the ceiling below are cut open over the flight. A flat roof slab with a
parapet caps the top storey.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.transform import Transform
from .floorplan import FloorPlan

STAIR_WIDTH = 1.2
DOOR_CLEAR = 0.9      # free floor between a stair room's door and the bottom step


def _box(lo: np.ndarray, hi: np.ndarray) -> Mesh:
    from .primitives import CubeGenerator

    size = hi - lo
    mesh = CubeGenerator(size_x=size[0], size_y=size[1], size_z=size[2], bevel=0).generate()
    m = np.eye(4)
    m[:3, 3] = (lo + hi) / 2
    return mesh.transform(m)


def _storey_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    storeys = []
    for entry in spec.get("storeys") or []:
        if "floorplan" not in entry:
            raise ValueError("each building storey needs a floorplan:")
        for _ in range(int(entry.get("repeat", 1))):
            storeys.append(entry)
    if not storeys:
        raise ValueError("building needs at least one storey under 'storeys:'")
    return storeys


def build_building(spec: dict[str, Any], name: str, material_loader, assets_dir: Path | None = None,
                   layout_loader=None) -> SceneNode:
    from ..core import csg, uvmap
    from .plan_layouts import LAYOUTS

    root = SceneNode(name=name)
    root.tags = ["building"]
    entries = _storey_specs(spec)
    facade = spec.get("facade")
    plans = []
    for index, entry in enumerate(entries):
        plan_spec = copy.deepcopy(entry["floorplan"])
        if facade:
            from .facade import STYLES

            style = STYLES.get(facade.get("style", "brick_hotel"))
            if style is not None:
                plan_spec.setdefault("materials", {})["exterior"] = style.cladding
        layout = LAYOUTS.get(plan_spec.get("generate", ""))
        if layout is not None and "floor" in inspect.signature(layout).parameters and "floor" not in plan_spec:
            plan_spec["floor"] = index
        plans.append(FloorPlan.from_spec(plan_spec))

    elevation = 0.0
    storeys: list[tuple[SceneNode, FloorPlan]] = []
    for index, (entry, plan) in enumerate(zip(entries, plans)):
        storey = plan.build(f"storey_{index}", material_loader)
        storey.tags = ["storey", f"storey.{index}"]
        storey.meta["storey"] = {"index": index, "elevation": round(elevation, 6), "height": plan.wall_height}
        storey.transform = Transform(translation=np.array([0.0, elevation, 0.0]))
        root.add_child(storey)
        storeys.append((storey, plan))
        if entry.get("furnish") and assets_dir is not None:
            from ..layout.furnish import furnish_plan

            seed = entry["furnish"] if isinstance(entry["furnish"], int) and entry["furnish"] is not True else index
            furnish_plan(storey, assets_dir, seed=seed, loader=layout_loader)
        if index + 1 < len(plans):
            # The next floor slab sits on top of these walls.
            elevation += plan.wall_height + plans[index + 1].floor_thickness

    for (lower, lower_plan), (upper, _) in zip(storeys, storeys[1:]):
        _connect_stairs(lower, lower_plan, upper, material_loader)

    if facade:
        from .facade import build_facade

        root.add_child(build_facade(root, storeys, facade, material_loader))

    top, top_plan = storeys[-1]
    roof = spec.get("roof", {})
    if roof is not False:
        root.add_child(_roof(top, top_plan, roof or {}, material_loader))

    x0, z0, x1, z1 = storeys[0][1].bounds
    t = storeys[0][1].exterior_wall
    top_y = elevation + top_plan.wall_height
    root.size = np.array([x1 - x0 + t, top_y + (roof or {}).get("parapet", 1.0), z1 - z0 + t])
    root.meta["building"] = {"storeys": len(storeys), "height": round(top_y, 6)}
    return root


def _connect_stairs(lower: SceneNode, lower_plan: FloorPlan, upper: SceneNode, material_loader) -> None:
    """A straight flight in each lower stair room up to the matching upper room."""
    from ..core import csg
    from .stairs import StairsGenerator

    rise = float(upper.transform.translation[1] - lower.transform.translation[1])
    uppers = {n.meta["room"]["id"]: n for n in upper.iter_nodes()
              if isinstance(n.meta.get("room"), dict) and n.meta.get("type") != "room_volume"}
    for room in [n for n in lower.iter_nodes() if isinstance(n.meta.get("room"), dict)
                 and n.meta.get("type") != "room_volume" and n.meta["room"]["type"] == "stair"]:
        above = uppers.get(room.meta["room"]["id"])
        if above is None:
            continue
        sx, sz = float(room.size[0]), float(room.size[2])
        along_z = sz >= sx
        length, across = (sz, sx) if along_z else (sx, sz)
        width = min(STAIR_WIDTH, across - 0.4)
        gen = StairsGenerator(style="straight", rise=rise, width=width, railing="both")
        run = gen.steps * gen.tread
        # Which half of the room holds the doors? Climb away from them.
        doors = [o for o in room.meta.get("openings", []) if o["kind"] == "door"]
        door_pos = np.mean([(o["lo"] + o["hi"]) / 2 for o in doors]) if doors else 0.0
        side_doors = [o for o in doors if o["side"] in (("east", "west") if along_z else ("north", "south"))]
        # The flight hugs the long wall without doors (or the first one).
        wall_side = -1.0
        if side_doors and all(o["side"] in ("west", "south") for o in side_doors):
            wall_side = 1.0
        offset_across = wall_side * (across / 2 - width / 2 - 0.02)
        climb = -1.0 if door_pos >= 0 else 1.0     # up and away from the doors
        start = door_pos + climb * DOOR_CLEAR if doors else climb * -length / 2 + climb * 0.2
        start = float(np.clip(start, -length / 2 + 0.2, length / 2 - 0.2))
        if abs(start + climb * run) > length / 2 - 0.2:
            continue  # the room is too short for a straight flight of this rise
        centre_along = start + climb * run / 2
        node = gen.to_node("stairs")
        node.mesh.material = material_loader.load("concrete")
        # Local stairs climb -Z; rotate so they climb along +/-Z or +/-X of the room.
        if along_z:
            yaw = 0.0 if climb < 0 else 180.0
            pos = np.array([offset_across, rise / 2, centre_along])
        else:
            yaw = 90.0 if climb < 0 else -90.0
            pos = np.array([centre_along, rise / 2, offset_across])
        node.transform = Transform(translation=pos, rotation=np.array([0.0, np.radians(yaw), 0.0]))
        room.add_child(node)

        # Open the slab above the flight (upper floor and lower ceiling).
        half = np.array([width / 2, run / 2]) + 0.05
        half_xz = half if along_z else half[::-1]
        lo = np.array([pos[0] - half_xz[0], 0.0, pos[2] - half_xz[1]])
        hi = np.array([pos[0] + half_xz[0], 0.0, pos[2] + half_xz[1]])
        for target, y0, y1 in ((room.find("ceiling"), -1.0, rise + 1.0), (above.find("floor"), -rise - 1.0, 1.0)):
            if target is None or target.mesh is None:
                continue
            cutter = _box(np.array([lo[0], y0, lo[2]]), np.array([hi[0], y1, hi[2]]))
            target.mesh = csg.difference(target.mesh, cutter, crease_angle=30.0)
        room.meta.setdefault("stairs", []).append({"to": above.meta["room"]["id"], "rise": rise,
                                                   "steps": gen.steps})


def _roof(top: SceneNode, plan: FloorPlan, roof: dict[str, Any], material_loader) -> SceneNode:
    """Flat roof slab over the top storey plus a parapet around its edge."""
    from ..core import csg, uvmap
    from .primitives import CubeGenerator

    size = plan.size
    t = plan.exterior_wall
    slab_t = float(roof.get("thickness", 0.25))
    parapet = float(roof.get("parapet", 1.0))
    y = float(top.transform.translation[1]) + plan.wall_height
    outer_lo = np.array([-size[0] / 2, y, -size[2] / 2])
    outer_hi = np.array([size[0] / 2, y + slab_t, size[2] / 2])
    slab = _box(outer_lo, outer_hi)
    parts = [slab]
    if parapet > 0:
        ring = csg.difference(
            _box(np.array([outer_lo[0], y + slab_t, outer_lo[2]]), np.array([outer_hi[0], y + slab_t + parapet, outer_hi[2]])),
            _box(np.array([outer_lo[0] + t, y, outer_lo[2] + t]),
                 np.array([outer_hi[0] - t, y + slab_t + parapet + 1, outer_hi[2] - t])))
        parts.append(ring)
    mesh = uvmap.box_project(csg.union(*parts, crease_angle=30.0))
    mesh.material = material_loader.load(roof.get("material", "concrete"))
    node = SceneNode(name="roof", mesh=mesh, tags=["roof"])
    node.meta["walkable"] = True
    return node
