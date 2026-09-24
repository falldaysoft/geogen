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
the storey number. Stair rooms stacked over several storeys become scissor
cores (see ``_connect_all_stairs``); the floor above and ceiling below are
cut open over each flight. A flat roof slab with a
parapet caps the top storey.
"""

from __future__ import annotations

import copy
import logging
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.transform import Transform
from .floorplan import FloorPlan

logger = logging.getLogger(__name__)

STAIR_WIDTH = 1.2
DOOR_CLEAR = 0.9      # free floor in front of a stair room's end-wall door
WALKWAY = 1.2         # clear width beside the flights (door side)


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

    _connect_all_stairs([s for s, _ in storeys], material_loader)
    _add_lifts(root, [s for s, _ in storeys], material_loader)
    _add_entrance_spawn(root, storeys[0][0])

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
    roof_spec = roof or {}
    if roof_spec.get("style", "flat") in ("gable", "hip", "shed"):
        cap = float(roof_spec.get("rise", 0.35 * min(x1 - x0 + t, z1 - z0 + t)))
    else:
        cap = float(roof_spec.get("parapet", 1.0))
    root.size = np.array([x1 - x0 + t, top_y + cap, z1 - z0 + t])
    root.meta["building"] = {"storeys": len(storeys), "height": round(top_y, 6)}
    return root


LIFT_SPEED = 1.2      # m/s
CAR_HEIGHT = 2.3


def _shaft_rooms(storey: SceneNode) -> dict[str, SceneNode]:
    return {n.meta["room"]["id"]: n for n in storey.iter_nodes()
            if isinstance(n.meta.get("room"), dict) and n.meta.get("type") != "room_volume"
            and n.meta["room"]["type"] == "lift_shaft"}


def _add_lifts(root: SceneNode, storeys: list[SceneNode], material_loader) -> None:
    """Turn lift shafts stacked over storeys into a working lift.

    The shaft is opened up (no floors above the bottom, no ceilings below
    the top, no fittings), a car rides it as a ``lift`` interaction whose
    states are the floors (using it goes to the next floor, wrapping to the
    bottom), and each floor's shaft doorway gets a gate that's solid unless
    the car is standing at that floor (``meta.gate``; see the Godot runtime).
    """
    from ..core import uvmap
    from ..layout.interactions import Interaction, Motion, State
    from .primitives import CubeGenerator

    for shaft_id in sorted(set().union(*(_shaft_rooms(s).keys() for s in storeys))):
        column = [(s, _shaft_rooms(s).get(shaft_id)) for s in storeys]
        column = [(s, r) for s, r in column if r is not None]
        if len(column) < 2:
            continue
        for i, (storey, room) in enumerate(column):
            drop = {"skirting", "cornice", f"{room.name}_light"} | ({"floor"} if i > 0 else set()) \
                | ({"ceiling"} if i < len(column) - 1 else set())
            room.children = [c for c in room.children if c.name not in drop and "switch.light" not in c.tags]
            room.meta["nav"] = False  # the shaft volume isn't somewhere to walk
            volume = room.find(f"{room.name}_volume")
            if volume is not None:
                volume.meta["nav"] = False
        bottom = column[0][1]
        to_root = np.linalg.inv(root.world_transform())
        m = to_root @ bottom.world_transform()
        sx, sz = float(bottom.size[0]), float(bottom.size[2])
        elevations = [float((to_root @ s.world_transform())[1, 3]) for s, _ in column]

        lift = SceneNode(name=f"lift_{shaft_id}", transform=Transform(translation=m[:3, 3].copy()))
        lift.tags = ["lift"]
        car = SceneNode(name="lift_car", transform=Transform(translation=np.array([0.0, elevations[0], 0.0])))
        car.tags = ["lift.car"]
        lift.add_child(car)
        opening = next((o for o in bottom.meta.get("openings", []) if o["kind"] == "door"), None)
        front = opening["side"] if opening else "south"
        inward = {"north": np.array([0, 0, -1.0]), "south": np.array([0, 0, 1.0]),
                  "east": np.array([-1.0, 0, 0]), "west": np.array([1.0, 0, 0])}[front]
        cw, cd = sx - 0.1, sz - 0.1
        steel = material_loader.load("metal")
        panel_mat = material_loader.load("wood_dark")

        def part(name: str, lo, hi, material, collider: str = "box") -> SceneNode:
            lo, hi = np.asarray(lo, float), np.asarray(hi, float)
            mesh = CubeGenerator(size_x=hi[0] - lo[0], size_y=hi[1] - lo[1], size_z=hi[2] - lo[2], bevel=0).generate()
            move = np.eye(4)
            move[:3, 3] = (lo + hi) / 2
            mesh = uvmap.box_project(mesh.transform(move))
            mesh.material = material
            node = SceneNode(name=name, mesh=mesh)
            node.meta["collider"] = collider
            return node

        # 4 mm proud of the landing floor so the two don't z-fight while the car is parked.
        car.add_child(part("car_floor", [-cw / 2, -0.12, -cd / 2], [cw / 2, 0.004, cd / 2], steel))
        car.add_child(part("car_ceiling", [-cw / 2, CAR_HEIGHT, -cd / 2], [cw / 2, CAR_HEIGHT + 0.06, cd / 2], steel))
        # Three walls; the open side faces the doorway.
        walls = {"north": ([-cw / 2, 0, cd / 2 - 0.04], [cw / 2, CAR_HEIGHT, cd / 2]),
                 "south": ([-cw / 2, 0, -cd / 2], [cw / 2, CAR_HEIGHT, -cd / 2 + 0.04]),
                 "east": ([cw / 2 - 0.04, 0, -cd / 2], [cw / 2, CAR_HEIGHT, cd / 2]),
                 "west": ([-cw / 2, 0, -cd / 2], [-cw / 2 + 0.04, CAR_HEIGHT, cd / 2])}
        for side, (lo, hi) in walls.items():
            if side != front:
                car.add_child(part(f"car_wall_{side}", lo, hi, panel_mat))
        # Button panel beside the opening, on the wall to its right.
        right = np.array([-inward[2], 0.0, inward[0]])
        panel_centre = -inward * 0.0 + right * (min(cw, cd) / 2 - 0.05) + np.array([0, 1.2, 0]) - inward * (
            (cd if front in ("north", "south") else cw) / 2 - 0.35)
        half = np.abs(right) * 0.02 + np.abs(inward) * 0.1 + np.array([0, 0.18, 0])
        panel = part("car_buttons", panel_centre - half, panel_centre + half, steel, collider="none")
        panel.tags = ["lift.buttons"]
        car.add_child(panel)

        states = {}
        for k in range(len(column)):
            nxt = (k + 1) % len(column)
            states[f"floor_{k}"] = State(next=f"floor_{nxt}", prompt=f"Go to floor {nxt}", emit=f"arrived_{k}")
        travel = max(elevations) - min(elevations)
        lift.interactions = [Interaction(
            name="lift", states=states,
            motions=[Motion([car], "translate", np.array([0.0, 1.0, 0.0]), np.zeros(3),
                            {f"floor_{k}": e - elevations[0] for k, e in enumerate(elevations)})],
            targets=[panel], initial="floor_0", duration=max(1.0, travel / LIFT_SPEED),
        )]
        # Gates in each floor's shaft doorway (the wall's centre line).
        for k, (storey, room) in enumerate(column):
            door = next((o for o in room.meta.get("openings", []) if o["kind"] == "door"), None)
            if door is None:
                continue
            mr = to_root @ room.world_transform()
            half_room = np.array([float(room.size[0]), 0.0, float(room.size[2])]) / 2
            axis = 0 if door["side"] in ("north", "south") else 2
            normal = -{"north": np.array([0, 0, -1.0]), "south": np.array([0, 0, 1.0]),
                       "east": np.array([-1.0, 0, 0]), "west": np.array([1.0, 0, 0])}[door["side"]]
            centre = mr[:3, 3].copy()
            centre[axis] += (door["lo"] + door["hi"]) / 2
            other = 2 - axis
            centre[other] += normal[other] * (half_room[other] + 0.06)
            extent = np.zeros(3)
            extent[axis] = (door["hi"] - door["lo"]) / 2
            extent[other] = 0.03
            extent[1] = door["top"] / 2
            centre[1] = elevations[k] + door["top"] / 2
            gate = part(f"lift_gate_{k}", centre - extent - m[:3, 3], centre + extent - m[:3, 3], steel)
            gate.tags = ["lift.gate"]
            gate.meta["gate"] = {"interaction": "lift", "open_in": f"floor_{k}"}
            lift.add_child(gate)
        root.add_child(lift)


def _add_entrance_spawn(root: SceneNode, ground: SceneNode) -> None:
    """A spawn point 2.5 m outside the first ground-floor exterior door, facing it."""
    to_root = np.linalg.inv(root.world_transform())
    for door in ground.iter_nodes():
        if "door.exterior" not in door.tags:
            continue
        m = to_root @ door.world_transform()
        # Door +Z faces the swing room (inside); outside is -Z.
        inward = m[:3, 2] / np.linalg.norm(m[:3, 2])
        position = m[:3, 3] - inward * 2.5
        position[1] = 0.0
        yaw = float(np.arctan2(inward[0], inward[2]))
        spawn = SceneNode(name="entrance_spawn", transform=Transform(translation=position,
                                                                   rotation=np.array([0.0, yaw, 0.0])))
        spawn.tags = ["spawn"]
        spawn.meta["type"] = "spawn"
        root.add_child(spawn)
        return


def _stair_rooms(storey: SceneNode) -> dict[str, SceneNode]:
    return {n.meta["room"]["id"]: n for n in storey.iter_nodes()
            if isinstance(n.meta.get("room"), dict) and n.meta.get("type") != "room_volume"
            and n.meta["room"]["type"] == "stair"}


def _connect_all_stairs(storeys: list[SceneNode], material_loader) -> None:
    """Scissor stair cores for stair rooms stacked over several storeys.

    Every flight runs along the long wall away from the room's doors, so the
    door side stays a clear walkway on every floor. Flights alternate halves:
    even ones climb toward one end, odd ones toward the other, each starting
    near the middle; a player arriving at the far end of one flight walks
    back along the walkway to the start of the next. Flights two storeys
    apart share a footprint but are separated by the slab in between.
    """
    from .stairs import StairsGenerator

    ids = set().union(*(_stair_rooms(s).keys() for s in storeys))
    for room_id in sorted(ids):
        flights = []
        for lo, hi in zip(storeys, storeys[1:]):
            room, above = _stair_rooms(lo).get(room_id), _stair_rooms(hi).get(room_id)
            if room is None or above is None:
                break
            flights.append((room, above, float(hi.transform.translation[1] - lo.transform.translation[1])))
        if not flights:
            continue
        first = flights[0][0]
        sx, sz = float(first.size[0]), float(first.size[2])
        along_z = sz >= sx
        length, across = (sz, sx) if along_z else (sx, sz)
        width = min(STAIR_WIDTH, across - WALKWAY)
        if width < 0.8:
            logger.warning("stair core %s: %.1f m is too narrow for a stair and a walkway", room_id, across)
            continue
        gens = [StairsGenerator(style="straight", rise=rise, width=width, railing="both") for _, _, rise in flights]
        runs = [g.steps * g.tread for g in gens]
        long_sides = ("east", "west") if along_z else ("north", "south")
        door_sides, end_zones = [], []
        for node in [f[0] for f in flights] + [flights[-1][1]]:
            for o in node.meta.get("openings", []):
                if o["kind"] != "door":
                    continue
                if o["side"] in long_sides:
                    door_sides.append(1.0 if o["side"] in ("east", "north") else -1.0)
                else:
                    end = length / 2 if o["side"] in ("north", "east") else -length / 2
                    end_zones.append((end - DOOR_CLEAR, end) if end > 0 else (end, end + DOOR_CLEAR))
        side = -1.0 if (door_sides and np.mean(door_sides) > 0) else 1.0
        lo_lim = max([-length / 2 + 0.2] + [b for a, b in end_zones if a < 0])
        hi_lim = min([length / 2 - 0.2] + [a for a, b in end_zones if a > 0])
        up_run = max(runs[0::2])                      # flights climbing toward +
        down_run = max(runs[1::2], default=0.0)       # flights climbing toward -
        a = hi_lim - up_run                           # start of the + flights
        b = lo_lim + down_run                         # start of the - flights
        if b > a + 1e-6:
            logger.warning("stair core %s: %.1f m is too short for flights of %.1f + %.1f m",
                           room_id, length, up_run, down_run)
            continue
        mid = (a + b) / 2                             # centre both starts on the slack
        a, b = max(a - (a - mid) / 2, b), min(b + (mid - b) / 2, a)
        offset = side * (across / 2 - width / 2 - 0.02)
        for i, ((room, above, rise), gen, run) in enumerate(zip(flights, gens, runs)):
            start, climb = (a, 1.0) if i % 2 == 0 else (b, -1.0)
            _place_flight(room, above, gen, rise, run, start, climb, offset, along_z, material_loader)


def _place_flight(room: SceneNode, above: SceneNode, gen, rise: float, run: float, start: float, climb: float,
                  across: float, along_z: bool, material_loader) -> None:
    from ..core import csg

    centre_along = start + climb * run / 2
    node = gen.to_node("stairs")
    node.mesh.material = material_loader.load("concrete")
    # Local stairs climb -Z; rotate so they climb along +/-Z or +/-X of the room.
    if along_z:
        yaw = 0.0 if climb < 0 else 180.0
        pos = np.array([across, rise / 2, centre_along])
    else:
        yaw = 90.0 if climb < 0 else -90.0
        pos = np.array([centre_along, rise / 2, across])
    node.transform = Transform(translation=pos, rotation=np.array([0.0, np.radians(yaw), 0.0]))
    room.add_child(node)

    # Open the slab above the flight (upper floor and lower ceiling).
    half = np.array([gen.width / 2, run / 2]) + 0.05
    half_xz = half if along_z else half[::-1]
    lo = np.array([pos[0] - half_xz[0], 0.0, pos[2] - half_xz[1]])
    hi = np.array([pos[0] + half_xz[0], 0.0, pos[2] + half_xz[1]])
    for target, y0, y1 in ((room.find("ceiling"), -1.0, rise + 1.0), (above.find("floor"), -rise - 1.0, 1.0)):
        if target is None or target.mesh is None:
            continue
        cutter = _box(np.array([lo[0], y0, lo[2]]), np.array([hi[0], y1, hi[2]]))
        target.mesh = csg.difference(target.mesh, cutter, crease_angle=30.0)
    room.meta.setdefault("stairs", []).append({"to": above.meta["room"]["id"], "rise": rise, "steps": gen.steps})


def _roof(top: SceneNode, plan: FloorPlan, roof: dict[str, Any], material_loader) -> SceneNode:
    """Flat roof slab over the top storey plus a parapet around its edge, or a
    pitched roof (``style: gable|hip|shed``, ``rise``, ``overhang``) with its
    gable ends filled in ``gable_material``."""
    from ..core import csg, uvmap
    from .primitives import CubeGenerator

    size = plan.size
    t = plan.exterior_wall
    style = roof.get("style", "flat")
    if style in ("gable", "hip", "shed"):
        return _pitched_roof(top, plan, roof, style, material_loader)
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


def _pitched_roof(top: SceneNode, plan: FloorPlan, roof: dict[str, Any], style: str, material_loader) -> SceneNode:
    from .architecture import PrismGenerator, RoofGenerator

    size = plan.size
    y = float(top.transform.translation[1]) + plan.wall_height
    rise = float(roof.get("rise", 0.35 * min(size[0], size[2])))
    gen = RoofGenerator(width=size[0], height=rise, depth=size[2], style=style,
                        overhang=float(roof.get("overhang", 0.4)), thickness=float(roof.get("thickness", 0.15)))
    m = np.eye(4)
    m[1, 3] = y + rise / 2          # the generator's wall-top plane is at -rise / 2
    mesh = gen.generate().transform(m)
    mesh.material = material_loader.load(roof.get("material", "roof_shingle"))
    node = SceneNode(name="roof", mesh=mesh, tags=["roof", f"roof.{style}"])
    if style in ("gable", "shed"):
        along_x = size[0] >= size[2]
        width, depth = (size[0], size[2]) if along_x else (size[2], size[0])
        prism = PrismGenerator(width=width, height=rise, depth=depth,
                               apex="back" if style == "shed" else "center").generate()
        m = np.eye(4)
        if not along_x:
            m[:3, :3] = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        m[1, 3] = y + rise / 2
        from ..core import meshops, uvmap

        gable = meshops.compute_normals(uvmap.box_project(prism.transform(m)), 30.0)
        gable.material = material_loader.load(roof.get("gable_material", "brick"))
        node.add_child(SceneNode(name="gables", mesh=gable, tags=["wall.gable"]))
    node.meta["roof"] = {"style": style, "rise": round(rise, 3)}
    return node
