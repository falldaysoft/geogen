"""Procedural floor-plan layouts: footprint + program -> ``floorplan:`` spec.

A layout returns the same mapping you'd write under ``floorplan:`` by hand
(rooms, doors, windows), so everything downstream (walls, doors, finishes,
furnishing, QA) is shared. In YAML::

    floorplan:
      generate: hotel_corridor      # or hotel_lobby
      length: 30
      depth: 16
      module: 3.6
      # ...any FloorPlan key (wall_height, materials, finishes) passes through

Coordinates: X along the building's length, Z across it; south (-Z) is the
street side. All values snap to the plan grid (0.1 m).
"""

from __future__ import annotations

from typing import Any, Callable

GRID = 0.1


def _snap(v: float) -> float:
    return round(round(v / GRID) * GRID, 6)


def _rect(x: float, z: float, w: float, d: float) -> list[float]:
    return [_snap(x), _snap(z), _snap(w), _snap(d)]


def hotel_corridor(length: float = 30.0, depth: float = 16.0, corridor: float = 1.8, module: float = 3.6,
                   ensuite: bool = True, bath_width: float = 2.0, bath_depth: float = 2.4,
                   core_width: float = 4.8, stair_width: float = 3.0, floor: int = 1,
                   lift: bool = True, shaft: float = 2.2) -> dict[str, Any]:
    """Double-loaded guest floor.

    ``[stair][rooms...][core][rooms...][stair]`` along X, a corridor down the
    middle, guest modules on both sides. With ``ensuite`` each module is an
    entry lobby and bathroom on the corridor side, then the bedroom with the
    window on the façade; without, the bedroom opens off the corridor.
    Rooms are named ``room_<floor><nn>`` (bedroom), ``..._bath``, ``..._entry``.
    """
    side_depth = _snap((depth - corridor) / 2)
    cz0 = side_depth                      # corridor z range
    cz1 = side_depth + corridor
    usable = length - 2 * stair_width - core_width
    per_half = int((usable / 2) // module)
    if per_half < 1:
        raise ValueError(f"hotel_corridor: {length} m is too short for a {module} m room module")
    # Stretch the core to take up what the modules leave.
    core = _snap(length - 2 * stair_width - 2 * per_half * module)
    rooms: dict[str, dict] = {}
    doors: list[dict] = []
    windows: list[dict] = []

    x0 = stair_width
    rooms["corridor"] = {"rect": _rect(x0, cz0, length - 2 * stair_width, corridor), "type": "hallway"}
    for name, sx in (("stair_west", 0.0), ("stair_east", length - stair_width)):
        rooms[name] = {"rect": _rect(sx, 0, stair_width, depth), "type": "stair"}
        doors.append({"between": [name, "corridor"], "width": 1.0})
    core_x = x0 + per_half * module
    lobby_depth = side_depth - shaft if lift else side_depth
    rooms["lift_lobby"] = {"rect": _rect(core_x, cz1, core, lobby_depth), "type": "lift_lobby"}
    rooms["service"] = {"rect": _rect(core_x, 0, core, side_depth), "type": "service"}
    doors.append({"between": ["corridor", "lift_lobby"], "width": min(2.4, core - 0.6), "style": "archway"})
    doors.append({"between": ["corridor", "service"], "width": 0.9})
    if lift:
        _add_lift(rooms, doors, "lift_lobby", core_x, core, cz1 + lobby_depth, shaft)

    number = 1
    for side, (z_bath, z_bed, bed_depth, window_side) in {
        "north": (cz1, cz1 + bath_depth, side_depth - bath_depth, "north"),
        "south": (cz0 - bath_depth, 0.0, side_depth - bath_depth, "south"),
    }.items():
        xs = [x0 + i * module for i in range(per_half)] + [core_x + core + i * module for i in range(per_half)]
        for x in xs:
            rid = f"room_{floor}{number:02d}"
            number += 1
            if ensuite:
                entry_w = module - bath_width
                # Mirror alternate modules so bathrooms share walls back to back.
                flip = (number % 2) == 0
                ex, bx = (x, x + entry_w) if not flip else (x + bath_width, x)
                rooms[f"{rid}_entry"] = {"rect": _rect(ex, z_bath, entry_w, bath_depth), "type": "corridor"}
                rooms[f"{rid}_bath"] = {"rect": _rect(bx, z_bath, bath_width, bath_depth), "type": "hotel_bathroom"}
                rooms[rid] = {"rect": _rect(x, z_bed, module, bed_depth), "type": "hotel_bedroom"}
                # Hinge the corridor door on the side away from the bathroom so the
                # open leaf doesn't park across the bathroom doorway. Door-local +X
                # is world +X on north-side modules (swinging +Z), -X on south ones.
                bath_east = not flip
                hinge_west = bath_east
                hinge = ("left" if hinge_west else "right") if side == "north" else \
                    ("right" if hinge_west else "left")
                doors.append({"name": f"door_{rid}", "between": ["corridor", f"{rid}_entry"], "width": 0.9,
                              "hinge": hinge})
                doors.append({"between": [f"{rid}_entry", rid], "width": 0.9})
                doors.append({"between": [f"{rid}_entry", f"{rid}_bath"], "width": 0.9})
            else:
                z = z_bath if side == "north" else 0.0
                rooms[rid] = {"rect": _rect(x, z, module, side_depth), "type": "hotel_bedroom"}
                doors.append({"name": f"door_{rid}", "between": ["corridor", rid], "width": 0.9})
            windows.append({"room": rid, "side": window_side, "width": _snap(min(1.8, module - 1.2)),
                            "height": 1.5, "sill": 0.7})
    # Stair windows light the landings.
    for name in ("stair_west", "stair_east"):
        windows.append({"room": name, "side": "south", "width": 1.2, "height": 1.2, "sill": 1.2})
    return {"rooms": rooms, "doors": doors, "windows": windows}


def _add_lift(rooms: dict, doors: list, lobby: str, x: float, width: float, z: float, shaft: float) -> None:
    """A lift shaft centred at the back of ``lobby`` with plant rooms either side.

    Shafts are centred on the building (x = x + width / 2) so they stack
    across storeys; building.py turns stacked shafts into a working lift.
    """
    side = _snap((width - shaft) / 2)
    rooms["lift_shaft"] = {"rect": _rect(x + side, z, shaft, shaft), "type": "lift_shaft"}
    doors.append({"between": [lobby, "lift_shaft"], "width": 1.1, "style": "opening"})
    for name, px in (("lift_plant_west", x), ("lift_plant_east", x + side + shaft)):
        if side >= 1.2:
            rooms[name] = {"rect": _rect(px, z, side, shaft), "type": "service"}
            doors.append({"between": [lobby, name], "width": 0.9})


def hotel_lobby(length: float = 30.0, depth: float = 16.0, entrance_width: float = 6.0,
                reception_depth: float = 5.0, restroom_width: float = 3.0,
                back_of_house: float = 7.0, stair_width: float = 3.0, lift: bool = True,
                shaft: float = 2.2) -> dict[str, Any]:
    """Ground floor: entrance lobby off the street (south), reception behind
    it, lounge and restaurant either side, restrooms and back-of-house to
    the north. Entry doors on the south façade. With ``stair_width`` > 0,
    full-depth stair rooms at both ends line up with ``hotel_corridor``
    floors of the same length (see building.py)."""
    front = _snap(depth - reception_depth)
    x0 = stair_width
    side_w = _snap((length - 2 * stair_width - entrance_width) / 2)
    rooms: dict[str, dict] = {
        "lobby": {"rect": _rect(x0 + side_w, 0, entrance_width, front), "type": "lobby"},
        "lounge": {"rect": _rect(x0, 0, side_w, front), "type": "lounge"},
        "restaurant": {"rect": _rect(x0 + side_w + entrance_width, 0, side_w, front), "type": "restaurant"},
        "reception": {"rect": _rect(x0 + side_w, front, entrance_width,
                                    reception_depth - (shaft if lift else 0.0)), "type": "reception"},
        "restroom_a": {"rect": _rect(x0 + side_w - restroom_width, front, restroom_width, reception_depth),
                       "type": "restroom"},
        "restroom_b": {"rect": _rect(x0, front, side_w - restroom_width, reception_depth), "type": "restroom"},
        "back_of_house": {"rect": _rect(x0 + side_w + entrance_width, front, side_w, reception_depth),
                          "type": "back_of_house"},
    }
    lift_doors: list[dict] = []
    if lift:
        _add_lift(rooms, lift_doors, "reception", x0 + side_w, entrance_width, depth - shaft, shaft)
    stair_doors = []
    if stair_width > 0:
        rooms["stair_west"] = {"rect": _rect(0, 0, stair_width, depth), "type": "stair"}
        rooms["stair_east"] = {"rect": _rect(length - stair_width, 0, stair_width, depth), "type": "stair"}
        stair_doors = [{"between": ["stair_west", "lounge"], "width": 1.0},
                       {"between": ["restaurant", "stair_east"], "width": 1.0}]
    if back_of_house <= 0:
        rooms["back_of_house"]["type"] = "office"
    doors = [
        {"name": "entrance", "room": "lobby", "side": "south", "width": 2.0, "height": 2.4, "swing": "lobby"},
        {"between": ["lounge", "lobby"], "width": 3.0, "style": "archway"},
        {"between": ["lobby", "restaurant"], "width": 2.4, "style": "archway"},
        {"between": ["lobby", "reception"], "width": 3.6, "style": "archway"},
        {"between": ["lounge", "restroom_a"], "width": 0.9},
        {"between": ["lounge", "restroom_b"], "width": 0.9},
        {"between": ["restaurant", "back_of_house"], "width": 1.2},
        *stair_doors,
        *lift_doors,
    ]
    windows = [
        {"room": "lounge", "side": "south", "width": _snap(side_w - 2.0), "height": 2.0, "sill": 0.4},
        {"room": "restaurant", "side": "south", "width": _snap(side_w - 2.0), "height": 2.0, "sill": 0.4},
    ]
    if stair_width <= 0:
        windows += [{"room": "lounge", "side": "west", "width": 3.0, "height": 2.0, "sill": 0.4},
                    {"room": "restaurant", "side": "east", "width": 3.0, "height": 2.0, "sill": 0.4}]
    return {"rooms": rooms, "doors": doors, "windows": windows}


LAYOUTS: dict[str, Callable[..., dict[str, Any]]] = {
    "hotel_corridor": hotel_corridor,
    "hotel_lobby": hotel_lobby,
}

# FloorPlan keys that pass straight through a generated spec.
PLAN_KEYS = {"grid", "wall_height", "exterior_wall", "interior_wall", "floor_thickness", "ceiling_thickness",
             "ceiling", "materials", "finishes"}


def expand(spec: dict[str, Any]) -> dict[str, Any]:
    """Turn ``{generate: <layout>, ...}`` into a plain floorplan spec."""
    if "generate" not in spec:
        return spec
    import inspect

    name = spec["generate"]
    if name not in LAYOUTS:
        raise ValueError(f"Unknown floor plan layout '{name}'. Available: {sorted(LAYOUTS)}")
    layout = LAYOUTS[name]
    accepted = set(inspect.signature(layout).parameters)
    args = {k: v for k, v in spec.items() if k in accepted}
    unknown = set(spec) - accepted - PLAN_KEYS - {"generate"}
    if unknown:
        raise ValueError(f"Unknown keys for layout '{name}': {sorted(unknown)}."
                         f" Layout params: {sorted(accepted)}; plan keys: {sorted(PLAN_KEYS)}")
    generated = layout(**args)
    generated.update({k: v for k, v in spec.items() if k in PLAN_KEYS})
    return generated
