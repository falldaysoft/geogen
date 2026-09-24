"""Parametric building recipes sized to a lot.

``build_recipe(name, params, loader)`` returns a building SceneNode (via the
multi-storey generator, ``generators/building.py``). Every recipe takes
``width`` (along x, the street frontage) and ``depth`` (along z) in metres,
``storeys``, ``interior`` and optionally ``style`` (façade) and ``seed``
(furnishing):

- ``detached_house``: living room, kitchen, bedroom(s), bathroom; one or two
  storeys (a stair hall along one side), gable roof; front door on +Z.
- ``shop_row``: shop and stock room on the ground floor, flats above
  reached by their own street door and stair; flat roof; front on +Z.
- ``apartment_block``: hotel-style lobby, then double-loaded corridors of
  flats (bedroom + bathroom units) with scissor stairs and a lift.
- ``office``: reception lobby and open offices around a stair; modern façade.
- ``hotel``: ``hotel_lobby`` + ``hotel_corridor`` floors.

``interior``: ``full`` furnishes every storey, ``lobby`` only the ground
floor (upper floors stay empty rooms), ``shell`` removes the rooms and
interior doors, locks the street doors and fills each storey with a dark
volume behind the windows: a cheap background building.

Recipes raise ``RecipeError`` when the lot is too small for them, so a
caller can try the next one. The building's front is whichever side its
entrance spawn is on (``front_of`` below).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.node import SceneNode

GRID = 0.1
STAIR = 2.8           # stair hall width (a 1.2 m flight + walkway)
INTERIORS = ("full", "lobby", "shell")


class RecipeError(ValueError):
    """The recipe can't be built at this size."""


def _snap(v: float) -> float:
    return round(round(v / GRID) * GRID, 6)


def _r(x, z, w, d) -> list[float]:
    return [_snap(x), _snap(z), _snap(w), _snap(d)]


def _cap_storeys(storeys: int, length: float, ground_h: float, upper_h: float, end_doors: int = 1) -> int:
    """Most storeys (up to ``storeys``) a scissor stair hall ``length`` long can serve."""
    from ..generators.building import DOOR_CLEAR
    from ..generators.floorplan import FloorPlan
    from ..generators.stairs import StairsGenerator

    slab = FloorPlan.__dataclass_fields__["floor_thickness"].default
    while storeys > 1:
        rises = [ground_h + slab] + [upper_h + slab] * (storeys - 2)
        runs = [(g := StairsGenerator(style="straight", rise=r, width=1.2)).steps * g.tread for r in rises]
        need = max(runs[0::2]) + max(runs[1::2], default=0.0) + 0.4 + DOOR_CLEAR * end_doors
        if need <= length - 0.3 + 1e-6:     # clear length inside the walls
            break
        storeys -= 1
    return storeys


def _need(name: str, width: float, depth: float, min_width: float, min_depth: float) -> None:
    if width < min_width - 1e-6 or depth < min_depth - 1e-6:
        raise RecipeError(f"{name} needs at least {min_width} x {min_depth} m, got {width:.1f} x {depth:.1f}")


def detached_house(width: float, depth: float, storeys: int = 1, style: str = "brick_hotel") -> dict[str, Any]:
    _need("detached_house", width, depth, 7.0 if storeys > 1 else 5.5, 7.0)
    w, d = _snap(width), _snap(depth)
    storeys = _cap_storeys(storeys, d, 2.7, 2.6)
    materials = {"walls": "paint_white", "floor": "hardwood_floor", "ceiling": "ceiling_white"}
    if storeys <= 1:
        split, front = _snap(w * 0.58), _snap(d * 0.45)
        ground = {
            "wall_height": 2.7, "materials": materials,
            "rooms": {
                "living": {"rect": _r(0, front, split, d - front), "type": "living_room"},
                "kitchen": {"rect": _r(split, front, w - split, d - front), "type": "kitchen"},
                "bedroom": {"rect": _r(0, 0, split, front), "type": "bedroom"},
                "bathroom": {"rect": _r(split, 0, w - split, front), "type": "bathroom"},
            },
            "doors": [
                {"name": "front_door", "room": "living", "side": "north", "at": 0.75, "width": 1.0},
                {"between": ["living", "kitchen"], "width": 0.9},
                {"between": ["living", "bedroom"], "width": 0.9},
                {"between": ["bedroom", "bathroom"], "width": 0.9},
            ],
            "windows": [
                {"room": "living", "side": "north", "at": 0.3, "width": 1.6, "height": 1.4, "sill": 0.8},
                {"room": "living", "side": "west", "width": 1.2, "height": 1.4, "sill": 0.8},
                {"room": "kitchen", "side": "north", "width": 1.2, "height": 1.2, "sill": 1.0},
                {"room": "bedroom", "side": "south", "width": 1.4, "height": 1.4, "sill": 0.8},
                {"room": "bathroom", "side": "south", "width": 0.6, "height": 0.7, "sill": 1.5},
            ],
        }
        storeys_spec = [{"floorplan": ground}]
    else:
        inner = _snap(w - STAIR)
        half = _snap(d / 2)
        stair = {"rect": _r(inner, 0, STAIR, d), "type": "stair"}
        ground = {
            "wall_height": 2.7, "materials": materials,
            "rooms": {"stair": stair,
                      "living": {"rect": _r(0, half, inner, d - half), "type": "living_room"},
                      "kitchen": {"rect": _r(0, 0, inner, half), "type": "kitchen"}},
            "doors": [
                {"name": "front_door", "room": "stair", "side": "north", "width": 1.0},
                {"between": ["stair", "living"], "width": 0.9},
                {"between": ["living", "kitchen"], "width": 0.9},
            ],
            "windows": [
                {"room": "living", "side": "north", "width": 1.8, "height": 1.4, "sill": 0.8},
                {"room": "living", "side": "west", "width": 1.2, "height": 1.4, "sill": 0.8},
                {"room": "kitchen", "side": "south", "width": 1.4, "height": 1.2, "sill": 1.0},
            ],
        }
        bed2 = _snap(inner * 0.55)
        upper = {
            "wall_height": 2.6, "materials": materials,
            "rooms": {"stair": stair,
                      "bedroom": {"rect": _r(0, half, inner, d - half), "type": "bedroom"},
                      "bedroom_2": {"rect": _r(0, 0, bed2, half), "type": "bedroom_single"},
                      "bathroom": {"rect": _r(bed2, 0, inner - bed2, half), "type": "bathroom"}},
            "doors": [
                {"between": ["stair", "bedroom"], "width": 0.9},
                {"between": ["stair", "bathroom"], "width": 0.9},
                {"between": ["bedroom", "bedroom_2"], "width": 0.9},
            ],
            "windows": [
                {"room": "bedroom", "side": "north", "width": 1.6, "height": 1.3, "sill": 0.9},
                {"room": "bedroom_2", "side": "south", "width": 1.2, "height": 1.3, "sill": 0.9},
                {"room": "bathroom", "side": "south", "width": 0.6, "height": 0.7, "sill": 1.5},
                {"room": "stair", "side": "east", "width": 0.8, "height": 1.0, "sill": 1.2},
            ],
        }
        storeys_spec = [{"floorplan": ground}] + [{"floorplan": upper}] * (storeys - 1)
    return {"storeys": storeys_spec, "facade": {"style": style},
            "roof": {"style": "gable", "rise": _snap(0.3 * min(w, d)), "material": "roof_shingle",
                     "gable_material": _cladding(style)}}


def shop_row(width: float, depth: float, storeys: int = 2, style: str = "stucco") -> dict[str, Any]:
    _need("shop_row", width, depth, 6.0 if storeys <= 1 else 7.5, 8.0 if storeys <= 1 else 9.0)
    w, d = _snap(width), _snap(depth)
    storeys = _cap_storeys(storeys, d, 3.4, 2.7)
    inner = _snap(w - STAIR) if storeys > 1 else w
    front = _snap(d * 0.35)
    rooms = {"shop": {"rect": _r(0, front, inner, d - front), "type": "shop"},
             "stock": {"rect": _r(0, 0, inner, front), "type": "storage", "floor": "laminate"}}
    doors = [{"name": "shop_door", "room": "shop", "side": "north", "at": 0.5, "width": 1.2},
             {"between": ["shop", "stock"], "at": 0.8, "width": 0.9}]
    pane = _snap(min(2.4, inner * 0.32))
    windows = [{"room": "shop", "side": "north", "at": 0.18, "width": pane, "height": 2.3, "sill": 0.4},
               {"room": "shop", "side": "north", "at": 0.82, "width": pane, "height": 2.3, "sill": 0.4}]
    stair = {"rect": _r(inner, 0, STAIR, d), "type": "stair"}
    if storeys > 1:
        rooms["stair"] = stair
        doors.append({"name": "flats_door", "room": "stair", "side": "north", "width": 1.0})
    ground = {"wall_height": 3.4, "rooms": rooms, "doors": doors, "windows": windows,
              "materials": {"walls": "paint_white", "floor": "tile_floor_grey", "ceiling": "ceiling_white"}}
    half = _snap(d / 2)
    bed = _snap(inner * 0.6)
    flat = {
        "wall_height": 2.7,
        "materials": {"walls": "paint_greige", "floor": "laminate", "ceiling": "ceiling_white"},
        "rooms": {"stair": stair,
                  "living": {"rect": _r(0, half, inner, d - half), "type": "living_room"},
                  "bedroom": {"rect": _r(0, 0, bed, half), "type": "bedroom"},
                  "bathroom": {"rect": _r(bed, 0, inner - bed, half), "type": "bathroom"}},
        "doors": [{"between": ["stair", "living"], "width": 0.9},
                  {"between": ["stair", "bathroom"], "width": 0.9},
                  {"between": ["living", "bedroom"], "width": 0.9}],
        "windows": [{"room": "living", "side": "north", "at": 0.25, "width": 1.4, "height": 1.5, "sill": 0.8},
                    {"room": "living", "side": "north", "at": 0.75, "width": 1.4, "height": 1.5, "sill": 0.8},
                    {"room": "bedroom", "side": "south", "width": 1.4, "height": 1.5, "sill": 0.8}],
    }
    return {"storeys": [{"floorplan": ground}] + [{"floorplan": flat}] * (storeys - 1),
            "facade": {"style": style}, "roof": {"parapet": 0.8, "material": "concrete"}}


def apartment_block(width: float, depth: float, storeys: int = 4, style: str = "brick_hotel") -> dict[str, Any]:
    _need("apartment_block", width, depth, 22.0, 13.0)
    w, d = _snap(width), _snap(depth)
    common = {"materials": {"walls": "paint_white", "floor": "laminate", "ceiling": "ceiling_white"}}
    lobby = {"generate": "hotel_lobby", "length": w, "depth": d, "wall_height": 3.4, **common}
    flats = {"generate": "hotel_corridor", "length": w, "depth": d, "module": 4.2, "unit_type": "bedroom",
             "bath_type": "bathroom", **common}
    return {"storeys": [{"floorplan": lobby}, {"floorplan": flats, "repeat": storeys - 1}],
            "facade": {"style": style, "balconies": True}, "roof": {"parapet": 1.0, "material": "concrete"}}


def office(width: float, depth: float, storeys: int = 3, style: str = "modern") -> dict[str, Any]:
    _need("office", width, depth, 10.0, 9.0)
    w, d = _snap(width), _snap(depth)
    storeys = _cap_storeys(storeys, d, 3.4, 3.0, end_doors=0)
    stair_w = 3.0
    half = _snap(d / 2)
    stair = {"rect": _r(0, 0, stair_w, d), "type": "stair"}
    materials = {"walls": "paint_white", "floor": "carpet", "ceiling": "ceiling_white"}
    ground = {
        "wall_height": 3.4, "materials": materials,
        "rooms": {"stair": stair,
                  "lobby": {"rect": _r(stair_w, half, w - stair_w, d - half), "type": "lobby",
                            "floor": "tile_floor_grey"},
                  "office": {"rect": _r(stair_w, 0, w - stair_w, half), "type": "office"}},
        "doors": [{"name": "entrance", "room": "lobby", "side": "north", "width": 1.8, "height": 2.4},
                  {"between": ["lobby", "stair"], "width": 1.0},
                  {"between": ["lobby", "office"], "width": 1.2}],
        "windows": [{"room": "lobby", "side": "north", "at": 0.2, "width": _snap(min(3.0, (w - stair_w) * 0.3)),
                     "height": 2.2, "sill": 0.4},
                    {"room": "lobby", "side": "north", "at": 0.8, "width": _snap(min(3.0, (w - stair_w) * 0.3)),
                     "height": 2.2, "sill": 0.4},
                    {"room": "office", "side": "south", "width": _snap((w - stair_w) * 0.6), "height": 1.6,
                     "sill": 0.8}],
    }
    upper = {
        "wall_height": 3.0, "materials": materials,
        "rooms": {"stair": stair,
                  "office_front": {"rect": _r(stair_w, half, w - stair_w, d - half), "type": "office"},
                  "office_back": {"rect": _r(stair_w, 0, w - stair_w, half), "type": "office"}},
        "doors": [{"between": ["stair", "office_front"], "width": 1.0},
                  {"between": ["office_front", "office_back"], "width": 1.8, "style": "archway"}],
        "windows": [{"room": "office_front", "side": "north", "width": _snap((w - stair_w) * 0.7), "height": 1.6,
                     "sill": 0.8},
                    {"room": "office_back", "side": "south", "width": _snap((w - stair_w) * 0.7), "height": 1.6,
                     "sill": 0.8}],
    }
    return {"storeys": [{"floorplan": ground}, {"floorplan": upper, "repeat": storeys - 1}],
            "facade": {"style": style}, "roof": {"parapet": 0.8, "material": "concrete"}}


def hotel(width: float, depth: float, storeys: int = 4, style: str = "brick_hotel") -> dict[str, Any]:
    _need("hotel", width, depth, 22.0, 13.0)
    w, d = _snap(width), _snap(depth)
    lobby = {"generate": "hotel_lobby", "length": w, "depth": d, "wall_height": 3.6,
             "materials": {"walls": "paint_white", "floor": "tile_floor_grey", "ceiling": "ceiling_white"}}
    rooms = {"generate": "hotel_corridor", "length": w, "depth": d, "module": 3.8,
             "materials": {"walls": "paint_white", "floor": "carpet", "ceiling": "ceiling_white"}}
    return {"storeys": [{"floorplan": lobby}, {"floorplan": rooms, "repeat": storeys - 1}],
            "facade": {"style": style, "balconies": True}, "roof": {"parapet": 1.0, "material": "concrete"}}


# Largest sensible (width, depth) per recipe when sizing to a lot, metres.
RECIPE_LIMITS = {
    "detached_house": (11.0, 10.0),
    "shop_row": (12.0, 14.0),
    "apartment_block": (36.0, 16.0),
    "office": (24.0, 16.0),
    "hotel": (36.0, 16.0),
}

RECIPES: dict[str, Callable[..., dict[str, Any]]] = {
    "detached_house": detached_house,
    "shop_row": shop_row,
    "apartment_block": apartment_block,
    "office": office,
    "hotel": hotel,
}


def _cladding(style: str) -> str:
    from ..generators.facade import STYLES

    return STYLES[style].cladding if style in STYLES else "brick"


def recipe_spec(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """The ``building:`` spec for a recipe (storey furnishing set from ``interior``)."""
    if name not in RECIPES:
        raise ValueError(f"Unknown building recipe '{name}'. Available: {sorted(RECIPES)}")
    params = dict(params)
    interior = params.pop("interior", "full")
    seed = int(params.pop("seed", 0))
    if interior not in INTERIORS:
        raise ValueError(f"interior must be one of {INTERIORS}, got {interior!r}")
    params["storeys"] = max(1, int(params.get("storeys", 1)))
    spec = RECIPES[name](**params)
    # Expand repeats so each storey can be furnished (or not) on its own.
    storeys = []
    for entry in spec["storeys"]:
        for _ in range(int(entry.get("repeat", 1))):
            storeys.append({k: v for k, v in entry.items() if k != "repeat"})
    for index, entry in enumerate(storeys):
        if interior == "full" or (interior == "lobby" and index == 0):
            entry["furnish"] = seed * 100 + index + 1
    spec["storeys"] = storeys
    return spec


def build_recipe(name: str, params: dict[str, Any], loader) -> SceneNode:
    """Build a recipe into a building node (see module docstring)."""
    interior = params.get("interior", "full")
    spec = recipe_spec(name, params)
    node = loader._build_hierarchy({"name": name, "tags": [f"building.{name}"], "building": spec})
    node.meta["recipe"] = {"name": name, **{k: v for k, v in params.items()}}
    if interior == "shell":
        make_shell(node, loader._material_loader)
    return node


def make_shell(building: SceneNode, material_loader) -> None:
    """Strip rooms and interior doors, lock street doors, and darken each storey behind its windows."""
    from ..chunks import split_interiors
    from ..core import meshops, uvmap
    from ..generators.building import _box

    split_interiors(building)
    for node in building.iter_nodes():
        for interaction in node.interactions:
            interaction.lock = {"key": "shell_no_entry", "locked": True}
    for storey in [n for n in building.children if "storey" in n.tags]:
        walls = next((c for c in storey.children if "wall" in c.tags and c.mesh is not None), None)
        if walls is None:
            continue
        lo, hi = walls.mesh.vertices.min(axis=0), walls.mesh.vertices.max(axis=0)
        inset = 0.45
        box = _box(np.array([lo[0] + inset, lo[1], lo[2] + inset]), np.array([hi[0] - inset, hi[1], hi[2] - inset]))
        box = meshops.compute_normals(uvmap.box_project(box), 30.0)
        box.material = material_loader.load("plastic_black")
        void = SceneNode(name="shell_void", mesh=box, tags=["shell.void"])
        void.meta["collider"] = "box"
        storey.add_child(void)
    building.tags = [*building.tags, "building.shell"]


def front_of(node: SceneNode) -> np.ndarray:
    """Unit (x, z) direction the building's entrance faces, in its own frame (+Z by default)."""
    spawn = next((n for n in node.iter_nodes() if n.name == "entrance_spawn"), None)
    if spawn is None:
        return np.array([0.0, 1.0])
    to_local = np.linalg.inv(node.world_transform())
    pts = []
    for n in node.iter_nodes():
        if n.mesh is not None and "wall" in n.tags:
            v = n.mesh.vertices
            pts.append((to_local @ n.world_transform() @ np.c_[v, np.ones(len(v))].T).T[:, [0, 2]])
    centre = (np.vstack(pts).min(axis=0) + np.vstack(pts).max(axis=0)) / 2 if pts else np.zeros(2)
    d = (to_local @ spawn.world_transform())[[0, 2], 3] - centre
    axis = int(np.argmax(np.abs(d)))
    out = np.zeros(2)
    out[axis] = np.sign(d[axis]) or 1.0
    return out
