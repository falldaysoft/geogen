"""Layout QA for generated interiors.

``check_layout(scene, player)`` inspects every floor-plan room in a composed
scene and reports:

- ``overlap``: two floor-standing items whose footprints intersect;
- ``outside_room``: an item poking through a room's walls;
- ``door_swing``: an item inside a door leaf's swing arc;
- ``window_blocked``: a tall item standing in front of a window;
- ``narrow_door``: a doorway narrower than the player's ``door_min_width``;
- ``unreachable``: an item's access point (in front of it) that the player
  can't reach from the room's entrances, found by flood-filling a 2D
  occupancy grid of free floor eroded by the player radius.

Items are asset roots with a ``furniture.*``/``bathroom.*`` tag (or anything
the furnishing solver placed). Items raised off the floor (a lamp on a
nightstand) and walk-on items (rugs) aren't obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.node import SceneNode

ITEM_TAG_PREFIXES = ("furniture.", "bathroom.")
FLOOR_EPS = 0.05          # items whose lowest point is above this aren't floor obstacles
CELL = 0.05               # occupancy grid resolution
NON_OBSTACLES = ("furniture.rug", "furniture.curtains")
HIGH_SILL = 1.2           # furniture may stand under windows this high (bathrooms, corridors)
# Pairs allowed to overlap in plan: seats tuck under desks and tables.
TUCKS = (("furniture.chair", "furniture.desk"), ("furniture.chair", "furniture.table"))


@dataclass
class Issue:
    kind: str
    room: str
    items: tuple[str, ...]
    message: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.room}: {self.message}"


@dataclass
class _Room:
    name: str
    node: SceneNode
    center: np.ndarray            # world (x, z)
    half: np.ndarray              # clear half size (x, z) (axis-aligned rooms)
    openings: list[dict]
    swings: list[dict]
    items: list["_Item"] = field(default_factory=list)


@dataclass
class _Item:
    name: str
    node: SceneNode
    lo: np.ndarray                # world footprint (x, z)
    hi: np.ndarray
    bottom: float
    top: float
    facing: np.ndarray            # world (x, z)
    front: float                  # clearance in front (m)


def _world_points(node: SceneNode) -> np.ndarray | None:
    pts = []
    for n in node.iter_nodes():
        if n.mesh is None or not len(n.mesh.vertices):
            continue
        v = n.mesh.vertices
        pts.append((n.world_transform() @ np.c_[v, np.ones(len(v))].T).T[:, :3])
    return np.concatenate(pts) if pts else None


def _is_item(node: SceneNode) -> bool:
    return node.meta.get("placed_by") == "furnish" or any(t.startswith(ITEM_TAG_PREFIXES) for t in node.tags)


def _rooms(scene: SceneNode) -> list[_Room]:
    rooms = []
    for node in scene.iter_nodes():
        room = node.meta.get("room")
        if not isinstance(room, dict) or node.meta.get("type") == "room_volume" or node.size is None:
            continue
        m = node.world_transform()
        inset = float(node.meta.get("wall_inset", {}).get("mount", 0.0))
        rooms.append(_Room(
            name=str(room.get("id", node.name)), node=node, center=m[[0, 2], 3],
            half=np.asarray(node.size, dtype=float)[[0, 2]] / 2 - inset,
            openings=list(node.meta.get("openings", [])),
            swings=list(node.meta.get("door_swings", [])),
        ))
    return rooms


def _collect_items(scene: SceneNode, rooms: list[_Room]) -> None:
    seen: set[int] = set()

    def visit(node: SceneNode) -> None:
        if _is_item(node) and id(node) not in seen:
            seen.add(id(node))
            pts = _world_points(node)
            if pts is not None:
                m = node.world_transform()
                facing = m[[0, 2], 2]
                facing = facing / max(np.linalg.norm(facing), 1e-9)
                item = _Item(node.name, node, pts[:, [0, 2]].min(axis=0), pts[:, [0, 2]].max(axis=0),
                             float(pts[:, 1].min()), float(pts[:, 1].max()), facing,
                             float(node.meta.get("clearance", {}).get("front", 0.3)))
                centre = (item.lo + item.hi) / 2
                for room in rooms:
                    floor_y = room.node.world_transform()[1, 3]
                    if np.all(np.abs(centre - room.center) <= room.half + 1e-6) and item.bottom - floor_y < 2.0:
                        item.bottom -= floor_y
                        item.top -= floor_y
                        room.items.append(item)
                        break
            return  # parts of an item aren't items
        for child in node.children:
            visit(child)

    visit(scene)


def _tucks(a: _Item, b: _Item) -> bool:
    return any(x in a.node.tags and y in b.node.tags for x, y in TUCKS)


def _obstacle(item: _Item) -> bool:
    return item.bottom <= FLOOR_EPS and not any(t in NON_OBSTACLES for t in item.node.tags)


def _wall_frame(room: _Room, side: str):
    """(along axis index, wall coordinate, inward normal) in world (x, z)."""
    if side == "north":
        return 0, room.center[1] + room.half[1], np.array([0.0, -1.0])
    if side == "south":
        return 0, room.center[1] - room.half[1], np.array([0.0, 1.0])
    if side == "east":
        return 1, room.center[0] + room.half[0], np.array([-1.0, 0.0])
    return 1, room.center[0] - room.half[0], np.array([1.0, 0.0])


def _opening_rect(room: _Room, o: dict, depth: float) -> tuple[np.ndarray, np.ndarray]:
    axis, coord, inward = _wall_frame(room, o["side"])
    lo, hi = np.zeros(2), np.zeros(2)
    lo[axis], hi[axis] = room.center[axis] + o["lo"], room.center[axis] + o["hi"]
    other = 1 - axis
    a, b = coord, coord + inward[other] * depth
    lo[other], hi[other] = min(a, b), max(a, b)
    return lo, hi


def check_layout(scene: SceneNode, player=None) -> list[Issue]:
    from ..player import load_player_spec

    player = player or load_player_spec()
    rooms = _rooms(scene)
    _collect_items(scene, rooms)
    issues: list[Issue] = []
    for room in rooms:
        issues += _check_room(room, player)
    # Reachability per floor level: rooms of different storeys overlap in plan.
    levels: dict[float, list[_Room]] = {}
    for room in rooms:
        levels.setdefault(round(float(room.node.world_transform()[1, 3]), 2), []).append(room)
    for level_rooms in levels.values():
        issues += _check_reachability(level_rooms, player)
    return issues


def _check_room(room: _Room, player) -> list[Issue]:
    issues = []
    obstacles = [i for i in room.items if _obstacle(i)]
    for a_i, a in enumerate(obstacles):
        for b in obstacles[a_i + 1:]:
            overlap = np.minimum(a.hi, b.hi) - np.maximum(a.lo, b.lo)
            if _tucks(a, b) or _tucks(b, a):
                continue
            if (overlap > 0.01).all():
                issues.append(Issue("overlap", room.name, (a.name, b.name),
                                    f"{a.name} and {b.name} overlap by {overlap.min():.2f} m"))
    for item in room.items:
        out = np.maximum(np.maximum(item.hi - (room.center + room.half), (room.center - room.half) - item.lo), 0)
        if out.max() > 0.01:
            issues.append(Issue("outside_room", room.name, (item.name,),
                                f"{item.name} extends {out.max():.2f} m into the wall"))
    to_world = room.node.world_transform()
    for swing in room.swings:
        hinge = (to_world @ np.r_[swing["hinge"], 1.0])[[0, 2]]
        lo_a, hi_a = sorted((swing["from_deg"], swing["to_deg"]))
        pts = np.array([hinge + r * np.array([np.cos(t), np.sin(t)])
                        for r in np.linspace(0.1, swing["radius"], 6)
                        for t in np.radians(np.linspace(lo_a, hi_a, 12))])
        for item in obstacles:
            if ((pts > item.lo + 0.01) & (pts < item.hi - 0.01)).all(axis=1).any():
                issues.append(Issue("door_swing", room.name, (item.name,),
                                    f"{item.name} is in the swing of {swing['door']}"))
    for o in room.openings:
        width = o["hi"] - o["lo"]
        if o["kind"] == "door" and width < player.door_min_width - 1e-6:
            issues.append(Issue("narrow_door", room.name, (),
                                f"door on the {o['side']} wall is {width:.2f} m wide"
                                f" (player needs {player.door_min_width:.2f} m)"))
        if o["kind"] == "window" and o["sill"] < HIGH_SILL:
            lo, hi = _opening_rect(room, o, 0.5)
            for item in obstacles:
                if item.top > o["sill"] + 0.4 and (np.minimum(item.hi, hi) - np.maximum(item.lo, lo) > 0.01).all():
                    issues.append(Issue("window_blocked", room.name, (item.name,),
                                        f"{item.name} ({item.top:.2f} m tall) stands in front of a window"))
    return issues


def _check_reachability(rooms: list[_Room], player) -> list[Issue]:
    """Flood-fill free floor from every door; each item's front must be within reach."""
    from scipy import ndimage

    if not rooms:
        return []
    lo = np.min([r.center - r.half for r in rooms], axis=0) - 0.5
    hi = np.max([r.center + r.half for r in rooms], axis=0) + 0.5
    # Snap to a world lattice so a room checked alone (furnishing) and the
    # same room checked with its whole storey (QA) rasterise identically.
    lo = np.floor(lo / CELL) * CELL
    hi = np.ceil(hi / CELL) * CELL
    shape = np.ceil((hi - lo) / CELL).astype(int) + 1
    xs = lo[0] + np.arange(shape[0]) * CELL
    zs = lo[1] + np.arange(shape[1]) * CELL
    gx, gz = np.meshgrid(xs, zs, indexing="ij")

    def cells(r_lo: np.ndarray, r_hi: np.ndarray) -> np.ndarray:
        return (gx >= r_lo[0]) & (gx <= r_hi[0]) & (gz >= r_lo[1]) & (gz <= r_hi[1])

    free = np.zeros(shape, dtype=bool)
    for room in rooms:
        free |= cells(room.center - room.half, room.center + room.half)
    seeds = np.zeros(shape, dtype=bool)
    for room in rooms:
        for o in room.openings:
            if o["kind"] != "door":
                continue
            # The doorway itself (through the wall) is floor too.
            d_lo, d_hi = _opening_rect(room, o, -0.4)
            free |= cells(d_lo, d_hi)
            seeds |= cells(*_opening_rect(room, o, 0.3))
    for room in rooms:
        for item in room.items:
            if _obstacle(item):
                free &= ~cells(item.lo, item.hi)
    # Erode by the player radius: where the body's centre can stand.
    radius_cells = int(np.ceil(player.radius / CELL))
    yy, xx = np.mgrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    disk = xx ** 2 + yy ** 2 <= radius_cells ** 2
    standable = ndimage.binary_erosion(free, structure=disk)
    labels, _ = ndimage.label(standable)
    reachable_labels = set(np.unique(labels[seeds & standable])) - {0}
    reachable = np.isin(labels, list(reachable_labels))

    issues = []
    for room in rooms:
        for item in room.items:
            if not _obstacle(item) or item.front <= 0:
                continue
            if not _reachable_front(item, reachable, gx, gz, player.reach):
                front = _front_point(item)
                issues.append(Issue("unreachable", room.name, (item.name,),
                                    f"{item.name} can't be reached from a door (front at"
                                    f" {front[0]:.2f}, {front[1]:.2f})"))
    return issues


def _front_point(item: _Item) -> np.ndarray:
    centre = (item.lo + item.hi) / 2
    extent = np.abs(item.facing) @ ((item.hi - item.lo) / 2)
    return centre + item.facing * extent


# Items used from any side (tables, chairs pulled out) rather than their front.
ANY_SIDE = ("furniture.table", "furniture.chair")


def _reachable_front(item: _Item, reachable: np.ndarray, gx: np.ndarray, gz: np.ndarray, reach: float) -> bool:
    """Is some standable, connected spot within arm's reach of the item's front edge (or any side)?"""
    centre = (item.lo + item.hi) / 2
    half = (item.hi - item.lo) / 2
    points = [_front_point(item)]
    if any(t in item.node.tags for t in ANY_SIDE):
        points = [centre + np.array(d) * half for d in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    for p in points:
        near = (gx - p[0]) ** 2 + (gz - p[1]) ** 2 <= reach ** 2
        if (near & reachable).any():
            return True
    return False


def room_reachability(room_node: SceneNode, player=None) -> list[str]:
    """Names of items under ``room_node`` whose front can't be reached (for the furnishing solver)."""
    from ..player import load_player_spec

    player = player or load_player_spec()
    rooms = [r for r in _rooms(room_node) if r.node is room_node]
    if not rooms:
        return []
    _collect_items(room_node, rooms)
    return [i.items[0] for i in _check_reachability(rooms, player)]

