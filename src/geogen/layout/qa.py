"""Layout QA for generated interiors.

``check_layout(scene, player)`` inspects every floor-plan room in a composed
scene and reports:

- ``overlap``: two floor-standing items whose footprints intersect;
- ``outside_room``: an item poking through a room's walls;
- ``door_swing``: an item inside a door leaf's swing arc;
- ``window_blocked``: a tall item standing in front of a window;
- ``narrow_door``: a doorway narrower than the player's ``door_min_width``;
- ``unreachable``: an item's access point (in front of it) that the player
  can't reach from the storey's entrances (doors from outdoors, stair and
  lift rooms), found by flood-filling a 2D occupancy grid of free floor
  eroded by the player radius. The furnishing solver asks the same of one
  room (``room_reachability``), from the doors people come into it by: a
  door to a dead end (a shop's stock room) is not a way in.

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
# Pairs allowed to overlap in plan: seats tuck under desks and tables, toilets stand in cubicles.
TUCKS = (("furniture.chair", "furniture.desk"), ("furniture.chair", "furniture.table"),
         ("bathroom.toilet", "bathroom.stall"))


@dataclass
class Issue:
    kind: str
    room: str
    items: tuple[str, ...]
    message: str

    @property
    def text(self) -> str:
        """Where and what, without the kind (``str`` puts it in front)."""
        return f"{self.room}: {self.message}"

    def __str__(self) -> str:
        return f"[{self.kind}] {self.text}"


@dataclass
class _Room:
    name: str
    node: SceneNode
    center: np.ndarray            # plan (x, z): the room's parent (storey) frame
    half: np.ndarray              # clear half size (x, z) (axis-aligned rooms)
    openings: list[dict]
    swings: list[dict]
    frame: np.ndarray             # world -> plan frame
    matrix: np.ndarray            # room -> plan frame
    items: list["_Item"] = field(default_factory=list)


@dataclass
class _Item:
    name: str
    node: SceneNode
    lo: np.ndarray                # footprint (x, z) in its room's plan frame
    hi: np.ndarray
    bottom: float
    top: float
    facing: np.ndarray            # plan (x, z)
    front: float                  # clearance in front (m)


def _world_points(node: SceneNode) -> np.ndarray | None:
    pts = []
    for n in node.iter_nodes():
        if n.mesh is None or not len(n.mesh.vertices):
            continue
        pts.append(n.world_vertices())
    return np.concatenate(pts) if pts else None


def _is_item(node: SceneNode) -> bool:
    return node.meta.get("placed_by") == "furnish" or any(t.startswith(ITEM_TAG_PREFIXES) for t in node.tags)


def _is_room(node: SceneNode) -> bool:
    return (isinstance(node.meta.get("room"), dict) and node.meta.get("type") != "room_volume"
            and node.size is not None)


def _rooms(scene: SceneNode) -> list[_Room]:
    rooms = []
    for node in scene.iter_nodes():
        if not _is_room(node):
            continue
        room = node.meta["room"]
        # Work in the parent's frame, so moving or turning a whole building
        # (a city lot) never changes the result.
        frame = np.linalg.inv(node.parent.world_transform()) if node.parent is not None else np.eye(4)
        m = frame @ node.world_transform()
        inset = float(node.meta.get("wall_inset", {}).get("mount", 0.0))
        rooms.append(_Room(
            name=str(room.get("id", node.name)), node=node, center=m[[0, 2], 3],
            half=np.asarray(node.size, dtype=float)[[0, 2]] / 2 - inset,
            openings=list(node.meta.get("openings", [])),
            swings=list(node.meta.get("door_swings", [])),
            frame=frame, matrix=m,
        ))
    return rooms


def _collect_items(scene: SceneNode, rooms: list[_Room]) -> None:
    seen: set[int] = set()

    def visit(node: SceneNode) -> None:
        if _is_item(node) and id(node) not in seen:
            seen.add(id(node))
            world_pts = _world_points(node)
            if world_pts is not None:
                homogeneous = np.c_[world_pts, np.ones(len(world_pts))]
                for room in rooms:
                    pts = (room.frame @ homogeneous.T).T
                    m = room.frame @ node.world_transform()
                    facing = m[[0, 2], 2]
                    facing = facing / max(np.linalg.norm(facing), 1e-9)
                    item = _Item(node.name, node, pts[:, [0, 2]].min(axis=0), pts[:, [0, 2]].max(axis=0),
                                 float(pts[:, 1].min()), float(pts[:, 1].max()), facing,
                                 float(node.meta.get("clearance", {}).get("front", 0.3)))
                    centre = (item.lo + item.hi) / 2
                    floor_y = room.matrix[1, 3]
                    inside = np.all(np.abs(centre - room.center) <= room.half + 1e-6)
                    if inside and -0.5 <= item.bottom - floor_y < 2.0:
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
    # Reachability per plan frame and floor level: rooms of different storeys overlap in plan.
    levels: dict[tuple[int, float], list[_Room]] = {}
    for room in rooms:
        key = (id(room.node.parent), round(float(room.matrix[1, 3]), 2))
        levels.setdefault(key, []).append(room)
    for level_rooms in levels.values():
        issues += _check_reachability(level_rooms, player, _storey_entrances(level_rooms))
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
    for swing in room.swings:
        hinge = (room.matrix @ np.r_[swing["hinge"], 1.0])[[0, 2]]
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


# Room types people arrive in from another storey.
ARRIVAL_ROOM_TYPES = ("stair", "lift_shaft")


def _room_name(node: SceneNode) -> str:
    return str(node.meta["room"].get("id", node.name))


def _room_doors(node: SceneNode) -> list[dict]:
    return [o for o in node.meta.get("openings", []) if o["kind"] == "door"]


def _arrivals(nodes: list[SceneNode]) -> tuple[dict[str, set[str]], set[str]]:
    """The door graph of one storey's rooms, and the rooms people come into the storey by.

    People arrive through a door from outdoors, or in a stair or lift room.
    A door that doesn't say where it leads (not from a floor plan), or that
    leads to a room that isn't in ``nodes``, counts as one from outdoors.
    """
    names = {_room_name(n): n for n in nodes}
    graph: dict[str, set[str]] = {name: set() for name in names}
    arrivals = set()
    for name, node in names.items():
        if node.meta["room"].get("type") in ARRIVAL_ROOM_TYPES:
            arrivals.add(name)
        for o in _room_doors(node):
            beyond = o.get("to")
            if beyond in names:
                graph[name].add(beyond)
                graph[beyond].add(name)
            else:
                arrivals.add(name)
    return graph, arrivals


def _reaches(graph: dict[str, set[str]], start: str, goals: set[str], without: str | None = None) -> bool:
    """Can you walk from ``start`` to one of ``goals`` room by room, never entering ``without``?"""
    seen, todo = {start, without}, [start]
    while todo:
        name = todo.pop()
        if name in goals:
            return True
        new = graph.get(name, set()) - seen
        seen |= new
        todo += new
    return False


def _room_entrances(node: SceneNode, siblings: list[SceneNode]) -> list[dict]:
    """The doors of a room that people come in by, given the rooms of its storey.

    Not its doors to dead ends: a shop's stock room (and a staff toilet off
    that) is reached through the shop, so nothing in the shop may count on
    being reached from there. The dead end's own door to the shop is its
    entrance. A room that no door joins to a way into the storey keeps every
    door: a plan with no way in is not a furnishing fault.
    """
    doors = _room_doors(node)
    graph, arrivals = _arrivals(siblings if any(s is node for s in siblings) else [*siblings, node])
    name = _room_name(node)
    if node.meta["room"].get("type") in ARRIVAL_ROOM_TYPES:
        return doors
    entrances = [o for o in doors
                 if o.get("to") not in graph or _reaches(graph, o["to"], arrivals, without=name)]
    return entrances or doors


def _storey_entrances(rooms: list[_Room]) -> list[tuple[_Room, dict]]:
    """Where a storey's flood fill starts: doors from outdoors and the doors of stair and lift rooms.

    Everything else is reached through the doorways from those. Rooms that
    no door joins to any of them start from their own doors.
    """
    graph, arrivals = _arrivals([r.node for r in rooms])
    seeds = []
    for room in rooms:
        cut_off = not _reaches(graph, room.name, arrivals)
        arrival_room = room.node.meta["room"].get("type") in ARRIVAL_ROOM_TYPES
        for o in _room_doors(room.node):
            if cut_off or arrival_room or o.get("to") not in graph:
                seeds.append((room, o))
    return seeds


def _check_reachability(rooms: list[_Room], player, entrances: list[tuple[_Room, dict]]) -> list[Issue]:
    """Flood-fill free floor from the ``entrances`` (room, door); each item's front must be within reach."""
    from scipy import ndimage

    if not rooms:
        return []
    lo = np.min([r.center - r.half for r in rooms], axis=0) - 0.5
    hi = np.max([r.center + r.half for r in rooms], axis=0) + 0.5
    # Snap to a lattice in the plan frame so a room checked alone (furnishing)
    # and the same room checked with its whole storey (QA) rasterise identically.
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
    for room in rooms:
        for o in room.openings:
            if o["kind"] == "door":
                # The doorway itself (through the wall) is floor too.
                free |= cells(*_opening_rect(room, o, -0.4))
    seeds = np.zeros(shape, dtype=bool)
    for room, o in entrances:
        seeds |= cells(*_opening_rect(room, o, 0.3))
    for room in rooms:
        for item in room.items:
            if _obstacle(item):
                free &= ~cells(item.lo, item.hi)
    # Erode by the player radius: where the body's centre can stand.
    # (plus half a cell: a gap exactly one body wide rasterises open or shut by luck)
    radius_cells = int(np.ceil((player.radius + CELL / 2) / CELL))
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
                continue
            # Where people stand to use it (a bed's sides, a chair pulled from a table).
            for a, p in _approach_points(item, room):
                near = (gx - p[0]) ** 2 + (gz - p[1]) ** 2 <= APPROACH_SNAP ** 2
                if not (near & reachable).any():
                    issues.append(Issue("unreachable", room.name, (item.name,),
                                        f"{item.name}: can't reach where you stand to {a['type']}"
                                        f" ({p[0]:.2f}, {p[1]:.2f})"))
                    break
    return issues


APPROACH_SNAP = 0.3      # an affordance's approach point may be this far from standable floor


def _approach_points(item: _Item, room: _Room) -> list[tuple[dict, np.ndarray]]:
    """The item's affordance approach points (plan frame)."""
    out = []
    m = room.frame @ item.node.world_transform()
    for a in item.node.meta.get("affordances") or []:
        if a["type"] == "perch":
            continue          # jumped up to, not walked to
        p = m @ np.r_[np.asarray(a["approach"], dtype=float), 1.0]
        out.append((a, p[[0, 2]]))
    return out


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
    """Names of items under ``room_node`` whose front can't be reached (for the furnishing solver).

    Reached from the doors people come into the room by (``_room_entrances``),
    which takes the room's siblings in its floor plan: alone, every door counts.
    """
    from ..player import load_player_spec

    player = player or load_player_spec()
    rooms = [r for r in _rooms(room_node) if r.node is room_node]
    if not rooms:
        return []
    _collect_items(room_node, rooms)
    parent = room_node.parent
    siblings = [c for c in parent.children if _is_room(c)] if parent is not None else []
    entrances = [(rooms[0], o) for o in _room_entrances(room_node, siblings)]
    return [i.items[0] for i in _check_reachability(rooms, player, entrances)]


# --- z-fighting --------------------------------------------------------------------------------

ZFIGHT_MIN_AREA = 0.01      # m²: smaller coplanar overlaps aren't visible
_ZFIGHT_GRID = 1e-5         # m: snapping for the 2D overlays (nearly coincident edges break GEOS otherwise)


@dataclass(eq=False)
class _PlaneFaces:
    """One mesh's triangles in one axis-aligned plane, facing one way (2D, in that plane)."""

    node: SceneNode
    axis: int
    plane: float
    sign: int
    transparent: bool
    tris: np.ndarray
    _shape: object = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        low, high = self.tris.reshape(-1, 2).min(axis=0), self.tris.reshape(-1, 2).max(axis=0)
        return float(low[0]), float(low[1]), float(high[0]), float(high[1])

    @property
    def shape(self):
        if self._shape is None:
            import shapely

            rings = np.concatenate([self.tris, self.tris[:, :1]], axis=1)
            self._shape = shapely.union_all(shapely.polygons(rings), grid_size=_ZFIGHT_GRID)
        return self._shape


def _plane_faces(node: SceneNode, mesh, min_area: float) -> list[_PlaneFaces]:
    """``mesh``'s (world-space) faces grouped by axis-aligned plane (1 mm) and facing."""
    if not len(mesh.faces):
        return []
    tri = mesh.vertices[mesh.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(normals, axis=1)
    transparent = mesh.material is not None and mesh.material.transparent
    out = []
    for axis in range(3):
        # Level to the plane's own 1 mm: a long face turned a degree (a train on a curve) isn't in one plane.
        sel = (np.abs(normals[:, axis]) > 0.999 * twice_area) & (twice_area > 2e-9) \
            & (np.ptp(tri[:, :, axis], axis=1) <= 0.001)
        if not sel.any():
            continue
        faces = tri[sel]
        keys, group = np.unique(np.c_[np.round(faces[:, :, axis].mean(axis=1), 3), np.sign(normals[sel, axis])],
                                axis=0, return_inverse=True)
        group = group.ravel()
        area = np.bincount(group, twice_area[sel] / 2)
        in_plane = [i for i in range(3) if i != axis]
        for k in np.flatnonzero(area >= min_area):
            out.append(_PlaneFaces(node, axis, float(keys[k, 0]), int(keys[k, 1]), transparent,
                                   faces[group == k][:, :, in_plane]))
    return out


class _PlaneIndex:
    """Faces by plane, to find the ones whose bounds overlap a given face."""

    def __init__(self, faces: list[_PlaneFaces]):
        self.planes: dict[tuple[int, float], list[_PlaneFaces]] = {}
        for f in faces:
            self.planes.setdefault((f.axis, f.plane), []).append(f)
        self.trees: dict[tuple[int, float], object] = {}

    def near(self, face: _PlaneFaces) -> list[_PlaneFaces]:
        import shapely

        key = (face.axis, face.plane)
        items = self.planes.get(key)
        if not items:
            return []
        if key not in self.trees:
            self.trees[key] = shapely.STRtree([shapely.box(*f.bounds) for f in items])
        return [items[i] for i in self.trees[key].query(shapely.box(*face.bounds))]


def _zfight_area(a: _PlaneFaces, b: _PlaneFaces, min_area: float) -> float:
    """How much of ``a`` and ``b`` (same plane) fight, 0 if they can't.

    Opaque faces are culled from behind, so only faces turned the same way fight; transparent
    materials export double-sided, so for them opposite facings do too. Undersides on or below
    the ground plane (y = 0, where scenes are authored) can't be seen.
    """
    if a.node is b.node or (a.sign != b.sign and not (a.transparent or b.transparent)):
        return 0.0
    if a.axis == 1 and a.sign < 0 and b.sign < 0 and a.plane <= 0.005:
        return 0.0
    ab, bb = a.bounds, b.bounds
    if (min(ab[2], bb[2]) - max(ab[0], bb[0])) * (min(ab[3], bb[3]) - max(ab[1], bb[1])) < min_area:
        return 0.0
    import shapely

    return float(shapely.area(shapely.intersection(a.shape, b.shape, grid_size=_ZFIGHT_GRID)))


def _zfight_issue(a: _PlaneFaces, b: _PlaneFaces, area: float, when: str = "") -> Issue:
    axis = "xyz"[a.axis]
    if a.sign != b.sign:
        facing = "opposite-facing, transparent"
    elif a.axis == 1:
        facing = "up-facing" if a.sign > 0 else "down-facing"
    else:
        facing = f"{'+' if a.sign > 0 else '-'}{axis}-facing"
    return Issue("zfight", a.node.name, (a.node.name, b.node.name),
                 f"{a.node.name} and {b.node.name} are coplanar at {axis}={a.plane:.3f}{when}"
                 f" ({facing}, {area:.2f} m² overlap)")


def _changed_parts(interaction, state: str) -> list[SceneNode]:
    """The parts ``interaction`` poses differently in ``state`` than in its initial state."""
    parts: list[SceneNode] = []
    for motion in interaction.motions:
        if motion.kind in ("rotate", "translate", "visible") \
                and interaction.value(motion, state) != interaction.value(motion, interaction.initial):
            parts += [p for p in motion.parts if p not in parts]
    return parts


def _shown(node: SceneNode) -> bool:
    while node is not None:
        if node.hidden:
            return False
        node = node.parent
    return True


def coplanar_overlaps(scene: SceneNode, min_area: float = ZFIGHT_MIN_AREA) -> list[Issue]:
    """Faces of different meshes that lie in the same axis-aligned plane and overlap: they
    z-fight in every renderer (a floor on top of a lot slab, a door seam level with its leaf).

    Faces are grouped by plane (x, y or z, to 1 mm) and facing; a transparent mesh also fights
    faces turned the other way (see :func:`_zfight_area`). Every state of every interaction is
    posed in turn, the others staying in their initial state, so a button cap that only sinks
    level with its base when pressed is found too; the scene is left as it was. Returns
    ``zfight`` issues naming both nodes, the overlap area and the state.
    """
    from .interactions import apply_state

    faces = [f for node, mesh in scene.iter_meshes() for f in _plane_faces(node, mesh, min_area)]
    index = _PlaneIndex(faces)
    issues: list[Issue] = []
    reported: set[tuple] = set()

    def report(moved: list[_PlaneFaces], against: _PlaneIndex, skip: set[int], when: str = "") -> None:
        for a in moved:
            for b in against.near(a):
                if id(b.node) in skip:
                    continue
                pair = tuple(sorted((id(a.node), id(b.node))))
                key = (pair, a.axis, a.plane, tuple(sorted((a.sign, b.sign))))
                if key in reported:
                    continue
                area = _zfight_area(a, b, min_area)
                if area >= min_area:
                    reported.add(key)
                    issues.append(_zfight_issue(a, b, area, when))

    report(faces, index, set())
    for owner in scene.iter_nodes():
        for interaction in owner.interactions:
            for state in interaction.states:
                parts = _changed_parts(interaction, state)
                if state == interaction.initial or not parts:
                    continue
                posed = [(p, p.transform, p.hidden) for m in interaction.motions for p in m.parts]
                applied = [m.applied for m in interaction.motions]
                try:
                    apply_state(owner, interaction, state)
                    nodes = {id(n): n for p in parts for n in p.iter_nodes()}
                    moved = [f for n in nodes.values()
                             if n.mesh is not None and n.meta.get("type") != "collider" and _shown(n)
                             for f in _plane_faces(n, n.world_mesh(), min_area)]
                    when = f" when {interaction.name} is {state}"
                    report(moved, index, set(nodes), when)   # against what stayed put
                    report(moved, _PlaneIndex(moved), set(), when)   # parts moving differently
                finally:
                    for part, transform, hidden in posed:
                        part.transform, part.hidden = transform, hidden
                    for motion, value in zip(interaction.motions, applied):
                        motion.applied = value
    return issues


OVERLAP_MIN_DEPTH = 0.02    # m: shallower interpenetration (feet on a rug, a base sunk into paving) is fine
OVERLAP_MIN_VOLUME = 2e-5   # m³ (20 cm³)


def object_overlaps(scene: SceneNode, min_depth: float = OVERLAP_MIN_DEPTH,
                    min_volume: float = OVERLAP_MIN_VOLUME) -> list[Issue]:
    """Solid parts of different placed objects that run into each other, e.g. a car through a wall.

    An object is the nearest asset root above a mesh (``meta.source``, set by the loader), or
    else the scene's top-level placement. Pairs whose world bounds overlap are intersected
    exactly (manifold3d); the depth is the smallest extent of the shared volume. Only closed
    meshes can be measured; skinned bodies and collider-only nodes are left out (affordance QA
    poses bodies). Returns ``overlap`` issues naming both nodes.

    Deliberate embedding isn't reported: openings (objects with host cutters: sills and steps
    set into their wall) and anything sunk into a walkable slab no higher than its top
    (foundations in the ground, a road over the ground plane).

    NPCs aren't meshes to measure (their bodies are skinned), but each one placed with its body
    inside something solid (a diner's slot in a booth, geogen.65) is an ``npc_blocked`` issue:
    the runtime would start it off the navmesh, wedged. See :class:`Solids`.
    """
    solids = Solids(scene)
    entries, solid = solids.entries, solids.solid
    entries.sort(key=lambda e: e[3][0])
    issues: list[Issue] = []
    unmeasured = 0
    active: list = []
    for e in entries:
        active = [a for a in active if a[4][0] > e[3][0] + min_depth]
        for a in active:
            if a[1] is e[1] or np.any(np.minimum(a[4], e[4]) - np.maximum(a[3], e[3]) < min_depth):
                continue
            if a[1].host_cutters or e[1].host_cutters:
                continue
            ma, me = solid(a[0], a[2]), solid(e[0], e[2])
            if ma is None or me is None:
                unmeasured += 1
                continue
            shared = ma ^ me
            if shared.is_empty():
                continue
            box = np.array(shared.bounding_box())
            extent = box[3:] - box[:3]
            depth, volume = float(extent.min()), float(shared.volume())
            if depth < min_depth or volume < min_volume:
                continue
            ground = [x for x in (a, e) if x[0].meta.get("walkable")]
            if any(box[4] <= g[4][1] + 0.01 for g in ground):
                continue
            centre = (box[:3] + box[3:]) / 2
            na, ne = _path(a[0], a[1]), _path(e[0], e[1])
            issues.append(Issue("overlap", a[1].name, (na, ne),
                                f"{na} and {ne} overlap {depth:.2f} m deep ({volume * 1000:.1f} L)"
                                f" around [{centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}]"))
        active.append(e)
    for node, world in solids.npcs:
        feet = world[:3, 3]
        for where in solids.blocking(feet, **npc_body(node)):
            issues.append(Issue("npc_blocked", node.name, (node.name, where),
                                f"{node.name} stands inside {where} at "
                                f"[{feet[0]:.2f}, {feet[1]:.2f}, {feet[2]:.2f}]: it would start wedged"))
    issues.sort(key=lambda i: i.message)
    if unmeasured:
        issues.append(Issue("overlap_unmeasured", scene.name, (),
                            f"{unmeasured} touching pair(s) not measured (open meshes)"))
    return issues


NPC_CLEAR_ABOVE = 0.2      # m over the feet: floors, rugs and a kerb's step don't block a body
NPC_BLOCK_DEPTH = 0.05     # m: something solid this far into a placed NPC's body column


def npc_body(node: SceneNode) -> dict[str, float]:
    """An NPC node's body column for :meth:`Solids.blocking` (the runtime's capsule radius)."""
    spec = node.meta.get("npc") or {}
    return {"radius": float(spec.get("radius", 0.25)), "height": float(spec.get("height", 1.7))}


class Solids:
    """A scene's fixed solid meshes, world-placed, with their owners (the nearest asset root, or
    the top-level placement) and world bounds: ``entries`` is (node, owner, world matrix, lo, hi).
    Skinned bodies, colliders, spawned-item templates and NPCs (``npcs``: (node, world matrix))
    are left out. ``solid`` makes a mesh's manifold (None for an open mesh), cached per mesh."""

    def __init__(self, scene: SceneNode):
        self.entries: list = []
        self.npcs: list = []
        self._local: dict[int, object] = {}

        def visit(node: SceneNode, owner: SceneNode | None, world: np.ndarray) -> None:
            if "item" in node.meta:
                return                      # a spawned item's template (geogen.flows): not in the world
            if "source" in node.meta or owner is None:
                owner = node
            world = world @ node.transform.to_matrix()
            if node.meta.get("type") == "npc":
                self.npcs.append((node, world))
                return                      # a character moves: its parts aren't fixed objects
            if (node.mesh is not None and len(node.mesh.faces) and node.skin is None
                    and node.meta.get("type") != "collider"):
                v = node.mesh.vertices @ world[:3, :3].T + world[:3, 3]
                self.entries.append((node, owner, world, v.min(axis=0), v.max(axis=0)))
            for child in node.children:
                visit(child, owner, world)

        for child in scene.children:
            visit(child, None, scene.world_transform())

    def solid(self, node: SceneNode, world: np.ndarray):
        from ..core import meshops

        key = id(node.mesh)
        if key not in self._local:
            mesh = node.mesh
            closed = meshops.validate(mesh).watertight
            self._local[key] = _solid(mesh.vertices, mesh.faces) if closed else None
        man = self._local[key]
        return None if man is None else man.transform(world[:3])

    def blocking(self, feet: np.ndarray, radius: float = 0.25, height: float = 1.7) -> list[str]:
        """What is solid inside a body standing at ``feet`` (scene frame), as ``owner`` names: a
        column ``radius`` wide from ``NPC_CLEAR_ABOVE`` to ``height``, walkable slabs aside, with
        anything at least ``NPC_BLOCK_DEPTH`` into it across."""
        import manifold3d

        lo = np.asarray(feet, dtype=np.float64) + np.array([-radius, NPC_CLEAR_ABOVE, -radius])
        hi = np.asarray(feet, dtype=np.float64) + np.array([radius, height, radius])
        column, found = None, {}
        for node, owner, world, e_lo, e_hi in self.entries:
            if (owner.name in found or node.meta.get("walkable")
                    or np.any(np.minimum(hi, e_hi) - np.maximum(lo, e_lo) < NPC_BLOCK_DEPTH)):
                continue
            body = self.solid(node, world)
            if body is None:
                continue
            if column is None:
                column = manifold3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
            shared = column ^ body
            if shared.is_empty():
                continue
            box = np.array(shared.bounding_box())
            if min(box[3] - box[0], box[5] - box[2]) >= NPC_BLOCK_DEPTH:
                found[owner.name] = True
        return list(found)


def _path(node: SceneNode, owner: SceneNode) -> str:
    """``owner/part`` naming for a mesh node (just the owner when the mesh is its root)."""
    names = []
    while node is not None and node is not owner:
        names.append(node.name)
        node = node.parent
    return "/".join([owner.name, *reversed(names)])


def _solid(v: np.ndarray, f: np.ndarray):
    import manifold3d

    mm = manifold3d.Mesh(vert_properties=np.ascontiguousarray(v, dtype=np.float32),
                         tri_verts=np.ascontiguousarray(f, dtype=np.uint32))
    mm.merge()     # stitch seams (vertices split for normals/UVs)
    man = manifold3d.Manifold(mm)
    return man if man.status() == manifold3d.Error.NoError and not man.is_empty() else None
