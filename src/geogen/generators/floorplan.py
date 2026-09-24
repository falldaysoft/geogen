"""Floor plans: rooms as grid rectangles, walls generated once per boundary.

A :class:`FloorPlan` holds axis-aligned rooms (``rect: [x, z, w, d]`` on the
plan's XZ grid, measured on wall centre lines) plus doors and windows.
:meth:`FloorPlan.build` turns it into a hierarchy:

    <plan>
    ├── walls              one watertight mesh for every wall, openings cut
    └── <room> ...         per room: floor / ceiling slabs, surfaces, tags

Walls are derived from room edges: an edge shared by two rooms becomes an
interior wall (``interior_wall`` thick), an edge with a room on one side
only becomes an exterior wall (``exterior_wall`` thick). Walls are centred
on the edge. Each boundary segment is extended at its ends just far enough
to meet the perpendicular walls there, and all segments are unioned in 2D
before extruding, so L/T/X junctions are clean: no overlaps, no gaps.

Sides follow :class:`RoomGenerator`: north = +Z, south = -Z, east = +X,
west = -X. Room wall surfaces face into the room; ``<side>_exterior``
surfaces face out of the building (for placing window/door assets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode
from ..core.transform import Transform
from ..layout.surfaces import Surface

SIDES = ("north", "south", "east", "west")

# side -> (axis the wall line is constant on, which rect edge, inward normal)
_SIDE_INFO = {
    "north": ("z", "max", np.array([0.0, 0.0, -1.0])),
    "south": ("z", "min", np.array([0.0, 0.0, 1.0])),
    "east": ("x", "max", np.array([-1.0, 0.0, 0.0])),
    "west": ("x", "min", np.array([1.0, 0.0, 0.0])),
}

_EPS = 1e-6


@dataclass
class PlanRoom:
    """A rectangular room: ``x, z`` is the min corner, ``w, d`` the size (centre lines)."""

    name: str
    x: float
    z: float
    w: float
    d: float
    type: str | None = None
    floor: str | None = None      # material names (default: room type finishes, then plan materials)
    walls: str | None = None
    ceiling: str | None = None

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def z1(self) -> float:
        return self.z + self.d

    def edge(self, side: str) -> tuple[str, float, float, float]:
        """(axis, line coordinate, lo, hi) of one side, lo/hi along the line."""
        axis, which, _ = _SIDE_INFO[side]
        if axis == "z":
            return "z", self.z1 if which == "max" else self.z, self.x, self.x1
        return "x", self.x1 if which == "max" else self.x, self.z, self.z1


@dataclass
class WallSegment:
    """A straight run of wall with the same room(s) on either side."""

    axis: str                 # "z": wall runs along X at constant z; "x": along Z at constant x
    line: float
    lo: float
    hi: float
    rooms: dict[str, str]     # room name -> side of that room this segment lies on
    thickness: float

    @property
    def exterior(self) -> bool:
        return len(self.rooms) == 1

    @property
    def length(self) -> float:
        return self.hi - self.lo


@dataclass
class PlanOpening:
    """A door or window cut through a wall.

    Doors either connect two rooms (``between``) or open a room to the
    outside (``room`` + ``side``). Windows use ``room`` + ``side``.
    ``at`` is the opening centre as a 0-1 fraction along the shared segment
    (doors between rooms, measured from the segment's low end) or along the
    room's side (u left -> right when facing the wall from inside).
    ``at_abs`` gives the same in metres instead.
    """

    kind: str                              # "door" | "window"
    width: float
    height: float
    sill: float = 0.0
    between: tuple[str, str] | None = None
    room: str | None = None
    side: str | None = None
    at: float = 0.5
    at_abs: float | None = None
    name: str | None = None
    # Doors only: 'door' (lining, architraves, leaf), 'archway' (no leaf) or
    # 'opening' (bare hole); which room the leaf swings into (default: the
    # second room of 'between', or the room itself for outside doors; 'out'
    # swings an outside door outward); hinge side seen from the swing side.
    style: str = "door"
    swing: str | None = None
    hinge: str = "left"
    open_deg: float = 0.0


@dataclass
class FloorPlan:
    """Rooms, openings and wall settings for one storey."""

    rooms: dict[str, PlanRoom]
    doors: list[PlanOpening] = field(default_factory=list)
    windows: list[PlanOpening] = field(default_factory=list)
    grid: float = 0.1
    wall_height: float = 2.7
    exterior_wall: float = 0.3
    interior_wall: float = 0.12
    floor_thickness: float = 0.05
    # Structural slab + finish; thin ceilings let shadow maps leak light at the seams.
    ceiling_thickness: float = 0.2
    ceiling: bool = True
    materials: dict[str, str] = field(default_factory=lambda: {
        "walls": "wall_plaster", "floor": "hardwood_floor", "ceiling": "ceiling_white",
    })
    # Interior finishes per room (see finishes.py): lining, skirting, cornice,
    # light, switches. False turns them all off.
    finishes: dict[str, bool] | bool = True
    # Where room archetypes live (their ``finishes:`` give per-type materials).
    room_types_dir: Path | None = None

    # ------------------------------------------------------------------ parsing

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> FloorPlan:
        """Build from the YAML ``floorplan:`` mapping."""
        known = {"grid", "wall_height", "exterior_wall", "interior_wall", "floor_thickness",
                 "ceiling_thickness", "ceiling", "materials", "rooms", "doors", "windows", "finishes"}
        unknown = set(spec) - known
        if unknown:
            raise ValueError(f"Unknown floorplan keys: {sorted(unknown)}. Known: {sorted(known)}")
        if not spec.get("rooms"):
            raise ValueError("floorplan needs at least one room under 'rooms:'")

        grid = float(spec.get("grid", 0.1))
        rooms = {}
        for name, r in spec["rooms"].items():
            if "rect" not in r or len(r["rect"]) != 4:
                raise ValueError(f"Room '{name}' needs rect: [x, z, width, depth]")
            x, z, w, d = (float(v) for v in r["rect"])
            if w <= 0 or d <= 0:
                raise ValueError(f"Room '{name}' has non-positive size {w} x {d}")
            extra = set(r) - {"rect", "type", "floor", "walls", "ceiling"}
            if extra:
                raise ValueError(f"Unknown keys in room '{name}': {sorted(extra)}")
            rooms[name] = PlanRoom(name, x, z, w, d, type=r.get("type"),
                                   floor=r.get("floor"), walls=r.get("walls"), ceiling=r.get("ceiling"))

        def opening(kind: str, o: dict[str, Any]) -> PlanOpening:
            between = o.get("between")
            at = o.get("at", 0.5)
            at_abs = None
            if isinstance(at, dict):
                at_abs = float(at["abs"]) if "abs" in at else None
                at = float(at.get("frac", 0.5))
            return PlanOpening(
                kind=kind,
                width=float(o.get("width", 0.9 if kind == "door" else 1.2)),
                height=float(o.get("height", 2.1 if kind == "door" else 1.3)),
                sill=float(o.get("sill", 0.0 if kind == "door" else 0.9)),
                between=tuple(between) if between else None,
                room=o.get("room"), side=o.get("side"),
                at=float(at), at_abs=at_abs, name=o.get("name"),
                style=o.get("style", "door"), swing=o.get("swing"),
                hinge=o.get("hinge", "left"), open_deg=float(o.get("open", 0.0)),
            )

        materials = {"walls": "wall_plaster", "floor": "hardwood_floor", "ceiling": "ceiling_white"}
        materials.update(spec.get("materials", {}))
        plan = cls(
            rooms=rooms,
            doors=[opening("door", o) for o in spec.get("doors", [])],
            windows=[opening("window", o) for o in spec.get("windows", [])],
            grid=grid,
            wall_height=float(spec.get("wall_height", 2.7)),
            exterior_wall=float(spec.get("exterior_wall", 0.3)),
            interior_wall=float(spec.get("interior_wall", 0.12)),
            floor_thickness=float(spec.get("floor_thickness", 0.05)),
            ceiling_thickness=float(spec.get("ceiling_thickness", 0.2)),
            ceiling=bool(spec.get("ceiling", True)),
            materials=materials,
            finishes=spec.get("finishes", True),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        """Check grid alignment, overlaps and opening references."""
        if self.grid > 0:
            for room in self.rooms.values():
                for v in (room.x, room.z, room.w, room.d):
                    if abs(v / self.grid - round(v / self.grid)) > 1e-6:
                        raise ValueError(f"Room '{room.name}' rect value {v} is off the {self.grid} m grid")
        names = list(self.rooms)
        for i, a in enumerate(names):
            ra = self.rooms[a]
            for b in names[i + 1:]:
                rb = self.rooms[b]
                ox = min(ra.x1, rb.x1) - max(ra.x, rb.x)
                oz = min(ra.z1, rb.z1) - max(ra.z, rb.z)
                if ox > _EPS and oz > _EPS:
                    raise ValueError(f"Rooms '{a}' and '{b}' overlap")
        for o in [*self.doors, *self.windows]:
            refs = list(o.between or []) + ([o.room] if o.room else [])
            if not refs:
                raise ValueError(f"{o.kind} needs 'between: [a, b]' or 'room' + 'side'")
            for r in refs:
                if r not in self.rooms:
                    raise ValueError(f"{o.kind} references unknown room '{r}'. Rooms: {sorted(self.rooms)}")
            if o.room and o.side not in SIDES:
                raise ValueError(f"{o.kind} in '{o.room}' needs side: one of {SIDES}")
            if o.kind == "door":
                if o.style not in ("door", "archway", "opening"):
                    raise ValueError(f"door style must be door|archway|opening, got {o.style!r}")
                if o.hinge not in ("left", "right"):
                    raise ValueError(f"door hinge must be left|right, got {o.hinge!r}")
                allowed = set(o.between or []) | ({"out"} if o.room else set()) | ({o.room} if o.room else set())
                if o.swing is not None and o.swing not in allowed:
                    raise ValueError(f"door swing must be one of {sorted(allowed)}, got {o.swing!r}")

    # ---------------------------------------------------------------- geometry

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x0, z0, x1, z1) of the room centre lines."""
        rs = self.rooms.values()
        return (min(r.x for r in rs), min(r.z for r in rs), max(r.x1 for r in rs), max(r.z1 for r in rs))

    @property
    def center(self) -> np.ndarray:
        x0, z0, x1, z1 = self.bounds
        return np.array([(x0 + x1) / 2, 0.0, (z0 + z1) / 2])

    @property
    def size(self) -> np.ndarray:
        """Outer size including exterior walls, floor slab and wall height."""
        x0, z0, x1, z1 = self.bounds
        t = self.exterior_wall
        return np.array([x1 - x0 + t, self.wall_height + self.floor_thickness, z1 - z0 + t])

    def segments(self) -> list[WallSegment]:
        """Split every room edge at every other edge's endpoints and pair rooms."""
        lines: dict[tuple[str, float], list[tuple[float, float, str, str]]] = {}
        for room in self.rooms.values():
            for side in SIDES:
                axis, line, lo, hi = room.edge(side)
                key = (axis, round(line, 6))
                lines.setdefault(key, []).append((lo, hi, room.name, side))

        segments: list[WallSegment] = []
        for (axis, line), edges in lines.items():
            cuts = sorted({round(v, 6) for lo, hi, _, _ in edges for v in (lo, hi)})
            pieces: list[tuple[float, float, dict[str, str]]] = []
            for a, b in zip(cuts, cuts[1:]):
                mid = (a + b) / 2
                rooms = {name: side for lo, hi, name, side in edges if lo < mid < hi}
                if not rooms:
                    continue
                if len(rooms) > 2:
                    raise ValueError(f"More than two rooms share wall {axis}={line} at {mid}")
                if pieces and abs(pieces[-1][1] - a) < _EPS and pieces[-1][2] == rooms:
                    pieces[-1] = (pieces[-1][0], b, rooms)
                else:
                    pieces.append((a, b, rooms))
            for a, b, rooms in pieces:
                t = self.exterior_wall if len(rooms) == 1 else self.interior_wall
                segments.append(WallSegment(axis, line, a, b, rooms, t))
        return segments

    def _wall_footprint(self, segments: list[WallSegment]):
        """2D union (shapely) of all wall rectangles in plan (x, z) coordinates."""
        from shapely.geometry import box
        from shapely.ops import unary_union

        def half_thickness_at(point: tuple[float, float], axis: str) -> float:
            """Half the thickness of the thickest perpendicular wall through ``point``."""
            best = 0.0
            for s in segments:
                if s.axis == axis:
                    continue
                # s is perpendicular: its line is the other coordinate
                if s.axis == "z":   # runs along X at z = line
                    hit = abs(point[1] - s.line) < _EPS and s.lo - _EPS <= point[0] <= s.hi + _EPS
                else:               # runs along Z at x = line
                    hit = abs(point[0] - s.line) < _EPS and s.lo - _EPS <= point[1] <= s.hi + _EPS
                if hit:
                    best = max(best, s.thickness / 2)
            return best

        rects = []
        for s in segments:
            h = s.thickness / 2
            if s.axis == "z":
                e0 = half_thickness_at((s.lo, s.line), "z")
                e1 = half_thickness_at((s.hi, s.line), "z")
                rects.append(box(s.lo - e0, s.line - h, s.hi + e1, s.line + h))
            else:
                e0 = half_thickness_at((s.line, s.lo), "x")
                e1 = half_thickness_at((s.line, s.hi), "x")
                rects.append(box(s.line - h, s.lo - e0, s.line + h, s.hi + e1))
        return unary_union(rects).simplify(1e-9)

    def _wall_mesh(self, segments: list[WallSegment], offset: np.ndarray) -> Mesh:
        from ..core.profile import Shape
        from .profiles import ExtrudeGenerator

        footprint = self._wall_footprint(segments)
        polys = [footprint] if footprint.geom_type == "Polygon" else list(footprint.geoms)  # type: ignore[attr-defined]
        height = self.wall_height + self.floor_thickness
        meshes = []
        for poly in polys:
            # Extrude axis "y" maps profile (u, v) to (X, -Z).
            def to_profile(coords) -> np.ndarray:
                pts = np.asarray(coords)[:-1]
                return np.column_stack([pts[:, 0] - offset[0], -(pts[:, 1] - offset[2])])

            shape = Shape(to_profile(poly.exterior.coords), [to_profile(i.coords) for i in poly.interiors])
            mesh = ExtrudeGenerator(shape=shape, depth=height, axis="y", crease_angle=30.0).generate()
            shift = np.eye(4)
            shift[1, 3] = -self.floor_thickness - mesh.vertices[:, 1].min()
            meshes.append(mesh.transform(shift))
        return meshes[0] if len(meshes) == 1 else Mesh.merge(meshes)

    def _opening_frame(self, o: PlanOpening, segments: list[WallSegment]) -> tuple[WallSegment, float]:
        """The wall segment an opening sits in and its centre along the wall line."""
        if o.between:
            a, b = o.between
            candidates = [s for s in segments if a in s.rooms and b in s.rooms]
            if not candidates:
                raise ValueError(f"door between '{a}' and '{b}': the rooms don't share a wall")
            seg = max(candidates, key=lambda s: s.length)
            lo, hi, reverse = seg.lo, seg.hi, False
        else:
            assert o.room is not None and o.side is not None  # checked in validate()
            room = self.rooms[o.room]
            axis, line, lo, hi = room.edge(o.side)
            # u runs left -> right facing the wall from inside the room.
            reverse = o.side in ("north", "west")
            seg = next(
                (s for s in segments if s.axis == axis and abs(s.line - line) < _EPS
                 and o.room in s.rooms and s.rooms[o.room] == o.side), None)
            if seg is None:
                raise ValueError(f"{o.kind} in '{o.room}': no wall on side '{o.side}'")
            thickest = max(s.thickness for s in segments if s.axis == axis and abs(s.line - line) < _EPS
                           and o.room in s.rooms and s.rooms[o.room] == o.side)
            seg = WallSegment(axis, line, lo, hi, seg.rooms, thickest)

        length = hi - lo
        along = o.at_abs if o.at_abs is not None else o.at * length
        if reverse:
            along = length - along
        centre = lo + along
        if along - o.width / 2 < -_EPS or along + o.width / 2 > length + _EPS:
            raise ValueError(f"{o.kind} ({o.width} m wide at {along:.2f} m) doesn't fit its {length:.2f} m wall")

        return seg, centre

    def _opening_box(self, o: PlanOpening, segments: list[WallSegment], offset: np.ndarray) -> Mesh:
        """Cutter box for one opening, in plan-root coordinates."""
        from .primitives import CubeGenerator

        seg, centre = self._opening_frame(o, segments)
        depth = seg.thickness + 0.2
        if seg.axis == "z":
            size = (o.width, o.height, depth)
            pos = np.array([centre, o.sill + o.height / 2, seg.line])
        else:
            size = (depth, o.height, o.width)
            pos = np.array([seg.line, o.sill + o.height / 2, centre])
        mesh = CubeGenerator(size_x=size[0], size_y=size[1], size_z=size[2], bevel=0).generate()
        move = np.eye(4)
        move[:3, 3] = pos - offset
        return mesh.transform(move)

    # ---------------------------------------------------------------- building

    def build(self, name: str = "floorplan", material_loader=None) -> SceneNode:
        """Build the plan hierarchy, centred on the plan bounds at floor level."""
        from ..core import csg, uvmap
        from ..materials.loader import MaterialLoader
        from .primitives import CubeGenerator

        loader = material_loader or MaterialLoader()
        offset = self.center
        segments = self.segments()

        root = SceneNode(name=name)
        root.size = self.size
        root.tags = ["floorplan"]

        walls_mesh = self._wall_mesh(segments, offset)
        cutters = [self._opening_box(o, segments, offset) for o in [*self.doors, *self.windows]]
        if cutters:
            walls_mesh = csg.difference(walls_mesh, *cutters, crease_angle=30.0)
        walls_mesh = uvmap.box_project(walls_mesh)
        walls_mesh.material = loader.load(self.materials["walls"])
        walls = SceneNode(name="walls", mesh=walls_mesh, tags=["wall"])
        root.add_child(walls)

        room_nodes = {}
        for room in self.rooms.values():
            root.add_child(self._room_node(room, segments, offset, walls, loader, CubeGenerator))
            node = root.children[-1]
            room_nodes[room.name] = node
            to_root = node.transform.to_matrix()
            for surf_name, surf in node.surfaces.items():
                root.surfaces[f"{room.name}.{surf_name}"] = _transformed(surf, to_root)

        # Openings per room, in the room's frame (for furnishing): which wall,
        # the interval along it and the vertical extent.
        for o in [*self.doors, *self.windows]:
            seg, centre = self._opening_frame(o, segments)
            for room_name, side in seg.rooms.items():
                room_node = room_nodes[room_name]
                origin = room_node.transform.translation + offset  # plan coords of the room centre
                along0 = origin[0] if seg.axis == "z" else origin[2]
                room_node.meta.setdefault("openings", []).append({
                    "kind": o.kind, "side": side,
                    "lo": round(float(centre - o.width / 2 - along0), 6),
                    "hi": round(float(centre + o.width / 2 - along0), 6),
                    "sill": o.sill, "top": o.sill + o.height,
                })

        for index, door in enumerate(self.doors):
            if door.style != "opening":
                root.add_child(self._door_node(door, index, segments, offset, loader, room_nodes))

        if self.finishes:
            from .finishes import add_room_finishes

            options = self.finishes if isinstance(self.finishes, dict) else {}
            for room in self.rooms.values():
                node = room_nodes[room.name]
                to_room = np.eye(4)
                to_room[:3, 3] = -node.transform.translation
                add_room_finishes(node, float(node.meta.get("clear_height", self.wall_height)),
                                  self.room_materials(room), [c.transform(to_room) for c in cutters],
                                  options, loader)
        return root

    def _door_node(self, o: PlanOpening, index: int, segments: list[WallSegment], offset: np.ndarray,
                   loader, room_nodes: dict[str, SceneNode]) -> SceneNode:
        """Place a DoorGenerator in an opening, +Z toward the room it swings into."""
        from .doors import DoorGenerator

        seg, centre = self._opening_frame(o, segments)
        exterior = o.between is None
        if o.between:
            swing_room = o.swing or o.between[1]
        else:
            swing_room = o.room if o.swing in (None, o.room) else None  # None: swings outside
        # Which way is the swing side? A room on its north edge lies toward -Z, etc.
        toward = {"north": -1.0, "south": 1.0, "east": -1.0, "west": 1.0}
        if swing_room is not None:
            direction = toward[seg.rooms[swing_room]]
        else:
            direction = -toward[seg.rooms[o.room]]
        if seg.axis == "z":
            position = np.array([centre, o.sill, seg.line])
            yaw = 0.0 if direction > 0 else 180.0
        else:
            position = np.array([seg.line, o.sill, centre])
            yaw = 90.0 if direction > 0 else -90.0

        name = o.name or (f"door_{o.between[0]}_{o.between[1]}" if o.between else f"door_{o.room}_{o.side}")
        gen = DoorGenerator(
            width=o.width, height=o.height, wall=seg.thickness, style=o.style, hinge=o.hinge,
            open_deg=o.open_deg, name=name,
            frame_material=self.materials.get("door_frame", "trim_white"),
            leaf_material=self.materials.get("door_leaf", "wood"),
        )
        node = gen.generate(loader)
        node.transform = Transform(translation=position - offset, rotation=np.array([0.0, np.radians(yaw), 0.0]))
        kind = "door" if o.style == "door" else "opening"
        node.tags = [kind, f"{kind}.{o.style if o.style != 'door' else ('exterior' if exterior else 'interior')}"]
        node.meta["door"] = {
            "style": o.style, "hinge": o.hinge, "width": o.width, "height": o.height,
            "rooms": list(o.between) if o.between else [o.room], "swing_room": swing_room,
        }

        # Record the leaf's swing arc in the swing room's frame so furnishing avoids it.
        if o.style == "door" and swing_room is not None:
            room_node = room_nodes[swing_room]
            swing = gen.swing()
            r = np.radians(yaw)
            rot = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
            hinge = rot @ np.array(swing["hinge"]) + node.transform.translation - room_node.transform.translation
            room_node.meta.setdefault("door_swings", []).append({
                "door": name,
                "hinge": [round(float(v), 6) for v in hinge],
                "radius": swing["radius"],
                "from_deg": swing["from_deg"] - yaw,
                "to_deg": swing["to_deg"] - yaw,
            })
        return node

    def _lining_thickness(self) -> float:
        from .finishes import LINING

        if not self.finishes:
            return 0.0
        options = self.finishes if isinstance(self.finishes, dict) else {}
        return LINING if options.get("lining", True) else 0.0

    def room_materials(self, room: PlanRoom) -> dict[str, str]:
        """floor/walls/ceiling materials: room keys, then its type's finishes, then the plan's."""
        result = dict(self.materials)
        if room.type:
            from ..layout.yaml_utils import safe_load_path

            types_dir = self.room_types_dir or Path(__file__).parents[3] / "assets" / "room_types"
            path = types_dir / f"{room.type}.yaml"
            if path.exists():
                result.update((safe_load_path(path) or {}).get("finishes") or {})
        for key in ("floor", "walls", "ceiling"):
            if getattr(room, key):
                result[key] = getattr(room, key)
        return result

    def _half(self, room: PlanRoom, side: str, segments: list[WallSegment]) -> float:
        """Half-thickness of the thickest wall along one side of a room."""
        axis, line, _, _ = room.edge(side)
        ts = [s.thickness for s in segments
              if s.axis == axis and abs(s.line - line) < _EPS and s.rooms.get(room.name) == side]
        return max(ts) / 2 if ts else 0.0

    def _room_node(self, room, segments, offset, walls, loader, CubeGenerator) -> SceneNode:
        # Clear (interior) rectangle: centre lines minus half the wall on each side.
        hw = {side: self._half(room, side, segments) for side in SIDES}
        cx0, cx1 = room.x + hw["west"], room.x1 - hw["east"]
        cz0, cz1 = room.z + hw["south"], room.z1 - hw["north"]
        sx, sz = cx1 - cx0, cz1 - cz0
        centre = np.array([(cx0 + cx1) / 2, 0.0, (cz0 + cz1) / 2]) - offset

        node = SceneNode(name=room.name, transform=Transform(translation=centre))
        node.size = np.array([sx, self.wall_height, sz])
        room_type = room.type or "room"
        node.tags = ["room", f"room.{room_type}"]
        node.meta = {"room": {"id": room.name, "type": room_type}}

        # Ceilings tuck into the walls so shadow maps don't leak light through
        # the seam; floors stop at the wall face so doorway thresholds (the
        # wall's top between two rooms) don't z-fight with them.
        tuck_ceiling = min(0.04, 0.8 * min(hw.values()))

        def slab(part: str, height: float, y: float, material: str, tuck: float = 0.0) -> SceneNode:
            mesh = CubeGenerator(size_x=sx + 2 * tuck, size_y=height, size_z=sz + 2 * tuck, bevel=0).generate()
            move = np.eye(4)
            move[1, 3] = y + height / 2
            mesh = mesh.transform(move)
            mesh.material = loader.load(material)
            return SceneNode(name=part, mesh=mesh, tags=[part])

        mats = self.room_materials(room)
        floor = slab("floor", self.floor_thickness, -self.floor_thickness, mats["floor"])
        floor.meta["walkable"] = True
        node.add_child(floor)
        clear_height = self.wall_height
        if self.ceiling:
            clear_height = self.wall_height - self.ceiling_thickness
            node.add_child(slab("ceiling", self.ceiling_thickness, clear_height, mats["ceiling"], tuck_ceiling))
        node.meta["clear_height"] = clear_height

        # Trigger volume filling the room's clear space (for "which room am I in").
        volume = SceneNode(name=f"{room.name}_volume",
                           transform=Transform(translation=np.array([0.0, clear_height / 2, 0.0])),
                           tags=["room_volume"])
        volume.meta = {"type": "room_volume", "room": dict(node.meta["room"]),
                       "size": [sx, clear_height, sz]}
        node.add_child(volume)

        hx, hz = sx / 2, sz / 2
        surfaces = {
            "floor": Surface("floor", np.array([-hx, 0.0, -hz]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                             np.array([0, 1.0, 0]), sx, sz),
            "ceiling": Surface("ceiling", np.array([-hx, clear_height, -hz]), np.array([1.0, 0, 0]),
                               np.array([0, 0, 1.0]), np.array([0, -1.0, 0]), sx, sz),
            "north_wall": Surface("north_wall", np.array([hx, 0, hz]), np.array([-1.0, 0, 0]),
                                  np.array([0, 1.0, 0]), np.array([0, 0, -1.0]), sx, clear_height),
            "south_wall": Surface("south_wall", np.array([-hx, 0, -hz]), np.array([1.0, 0, 0]),
                                  np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), sx, clear_height),
            "east_wall": Surface("east_wall", np.array([hx, 0, -hz]), np.array([0, 0, 1.0]),
                                 np.array([0, 1.0, 0]), np.array([-1.0, 0, 0]), sz, clear_height),
            "west_wall": Surface("west_wall", np.array([-hx, 0, hz]), np.array([0, 0, -1.0]),
                                 np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), sz, clear_height),
        }
        # Outside faces of exterior walls (u left -> right seen from outside).
        for side in SIDES:
            axis, line, _, _ = room.edge(side)
            if not any(s.exterior and s.axis == axis and abs(s.line - line) < _EPS
                       and s.rooms.get(room.name) == side for s in segments):
                continue
            inner = surfaces[f"{side}_wall"]
            thick = 2 * hw[side]
            surfaces[f"{side}_exterior"] = Surface(
                f"{side}_exterior",
                origin=inner.origin + inner.u_axis * inner.u_extent - inner.normal * thick,
                u_axis=-inner.u_axis, v_axis=inner.v_axis, normal=-inner.normal,
                u_extent=inner.u_extent, v_extent=clear_height,
            )
        # Interior wall surfaces sit on the finish lining when there is one.
        lining = self._lining_thickness()
        if lining:
            for side in SIDES:
                surf = surfaces[f"{side}_wall"]
                surf.origin = surf.origin + surf.normal * lining
        for surf in surfaces.values():
            surf.source = walls if surf.name.endswith(("_wall", "_exterior")) else node.children[0]
        node.surfaces = surfaces
        return node


def _transformed(surf: Surface, matrix: np.ndarray) -> Surface:
    """Copy of ``surf`` moved by a 4x4 rigid transform."""
    r, t = matrix[:3, :3], matrix[:3, 3]
    return Surface(
        name=surf.name, origin=r @ surf.origin + t, u_axis=r @ surf.u_axis, v_axis=r @ surf.v_axis,
        normal=r @ surf.normal, u_extent=surf.u_extent, v_extent=surf.v_extent,
        source=surf.source, also_cut=surf.also_cut, reveal=surf.reveal,
    )
