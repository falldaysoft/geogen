"""Furnish floor-plan rooms from declarative room archetypes.

A room archetype (``assets/room_types/<type>.yaml``) lists items as rules
rather than coordinates::

    name: hotel_bedroom
    items:
      bed:        { asset: bed.yaml, against: wall, prefer: [longest_wall, no_door_wall, no_window_wall, centered] }
      nightstand: { asset: nightstand.yaml, flank: bed, count: 2 }
      lamp:       { asset: table_lamp.yaml, on: nightstand }
      tv_console: { asset: tv_console.yaml, against: wall, prefer: [{opposite: bed}] }
      tv:         { asset: tv.yaml, mount: wall, above: tv_console, height: 0.95 }
      desk_chair: { asset: desk_chair.yaml, front_of: desk, distance: -0.15 }
      rug:        { asset: rug.yaml, under: bed, offset: 0.6 }
      curtains:   { asset: curtains.yaml, mount: window }
      armchair:   { asset: armchair.yaml, against: none, facing: center, optional: true }

Items are placed in order (earlier = higher priority) by a greedy solver on
the room's floor rectangle. Candidates run along each wall (``against:
wall``) or over a grid (``against: none``); a candidate must stay inside the
room, off other items (footprints plus their declared ``clearance``), out of
door swing arcs and the approach zone in front of each door, and items
rising more than 0.4 m above a window's sill (wardrobes, shelves) stay out
of the zone in front of it. Remaining
candidates are scored by ``prefer`` terms; ties break by a seeded jitter, so
results are deterministic for a seed. A candidate is rejected if it would
leave an already-placed item out of reach from the room's doors (see
``qa.room_reachability``). Rules that can't be met are reported
(``furnish_room`` returns them); ``optional: true`` items are skipped
quietly.

Prefer terms: ``longest_wall``, ``no_door_wall``, ``door_wall``,
``window_wall``, ``no_window_wall``, ``near_door``, ``far_door``,
``near_window``, ``centered``, ``corner``, ``{opposite: <item>}`` (the wall
the item faces, lined up with it), ``{near: <item>}``, ``{far: <item>}``.

Relations: ``flank: <item>`` (beside it on the same wall, ``count`` sides),
``front_of: <item>`` (facing it, ``distance`` beyond its front; negative
tucks in), ``under: <item>`` (centred, ``offset`` toward its front),
``on: <item>`` (its ``surface``, default ``top``), ``mount: wall`` with
``above: <item>`` or prefer terms (``height`` above the floor),
``mount: window`` (one per window, sized with ``fit: window``).
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.node import SceneNode
from ..core.transform import Transform

logger = logging.getLogger(__name__)

# side -> (along axis index (0 = x, 1 = z), sign of the wall coordinate, inward normal (x, z))
_WALLS = {
    "north": (0, +1, np.array([0.0, -1.0])),
    "south": (0, -1, np.array([0.0, 1.0])),
    "east": (1, +1, np.array([-1.0, 0.0])),
    "west": (1, -1, np.array([1.0, 0.0])),
}
GAP = 0.02          # air between neighbouring items
STEP = 0.05         # candidate spacing along walls / grid
DOOR_APPROACH = 0.9  # free depth in front of every door
MAX_REACH_TRIES = 300 # candidates tried per item before giving up on reachability
WINDOW_OVERLAP = 0.4  # items may rise this far above a sill in front of a window


@dataclass
class Rect:
    x0: float
    z0: float
    x1: float
    z1: float

    def overlaps(self, other: "Rect", gap: float = 0.0) -> bool:
        return (self.x0 < other.x1 + gap - 1e-9 and other.x0 < self.x1 + gap - 1e-9
                and self.z0 < other.z1 + gap - 1e-9 and other.z0 < self.z1 + gap - 1e-9)

    def inside(self, other: "Rect", tol: float = 1e-6) -> bool:
        return (self.x0 >= other.x0 - tol and self.x1 <= other.x1 + tol
                and self.z0 >= other.z0 - tol and self.z1 <= other.z1 + tol)

    @staticmethod
    def around(center: np.ndarray, half: np.ndarray) -> "Rect":
        return Rect(center[0] - half[0], center[1] - half[1], center[0] + half[0], center[1] + half[1])


@dataclass
class Placed:
    name: str
    node: SceneNode
    center: np.ndarray        # (x, z) in the room frame
    facing: np.ndarray        # unit (x, z): the item's +Z
    size: np.ndarray          # (w, h, d) in the item's frame
    rect: Rect | None         # floor footprint (None: doesn't occupy the floor)
    keepout: list[Rect] = field(default_factory=list)  # clearance zones
    wall: str | None = None
    along: float = 0.0


@dataclass
class Unsatisfied:
    room: str
    item: str
    reason: str

    def __str__(self) -> str:
        return f"{self.room}: {self.item}: {self.reason}"


def load_room_type(name: str, assets_dir: Path) -> dict[str, Any] | None:
    from .yaml_utils import safe_load_path

    path = assets_dir / "room_types" / f"{name}.yaml"
    return safe_load_path(path) if path.exists() else None


def furnish_plan(plan_root: SceneNode, assets_dir: Path, seed: int = 0, loader=None) -> list[Unsatisfied]:
    """Furnish every room node (``meta.room``) under ``plan_root`` that has an archetype."""
    from .loader import LayoutLoader

    loader = loader or LayoutLoader()
    report: list[Unsatisfied] = []
    for room in [n for n in plan_root.iter_nodes() if isinstance(n.meta.get("room"), dict) and n.size is not None
                 and n.meta.get("type") != "room_volume"]:
        rtype = room.meta["room"].get("type")
        spec = load_room_type(rtype, assets_dir) if rtype else None
        if spec is None:
            continue
        report += furnish_room(room, spec, assets_dir, seed, loader)
    for item in report:
        logger.warning("furnish: %s", item)
    plan_root.meta["furnish_report"] = [str(r) for r in report]
    return report


def furnish_room(room: SceneNode, spec: dict[str, Any], assets_dir: Path, seed: int, loader) -> list[Unsatisfied]:
    return _RoomSolver(room, spec, assets_dir, seed, loader).run()


class _RoomSolver:
    def __init__(self, room: SceneNode, spec: dict[str, Any], assets_dir: Path, seed: int, loader) -> None:
        self.room = room
        self.name = str(room.meta["room"].get("id", room.name))
        self.spec = spec
        self.assets_dir = assets_dir
        self.loader = loader
        self.rng = np.random.default_rng(seed + zlib.crc32(self.name.encode()))
        inset = room.meta.get("wall_inset", {})
        self.floor_inset = float(inset.get("floor", 0.0))
        self.mount_inset = float(inset.get("mount", 0.0))
        # Floor items plan on the rectangle inside the lining and skirting.
        self.hx = float(room.size[0]) / 2 - self.floor_inset
        self.hz = float(room.size[2]) / 2 - self.floor_inset
        self.bounds = Rect(-self.hx, -self.hz, self.hx, self.hz)
        self.placed: dict[str, list[Placed]] = {}
        self.report: list[Unsatisfied] = []
        openings = room.meta.get("openings", [])
        self.doors = [o for o in openings if o["kind"] == "door"]
        self.windows = [o for o in openings if o["kind"] == "window"]
        self.door_zones = self._door_zones()

    # ------------------------------------------------------------- geometry

    def wall_length(self, side: str) -> float:
        return 2 * (self.hx if _WALLS[side][0] == 0 else self.hz)

    def wall_point(self, side: str, along: float) -> np.ndarray:
        """(x, z) on the wall face."""
        axis, sign, _ = _WALLS[side]
        return np.array([along, sign * self.hz]) if axis == 0 else np.array([sign * self.hx, along])

    def along_of(self, side: str, point: np.ndarray) -> float:
        return float(point[0] if _WALLS[side][0] == 0 else point[1])

    def opening_center(self, o: dict) -> np.ndarray:
        return self.wall_point(o["side"], (o["lo"] + o["hi"]) / 2)

    def _door_zones(self) -> list[Rect]:
        zones = []
        for d in self.doors:
            inward = _WALLS[d["side"]][2]
            a = self.wall_point(d["side"], d["lo"] - 0.1)
            b = self.wall_point(d["side"], d["hi"] + 0.1) + inward * DOOR_APPROACH
            zones.append(Rect(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))
        for s in self.room.meta.get("door_swings", []):
            hinge = np.array([s["hinge"][0], s["hinge"][2]])
            angles = np.radians(np.linspace(s["from_deg"], s["to_deg"], 10))
            pts = np.vstack([hinge, hinge + s["radius"] * np.c_[np.cos(angles), np.sin(angles)]])
            zones.append(Rect(*pts.min(axis=0), *pts.max(axis=0)))
        return zones

    def window_zones(self, height: float) -> list[Rect]:
        zones = []
        for w in self.windows:
            if height <= w["sill"] + WINDOW_OVERLAP:
                continue  # low enough not to block the view (desk, armchair)
            inward = _WALLS[w["side"]][2]
            a = self.wall_point(w["side"], w["lo"])
            b = self.wall_point(w["side"], w["hi"]) + inward * 0.5
            zones.append(Rect(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))
        return zones

    @staticmethod
    def footprint(center: np.ndarray, facing: np.ndarray, size: np.ndarray) -> Rect:
        w, d = size[0], size[2]
        half = np.array([d / 2, w / 2]) if abs(facing[0]) > 0.5 else np.array([w / 2, d / 2])
        return Rect.around(center, half)

    @staticmethod
    def keepout(center: np.ndarray, facing: np.ndarray, size: np.ndarray, clearance: dict[str, float]) -> list[Rect]:
        """Clearance strips beside the footprint (front = facing)."""
        w, d = size[0], size[2]
        right = np.array([facing[1], -facing[0]])  # local +X for an axis-aligned facing
        rects = []
        for side, depth in clearance.items():
            if depth <= 0:
                continue
            if side in ("front", "back"):
                n = facing if side == "front" else -facing
                c = center + n * (d / 2 + depth / 2)
                ext_n, ext_t = depth / 2, w / 2
            else:
                n = right if side == "right" else -right
                c = center + n * (w / 2 + depth / 2)
                ext_n, ext_t = depth / 2, d / 2
            half = np.abs(n) * ext_n + np.abs(np.array([n[1], n[0]])) * ext_t
            rects.append(Rect.around(c, half))
        return rects

    # ------------------------------------------------------------ checking

    def fits(self, rect: Rect, keepout: list[Rect], height: float, ignore: set[str]) -> bool:
        if not rect.inside(self.bounds):
            return False
        if any(rect.overlaps(z) for z in self.door_zones):
            return False
        if any(rect.overlaps(z) for z in self.window_zones(height)):
            return False
        for name, items in self.placed.items():
            if name in ignore:
                continue
            for p in items:
                if p.rect is None:
                    continue
                if rect.overlaps(p.rect, GAP) or any(rect.overlaps(k) for k in p.keepout):
                    return False
                if any(k.overlaps(p.rect) for k in keepout):
                    return False
        return True

    # ------------------------------------------------------------- scoring

    def score(self, prefer: list, center: np.ndarray, wall: str | None, along: float, size: np.ndarray) -> float:
        total = float(self.rng.uniform(0, 1e-3))
        doors = [self.opening_center(d) for d in self.doors]
        windows = [self.opening_center(w) for w in self.windows]
        for term in prefer:
            if isinstance(term, dict):
                (key, target), = term.items()
                others = self.placed.get(target, [])
                if not others:
                    continue
                other = others[0]
                if key == "opposite" and wall is not None:
                    if np.allclose(_WALLS[wall][2], -other.facing):
                        total += 10.0
                    offset = (center - other.center) @ np.array([other.facing[1], -other.facing[0]])
                    total -= abs(offset)
                elif key == "near":
                    total -= float(np.linalg.norm(center - other.center))
                elif key == "far":
                    total += float(np.linalg.norm(center - other.center))
                continue
            has_door = wall is not None and any(d["side"] == wall for d in self.doors)
            has_window = wall is not None and any(w["side"] == wall for w in self.windows)
            if term == "longest_wall" and wall is not None:
                total += self.wall_length(wall)
            elif term == "no_door_wall" and has_door:
                total -= 5
            elif term == "door_wall" and has_door:
                total += 5
            elif term == "window_wall" and has_window:
                total += 5
            elif term == "no_window_wall" and has_window:
                total -= 5
            elif term == "near_door" and doors:
                total -= min(float(np.linalg.norm(center - d)) for d in doors)
            elif term == "far_door" and doors:
                total += min(float(np.linalg.norm(center - d)) for d in doors)
            elif term == "near_window" and windows:
                total -= min(float(np.linalg.norm(center - w)) for w in windows)
            elif term == "centered":
                total -= abs(along) if wall is not None else float(np.linalg.norm(center))
            elif term == "corner" and wall is not None:
                total -= (self.wall_length(wall) / 2 - size[0] / 2) - abs(along)
        return total

    # ------------------------------------------------------------- placing

    def load(self, spec: dict[str, Any], extra_params: dict[str, float] | None = None) -> SceneNode:
        params = dict(spec.get("params", {}))
        params.update(extra_params or {})
        return self.loader.load(self.assets_dir / spec["asset"], params=params or None)

    def add(self, name: str, index: int, node: SceneNode, center: np.ndarray, facing: np.ndarray, y: float,
            rect: Rect | None, keepout: list[Rect], wall: str | None = None, along: float = 0.0) -> None:
        yaw = float(np.arctan2(facing[0], facing[1]))
        node.name = name if index == 0 else f"{name}_{index + 1}"
        node.transform = Transform(translation=np.array([center[0], y, center[1]]), rotation=np.array([0.0, yaw, 0.0]))
        node.meta["placed_by"] = "furnish"
        self.room.add_child(node)
        size = np.asarray(node.size if node.size is not None else [0, 0, 0], dtype=float)
        self.placed.setdefault(name, []).append(Placed(name, node, center, facing, size, rect, keepout, wall, along))

    def run(self) -> list[Unsatisfied]:
        for name, spec in (self.spec.get("items") or {}).items():
            try:
                placed = self.place_item(name, spec)
            except (FileNotFoundError, ValueError) as exc:
                self.report.append(Unsatisfied(self.name, name, f"error: {exc}"))
                continue
            want = int(spec.get("count", 1)) if not spec.get("mount") == "window" else len(self.windows)
            if placed < want and not spec.get("optional"):
                self.report.append(Unsatisfied(self.name, name, f"placed {placed} of {want}"))
        return self.report

    def place_item(self, name: str, spec: dict[str, Any]) -> int:
        if spec.get("mount") == "window":
            return self.place_curtains(name, spec)
        if spec.get("mount") == "wall":
            return self.place_on_wall(name, spec)
        if "on" in spec:
            return self.place_on(name, spec)
        if "under" in spec:
            return self.place_under(name, spec)
        if "front_of" in spec:
            return self.place_front_of(name, spec)
        if "flank" in spec:
            return self.place_flank(name, spec)
        return self.place_free(name, spec)

    def _clearance(self, node: SceneNode, spec: dict[str, Any]) -> dict[str, float]:
        return dict(spec.get("clearance", node.meta.get("clearance", {})))

    def place_free(self, name: str, spec: dict[str, Any]) -> int:
        count = int(spec.get("count", 1))
        prefer = list(spec.get("prefer", []))
        against = spec.get("against", "wall")
        placed = 0
        for i in range(count):
            node = self.load(spec)
            size = np.asarray(node.size, dtype=float)
            clearance = self._clearance(node, spec)
            options = []
            for center, facing, wall, along in self._candidates(size, against, spec.get("facing")):
                rect = self.footprint(center, facing, size)
                keep = self.keepout(center, facing, size, clearance)
                if not self.fits(rect, keep, float(size[1]), set()):
                    continue
                options.append((self.score(prefer, center, wall, along, size), center, facing, rect, keep, wall, along))
            options.sort(key=lambda o: -o[0])
            # Best first, but never cut off access to something already placed.
            before = set(self.unreachable())
            for _, center, facing, rect, keep, wall, along in options[:MAX_REACH_TRIES]:
                self.add(name, i, node, center, facing, 0.0, rect, keep, wall, along)
                if set(self.unreachable()) <= before:
                    placed += 1
                    break
                self.remove_last(name)
            else:
                break
        return placed

    def unreachable(self) -> list[str]:
        from .qa import room_reachability

        return room_reachability(self.room)

    def remove_last(self, name: str) -> None:
        placed = self.placed[name].pop()
        if not self.placed[name]:
            del self.placed[name]
        self.room.remove_child(placed.node)

    def _candidates(self, size: np.ndarray, against: str, facing_spec: str | None):
        w, d = float(size[0]), float(size[2])
        if against == "wall":
            for side, (_, _, inward) in _WALLS.items():
                half = self.wall_length(side) / 2 - w / 2
                if half < 0:
                    continue
                steps = np.unique(np.r_[np.arange(-half, half + 1e-9, STEP), -half, half])
                for along in steps:
                    center = self.wall_point(side, float(along)) + inward * (d / 2 + 0.005)
                    yield center, inward.copy(), side, float(along)
        else:
            for x in np.arange(-self.hx, self.hx + 1e-9, 2 * STEP):
                for z in np.arange(-self.hz, self.hz + 1e-9, 2 * STEP):
                    center = np.array([x, z])
                    for facing in self._free_facings(center, facing_spec):
                        yield center, facing, None, 0.0

    def _free_facings(self, center: np.ndarray, spec: str | None) -> list[np.ndarray]:
        axes = [np.array(v) for v in ([0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0])]
        if spec in (None, "any"):
            return axes
        if spec == "center":
            target = np.zeros(2)
        elif spec in self.placed:
            target = self.placed[spec][0].center
        else:
            return axes
        to = target - center
        if np.linalg.norm(to) < 1e-6:
            return axes
        # Nearest axis-aligned facing toward the target.
        return [max(axes, key=lambda a: float(a @ to))]

    def place_flank(self, name: str, spec: dict[str, Any]) -> int:
        parent = self.placed.get(spec["flank"], [None])[0]
        if parent is None or parent.wall is None:
            return 0
        count = int(spec.get("count", 2))
        placed = 0
        for i, sign in enumerate((-1.0, 1.0)[:count]):
            node = self.load(spec)
            size = np.asarray(node.size, dtype=float)
            along = parent.along + sign * (parent.size[0] / 2 + size[0] / 2 + GAP)
            inward = _WALLS[parent.wall][2]
            center = self.wall_point(parent.wall, along) + inward * (size[2] / 2 + 0.005)
            rect = self.footprint(center, inward, size)
            keep = self.keepout(center, inward, size, self._clearance(node, spec))
            # Flanking items stand in the parent's side clearance by design.
            if self.fits(rect, keep, float(size[1]), {parent.name}) and not rect.overlaps(parent.rect, GAP / 2):
                self.add(name, placed, node, center, inward.copy(), 0.0, rect, keep, parent.wall, along)
                placed += 1
        return placed

    def place_front_of(self, name: str, spec: dict[str, Any]) -> int:
        parent = self.placed.get(spec["front_of"], [None])[0]
        if parent is None:
            return 0
        node = self.load(spec)
        size = np.asarray(node.size, dtype=float)
        distance = float(spec.get("distance", 0.1))
        center = parent.center + parent.facing * (parent.size[2] / 2 + distance + size[2] / 2)
        facing = -parent.facing
        rect = self.footprint(center, facing, size)
        ignore = {spec["front_of"]} if distance < 0 else set()
        if not self.fits(rect, [], float(size[1]), ignore):
            return 0
        self.add(name, 0, node, center, facing, 0.0, rect, [])
        return 1

    def place_under(self, name: str, spec: dict[str, Any]) -> int:
        parent = self.placed.get(spec["under"], [None])[0]
        if parent is None:
            return 0
        node = self.load(spec)
        center = parent.center + parent.facing * float(spec.get("offset", 0.0))
        self.add(name, 0, node, center, parent.facing.copy(), 0.0, None, [])
        return 1

    def place_on(self, name: str, spec: dict[str, Any]) -> int:
        placed = 0
        surface_name = spec.get("surface", "top")
        for parent in self.placed.get(spec["on"], []):
            surface = parent.node.surfaces.get(surface_name)
            if surface is None:
                continue
            node = self.load(spec)
            local = surface.resolve(0.5, 0.5).translation
            world = parent.node.transform.to_matrix() @ np.r_[local, 1.0]
            self.add(name, placed, node, np.array([world[0], world[2]]), parent.facing.copy(), float(world[1]),
                     None, [])
            placed += 1
        return placed

    def place_on_wall(self, name: str, spec: dict[str, Any]) -> int:
        node = self.load(spec)
        size = np.asarray(node.size, dtype=float)
        height = float(spec.get("height", 1.2))
        candidates = []
        if "above" in spec:
            parent = self.placed.get(spec["above"], [None])[0]
            if parent is None or parent.wall is None:
                return 0
            candidates = [(parent.wall, parent.along)]
        else:
            for side in _WALLS:
                half = self.wall_length(side) / 2 - size[0] / 2
                for along in np.arange(-half, half + 1e-9, STEP):
                    candidates.append((side, float(along)))
        best = None
        for side, along in candidates:
            lo, hi = along - size[0] / 2, along + size[0] / 2
            blocked = any(o["side"] == side and o["lo"] < hi and lo < o["hi"]
                          and o["sill"] < height + size[1] and height < o["top"]
                          for o in self.doors + self.windows)
            # Not behind floor items on that wall that reach up to the mount.
            blocked = blocked or any(
                p.wall == side and p.rect is not None and p.size[1] > height
                and abs(p.along - along) < (p.size[0] + size[0]) / 2
                for items in self.placed.values() for p in items)
            if blocked or abs(along) > self.wall_length(side) / 2 - size[0] / 2 + 1e-6:
                continue
            center = self.wall_point(side, along)
            s = self.score(list(spec.get("prefer", [])), center, side, along, size)
            if best is None or s > best[0]:
                best = (s, side, along, center)
        if best is None:
            return 0
        _, side, along, center = best
        center = center - _WALLS[side][2] * (self.floor_inset - self.mount_inset)
        self.add(name, 0, node, center, _WALLS[side][2].copy(), height, None, [], side, along)
        return 1

    def place_curtains(self, name: str, spec: dict[str, Any]) -> int:
        placed = 0
        for window in self.windows:
            width = window["hi"] - window["lo"]
            params = {"width": width + float(spec.get("overhang", 0.5))} if spec.get("fit") == "window" else None
            node = self.load(spec, params)
            along = (window["lo"] + window["hi"]) / 2
            side = window["side"]
            if abs(along) + node.size[0] / 2 > self.wall_length(side) / 2 + 1e-6:
                node = self.load(spec, {"width": width + 0.1} if params else None)
            wall_face = self.wall_point(side, along) - _WALLS[side][2] * (self.floor_inset - self.mount_inset)
            self.add(name, placed, node, wall_face, _WALLS[side][2].copy(), 0.0, None, [], side, along)
            placed += 1
        return placed
