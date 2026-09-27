"""Traffic routes as data: a lane graph for vehicles to drive (see epic geogen-r65).

Everything is precomputed here so a runtime only follows curves:

- **City lanes** (``city: {traffic: {...}}``): every street segment between
  two intersections gets directed lanes (``lanes`` per way, ``avenue_lanes``
  on avenues), laid across the carriageway left after kerbside parking.
  Lanes run between the crosswalks; the end of a lane is its stop line.
- **Connectors** join each lane's stop line to the lanes leaving the
  intersection (straight, left, right; no U-turns): cubic Béziers from lane
  end to lane start, tagged with their ``turn`` and ``intersection``.
- **Routes** (scene ``routes: {name: {path, lanes, drive, speed, loop}}``):
  explicit roads anywhere (a lane past the cottage, a bus loop). Open
  routes start and end in the void: vehicles spawn and despawn there.
- **Resampling**: every lane is resampled at ``STEP`` metres; a lane's
  ``speed`` is its limit capped by curvature (``sqrt(LATERAL_ACCEL / k)``).
- **Conflicts**: connector pairs in one intersection whose paths come
  closer than a vehicle width, with the s-range on each (claim before entering).
- **Crosswalks**: s-ranges where a lane crosses a crosswalk (yield to people).

``build_traffic(scene)`` returns the graph as a JSON-ready dict (the manifest's
``traffic`` section, schema ``docs/schema/geogen-traffic.v1.schema.json``);
``check_clearance(scene, graph)`` sweeps vehicle boxes along it;
``lane_overlay(graph)`` makes ribbons for renders (``-r out.png --lanes``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core.node import SceneNode

VERSION = 1
STEP = 0.5                  # resampling step (m)
LATERAL_ACCEL = 2.5         # m/s² for the curvature speed cap
LANE_WIDTH = 3.0            # default lane width (m)
LANE_ENTRY = 4.0            # lanes start this far past the crosswalk (m): turning buses straighten first
STOP_SETBACK = 3.0          # stop lines sit this far before the crosswalk (m)
PARKING_DEPTH = 2.3         # kerb to the outer edge of a parked car (m)
CONFLICT_GAP = 3.6          # connector paths closer than this conflict (m): a bus plus its off-tracking in turns
TRAFFIC_KEYS = {"drive", "lanes", "avenue_lanes", "speed", "avenue_speed", "turns", "control", "signals", "cycle"}
SIGNAL_CYCLE = {"green": 16.0, "amber": 3.0, "all_red": 1.5}
ROUTE_KEYS = {"path", "lanes", "drive", "speed", "loop", "lane_width", "y", "surface"}
# Vehicle classes swept along every lane for clearance: (width, height) in metres.
CLEARANCE_CLASSES = {"car": (1.9, 1.6), "van": (2.1, 2.4), "bus": (2.6, 3.2)}


@dataclass
class Lane:
    id: str
    kind: str                         # lane | connector | route
    points: np.ndarray                # (n, 3), resampled
    speed: float
    road: str = ""
    turn: str = ""                    # connectors: straight | left | right
    intersection: str = ""
    successors: list[str] = field(default_factory=list)
    crosswalks: list[list[float]] = field(default_factory=list)
    end: str = ""                     # "exit" when vehicles leave the world at the lane's end
    start: str = ""                   # "entry" when vehicles arrive from outside the world here
    signal: dict | None = None        # {intersection, phase} for lanes ending at traffic lights
    kerb: float = 0.0                 # distance from the lane centre to the kerb on its right

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind, "speed": round(self.speed, 3),
                               "length": round(self.length, 3),
                               "points": [[round(float(v), 3) for v in p] for p in self.points],
                               "successors": list(self.successors)}
        for key in ("road", "turn", "intersection", "end", "start"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.crosswalks:
            out["crosswalks"] = [[round(a, 3), round(b, 3)] for a, b in self.crosswalks]
        if self.signal:
            out["signal"] = dict(self.signal)
        if self.kerb:
            out["kerb"] = round(self.kerb, 3)
        return out


# --- curves -------------------------------------------------------------------------------------


def resample(points: np.ndarray, step: float = STEP) -> np.ndarray:
    """A polyline resampled at ``step`` (the last point kept)."""
    points = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    points, seg = points[keep], seg[seg > 1e-9]
    if len(points) < 2:
        return points
    s = np.r_[0.0, np.cumsum(seg)]
    n = max(int(np.ceil(s[-1] / step)), 1)
    t = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(t, s, points[:, k]) for k in range(3)])


def bezier(p0, p1, p2, p3, samples: int = 24) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def max_curvature(points: np.ndarray) -> float:
    """Largest curvature (1/m) along a resampled plan polyline."""
    if len(points) < 3:
        return 0.0
    p = points[:, [0, 2]]
    a, b, c = p[:-2], p[1:-1], p[2:]
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1))
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    k = 2 * cross / np.maximum(ab * bc * ca, 1e-12)
    # Average over a few samples: one kink between straight pieces isn't a tight curve.
    if len(k) >= 5:
        k = np.convolve(k, np.ones(5) / 5, mode="same")
    return float(k.max())


def capped_speed(points: np.ndarray, limit: float) -> float:
    k = max_curvature(points)
    return min(limit, float(np.sqrt(LATERAL_ACCEL / k))) if k > 1e-6 else limit


def _cumulative(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


# --- city ---------------------------------------------------------------------------------------


def _city_lanes(builder, spec: dict[str, Any]) -> list[Lane]:
    """Lanes and connectors for a CityBuilder's street grid."""
    unknown = set(spec) - TRAFFIC_KEYS
    if unknown:
        raise ValueError(f"city traffic: unknown keys {sorted(unknown)}. Known: {sorted(TRAFFIC_KEYS)}")
    hand = 1.0 if spec.get("drive", "right") == "right" else -1.0
    per_way = int(spec.get("lanes", 1))
    avenue_per_way = int(spec.get("avenue_lanes", per_way))
    speed = float(spec.get("speed", 11.0))
    avenue_speed = float(spec.get("avenue_speed", 14.0))
    turns = spec.get("turns", True)
    parking = builder.spec.get("parking") or {}
    both = float(parking.get("both_sides", 11.0))
    sw = builder.sidewalk

    ns_spans = builder._street_spans(builder.x_edges, builder.ns_width, builder.lo[0], builder.hi[0])
    ew_spans = builder._street_spans(builder.z_edges, builder.ew_width, builder.lo[1], builder.hi[1])

    lanes: list[Lane] = []
    # arms[(i, j)][direction] = (incoming lanes, outgoing lanes), direction = side of the node.
    arms: dict[tuple[int, int], dict[str, tuple[list[Lane], list[Lane]]]] = {}

    def arm(node, side):
        return arms.setdefault(node, {}).setdefault(side, ([], []))

    def street(axis: str, k: int, a: float, b: float, lo: float, hi: float, n0, n1, avenue: bool,
               parked: bool = True):
        """Lanes on one segment. axis 'z': a north-south street (x in [a, b]) whose crosswalks
        end at z=lo and start at z=hi."""
        width = b - a
        # Parking lanes: on streets narrower than `both`, only the north/east kerb (x = b / z = b).
        park_lo = PARKING_DEPTH if parking and parked and width >= both else 0.0
        park_hi = PARKING_DEPTH if parking and parked else 0.0
        c0, c1 = a + park_lo, b - park_hi
        n_way = avenue_per_way if avenue else per_way
        lane_w = min(LANE_WIDTH, (c1 - c0) / (2 * n_way))
        mid = (c0 + c1) / 2
        limit = avenue_speed if avenue else speed
        road = f"{'ns' if axis == 'z' else 'ew'}_{k}_{min(n0[1 if axis == 'z' else 0], n1[1 if axis == 'z' else 0])}"
        for direction in (1.0, -1.0):          # +z/+x or -z/-x
            for m in range(n_way):
                # Left of travel is +x heading +z and -z heading +x, so right-hand traffic keeps
                # to -x on north-south streets and +z on east-west ones. The kerb lane (m = 0)
                # is the outermost.
                sign = -direction * hand if axis == "z" else direction * hand
                lateral = mid + sign * (n_way - m - 0.5) * lane_w
                # From just past the crosswalk to the stop line, set back so a bus turning
                # across the end of the lane clears the car waiting there.
                s0, s1 = (lo + LANE_ENTRY, hi - STOP_SETBACK) if direction > 0 else (hi - LANE_ENTRY, lo + STOP_SETBACK)
                if axis == "z":
                    pts = np.array([[lateral, 0.0, s0], [lateral, 0.0, s1]])
                else:
                    pts = np.array([[s0, 0.0, lateral], [s1, 0.0, lateral]])
                src, dst = (n0, n1) if direction > 0 else (n1, n0)
                lane = Lane(f"{road}_{'p' if direction > 0 else 'n'}{m}", "lane", resample(pts), limit, road=road)
                # The kerb on the lane's right: the street edge on the side it keeps to.
                lane.kerb = float(lateral - a if sign < 0 else b - lateral)
                lane.kerb_index = m  # type: ignore[attr-defined]  # 0 = kerb lane
                lanes.append(lane)
                # Leaving src (outgoing on src's arm), arriving at dst (incoming on dst's arm).
                src_side = ("north" if direction > 0 else "south") if axis == "z" else ("east" if direction > 0 else "west")
                dst_side = {"north": "south", "south": "north", "east": "west", "west": "east"}[src_side]
                arm(src, src_side)[1].append(lane)
                arm(dst, dst_side)[0].append(lane)

    for i, (a, b) in enumerate(ns_spans):
        for j in range(builder.nz):
            z0, z1 = builder.z_edges[j]
            street("z", i, a, b, z0 + sw, z1 - sw, (i, j), (i, j + 1), i in builder.ns_avenues)
    for j, (a, b) in enumerate(ew_spans):
        for i in range(builder.nx):
            x0, x1 = builder.x_edges[i]
            street("x", j, a, b, x0 + sw, x1 - sw, (i, j), (i + 1, j), j in builder.ew_avenues)

    # Roads out of town: each avenue end runs on to a node outside the grid, where the outbound
    # lanes end (vehicles leave the world and come back on an inbound lane).
    outside = []
    L = getattr(builder, "exits", 0.0)
    for side, idx, a, b in builder.exit_spans() if L > 0 else []:
        if side == "west":
            out = (-1, idx)
            street("x", idx, a, b, builder.lo[0] - L, builder.lo[0], out, (0, idx), True, parked=False)
        elif side == "east":
            out = (builder.nx + 1, idx)
            street("x", idx, a, b, builder.hi[0], builder.hi[0] + L, (builder.nx, idx), out, True, parked=False)
        elif side == "south":
            out = (idx, -1)
            street("z", idx, a, b, builder.lo[1] - L, builder.lo[1], out, (idx, 0), True, parked=False)
        else:
            out = (idx, builder.nz + 1)
            street("z", idx, a, b, builder.hi[1], builder.hi[1] + L, (idx, builder.nz), out, True, parked=False)
        outside.append(out)
    for node in outside:
        for incoming, outgoing in arms.get(node, {}).values():
            for ln in incoming:
                ln.end = "exit"
            for ln in outgoing:
                ln.start = "entry"

    heading = {"north": np.array([0.0, 1.0]), "south": np.array([0.0, -1.0]),
               "east": np.array([1.0, 0.0]), "west": np.array([-1.0, 0.0])}
    # Traffic lights: at every intersection ("all") or where an avenue meets ("avenues").
    signals = spec.get("signals", "none")
    if signals not in ("none", "avenues", "all"):
        raise ValueError(f"city traffic: signals must be none | avenues | all, got {signals!r}")
    for node, sides in arms.items():
        i, j = node
        avenue = i in builder.ns_avenues or j in builder.ew_avenues
        if len(sides) < 3 or signals == "none" or (signals == "avenues" and not avenue):
            continue
        for side, (incoming, _) in sides.items():
            for ln in incoming:     # phase 0: arriving from north/south; phase 1: east/west
                ln.signal = {"intersection": f"x_{i}_{j}", "phase": 0 if side in ("north", "south") else 1}
    for node, sides in arms.items():
        iid = f"x_{node[0]}_{node[1]}"
        for in_side, (incoming, _) in sides.items():
            travel = -heading[in_side]            # arriving from the in_side, moving inward
            for out_side, (_, outgoing) in sides.items():
                if out_side == in_side or not outgoing or not incoming:
                    continue
                out_dir = heading[out_side]
                cross = travel[0] * out_dir[1] - travel[1] * out_dir[0]
                turn = "straight" if abs(cross) < 0.5 else ("left" if cross < 0 else "right")
                if turn != "straight" and not turns:
                    continue
                ins = sorted(incoming, key=lambda ln: ln.kerb_index)   # type: ignore[attr-defined]
                outs = sorted(outgoing, key=lambda ln: ln.kerb_index)  # type: ignore[attr-defined]
                kerb_turn = "right" if hand > 0 else "left"     # turns across no traffic
                if turn == kerb_turn:
                    pairs = [(ins[0], outs[0])]
                elif turn != "straight":
                    pairs = [(ins[-1], outs[-1])]
                else:
                    pairs = [(ln, outs[min(k, len(outs) - 1)]) for k, ln in enumerate(ins)]
                for src, dst in pairs:
                    p0, p3 = src.points[-1], dst.points[0]
                    d0 = src.points[-1] - src.points[-2]
                    d1 = dst.points[1] - dst.points[0]
                    d0, d1 = d0 / np.linalg.norm(d0), d1 / np.linalg.norm(d1)
                    reach = np.linalg.norm(p3 - p0) * (0.55 if turn != "straight" else 0.33)
                    pts = resample(bezier(p0, p0 + d0 * reach, p3 - d1 * reach, p3, 32))
                    limit = min(src.speed, dst.speed)
                    con = Lane(f"{src.id}__{dst.id}", "connector", pts, capped_speed(pts, limit),
                               turn=turn, intersection=iid, successors=[dst.id])
                    lanes.append(con)
                    src.successors.append(con.id)
    # Crosswalk zones: the first/last `sidewalk` metres beyond each intersection box are crossings.
    boxes = [(np.array([ns_spans[i][0], ew_spans[j][0]]), np.array([ns_spans[i][1], ew_spans[j][1]]))
             for i in range(len(ns_spans)) for j in range(len(ew_spans))]
    zones = []
    for lo, hi in boxes:
        for d in (np.array([sw, 0.0]), np.array([0.0, sw])):
            # Crosswalks run across the streets leaving the box on each side.
            if d[0]:
                zones += [(np.array([lo[0] - sw, lo[1]]), np.array([lo[0], hi[1]])),
                          (np.array([hi[0], lo[1]]), np.array([hi[0] + sw, hi[1]]))]
            else:
                zones += [(np.array([lo[0], lo[1] - sw]), np.array([hi[0], lo[1]])),
                          (np.array([lo[0], hi[1]]), np.array([hi[0], hi[1] + sw]))]
    for lane in lanes:
        if lane.kind == "connector":
            lane.crosswalks = _zone_ranges(lane.points, zones)
    for lane in lanes:
        if hasattr(lane, "kerb_index"):
            del lane.kerb_index  # type: ignore[attr-defined]
    # The painted crossings (inside the district) for pedestrians to use.
    crossings = [[round(float(v), 3) for v in (*lo, *hi)] for lo, hi in zones
                 if np.all(lo >= builder.lo - 1e-6) and np.all(hi <= builder.hi + 1e-6)]
    builder.crosswalk_zones = crossings
    return lanes


def _zone_ranges(points: np.ndarray, zones) -> list[list[float]]:
    """s-ranges of ``points`` inside any of the (lo, hi) plan rectangles."""
    s = _cumulative(points)
    p = points[:, [0, 2]]
    inside = np.zeros(len(p), dtype=bool)
    for lo, hi in zones:
        inside |= np.all((p >= lo) & (p <= hi), axis=1)
    ranges, start = [], None
    for k, flag in enumerate(inside):
        if flag and start is None:
            start = s[k]
        if not flag and start is not None:
            ranges.append([float(start), float(s[k - 1])])
            start = None
    if start is not None:
        ranges.append([float(start), float(s[-1])])
    return ranges


# --- explicit routes ----------------------------------------------------------------------------


def route_lanes(name: str, spec: dict[str, Any]) -> list[Lane]:
    """Lanes along a declared route: ``{path: [[x, y, z], ...] | {spline: [...], samples}, lanes: 2,
    drive: right, speed: 13, loop: false, lane_width: 3}`` (the path is the road's centre line;
    with ``lanes: 1`` a one-way lane on the path itself)."""
    from .core.profile import catmull_rom

    unknown = set(spec) - ROUTE_KEYS
    if unknown:
        raise ValueError(f"route '{name}': unknown keys {sorted(unknown)}. Known: {sorted(ROUTE_KEYS)}")
    loop = bool(spec.get("loop", False))
    path = spec["path"]
    if isinstance(path, dict):
        pts = catmull_rom(np.asarray(path["spline"], dtype=np.float64), int(path.get("samples", 12)), closed=loop)
    else:
        pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError(f"route '{name}': path must be [[x, y, z], ...]")
    if loop:
        pts = np.vstack([pts, pts[:1]])
    centre = resample(pts, STEP / 2)
    hand = 1.0 if spec.get("drive", "right") == "right" else -1.0
    n_lanes = int(spec.get("lanes", 2))
    width = float(spec.get("lane_width", LANE_WIDTH))
    speed = float(spec.get("speed", 11.0))
    # Plan normals (left of travel = +X when heading +Z, i.e. rotate the tangent by +90°).
    t = np.gradient(centre[:, [0, 2]], axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    left = np.column_stack([t[:, 1], np.zeros(len(t)), -t[:, 0]])
    lanes = []
    directions = [1] if n_lanes == 1 else [1, -1]
    per_way = 1 if n_lanes == 1 else n_lanes // 2
    for direction in directions:
        base = centre if direction > 0 else centre[::-1]
        base_left = left if direction > 0 else -left[::-1]
        for m in range(per_way):
            offset = 0.0 if n_lanes == 1 else -hand * (per_way - m - 0.5) * width
            pts_m = resample(base + base_left * offset)
            lane = Lane(f"{name}_{'p' if direction > 0 else 'n'}{m}", "route", pts_m,
                        capped_speed(pts_m, speed), road=name)
            if loop:
                lane.successors = [lane.id]
            else:
                lane.end = "exit"
            lanes.append(lane)
    return lanes


# --- the graph ----------------------------------------------------------------------------------


def _conflicts(lanes: list[Lane]) -> list[dict[str, Any]]:
    """Connector pairs in one intersection whose paths come within CONFLICT_GAP (or merge)."""
    from scipy.spatial import cKDTree

    by_x: dict[str, list[Lane]] = {}
    for lane in lanes:
        if lane.kind == "connector":
            by_x.setdefault(lane.intersection, []).append(lane)
    out = []
    for group in by_x.values():
        for ai in range(len(group)):
            for bi in range(ai + 1, len(group)):
                a, b = group[ai], group[bi]
                if a.id.split("__")[0] == b.id.split("__")[0]:
                    continue          # same source lane: ordinary car following handles it
                pa, pb = a.points[:, [0, 2]], b.points[:, [0, 2]]
                d, idx = cKDTree(pb).query(pa)
                close = d < CONFLICT_GAP
                if not close.any():
                    continue
                sa, sb = _cumulative(a.points), _cumulative(b.points)
                ka = np.where(close)[0]
                kb = idx[close]
                out.append({"a": a.id, "b": b.id,
                            "a_range": [round(float(sa[ka.min()]), 3), round(float(sa[ka.max()]), 3)],
                            "b_range": [round(float(sb[kb.min()]), 3), round(float(sb[kb.max()]), 3)]})
    return out


def graph(lanes: list[Lane], drive: str = "right", cycle: dict | None = None,
          crosswalks: list | None = None) -> dict[str, Any]:
    """A JSON-ready lane graph from lanes (points in one frame)."""
    ids = {lane.id for lane in lanes}
    for lane in lanes:
        lane.successors = [s for s in lane.successors if s in ids]
    intersections: dict[str, dict[str, Any]] = {}
    by_id = {lane.id: lane for lane in lanes}
    signalled = {ln.signal["intersection"] for ln in lanes if ln.signal}
    for lane in lanes:
        if lane.intersection:
            x = intersections.setdefault(lane.intersection, {"id": lane.intersection, "control": "all_way",
                                                               "connectors": []})
            x["connectors"].append(lane.id)
            if lane.intersection in signalled:
                x["control"] = "signals"
                x["cycle"] = dict(cycle or SIGNAL_CYCLE)
                # Stagger neighbouring lights so the town doesn't flash in unison.
                x["offset"] = round(float(sum(ord(c) for c in lane.intersection) % 7) * 3.0, 3)
    for x in intersections.values():
        pts = np.vstack([by_id[c].points for c in x["connectors"]])
        x["center"] = [round(float(v), 3) for v in pts.mean(axis=0)]
    out = {"version": VERSION, "drive": drive, "step": STEP,
           "lanes": [lane.to_dict() for lane in lanes],
           "conflicts": _conflicts(lanes),
           "intersections": list(intersections.values())}
    if crosswalks:
        out["crosswalks"] = crosswalks
    return out


def city_graph(builder) -> dict[str, Any] | None:
    spec = builder.spec.get("traffic")
    if not spec:
        return None
    spec = {} if spec is True else dict(spec)
    cycle = {**SIGNAL_CYCLE, **(spec.get("cycle") or {})}
    lanes = _city_lanes(builder, spec)
    return graph(lanes, spec.get("drive", "right"), cycle, getattr(builder, "crosswalk_zones", None))


def route_surface(name: str, spec: dict[str, Any], materials) -> SceneNode | None:
    """Asphalt along a route (lanes plus 0.75 m shoulders, top 1 cm above its path) with a centre
    line; ``surface: false`` for routes over existing roads."""
    if spec.get("surface", True) is False:
        return None
    from .core.profile import catmull_rom
    from .railway import _sweep

    loop = bool(spec.get("loop", False))
    path = spec["path"]
    if isinstance(path, dict):
        pts = catmull_rom(np.asarray(path["spline"], dtype=np.float64), int(path.get("samples", 12)), closed=loop)
    else:
        pts = np.asarray(path, dtype=np.float64)
    if loop:
        pts = np.vstack([pts, pts[:1]])
    pts = resample(pts, 1.0)
    half = int(spec.get("lanes", 2)) * float(spec.get("lane_width", LANE_WIDTH)) / 2 + 0.75
    node = SceneNode(f"route_{name}", tags=["street.road"])
    road = SceneNode(f"route_{name}_surface",
                     mesh=_sweep([[-half, -0.15], [half, -0.15], [half, 0.01], [-half, 0.01]], pts, loop,
                                 materials.load("asphalt")), tags=["street.road"])
    road.meta["walkable"] = True
    node.add_child(road)
    if int(spec.get("lanes", 2)) > 1:
        line = SceneNode(f"route_{name}_line", mesh=_sweep([[-0.06, 0.01], [0.06, 0.01], [0.06, 0.014], [-0.06, 0.014]],
                                                           pts, loop, materials.load("road_paint_yellow")))
        line.meta["collider"] = "none"
        node.add_child(line)
    return node


def routes_graph(routes: dict[str, Any]) -> dict[str, Any] | None:
    lanes, drive = [], "right"
    for name, spec in (routes or {}).items():
        lanes += route_lanes(name, spec)
        drive = spec.get("drive", drive)
    return graph(lanes, drive) if lanes else None


def build_traffic(scene: SceneNode) -> dict[str, Any] | None:
    """The scene's whole lane graph: every ``meta.traffic`` graph under ``scene`` (a city's
    streets, a scene's routes), moved into the scene frame and merged; None if there are none.
    Ids of graphs from nested nodes are prefixed with the node name when they'd clash."""
    to_scene = np.linalg.inv(scene.world_transform())
    lanes: list = []
    conflicts, intersections, seen, crosswalks = [], [], set(), []
    drive = "right"
    for node in scene.iter_nodes():
        g = node.meta.get("traffic")
        if not isinstance(g, dict):
            continue
        drive = g.get("drive", drive)
        m = to_scene @ node.world_transform()
        prefix = "" if not ({ln["id"] for ln in g["lanes"]} & seen) else f"{node.name}/"

        def rid(i: str) -> str:
            return prefix + i

        for ln in g["lanes"]:
            p = np.asarray(ln["points"], dtype=np.float64)
            p = (m @ np.c_[p, np.ones(len(p))].T).T[:, :3]
            lanes.append({**ln, "id": rid(ln["id"]), "successors": [rid(s) for s in ln["successors"]],
                          "points": [[round(float(v), 3) for v in q] for q in p]})
            seen.add(rid(ln["id"]))
        conflicts += [{**c, "a": rid(c["a"]), "b": rid(c["b"])} for c in g.get("conflicts", [])]
        for x0, z0, x1, z1 in g.get("crosswalks", []):
            corners = (m @ np.array([[x0, 0, z0, 1], [x1, 0, z1, 1]], dtype=np.float64).T).T
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            crosswalks.append([round(float(lo[0]), 3), round(float(lo[2]), 3), round(float(hi[0]), 3),
                               round(float(hi[2]), 3)])
        for x in g.get("intersections", []):
            c = (m @ np.r_[np.asarray(x["center"], dtype=np.float64), 1.0])[:3]
            intersections.append({**x, "id": rid(x["id"]), "connectors": [rid(i) for i in x["connectors"]],
                                  "center": [round(float(v), 3) for v in c]})
    railways = []
    for node in scene.iter_nodes():
        rail = node.meta.get("railway")
        if not isinstance(rail, dict):
            continue
        m = to_scene @ node.world_transform()
        p = np.asarray(rail["points"], dtype=np.float64)
        p = (m @ np.c_[p, np.ones(len(p))].T).T[:, :3]
        railways.append({**rail, "points": [[round(float(v), 3) for v in q] for q in p]})
    if not lanes and not railways:
        return None
    out = {"version": VERSION, "drive": drive, "step": STEP, "lanes": lanes, "conflicts": conflicts,
           "intersections": intersections}
    if crosswalks:
        out["crosswalks"] = crosswalks
    if railways:
        out["railways"] = railways
    return out


def check_graph(graph: dict[str, Any]) -> list[str]:
    """Structural problems: dead ends (a lane with no successor that isn't an exit), unknown
    successors, and lanes nothing leads to (except route starts)."""
    lanes = {ln["id"]: ln for ln in graph["lanes"]}
    issues = []
    reached = {s for ln in lanes.values() for s in ln["successors"]}
    for lid, ln in lanes.items():
        for s in ln["successors"]:
            if s not in lanes:
                issues.append(f"{lid}: unknown successor {s}")
        if not ln["successors"] and ln.get("end") != "exit":
            issues.append(f"{lid}: dead end")
        if lid not in reached and ln["kind"] != "route" and ln.get("start") != "entry":
            issues.append(f"{lid}: nothing leads here")
    return issues


# --- clearance ----------------------------------------------------------------------------------


def check_clearance(scene: SceneNode, graph: dict[str, Any], classes=None, margin: float = 0.15) -> list[str]:
    """Sweep each vehicle class's box along every lane: geometry inside the corridor
    (half its width plus ``margin`` either side, from 0.2 m up to its height) is reported
    as '<lane>: <class> hits <node>'. Walkable surfaces (roads, kerbs) and vehicles don't count."""
    from shapely import STRtree
    from shapely.geometry import LineString, MultiPoint

    classes = classes or CLEARANCE_CLASSES
    to_scene = np.linalg.inv(scene.world_transform())
    skip: set[int] = set()
    for node in scene.iter_nodes():
        if node.meta.get("type") in ("vehicle", "npc") or node.meta.get("walkable"):
            skip.update(id(n) for n in node.iter_nodes())
    obstacles = []     # (name, bottom, top, footprint polygon)
    for node in scene.iter_nodes():
        if node.mesh is None or not len(node.mesh.faces) or id(node) in skip:
            continue
        if node.meta.get("collider") == "none" and not node.name.startswith(("foliage", "branches")):
            continue
        m = to_scene @ node.world_transform()
        v = (m @ np.c_[node.mesh.vertices, np.ones(len(node.mesh.vertices))].T).T[:, :3]
        obstacles.append((node, v))
    issues = []
    for cls, (width, height) in classes.items():
        polys, owners = [], []
        for node, v in obstacles:
            band = v[(v[:, 1] > 0.2) & (v[:, 1] < height)]
            if len(band) < 3:
                continue
            hull = MultiPoint(band[:, [0, 2]]).convex_hull
            if hull.area < 1e-4:
                hull = hull.buffer(0.02)
            polys.append(hull)
            owners.append(node)
        if not polys:
            continue
        tree = STRtree(polys)
        for lane in graph["lanes"]:
            pts = np.asarray(lane["points"])[:, [0, 2]]
            corridor = LineString(pts).buffer(width / 2 + margin, cap_style="flat")
            for k in tree.query(corridor, predicate="intersects"):
                name = owners[int(k)].name
                parent = owners[int(k)].parent
                where = f"{parent.name}/{name}" if parent is not None else name
                issues.append(f"{lane['id']}: {cls} hits {where}")
    return sorted(set(issues))


# --- overlay ------------------------------------------------------------------------------------


def lane_overlay(graph: dict[str, Any], material_loader=None) -> SceneNode:
    """Ribbons along every lane (lanes blue, connectors by turn) for review renders."""
    from .core.mesh import Mesh

    colours = {"lane": (0.15, 0.45, 1.0), "route": (0.15, 0.45, 1.0), "straight": (0.2, 0.8, 0.3),
               "left": (1.0, 0.6, 0.1), "right": (0.9, 0.2, 0.8)}
    group = SceneNode("traffic_lanes", tags=["traffic.overlay"])
    meshes = []
    for lane in graph["lanes"]:
        p = np.asarray(lane["points"])
        if len(p) < 2:
            continue
        t = np.gradient(p[:, [0, 2]], axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        side = np.column_stack([t[:, 1], np.zeros(len(t)), -t[:, 0]]) * 0.18
        y = np.array([0.0, 0.03, 0.0])
        verts = np.vstack([p - side + y, p + side + y])
        n = len(p)
        faces = [[k, k + 1, k + n] for k in range(n - 1)] + [[k + 1, k + n + 1, k + n] for k in range(n - 1)]
        mesh = Mesh(vertices=verts, faces=np.array(faces, dtype=np.int64))
        colour = colours[lane.get("turn") or lane["kind"]]
        # Fade toward the lane end so direction reads.
        fade = np.r_[np.linspace(1.0, 0.35, n), np.linspace(1.0, 0.35, n)][:, None]
        mesh.colors = np.c_[np.asarray(colour) * fade, np.ones(2 * n)]
        meshes.append(mesh)
    if meshes:
        node = SceneNode("lanes", mesh=Mesh.merge(meshes))
        node.meta["collider"] = "none"
        if material_loader is not None:
            node.mesh.material = material_loader.load("lamp_shade")
        group.add_child(node)
    return group


# --- fleets (what moves) ------------------------------------------------------------------------

FLEET_KIND = "traffic"
SPAWN_CLEARANCE = 12.0      # starting vehicles keep this far from spawns and NPCs (m)
FLEET_KEYS = {"kind", "version", "fleet", "count", "spacing", "driving", "turns", "yield", "radius", "seed",
              "description", "schedule"}
FLEET_ENTRY_KEYS = {"asset", "weight", "params", "lanes"}
DRIVING_DEFAULTS = {"accel": 1.8, "decel": 3.0, "headway": 1.3, "gap": 2.5, "speed_factor": [0.85, 1.1],
                    "stop": "all_way", "stop_wait": 0.8, "look_ahead": 14.0, "give_up": 25.0}
TURN_DEFAULTS = {"straight": 0.6, "left": 0.2, "right": 0.2}


def load_fleet(path) -> dict[str, Any]:
    """Validate a ``kind: traffic`` definition (assets/traffic/*.yaml)."""
    from pathlib import Path

    from .layout.yaml_utils import safe_load_path

    data = safe_load_path(path) or {}
    if data.get("kind") != FLEET_KIND or int(data.get("version", 0)) != VERSION:
        raise ValueError(f"{path}: expected kind '{FLEET_KIND}' version {VERSION}")
    unknown = set(data) - FLEET_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; known: {sorted(FLEET_KEYS)}")
    fleet = []
    for i, entry in enumerate(data.get("fleet") or []):
        if set(entry) - FLEET_ENTRY_KEYS or "asset" not in entry:
            raise ValueError(f"{path}: fleet[{i}] needs asset (keys: {sorted(FLEET_ENTRY_KEYS)})")
        fleet.append({"asset": entry["asset"], "weight": float(entry.get("weight", 1.0)),
                      "params": entry.get("params") or {}, "lanes": entry.get("lanes")})
    if not fleet:
        raise ValueError(f"{path}: fleet is empty")
    driving = {**DRIVING_DEFAULTS, **(data.get("driving") or {})}
    if set(driving) - set(DRIVING_DEFAULTS):
        raise ValueError(f"{path}: driving keys are {sorted(DRIVING_DEFAULTS)}")
    turns = {**TURN_DEFAULTS, **(data.get("turns") or {})}
    yields = list(data.get("yield", ["player", "npc"]))
    if set(yields) - {"player", "npc"}:
        raise ValueError(f"{path}: yield to player and/or npc")
    from .npc import parse_time

    schedule = sorted([parse_time(t, f"{path}: schedule"), float(f)] for t, f in (data.get("schedule") or []))
    if any(not 0.0 <= f <= 1.0 for _, f in schedule):
        raise ValueError(f"{path}: schedule shares are 0..1")
    return {"definition": Path(path).stem, "fleet": fleet, "count": int(data.get("count", 12)),
            "schedule": schedule,
            "spacing": float(data.get("spacing", 14.0)), "driving": driving, "turns": turns,
            "yield": yields, "radius": float(data.get("radius", 0.0)), "seed": int(data.get("seed", 0))}


def _pose_on_lane(points: np.ndarray, s: float, half_base: float) -> tuple[np.ndarray, float]:
    """Position (midway between axles) and yaw of a vehicle at ``s`` on a lane: the axles sit
    on the curve ``half_base`` behind and ahead (clamped to the lane)."""
    cum = _cumulative(points)

    def at(t: float) -> np.ndarray:
        t = min(max(t, 0.0), cum[-1])
        return np.array([np.interp(t, cum, points[:, k]) for k in range(3)])

    rear, front = at(s - half_base), at(s + half_base)
    d = front - rear
    return (rear + front) / 2, float(np.arctan2(d[0], d[2]))


def place_traffic(root: SceneNode, name: str, spec: dict[str, Any], load, graph: dict[str, Any] | None,
                  assets_dir) -> SceneNode:
    """The traffic placement: a node carrying ``meta.fleet`` whose children are the starting
    vehicles, each on a lane (``meta.driving = {lane, s, factor}``). Runtimes drive these
    and clone them for later arrivals; renders show the traffic as a snapshot."""
    from pathlib import Path

    from .characters import draw_params

    fleet = load_fleet(Path(assets_dir) / spec["traffic"])
    if "count" in spec:        # the placement can size the fleet to its streets
        fleet["count"] = int(spec["count"])
    seed = int(spec.get("seed", fleet["seed"]))
    rng = np.random.default_rng(seed)
    node = SceneNode(name, tags=["traffic"])
    node.meta["type"] = "traffic"
    node.meta["fleet"] = {k: v for k, v in fleet.items() if k != "fleet"}
    node.meta["fleet"]["seed"] = seed
    if graph is None:
        raise ValueError(f"traffic '{name}': the scene has no lanes (city traffic: or routes:)")
    lanes = [ln for ln in graph["lanes"] if ln["kind"] in ("lane", "route") and ln["length"] > 8.0]
    # Starting vehicles keep clear of where players and NPCs start.
    to_root = np.linalg.inv(root.world_transform())
    keep_clear = [(to_root @ n.world_transform())[:3, 3] for n in root.iter_nodes()
                  if n.meta.get("type") in ("spawn", "npc")]
    weights = np.array([e["weight"] for e in fleet["fleet"]])
    taken: dict[str, list[tuple[float, float]]] = {}      # lane -> [(s, half length)]
    count = fleet["count"]
    tries = 0
    while len(node.children) < count and tries < count * 40:
        tries += 1
        entry = fleet["fleet"][int(rng.choice(len(weights), p=weights / weights.sum()))]
        pool = [ln for ln in lanes if not entry["lanes"] or any(ln["id"].startswith(p) for p in entry["lanes"])]
        if not pool:
            continue
        lengths = np.array([ln["length"] for ln in pool])
        lane = pool[int(rng.choice(len(pool), p=lengths / lengths.sum()))]
        params = draw_params(entry["params"], rng)
        vehicle = load({"asset": entry["asset"], "params": params})
        v = vehicle.meta.get("vehicle")
        if v is None:
            raise ValueError(f"traffic '{name}': {entry['asset']} has no vehicle: block")
        half = v["clearance"][0] / 2
        if lane["length"] < 2 * half + 4.0:
            continue          # too short for this vehicle
        s = float(rng.uniform(half + 1.0, lane["length"] - half - 1.0))
        if any(abs(s - s2) < half + h2 + fleet["spacing"] for s2, h2 in taken.get(lane["id"], [])):
            continue
        taken.setdefault(lane["id"], []).append((s, half))
        pts = np.asarray(lane["points"], dtype=np.float64)
        position, yaw = _pose_on_lane(pts, s, v.get("wheelbase", 2.6) / 2)
        if any(np.linalg.norm((position - p)[[0, 2]]) < half + SPAWN_CLEARANCE for p in keep_clear):
            taken[lane["id"]].pop()
            continue
        from .core.transform import Transform

        vehicle.name = f"{name}_{len(node.children) + 1}"
        vehicle.transform = Transform(translation=position, rotation=np.array([0.0, yaw, 0.0]))
        lo, hi = fleet["driving"]["speed_factor"]
        vehicle.meta["driving"] = {"lane": lane["id"], "s": round(s, 3), "factor": round(float(rng.uniform(lo, hi)), 3)}
        node.add_child(vehicle)
    return node
