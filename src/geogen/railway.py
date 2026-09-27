"""Railways as data: track geometry, stations and level crossings (see epic geogen-r65, M4).

A scene declares railways next to its roads::

    railways:
      main:
        path: { spline: [[-120, 0, 30], [0, 0, 36], [120, 0, 30]], samples: 16 }
        loop: false
        speed: 22                       # line speed, m/s (curves cap it)
        stations:
          - { name: west, at: 0.25, length: 70, side: right }   # at: fraction of the line or {s: metres}

``build_railway`` sweeps ballast, sleepers and rails along the path (the path is
at ground level; the rail head is ``RAIL_TOP`` above it), adds low platforms
with a bench and lamps at each station, and records ``meta.railway``:
``{id, points (rail head, resampled), length, loop, speed, stations: [{name, s, length}]}``.

``level_crossings(scene, graph)`` finds where railways cross road lanes of the
lane graph (geogen/traffic.py): each crossing gets a barrier (crossing_barrier.yaml)
on the kerb side of every crossing lane, the rail s-range to guard, and the
lane s-ranges that must stay clear while it's closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.mesh import Mesh
from .core.node import SceneNode
from .core.profile import Shape
from .core.transform import Transform

RAIL_TOP = 0.16            # rail head above the ground: a low bed, so road crossings are near flush
BED = 0.03                 # ballast top
SLEEPER = 0.06             # sleeper thickness
STEP = 0.5
RAILWAY_KEYS = {"path", "loop", "gauge", "speed", "stations"}
STATION_KEYS = {"name", "at", "length", "side", "width"}
PLATFORM_HEIGHT = 0.3      # low platforms: a single step up from the ground
PLATFORM_EDGE = 1.75       # track centre to platform edge (clears a 2.85 m wide car)


def _path(spec: dict[str, Any], loop: bool) -> np.ndarray:
    from .core.profile import catmull_rom
    from .traffic import resample

    path = spec["path"]
    if isinstance(path, dict):
        pts = catmull_rom(np.asarray(path["spline"], dtype=np.float64), int(path.get("samples", 16)), closed=loop)
    else:
        pts = np.asarray(path, dtype=np.float64)
    if loop:
        pts = np.vstack([pts, pts[:1]])
    return resample(pts, STEP)


def _frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents and right vectors (plan) along a polyline."""
    t = np.gradient(points, axis=0)
    t[:, 1] = 0.0
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    right = np.column_stack([-t[:, 2], np.zeros(len(t)), t[:, 0]])   # left of +Z travel is +X
    return t, right


def _sweep(profile: list[list[float]], path: np.ndarray, closed: bool, material) -> Mesh:
    from .core import meshops, uvmap
    from .generators.sweep import SweepGenerator

    mesh = SweepGenerator(profile=Shape(np.asarray(profile, dtype=np.float64)), path=path, closed=closed,
                          center=False, crease_angle=35.0).generate()
    mesh = meshops.compute_normals(uvmap.box_project(mesh), 35.0)
    mesh.material = material
    return mesh


def _sleepers(points: np.ndarray, spacing: float, material) -> Mesh:
    from .core import meshops, uvmap
    from .generators.primitives import CubeGenerator

    box = CubeGenerator(size_x=2.5, size_y=SLEEPER, size_z=0.24, bevel=0).generate()
    t, _ = _frames(points)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    meshes = []
    for d in np.arange(spacing / 2, s[-1], spacing):
        k = min(int(np.searchsorted(s, d)), len(points) - 1)
        yaw = float(np.arctan2(t[k, 0], t[k, 2]))
        m = Transform(translation=points[k] + np.array([0.0, BED + SLEEPER / 2, 0.0]),
                      rotation=np.array([0.0, yaw, 0.0])).to_matrix()
        meshes.append(box.transform(m))
    mesh = meshops.compute_normals(uvmap.box_project(Mesh.merge(meshes)), 35.0)
    mesh.material = material
    return mesh


def _platform(points: np.ndarray, s: np.ndarray, s0: float, s1: float, side: float, width: float,
              material) -> Mesh | None:
    """A low platform beside the track between s0 and s1 (side +1 right, -1 left)."""
    from shapely.geometry import Polygon

    from .core import meshops, uvmap
    from .generators.profiles import ExtrudeGenerator

    sel = (s >= s0) & (s <= s1)
    if sel.sum() < 2:
        return None
    _, right = _frames(points)
    inner = points[sel] + right[sel] * side * PLATFORM_EDGE
    outer = points[sel] + right[sel] * side * (PLATFORM_EDGE + width)
    ring = np.vstack([inner[:, [0, 2]], outer[::-1][:, [0, 2]]])
    poly = Polygon(ring).buffer(0)
    if poly.is_empty or poly.geom_type != "Polygon":
        return None
    poly = poly.simplify(0.02)
    shape = Shape(np.asarray(poly.exterior.coords)[:-1])
    # ExtrudeGenerator (axis y) maps profile (x, y) to world (x, -z) and centres on the origin.
    shape = Shape(np.column_stack([shape.outer[:, 0], -shape.outer[:, 1]]))
    mesh = ExtrudeGenerator(shape=shape, depth=PLATFORM_HEIGHT, axis="y", bevel=0.02).generate()
    base_y = float(points[sel][:, 1].mean())
    mesh = mesh.transform(Transform(translation=np.array([0.0, base_y + PLATFORM_HEIGHT / 2, 0.0])).to_matrix())
    mesh = meshops.compute_normals(uvmap.box_project(mesh), 35.0)
    mesh.material = material
    return mesh


def build_railway(name: str, spec: dict[str, Any], materials, load, assets_dir) -> SceneNode:
    """The railway's node: track meshes, station platforms (with furniture) and ``meta.railway``."""
    unknown = set(spec) - RAILWAY_KEYS
    if unknown:
        raise ValueError(f"railway '{name}': unknown keys {sorted(unknown)}. Known: {sorted(RAILWAY_KEYS)}")
    loop = bool(spec.get("loop", False))
    ground = _path(spec, loop)
    gauge = float(spec.get("gauge", 1.435))
    head = ground + np.array([0.0, RAIL_TOP, 0.0])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ground, axis=0), axis=1))]
    node = SceneNode(f"railway_{name}", tags=["railway"])
    ballast = SceneNode(f"railway_{name}_ballast", mesh=_sweep(
        [[-2.0, -0.05], [2.0, -0.05], [1.7, BED], [-1.7, BED]], ground, loop, materials.load("rock")))
    ballast.meta["walkable"] = True
    node.add_child(ballast)
    sleepers = SceneNode(f"railway_{name}_sleepers", mesh=_sleepers(ground, 0.65, materials.load("wood_dark")))
    sleepers.meta["collider"] = "none"
    node.add_child(sleepers)
    for side in (-1.0, 1.0):
        x = side * (gauge / 2 + 0.035)
        rail = SceneNode(f"railway_{name}_rail_{'l' if side > 0 else 'r'}", mesh=_sweep(
            [[x - 0.035, BED + SLEEPER], [x + 0.035, BED + SLEEPER], [x + 0.035, RAIL_TOP], [x - 0.035, RAIL_TOP]],
            ground, loop, materials.load("metal")))
        rail.meta["collider"] = "none"
        node.add_child(rail)

    stations = []
    for i, st in enumerate(spec.get("stations") or []):
        if set(st) - STATION_KEYS:
            raise ValueError(f"railway '{name}' station {i}: keys are {sorted(STATION_KEYS)}")
        at = st.get("at", 0.5)
        centre = float(at["s"]) if isinstance(at, dict) else float(at) * s[-1]
        length = float(st.get("length", 60.0))
        station = {"name": str(st.get("name", f"station_{i}")), "s": round(centre, 3), "length": length}
        stations.append(station)
        sides = {"right": [1.0], "left": [-1.0], "both": [-1.0, 1.0]}[st.get("side", "right")]
        width = float(st.get("width", 3.5))
        for side in sides:
            mesh = _platform(ground, s, centre - length / 2, centre + length / 2, side, width,
                             materials.load("concrete"))
            if mesh is None:
                continue
            plat = SceneNode(f"platform_{station['name']}_{'r' if side > 0 else 'l'}", mesh=mesh,
                             tags=["railway.platform", "street.sidewalk"])
            plat.meta["walkable"] = True
            node.add_child(plat)
            _station_furniture(node, plat.name, ground, s, centre, length, side, width, load)
    node.meta["railway"] = {
        "id": name, "loop": loop, "gauge": gauge, "length": round(float(s[-1]), 3),
        "speed": float(spec.get("speed", 22.0)),
        "points": [[round(float(v), 3) for v in p] for p in head],
        "stations": stations,
    }
    return node


def _station_furniture(node: SceneNode, platform: str, ground, s, centre, length, side, width, load) -> None:
    """Benches facing the track and lamps along the back of a platform."""
    t, right = _frames(ground)

    def at(d: float, lateral: float) -> tuple[np.ndarray, float]:
        k = min(int(np.searchsorted(s, d)), len(ground) - 1)
        p = ground[k] + right[k] * side * lateral + np.array([0.0, PLATFORM_HEIGHT, 0.0])
        face = -right[k] * side                    # toward the track
        return p, float(np.arctan2(face[0], face[2]))

    for j, off in enumerate((-length / 4, length / 4)):
        p, yaw = at(centre + off, PLATFORM_EDGE + width - 0.9)
        bench = load({"asset": "bench.yaml"})
        bench.name = f"{platform}_bench_{j + 1}"
        bench.transform = Transform(translation=p, rotation=np.array([0.0, yaw, 0.0]))
        node.add_child(bench)
    for j, off in enumerate((-length * 0.4, 0.0, length * 0.4)):
        p, yaw = at(centre + off, PLATFORM_EDGE + width - 0.35)
        lamp = load({"asset": "street_lamp.yaml"})
        lamp.name = f"{platform}_lamp_{j + 1}"
        lamp.transform = Transform(translation=p, rotation=np.array([0.0, yaw, 0.0]))
        node.add_child(lamp)


# --- level crossings -----------------------------------------------------------------------------

CROSSING_STOP = 4.5        # road vehicles stop this far before the track centre
CROSSING_GUARD = 7.0       # rail s-range guarded either side of the outermost crossing lane


def level_crossings(root: SceneNode, railway_node: SceneNode, graph: dict[str, Any] | None, load) -> None:
    """Find where the railway crosses road lanes; add barriers and ``meta.railway.crossings``.

    Each crossing: ``{id, s: [s0, s1] (rail, guarded), barriers: [node names], lanes:
    [{lane, stop}]}`` where ``stop`` is the s on the road lane where vehicles wait
    while the barriers are down. Positions are in the root (scene) frame, like the graph.
    """
    from shapely.geometry import LineString

    rail = railway_node.meta["railway"]
    rail_pts = np.asarray(rail["points"], dtype=np.float64)
    rail_line = LineString(rail_pts[:, [0, 2]])
    hits = []           # (rail s, lane dict, lane s)
    for lane in (graph or {}).get("lanes", []):
        if lane["kind"] not in ("lane", "route", "connector"):
            continue
        pts = np.asarray(lane["points"], dtype=np.float64)
        line = LineString(pts[:, [0, 2]])
        inter = rail_line.intersection(line)
        points = [inter] if inter.geom_type == "Point" else list(getattr(inter, "geoms", []))
        for p in points:
            if p.geom_type != "Point":
                continue
            hits.append((rail_line.project(p), lane, line.project(p)))
    if not hits:
        rail["crossings"] = []
        return
    hits.sort(key=lambda h: h[0])
    groups: list[list] = [[hits[0]]]
    for h in hits[1:]:
        if h[0] - groups[-1][-1][0] < 15.0:
            groups[-1].append(h)          # the same road: one crossing
        else:
            groups.append([h])
    crossings = []
    for gi, group in enumerate(groups):
        cid = f"{rail['id']}_x{gi + 1}"
        s0 = max(min(h[0] for h in group) - CROSSING_GUARD, 0.0)
        s1 = min(max(h[0] for h in group) + CROSSING_GUARD, rail["length"])
        lanes, barriers = [], []
        for _, lane, ls in group:
            stop = max(ls - CROSSING_STOP, 0.0)
            lanes.append({"lane": lane["id"], "stop": round(float(stop), 3)})
            if lane["kind"] == "connector":
                continue           # a barrier on the approach lane covers it
            pts = np.asarray(lane["points"], dtype=np.float64)
            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            k = min(int(np.searchsorted(cum, stop + 0.5)), len(pts) - 1)
            p = np.array([np.interp(stop + 0.5, cum, pts[:, j]) for j in range(3)])
            d = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
            d = d / max(np.linalg.norm(d), 1e-9)
            right = np.array([-d[2], 0.0, d[0]])
            barrier = load({"asset": "crossing_barrier.yaml", "params": {"reach": 3.4}})
            barrier.name = f"crossing_{cid}_{len(barriers) + 1}"
            arm = -right                            # across the lane, toward the centre line
            yaw = float(np.arctan2(-arm[2], arm[0]))
            barrier.transform = Transform(translation=p + right * 1.75, rotation=np.array([0.0, yaw, 0.0]))
            barrier.meta["crossing"] = cid
            railway_node.add_child(barrier)
            barriers.append(barrier.name)
        crossings.append({"id": cid, "s": [round(float(s0), 3), round(float(s1), 3)],
                          "barriers": barriers, "lanes": lanes})
    rail["crossings"] = crossings
