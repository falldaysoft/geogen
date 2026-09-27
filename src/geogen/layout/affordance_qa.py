"""Affordance QA: pose a character at every affordance, and check each can be used.

Affordances (``layout.loader._affordance``) are what the declarative NPC
approach rests on, so they are checked the way the runtime will use them
(``runtime/godot/scripts/npc.gd``): walk to the ``approach`` point, face the
anchor's heading, and for actions with a ``pose:`` step (sit, lie) put the
body in that pose at the anchor.

- ``stage_actors(scene, body)`` returns a copy of the scene with a posed
  body at every affordance (``-r out.png --affordances``): a scale check
  for seat heights, table clearance and sill heights.
- ``check_affordances(scene)`` reports:

  - ``approach_blocked``: nowhere to stand within ``SNAP`` of the approach
    point (the runtime snaps it onto the navmesh);
  - ``approach_unreachable``: the approach can't be walked to from the
    scene's spawns and NPCs (a 2D occupancy grid of the collidable meshes
    sliced at body heights, eroded by the actor radius; parts moved by
    interactions -- door leaves -- don't block, as NPCs open them);
  - ``pose_clash``: the posed body intersects the asset it uses by more than
    ``CLASH_OWN`` (sinking into a cushion is fine) or anything else by more
    than ``CLASH_OTHER`` (knees through a table, a head in a shelf).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.node import SceneNode
from ..core.transform import Transform
from .qa import Issue

CELL = 0.05              # occupancy grid resolution (m)
SNAP = 0.3               # an approach point may snap this far to standable floor
MARGIN = 3.0             # grid extends this far beyond the spots and seeds
CLASH_OWN = 0.012        # m³: the body may sink this far into the asset it uses (a mattress, a cushion)
CLASH_OTHER = 0.0005     # m³: ...but only half a litre into anything else (a thin table top)
FLOOR_TOL = 0.06         # m: feet may press this far into the floor (a lawn slab, a rug)
SEAT_CLEAR = 0.03        # m the calves keep in front of a seat's front edge (generators/humanoid.py)
DEFAULT_BODY = {"asset": "characters/humanoid.yaml", "params": {"preset": "feminine"}}


@dataclass
class Spot:
    """One affordance, resolved in the scene frame."""

    owner: SceneNode           # the asset node that advertises it
    index: int
    affordance: dict
    anchor: np.ndarray         # 4x4: anchor position, turned to its yaw
    approach: np.ndarray       # (3,) where the actor stands first (asset floor level)

    @property
    def name(self) -> str:
        return f"{self.owner.name}[{self.index}] {self.affordance['type']}"


def affordance_spots(scene: SceneNode) -> list[Spot]:
    """Every affordance under ``scene`` (except those on NPC bodies), in the scene frame."""
    to_scene = np.linalg.inv(scene.world_transform())
    spots = []
    for node in scene.iter_nodes():
        if node.meta.get("type") == "npc" or _inside_npc(node):
            continue
        for i, a in enumerate(node.meta.get("affordances") or []):
            m = to_scene @ node.world_transform()
            yaw = np.radians(a["yaw"])
            local = Transform(translation=np.asarray(a["position"], dtype=float),
                              rotation=np.array([0.0, yaw, 0.0])).to_matrix()
            approach = (m @ np.r_[np.asarray(a["approach"], dtype=float), 1.0])[:3]
            spots.append(Spot(node, i, a, m @ local, approach))
    return spots


def _inside_npc(node: SceneNode) -> bool:
    p = node.parent
    while p is not None:
        if p.meta.get("type") == "npc":
            return True
        p = p.parent
    return False


def _pose_step(action: str) -> tuple[str, str] | None:
    """The (pose, at) an action puts the body in, if any (its first non-stand pose step)."""
    from ..npc import load_actions

    for step in load_actions().get(action, {}).get("steps", []):
        if "pose" in step and step["pose"] != "stand":
            return str(step["pose"]), str(step.get("at", "anchor"))
    return None


class _Floors:
    """Upward-facing triangles, for 'what is the floor under this point' queries."""

    def __init__(self, meshes):
        tris = [v[f] for _, v, f in meshes]
        tri = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        up = n[:, 1] > 0.5 * np.linalg.norm(n, axis=1)
        self.tri = tri[up]

    def y(self, x: float, z: float, below: float) -> float:
        """Height of the highest floor under (x, z) at or below ``below`` (the ground, 0, if none)."""
        t = self.tri
        if not len(t):
            return 0.0
        a, b, c = t[:, 0][:, [0, 2]], t[:, 1][:, [0, 2]], t[:, 2][:, [0, 2]]
        p = np.array([x, z])
        v0, v1, v2 = c - a, b - a, p - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        u = (d11 * d20 - d01 * d21) / den
        v = (d00 * d21 - d01 * d20) / den
        inside = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        ys = t[:, 0, 1] + u * (t[:, 2, 1] - t[:, 0, 1]) + v * (t[:, 1, 1] - t[:, 0, 1])
        ys = ys[inside & (ys <= below)]
        return float(max(ys.max(), 0.0)) if len(ys) else 0.0


def _floor_y(floors: "_Floors", x: float, z: float, y: float) -> float:
    return floors.y(x, z, y + 0.1)


def _load_body(body, loader, assets_dir: Path) -> SceneNode:
    if isinstance(body, SceneNode):
        return body
    spec = body or DEFAULT_BODY
    if "archetype" in spec:
        from ..characters import load_character

        return load_character(loader, assets_dir, spec["archetype"], int(spec.get("seed", 0)), spec.get("params"))
    return loader.load(Path(assets_dir) / spec["asset"], params=spec.get("params"))


def pose_actor(body: SceneNode, spot: Spot, floor: float | None = None) -> SceneNode:
    """An instance of ``body`` doing ``spot``'s action, in the scene frame."""
    from ..core.skin import pose_clips

    actor = body.instance()
    posed = _pose_step(spot.affordance["action"])
    poses = body.meta.get("poses") or {}
    if posed is not None and posed[0] in poses:
        name, at = posed
        place = spot.anchor.copy()
        if at == "approach":
            place[:3, 3] = spot.approach
        p = poses[name]
        rot = np.radians(p["rotation"])
        offset = np.asarray(p["offset"], dtype=float).copy()
        if at == "anchor" and "reach" in p and "depth" in spot.affordance:
            # Deep seats: sit forward until the shins clear the front edge (as the runtime does).
            offset[2] += max(0.0, spot.affordance["depth"] + SEAT_CLEAR - p["reach"])
        m = place @ Transform(translation=offset, rotation=rot, scale=np.asarray(p["scale"])).to_matrix()
        pose_clips(actor, f"pose_{name}", 0.0)
    else:
        m = spot.anchor.copy()
        m[:3, 3] = spot.approach
        if floor is not None:
            m[1, 3] = floor
    actor.transform = _transform(m)
    actor.name = f"actor_{spot.owner.name}_{spot.index}"
    actor.meta["affordance_actor"] = spot.name
    return actor


def _transform(m: np.ndarray) -> Transform:
    return Transform.from_matrix(m)


def stage_actors(scene: SceneNode, body=None, loader=None, assets_dir: Path | None = None) -> SceneNode:
    """A copy of ``scene`` with a posed body at every affordance (NPCs' own bodies hidden)."""
    loader, assets_dir = _defaults(loader, assets_dir)
    body_node = _load_body(body, loader, assets_dir)
    staged = scene.instance()
    for node in list(staged.iter_nodes()):
        if node.meta.get("type") == "npc" and node.parent is not None:
            node.parent.remove_child(node)
    world = _Floors(_obstacle_meshes(staged))
    group = SceneNode("affordance_actors")
    for spot in affordance_spots(staged):
        floor = _floor_y(world, spot.approach[0], spot.approach[2], spot.approach[1] + 1.0)
        group.add_child(pose_actor(body_node, spot, floor))
    staged.add_child(group)
    return staged


def _defaults(loader, assets_dir):
    from .loader import LayoutLoader

    return loader or LayoutLoader(), Path(assets_dir or Path(__file__).parents[3] / "assets")


# --- checks ------------------------------------------------------------------------------------


def _moving_parts(scene: SceneNode) -> set[int]:
    """ids of nodes moved by interactions (door leaves, drawers) and everything under them."""
    out: set[int] = set()
    for node in scene.iter_nodes():
        for interaction in node.interactions:
            for motion in interaction.motions:
                for part in motion.parts:
                    out.update(id(n) for n in part.iter_nodes())
    return out


def _obstacle_meshes(scene: SceneNode):
    """(node, world vertices, faces) for everything an actor bumps into."""
    moving = _moving_parts(scene)
    to_scene = np.linalg.inv(scene.world_transform())
    skip: set[int] = set()
    for node in scene.iter_nodes():
        if node.meta.get("type") == "npc" or node.meta.get("affordance_actor"):
            skip.update(id(n) for n in node.iter_nodes())
    out = []
    for node in scene.iter_nodes():
        if node.mesh is None or not len(node.mesh.faces) or id(node) in skip or id(node) in moving:
            continue
        if node.meta.get("collider") == "none" or node.meta.get("type") == "collider":
            continue
        m = to_scene @ node.world_transform()
        v = (m @ np.c_[node.mesh.vertices, np.ones(len(node.mesh.vertices))].T).T[:, :3]
        out.append((node, v, node.mesh.faces))
    return out


class _Grid:
    """Free floor at one level: collidable geometry between step and head height (projected
    to plan: a table top blocks, a floor or a ceiling doesn't), eroded by the actor radius."""

    def __init__(self, meshes, lo: np.ndarray, hi: np.ndarray, floor: float, band, radius: float):
        from scipy import ndimage

        self.lo = np.floor(lo / CELL) * CELL
        shape = np.ceil((hi - self.lo) / CELL).astype(int) + 1
        blocked = np.zeros(shape, dtype=bool)
        y0, y1 = floor + band[0], floor + band[1]
        for _, v, f in meshes:
            tri = v[f]
            ys = tri[:, :, 1]
            sel = (ys.max(axis=1) >= y0) & (ys.min(axis=1) <= y1)
            for t in tri[sel]:
                self._fill(blocked, t[:, [0, 2]])
        free = ~blocked
        r = int(np.ceil(radius / CELL))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        self.standable = ndimage.binary_erosion(free, structure=xx ** 2 + yy ** 2 <= r ** 2, border_value=1)
        self.labels, _ = ndimage.label(self.standable)

    def _fill(self, grid: np.ndarray, t: np.ndarray) -> None:
        """Mark a plan triangle's edges, and its inside if it isn't edge-on (a wall)."""
        for i in range(3):
            self._mark(grid, t[i], t[(i + 1) % 3])
        a, b, c = t
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(area) < CELL * CELL:
            return
        i0 = np.maximum(np.floor((t.min(axis=0) - self.lo) / CELL).astype(int), 0)
        i1 = np.minimum(np.ceil((t.max(axis=0) - self.lo) / CELL).astype(int), np.array(grid.shape) - 1)
        if np.any(i1 < i0):
            return
        gx, gz = np.mgrid[i0[0]:i1[0] + 1, i0[1]:i1[1] + 1]
        px, pz = self.lo[0] + gx * CELL, self.lo[1] + gz * CELL
        d0 = (b[0] - a[0]) * (pz - a[1]) - (b[1] - a[1]) * (px - a[0])
        d1 = (c[0] - b[0]) * (pz - b[1]) - (c[1] - b[1]) * (px - b[0])
        d2 = (a[0] - c[0]) * (pz - c[1]) - (a[1] - c[1]) * (px - c[0])
        inside = ((d0 >= 0) & (d1 >= 0) & (d2 >= 0)) | ((d0 <= 0) & (d1 <= 0) & (d2 <= 0))
        grid[gx[inside], gz[inside]] = True

    def _mark(self, grid: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
        n = max(int(np.ceil(np.linalg.norm(b - a) / (CELL / 2))), 1)
        pts = a + (b - a) * np.linspace(0, 1, n + 1)[:, None]
        idx = np.round((pts - self.lo) / CELL).astype(int)
        ok = np.all((idx >= 0) & (idx < grid.shape), axis=1)
        grid[idx[ok, 0], idx[ok, 1]] = True

    def cell(self, p: np.ndarray) -> tuple[int, int] | None:
        i = np.round((np.asarray(p)[[0, 2]] - self.lo) / CELL).astype(int)
        if np.any(i < 0) or np.any(i >= self.standable.shape):
            return None
        return int(i[0]), int(i[1])

    def near_labels(self, p: np.ndarray, radius: float) -> set[int]:
        """Labels of standable cells within ``radius`` of ``p`` (plan)."""
        c = self.cell(p)
        if c is None:
            return set()
        r = int(np.ceil(radius / CELL))
        x0, z0 = max(c[0] - r, 0), max(c[1] - r, 0)
        window = self.labels[x0:c[0] + r + 1, z0:c[1] + r + 1]
        gx, gz = np.mgrid[x0:x0 + window.shape[0], z0:z0 + window.shape[1]]
        near = (gx - c[0]) ** 2 + (gz - c[1]) ** 2 <= r * r
        return set(np.unique(window[near & (window > 0)]).tolist())


def _seeds(scene: SceneNode) -> list[np.ndarray]:
    """Where actors come from: spawns and NPC placements (scene frame)."""
    to_scene = np.linalg.inv(scene.world_transform())
    return [(to_scene @ n.world_transform())[:3, 3] for n in scene.iter_nodes()
            if n.meta.get("type") in ("spawn", "npc")]


def check_affordances(scene: SceneNode, player=None, body=None, loader=None,
                      assets_dir: Path | None = None, clash: bool = True) -> list[Issue]:
    """Approach points that can't be stood on or reached, and poses that clash with geometry."""
    from ..player import load_player_spec

    player = player or load_player_spec()
    spots = affordance_spots(scene)
    if not spots:
        return []
    meshes = _obstacle_meshes(scene)
    world = _Floors(meshes)
    seeds = _seeds(scene)
    npc_radius = [float(n.meta["npc"]["radius"]) for n in scene.iter_nodes() if n.meta.get("type") == "npc"]
    radius = min([player.radius, *npc_radius])
    band = (player.step_height + 0.05, player.height)
    issues: list[Issue] = []

    # One grid per floor level the approach points stand on.
    levels: dict[float, list[tuple[Spot, float]]] = {}
    for spot in spots:
        floor = _floor_y(world, spot.approach[0], spot.approach[2], spot.approach[1] + 1.0)
        levels.setdefault(round(floor, 1), []).append((spot, floor))
    for level, items in levels.items():
        pts = [s.approach[[0, 2]] for s, _ in items]
        level_seeds = [p for p in seeds if abs(_floor_y(world, p[0], p[2], p[1] + 1.0) - level) < 0.3]
        pts += [p[[0, 2]] for p in level_seeds]
        lo, hi = np.min(pts, axis=0) - MARGIN, np.max(pts, axis=0) + MARGIN
        nearby = [(n, v, f) for n, v, f in meshes
                  if v[:, 0].max() >= lo[0] and v[:, 0].min() <= hi[0]
                  and v[:, 2].max() >= lo[1] and v[:, 2].min() <= hi[1]]
        grid = _Grid(nearby, lo, hi, level, band, radius)
        reachable = set()
        for p in level_seeds:
            reachable |= grid.near_labels(p, SNAP)
        for spot, _ in items:
            labels = grid.near_labels(spot.approach, SNAP)
            where = f"({spot.approach[0]:.2f}, {spot.approach[2]:.2f})"
            if not labels:
                issues.append(Issue("approach_blocked", spot.owner.name, (spot.owner.name,),
                                    f"{spot.name}: nowhere to stand near its approach {where}"))
            elif level_seeds and not labels & reachable:
                issues.append(Issue("approach_unreachable", spot.owner.name, (spot.owner.name,),
                                    f"{spot.name}: approach {where} can't be reached from a spawn or NPC"))
    if clash:
        issues += _pose_clashes(scene, spots, meshes, body, loader, assets_dir)
    return issues


def _users(scene: SceneNode) -> list[tuple[list[str], SceneNode]]:
    """(affordance_tags, body) of the scene's NPCs, one per distinct body (up to 3 unrestricted
    and 3 per tag filter): whoever might actually use an affordance."""
    out, seen = [], set()
    for node in scene.iter_nodes():
        if node.meta.get("type") != "npc":
            continue
        body = next((c for c in node.children if c.name == node.meta["npc"]["body"].get("node", "body")), None)
        if body is None:
            continue
        key = frozenset(id(n.mesh) for n in body.iter_nodes() if n.mesh is not None)
        tags = list(node.meta["npc"].get("affordance_tags") or [])
        if key in seen or sum(1 for t, _ in out if t == tags) >= 3:
            continue
        seen.add(key)
        out.append((tags, body))
    return out


def _pose_clashes(scene, spots, meshes, body, loader, assets_dir) -> list[Issue]:
    """Posed bodies (sit, lie) that sink into geometry. Without ``body``, each affordance is
    tried with the bodies of the scene's NPCs that may use it (their ``affordance_tags``), or
    the default body if none may."""
    posed = [s for s in spots if _pose_step(s.affordance["action"]) is not None]
    if not posed:
        return []
    loader, assets_dir = _defaults(loader, assets_dir)
    users = [] if body is not None else _users(scene)
    fallback = _load_body(body, loader, assets_dir) if body is not None or not users else None
    floors = _Floors(meshes)
    issues = []
    for spot in posed:
        tags = spot.affordance.get("tags") or []
        bodies = [b for t, b in users if not t or set(t) & set(tags)]
        if not bodies:
            fallback = fallback or _load_body(None, loader, assets_dir)
            bodies = [fallback]
        for body_node in bodies:
            issues += _clash_one(spot, body_node, meshes, floors)
    return issues


def _clash_one(spot: Spot, body_node: SceneNode, meshes, floors) -> list[Issue]:
    issues = []
    actor = pose_actor(body_node, spot)
    skin = next((n for n in actor.iter_nodes() if n.skin is not None and n.name == "body"), None)
    if skin is None:
        return []
    mesh = skin.world_mesh()          # posed, in the actor's (scene) frame
    v = mesh.vertices
    body_man = _manifold(v, mesh.faces)
    if body_man is None:
        return []
    floor = floors.y(spot.approach[0], spot.approach[2], spot.approach[1] + 1.0)
    above = body_man.trim_by_plane((0.0, 1.0, 0.0), floor + FLOOR_TOL)   # for other geometry
    lo, hi = v.min(axis=0), v.max(axis=0)
    own_nodes = {id(n) for n in spot.owner.iter_nodes()}
    own, other, worst = 0.0, 0.0, ("", 0.0)
    for node, ov, of in meshes:
        if np.any(ov.max(axis=0) < lo) or np.any(ov.min(axis=0) > hi):
            continue
        solid = _manifold(ov, of)
        if solid is None:
            continue
        if id(node) in own_nodes:
            own += float((body_man ^ solid).volume())
        else:
            vol = float((above ^ solid).volume())
            other += vol
            if vol > worst[1]:
                worst = (node.name, vol)
    if own > CLASH_OWN:
        issues.append(Issue("pose_clash", spot.owner.name, (spot.owner.name,),
                            f"{spot.name}: the posed body sinks {own * 1000:.1f} L into {spot.owner.name}"))
    if other > CLASH_OTHER:
        issues.append(Issue("pose_clash", spot.owner.name, (spot.owner.name, worst[0]),
                            f"{spot.name}: the posed body is {other * 1000:.1f} L inside other geometry"
                            f" (most in {worst[0]}: {worst[1] * 1000:.1f} L)"))
    return issues


def _manifold(v: np.ndarray, f: np.ndarray):
    import manifold3d

    mm = manifold3d.Mesh(vert_properties=np.ascontiguousarray(v, dtype=np.float32),
                         tri_verts=np.ascontiguousarray(f, dtype=np.uint32))
    mm.merge()     # stitch seams (vertices split for normals/UVs)
    man = manifold3d.Manifold(mm)
    return man if man.status() == manifold3d.Error.NoError and not man.is_empty() else None
