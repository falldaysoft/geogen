"""Pose correctives for loose garment shells (skirts, dresses): a geometric drape, baked to morphs.

A loose garment is a ring-lofted tube skinned to the pelvis and thighs. Linear blend skinning
alone can't keep it on the body: where the hips bend it loses volume, a seated skirt stands off
the lap as a box and hangs through the seat, and walking legs fold it. There is no cloth
simulation at runtime, so the correction is worked out here, frame by frame, and stored as a few
morph targets whose weights the clips key (``fit``).

``drape_frame`` re-shapes the shell in one pose, ring by ring. A ring of the outer wall is a
closed loop round the body; skinned into the pose it still lies near a plane. In that plane:

- the body it has to go round is the convex hull of the pelvis and thighs' section (cloth spans
  the gap between the legs rather than dipping into it; a ring on the thighs passes under the
  belly, not round it);
- a ring vertex never comes nearer that hull than the clearance it had at rest (legs that move
  under the cloth push it out);
- what the skinned ring has to spare beyond that is its *looseness*, which it keeps, except that
  in a ring on edge it hangs below the body (gravity: cloth lies on a lap), and that the ring never gets longer
  than it was cut (legs pushing out one side pull the others in: the factor ``slack`` is the
  largest that keeps its length);
- seated, nothing behind the knees hangs below the seat.

Everything that is not a ring vertex (the lining, the rolled hem, the buried top cap) moves with
the ring vertex it is nearest to. The rest pose comes out unchanged, exactly.

``fit`` drapes every static pose and sampled frames of every clip, turns the displacements into
rest-space deltas (through each vertex's blended skinning matrix) and keeps at most
``MAX_MORPHS`` morph targets ``drape_<n>``: the static poses' own drapes (exact: they are held
for minutes), then the principal components of what the clips' frames still lack; each clip gets
a weights track. Runtime cost: linear blend skinning plus at most that many blend shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import Mesh
from ..core.node import SceneNode

CLEARANCE = 0.01          # m the outer wall keeps off the skin (the lining is a cloth thickness inside it)
SEAT_EDGE = 0.02          # m before a seat's front over which the cloth on it eases off
SEAT_CLOTH = 0.012        # m of cloth between the seat flesh and the seat
LEGS = ("LeftUpperLeg", "RightUpperLeg")       # what a skirt's rings on the thighs go round, with the seat
SHINS = ("LeftLowerLeg", "RightLowerLeg")
TRUNK_SHARE = 0.5                              # a ring with less of its weight than this on the thighs is on the pelvis
KNEE = 0.06                                    # m below the knee joints that still counts as knee
TRUNK = ("Hips", "Spine")                      # and its rings on the pelvis: all of that and the belly too
MARGIN = 0.003            # m more where the body has moved under the cloth (the morphs only approximate the drape)
SLAB = 0.05               # m either side of a ring's plane that it answers for
REACH = 1.8               # body further than this many ring radii from a ring's centre is not its business
MAX_MORPHS = 8
MORPH = "drape_{}"
MAX_OWN = 3               # static poses that may have a target of their own (the rest share the clips')
OWN_TARGET = 0.004        # m: a static pose this far from what the targets so far give gets its own
RESIDUAL = 0.001          # m: stop adding morphs once every frame is this close to its drape
SAMPLES = {"walk": 16, "idle": 8}      # frames draped per clip (others: DEFAULT_SAMPLES)
DEFAULT_SAMPLES = 10
EMPHASIS = {"walk": 3.0}               # how much more a clip's frames count when the targets are chosen


@dataclass
class ShellGrid:
    """The outer wall of one lofted shell as rings of vertices (waist first), within a mesh."""

    rings: list[np.ndarray]                 # per ring: vertex indices ordered round the ring
    s: np.ndarray                           # arc length of each ring
    follow: np.ndarray                      # per mesh vertex: the ring vertex it moves with (-1: stays)
    gain: np.ndarray                        # per ring: 0 (the waistband: as skinned, but never nearer the body) .. 1
    trunk: np.ndarray = None                # per ring: it rides the pelvis more than the thighs
    rest: list[dict] = field(default_factory=list)       # per ring, from the rest pose (see _measure)


def shell_grid(mesh: Mesh, vertices: np.ndarray, s_range: tuple[float, float]) -> ShellGrid | None:
    """Find the rings of the shell made of ``vertices`` of ``mesh`` (a ``ringloft`` tube: uv v is
    the arc length, u runs round the ring), between arc lengths ``s_range`` (its key rings; the
    start cap is before them and the rolled hem after). None if it isn't such a tube."""
    from scipy.spatial import cKDTree

    from .humanoid import lining_faces

    if mesh.uvs is None or not len(vertices):
        return None
    mine = np.zeros(len(mesh.vertices), bool)
    mine[vertices] = True
    lining = lining_faces(mesh)
    outer = np.zeros(len(mesh.vertices), bool)
    outer[mesh.faces[~lining].ravel()] = True
    outer &= mine
    s_all = mesh.uvs[:, 1]
    rings, ring_s = _rings(mesh, outer & (s_all >= s_range[0] - 1e-6) & (s_all <= s_range[1] + 1e-6))
    if any(len(r) < 6 for r in rings) or len(rings) < 3:
        return None
    grid_vertices = np.concatenate(rings)
    follow = np.full(len(mesh.vertices), -1)
    rest_of = np.flatnonzero(mine)
    follow[rest_of] = grid_vertices[cKDTree(mesh.vertices[grid_vertices]).query(mesh.vertices[rest_of])[1]]
    gain = np.ones(len(rings))
    gain[0] = 0.0                                             # the waistband: under the top's hem
    return ShellGrid(rings, np.array(ring_s), follow, gain)


def _rings(mesh: Mesh, wall: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """The rings of the vertices ``wall`` (a mask) of a ``ringloft`` tube, by arc length: each
    one vertex per position (the UV seam and creases duplicate them), in order round the ring."""
    s_all = mesh.uvs[:, 1]
    rings, ring_s = [], []
    for s in np.unique(np.round(s_all[wall], 5)):
        idx = np.flatnonzero(wall & (np.abs(s_all - s) < 2e-5))
        _, first = np.unique(np.round(mesh.vertices[idx], 5), axis=0, return_index=True)
        idx = idx[np.sort(first)]
        rings.append(idx[np.argsort(mesh.uvs[idx, 0], kind="stable")])
        ring_s.append(float(s))
    return rings, ring_s


def solid_rings(mesh: Mesh, vertices: np.ndarray, s_range: tuple[float, float]) -> list[np.ndarray]:
    """The rings that bound the solid of a draped shell, as clipping QA caps it: its key rings
    and those of the start cap before them (buried under the top), but not the rolled hem's."""
    from .humanoid import lining_faces

    outer = np.zeros(len(mesh.vertices), bool)
    outer[mesh.faces[~lining_faces(mesh)].ravel()] = True
    mine = np.zeros(len(mesh.vertices), bool)
    mine[vertices] = True
    rings, _ = _rings(mesh, outer & mine & (mesh.uvs[:, 1] <= s_range[1] + 1e-6))
    return [r for r in rings if len(r) >= 6]


def _plane(points: np.ndarray, toward: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit plane of a ring: (centre, normal, two in-plane axes)."""
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    normal = vt[2]
    if toward is not None and np.dot(normal, toward) < 0:
        normal = -normal
    e1 = vt[0]
    return centre, normal, e1, np.cross(normal, e1)


def _fade(x: np.ndarray, start: float, end: float) -> np.ndarray:
    """1 up to ``start``, easing to 0 at ``end``."""
    t = np.clip((x - start) / (end - start), 0.0, 1.0)
    return 1.0 - t * t * (3 - 2 * t)


def _section(tris: np.ndarray, centre, normal, e1, e2, reach: float) -> np.ndarray:
    """What of the triangles a ring in this plane has to go round, as 2D points about the centre:
    where they cross the plane, and their corners within ``SLAB`` of it (the cloth's faces run on
    to the next ring: the body bulging between two rings counts for both). Points fade toward
    the centre as they near the slab's faces and beyond ``reach``, so nothing pops in or out as
    the body moves."""
    d = (tris - centre) @ normal                               # (f, 3)
    near = np.abs(d) < SLAB
    points, weights = [tris[near]], [_fade(np.abs(d[near]), 0.6 * SLAB, SLAB)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cross = (d[:, i] * d[:, j]) < 0
        t = d[cross, i] / (d[cross, i] - d[cross, j])
        points.append(tris[cross, i] + (tris[cross, j] - tris[cross, i]) * t[:, None])
        weights.append(np.ones(int(cross.sum())))
    p = np.concatenate(points) - centre
    flat = np.column_stack([p @ e1, p @ e2])
    w = np.concatenate(weights) * _fade(np.linalg.norm(flat, axis=1), reach, 1.25 * reach)
    return flat[w > 0] * w[w > 0, None]


def _hull_reach(points: np.ndarray, directions: np.ndarray) -> np.ndarray | None:
    """How far the convex hull of 2D ``points`` reaches from the origin along each unit direction
    (its far side; 0 where the ray misses it). None without a hull."""
    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    n, b = hull.equations[:, :2], -hull.equations[:, 2]        # n . x <= b inside
    along = directions @ n.T                                    # (rays, planes)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = b[None, :] / along
    leave = np.where(along > 1e-9, t, np.inf).min(axis=1)
    enter = np.where(along < -1e-9, t, -np.inf).max(axis=1)
    outside = (np.abs(along) <= 1e-9) & (b[None, :] < 0)        # parallel to a side it is outside of
    hit = (leave >= np.maximum(enter, 0.0)) & np.isfinite(leave) & ~outside.any(axis=1)
    return np.where(hit, leave, 0.0)


def _measure(grid: ShellGrid, x: np.ndarray, tris: tuple[np.ndarray, np.ndarray],
             radii: list[float] | None = None) -> list[dict]:
    """Per ring of the shell at positions ``x``: its plane, each vertex's direction and distance
    from the centre in it (and height off it), and how far the body's hull reaches that way."""
    out = []
    toward = None
    hull = None
    for k, ring in enumerate(grid.rings):
        centre, normal, e1, e2 = _plane(x[ring], toward)
        toward = normal
        rel = x[ring] - centre
        off = rel @ normal
        flat = np.column_stack([rel @ e1, rel @ e2])
        r = np.maximum(np.linalg.norm(flat, axis=1), 1e-9)
        u = flat / r[:, None]
        size = radii[k] if radii is not None else float(r.max())
        found = _hull_reach(_section(tris[int(grid.trunk[k])], centre, normal, e1, e2, REACH * size), u)
        if found is not None and found.max() > 0:
            hull = found
        elif hull is None:
            hull = np.zeros(len(ring))                          # nothing to go round yet
        # (a ring past the end of the body, a hem below the thighs, goes round what the last one did)
        if len(hull) != len(ring):
            hull = np.interp(np.linspace(0, 1, len(ring), endpoint=False),
                             np.linspace(0, 1, len(hull), endpoint=False), hull, period=1.0)
        out.append({"centre": centre, "normal": normal, "axes": (e1, e2), "u": u, "r": r, "off": off,
                    "hull": hull.copy(), "size": size})
    return out


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum())


def _smooth_ring(values: np.ndarray, passes: int = 1) -> np.ndarray:
    for _ in range(passes):
        values = 0.5 * values + 0.25 * (np.roll(values, 1) + np.roll(values, -1))
    return values


def prepare(grid: ShellGrid, rest_shell: np.ndarray, rest_tris: tuple[np.ndarray, np.ndarray], thigh: np.ndarray) -> None:
    """Record what each ring is like at rest: its looseness round the body, and its length.
    ``rest_tris``: the body's triangles for thigh rings and for trunk rings (``_body_faces``);
    ``thigh``: each shell vertex's weight on the thighs."""
    grid.trunk = np.array([thigh[ring].mean() < TRUNK_SHARE for ring in grid.rings])
    grid.rest = _measure(grid, rest_shell, rest_tris)
    for entry, ring in zip(grid.rest, grid.rings):
        gap = entry["r"] - entry["hull"]
        entry["clear"] = np.minimum(gap, CLEARANCE)             # the clearance it actually has
        entry["length"] = _length(rest_shell[ring])


def drape_frame(grid: ShellGrid, shell: np.ndarray, tris: tuple[np.ndarray, np.ndarray], up=(0.0, 1.0, 0.0),
                seat: tuple[float, float, float] | None = None) -> np.ndarray:
    """The shell (skinned vertex positions ``shell``) draped over the posed body (``tris``: its
    triangles for thigh rings and for trunk rings, see ``_body_faces``). ``seat``: (height, front z, hip height) of a seat under it: what is
    below the hips and behind the front edge rests on it. Returns new positions for every vertex."""
    up = np.asarray(up, dtype=np.float64)
    now = _measure(grid, shell, tris, [e["size"] for e in grid.rest])
    out = shell.copy()
    for k, (ring, rest, posed, gain) in enumerate(zip(grid.rings, grid.rest, now, grid.gain)):
        if gain <= 0:
            # The waistband stays as skinned, except that it never comes nearer the body than at
            # rest (bent far forward, skinning pulls the inside of the bend into the belly).
            e1, e2 = posed["axes"]
            rays = posed["u"][:, :1] * e1 + posed["u"][:, 1:] * e2
            radius = np.maximum(posed["r"], posed["hull"] + rest["clear"])
            out[ring] = posed["centre"] + radius[:, None] * rays + posed["off"][:, None] * posed["normal"]
            continue
        # Round a bend the rings fan apart on its outside, and the cloth's faces cut the corner
        # between them: stand off by the sagitta of the chord to the next ring (none when level).
        bulge = np.zeros(len(ring))
        for other in (k - 1, k + 1):
            if 0 <= other < len(grid.rings) and len(grid.rings[other]) == len(ring):
                turn = float(np.arccos(np.clip(abs(posed["normal"] @ now[other]["normal"]), -1.0, 1.0)))
                turn -= float(np.arccos(np.clip(abs(rest["normal"] @ grid.rest[other]["normal"]), -1.0, 1.0)))
                chord = np.linalg.norm(shell[ring] - shell[grid.rings[other]], axis=1)
                bulge = np.maximum(bulge, chord * max(turn, 0.0) / 8)
        e1, e2 = posed["axes"]
        rays = posed["u"][:, :1] * e1 + posed["u"][:, 1:] * e2          # 3D directions
        # Gravity, as far as it acts in the ring's plane (not at all on a level ring: a standing
        # skirt keeps its cut): the looseness of a ring on edge hangs below the body, so cloth on
        # top lies on it (a lap) and little stands out at the sides.
        lift = rays @ up
        tilt = float(np.linalg.norm(up - (up @ posed["normal"]) * posed["normal"]))
        t = np.clip((0.3 - lift) / 0.9, 0.0, 1.0)
        hang = 1.0 - tilt * (1.0 - t * t * (3 - 2 * t))
        floor = posed["hull"] + rest["clear"] + np.minimum(np.abs(posed["hull"] - rest["hull"]), MARGIN) + bulge
        # What the skinned ring has to spare over the body: legs that move under it use it up.
        loose = hang * np.maximum(posed["r"] - floor, 0.0)

        def place(slack: float) -> np.ndarray:
            return posed["centre"] + (floor + slack * loose)[:, None] * rays + posed["off"][:, None] * posed["normal"]

        slack = 1.0
        limit = rest["length"] * 1.02
        if _length(place(1.0)) > limit:
            lo, hi = 0.0, 1.0
            for _ in range(12):
                slack = (lo + hi) / 2
                lo, hi = (slack, hi) if _length(place(slack)) <= limit else (lo, slack)
            slack = lo
        radius = floor + slack * loose
        change = _smooth_ring(radius - posed["r"])
        radius = np.maximum(posed["r"] + gain * change, np.where(gain >= 1, floor, 0.0))
        out[ring] = posed["centre"] + radius[:, None] * rays + posed["off"][:, None] * posed["normal"]
    def rest_on(points: np.ndarray) -> np.ndarray:
        """Nothing below the hips and behind the seat's front hangs below the seat (easing off
        over the last ``SEAT_EDGE`` before it)."""
        height, front, hips = seat
        on = _fade(points[:, 2], front - SEAT_EDGE, front) * (points[:, 1] < hips)
        points = points.copy()
        points[:, 1] += on * np.maximum(height - points[:, 1], 0.0)
        return points

    if seat is not None:
        for ring in grid.rings:
            out[ring] = rest_on(out[ring])
    moved = grid.follow >= 0
    out[moved] = shell[moved] + (out[grid.follow[moved]] - shell[grid.follow[moved]])
    if seat is not None:                                        # (the rolled hem too)
        out[moved] = rest_on(out[moved])
    return out


def _ancestors(node: SceneNode):
    while node.parent is not None:
        node = node.parent
        yield node


def _named(root: SceneNode, name: str) -> SceneNode | None:
    return next((n for n in root.iter_nodes() if n.name == name and n.mesh is not None), None)


def _body_faces(body: SceneNode, dominant: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The body's faces that (a ring on the thighs, a ring on the pelvis) goes round. A thigh ring
    goes round the thighs and the seat (the back half of the pelvis), not the belly: cloth on a
    lap passes under a belly that overhangs it. The knees count too (a hem at the knee goes over
    them), not the shins below."""
    faces = body.mesh.faces
    rest = body.mesh.vertices[faces]
    of = lambda bones: np.isin(dominant, bones)[faces].all(axis=1)
    names = [j.name for j in body.skin.joints]
    knee_y = float(np.linalg.inv(body.skin.inverse_bind[names.index("LeftLowerLeg")])[1, 3])
    knees = of(LEGS + SHINS) & (rest[:, :, 1].min(axis=1) > knee_y - KNEE)
    behind = rest[:, :, 2].mean(axis=1) < 0
    return faces[of(LEGS) | knees | (of(LEGS + ("Hips",)) & behind)], faces[of(LEGS + TRUNK) | knees]


def seat_under(actor: SceneNode, flesh: np.ndarray) -> tuple[float, float, float]:
    """(height, front z, hip height) of the seat under a seated ``actor``, from the posed points of
    its seat flesh (the pelvis and thighs): their underside less the cloth on it, as far forward
    as the knees. (``humanoid.fit_seats`` puts the seat ``SEAT_SINK`` above that. Seats differ in
    depth: stopping at the front edge the seat flesh is measured to, ``humanoid.KNEE_OVERHANG``,
    leaves cloth hanging in deeper ones.)"""
    from .humanoid import KNEE_OVERHANG

    hip = actor.find("LeftUpperLeg").world_transform()[:3, 3]
    knee = actor.find("LeftLowerLeg").world_transform()[:3, 3]
    front = knee[2] - KNEE_OVERHANG * np.linalg.norm(knee - hip)
    height = float(flesh[flesh[:, 2] < front][:, 1].min()) - SEAT_CLOTH
    return height, float(knee[2]), float(hip[1])


def _skinned(node: SceneNode, rows: np.ndarray | slice = slice(None)) -> tuple[np.ndarray, np.ndarray]:
    """World positions of ``rows`` of a skinned node's vertices as posed, and their blended
    skinning matrices (3x3)."""
    mesh, world = node.mesh, node.world_transform()
    blended = np.einsum("vk,vkij->vij", mesh.weights[rows], node.skin.matrices(node)[mesh.joints[rows]])
    local = np.einsum("vij,vj->vi", blended[:, :3, :3], mesh.vertices[rows]) + blended[:, :3, 3]
    return local @ world[:3, :3].T + world[:3, 3], blended[:, :3, :3]


class Draper:
    """Drapes the ``clothes`` shells of a built humanoid in any pose of it. Of the body it skins
    only what the cloth goes round and sits on."""

    def __init__(self, root: SceneNode, shells: list[dict]) -> None:
        from .humanoid import dominant_bones

        self.grids: list[ShellGrid] = []
        clothes = next((n for n in root.children if n.name == "clothes" and n.mesh is not None), None)
        body = next((n for n in root.children if n.name == "body" and n.skin is not None), None)
        if clothes is None or body is None or clothes.skin is None:
            return
        self.grids = [g for g in (shell_grid(clothes.mesh, s["vertices"], s["s_range"]) for s in shells)
                      if g is not None]
        dominant = dominant_bones(body.mesh, [j.name for j in body.skin.joints])
        faces = _body_faces(body, dominant)
        flesh = np.flatnonzero(np.isin(dominant, ["Hips", *LEGS]))
        self.used = np.union1d(np.concatenate([f.ravel() for f in faces]), flesh)
        # The joints a drape depends on: what moves the clothes or the body it reads.
        moving = np.concatenate([clothes.mesh.joints[clothes.mesh.weights > 0], body.mesh.joints[self.used][body.mesh.weights[self.used] > 0]])
        self.joints = {body.skin.joints[j].name for j in np.unique(moving)}
        self.faces = [np.searchsorted(self.used, f) for f in faces]
        self.flesh = np.searchsorted(self.used, flesh)
        rest = root.instance(share_clips=True)
        for node in rest.iter_nodes():
            node.morph_weights = {}
        body_rest = _skinned(_named(rest, "body"), self.used)[0]
        rest_shell = _skinned(_named(rest, "clothes"))[0]
        names = [j.name for j in clothes.skin.joints]
        thigh = np.where(np.isin(names, LEGS)[clothes.mesh.joints], clothes.mesh.weights, 0.0).sum(axis=1)
        for grid in self.grids:
            prepare(grid, rest_shell, tuple(body_rest[f] for f in self.faces), thigh)

    def positions(self, actor: SceneNode, seat: bool | tuple = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(skinned, draped) positions of the clothes vertices of ``actor`` (a posed instance,
        morph weights at 0) and their blended skinning matrices. ``seat``: it sits on a seat
        (True: the one under its seat flesh; or ``seat_under``'s (height, front, hips))."""
        body = _skinned(_named(actor, "body"), self.used)[0]
        tris = tuple(body[f] for f in self.faces)
        shell, blended = _skinned(_named(actor, "clothes"))
        under = seat if isinstance(seat, tuple) else seat_under(actor, body[self.flesh]) if seat else None
        out = shell.copy()
        for grid in self.grids:
            mine = grid.follow >= 0
            out[mine] = drape_frame(grid, shell, tris, seat=under)[mine]
        return shell, out, blended


def fit(root: SceneNode, shells: list[dict], seated: set[str], max_morphs: int = MAX_MORPHS) -> dict:
    """Bake drape correctives for the ``clothes`` node of a built humanoid ``root``.

    ``shells``: per loose garment {"vertices": its vertices of the clothes mesh, "s_range": its
    key rings' arc lengths}. ``seated``: names of the poses that sit on a seat. Every
    ``pose_<name>`` clip and sampled frames of the other clips are draped; the clothes mesh gets
    morph targets ``drape_0..`` and every clip a weights track for them. Returns a report
    ({"morphs", "frames", "residual_mm"}), empty when there is nothing to drape."""
    draper = Draper(root, shells)
    if not draper.grids:
        return {}
    clothes = next(n for n in root.children if n.name == "clothes")

    actor = root.instance(share_clips=True)                    # clips key every joint: one actor does
    # Frames where nothing the drape depends on moves (a wave, a work clip: arms and chest) drape
    # the same: keyed by those joints' and their ancestors' local transforms (and the seat), done once.
    joints = [n for n in actor.iter_nodes() if n.name in draper.joints]
    joints += {id(a): a for n in joints for a in _ancestors(n) if a.name not in draper.joints}.values()
    done: dict = {}

    def delta_at(clip, t: float, seat: bool) -> np.ndarray:
        clip.apply(actor, t)
        key = (seat, *(np.round(n.transform.to_matrix(), 9).tobytes() for n in joints))
        if key not in done:
            done[key] = _delta(clip, seat)
        return done[key]

    def _delta(clip, seat: bool) -> np.ndarray:
        shell_node = _named(actor, "clothes")
        shell, out, blended = draper.positions(actor, seat)
        # Posed displacement -> rest-space delta, through each vertex's blended skinning matrix.
        world = out - shell
        local = world @ np.linalg.inv(shell_node.world_transform()[:3, :3]).T
        # (a vertex blended half and half across a right angle has a weak matrix: damp, don't invert it blindly)
        a = np.einsum("vji,vjk->vik", blended, blended) + 1e-3 * np.eye(3)
        return np.linalg.solve(a, np.einsum("vji,vj->vi", blended, local)[:, :, None])[:, :, 0]

    def seats_along(clip, times) -> list:
        """Which frames of a clip sit on a seat: the seated end of a transition. (Carrying the
        seat along under the frames before it, as the root moves, was tried: it takes more morph
        targets than there are to approximate, and the cloth wobbles.)"""
        at_end = [False] * len(times)
        at_end[0], at_end[-1] = [clip.meta.get(key) in seated for key in ("from", "to")]
        return at_end

    frames = []                                                # (clip, key times, row range in `deltas`)
    deltas, static = [], []
    for clip in root.clips:
        if clip.name.startswith("pose_"):
            times = np.array([0.0, max(clip.duration, 1e-3)])
            d = delta_at(clip, 0.0, clip.name[5:] in seated)
            rows = [d, d]
            static.append(d.ravel())
        else:
            count = clip.samples or SAMPLES.get(clip.name, DEFAULT_SAMPLES)
            times = np.linspace(0.0, clip.duration, count + 1)
            seats = seats_along(clip, times)
            rows = [delta_at(clip, float(t), seats[i]) for i, t in enumerate(times[:-1] if clip.loop else times)]
            if clip.loop:
                rows.append(rows[0])
        frames.append((clip, times, len(deltas), len(deltas) + len(rows)))
        deltas += rows
    data = np.array([d.ravel() for d in deltas])               # (frames, 3 * vertices)
    if not len(data) or np.abs(data).max() < 1e-5:
        return {}
    # The targets: first the static poses' own drapes (held for minutes: exact), biggest first,
    # each less what the ones before it already give; then the principal components of what is
    # left of the clips' frames.
    basis: list[np.ndarray] = []

    def left(rows: np.ndarray) -> np.ndarray:
        if not basis:
            return rows
        q = np.array(basis)
        return rows - (rows @ q.T) @ q

    for d in sorted(static, key=lambda d: -float(np.abs(d).max())):
        r = left(d[None])[0]
        if np.abs(r).max() > OWN_TARGET and len(basis) < min(MAX_OWN, max_morphs):
            basis.append(r / np.linalg.norm(r))
    rest_of = left(data)
    residual = float(np.abs(rest_of).max())
    if residual > RESIDUAL and len(basis) < max_morphs:
        # (the walk is what is seen most: its frames count for more)
        emphasis = np.ones(len(data))
        for clip, _, start, end in frames:
            emphasis[start:end] = EMPHASIS.get(clip.name, 1.0)
        _, _, vt = np.linalg.svd(rest_of * emphasis[:, None], full_matrices=False)
        for v in vt:
            if len(basis) >= max_morphs or residual <= RESIDUAL:
                break
            basis.append(v)
            residual = float(np.abs(left(data)).max())
    basis = np.array(basis)
    count = len(basis)
    coefficients = data @ basis.T                              # (frames, morphs)
    # Scale each target so its weights stay within -1..1.
    peak = np.maximum(np.abs(coefficients).max(axis=0), 1e-9)
    basis, coefficients = basis * peak[:, None], coefficients / peak
    mesh = clothes.mesh.copy()
    mesh.morphs = {MORPH.format(i): basis[i].reshape(-1, 3) for i in range(count)}
    clothes.mesh = mesh
    for clip, times, start, end in frames:
        from ..core.skin import MorphTrack

        clip.morphs["clothes"] = MorphTrack(times, {MORPH.format(i): coefficients[start:end, i] for i in range(count)})
    error = np.abs(data - coefficients * peak @ (basis / peak[:, None])).reshape(len(data), -1)
    return {"morphs": count, "frames": len(data), "residual_mm": round(float(residual) * 1000, 2),
            "clips": {clip.name: [round(float(np.abs(data[a:b]).max()) * 1000, 1), round(float(error[a:b].max()) * 1000, 1)]
                      for clip, _, a, b in frames}}
