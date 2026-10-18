"""Arms that keep clear of the body and its clothes (geogen.11.5.5, ``docs/clothing.md``).

Poses and clips are authored once for every build, so a hand that hangs beside a slim hip is
inside a wide one, and a blend from standing to sitting takes the hands through the thighs.
``clear_arms`` runs on a built humanoid, after its clips exist and its loose clothes are draped:
in each static pose and at sampled frames of each clip it finds how far each arm has to swing
out from the shoulder for its hand and forearm to be outside the rest of the body and the
clothes, and adds that to the upper arm's keys. Nothing else moves, and a pose that is already
clear is left exactly as authored.

The swing is a rotation about the shoulder joint that carries the hand outward, away from the
body's centre plane: the same way in every frame, so the swings of neighbouring frames blend.
What it has to clear is the body's surface other than the arms, and the loose shell: its outer
wall, with the hems' openings capped. A static pose is solved exactly; a clip's sampled frames
are spread over their neighbours (``_spread``) and interpolated onto its keys.
"""

from __future__ import annotations

import numpy as np

from ..core.node import SceneNode
from ..core.transform import matrix_from_quat, quat_axis_angle, quat_from_matrix

SIDES = {"Left": 1.0, "Right": -1.0}                    # and which way along the body's X is out
ARM = ("UpperArm", "LowerArm", "Hand", "Thumb", "Index", "Middle", "Ring", "Little")
OUT_OF_REACH = ("Head", "Neck", "Jaw", "Eye", "Foot", "Toes")       # the body's parts no hand comes near in a clip
# (the bones compared to tell that one clip is another played backwards)
CHECKED = ("Hips", "Spine", "LeftUpperArm", "LeftLowerArm", "RightUpperArm", "LeftUpperLeg", "LeftLowerLeg")
KEPT_CLEAR = ARM[1:]         # the forearm and hand (what clipping QA's `hands` measures)
CLEARANCE = 0.004            # m the hand and forearm keep off the body and the cloth, once they are moved
SKIN = 0.001                 # m a vertex may be inside before the arm is moved at all (it rests on the body)
LIMIT = np.radians(60.0)     # the most an arm is swung
STEPS = 10                   # tries at finding the swing
OVERSHOOT = 1.15             # each try goes this much further than the depth left suggests
MIN_STEP = np.radians(0.5)
TIE = 3e-4                   # m: triangles this nearly as near as the nearest count as meeting at the nearest point
NEAREST = 8                  # sample points whose triangles are tried for the nearest
SAMPLES = {"walk": 12, "idle": 3, "wave": 6}      # frames looked at per clip (others: DEFAULT_SAMPLES)
DEFAULT_SAMPLES = 14
_BARY = np.array([[2, 2, 2], [0, 3, 3], [3, 0, 3], [3, 3, 0]]) / 6.0       # points a face is sampled at
_DENSE = np.array([[i, j, 6 - i - j] for i in range(7) for j in range(7 - i)]) / 6.0


class _Surface:
    """What the arms keep out of, as triangles of a skinned mesh: its topology, worked out once.
    ``at(vertices)`` is the surface in a pose. ``neutral`` faces are along an open edge that is
    left open (an armpit); ``fans`` are openings to close, each with a fan from its middle."""

    def __init__(self, faces: np.ndarray, rest: np.ndarray, neutral: np.ndarray | None = None, fans=()) -> None:
        self.loops = [loop for _, _, loop in fans]
        self.closed = bool(fans)                               # (a shell with its openings capped)
        caps = [np.stack([b, a, np.full(len(a), len(rest) + i)], axis=1) for i, (a, b, _) in enumerate(fans)]
        self.big = sum(len(c) for c in caps)                   # the caps' triangles are large: sampled densely
        self.faces = np.concatenate([faces, *caps])
        self.neutral = np.zeros(len(self.faces), bool)
        if neutral is not None:
            self.neutral[:len(faces)] = neutral
        # Each face's neighbours: the faces it shares a vertex with (vertices doubled along a UV seam are one).
        points = np.concatenate([rest, [rest[loop].mean(axis=0) for loop in self.loops]]) if fans else rest
        same = np.unique(np.round(points, 6), axis=0, return_inverse=True)[1].ravel()[self.faces]
        at: dict[int, list[int]] = {}
        for f, corners in enumerate(same.tolist()):
            for v in corners:
                at.setdefault(v, []).append(f)
        near = [sorted({g for v in corners for g in at[v]}) for corners in same.tolist()]
        self.near = np.array([n + [n[0]] * (max(map(len, near)) - len(n)) for n in near])

    def at(self, vertices: np.ndarray) -> _Obstacle:
        if self.loops:
            vertices = np.concatenate([vertices, [vertices[loop].mean(axis=0) for loop in self.loops]])
        return _Obstacle(self, vertices[self.faces])


class _Obstacle:
    """A ``_Surface`` in a pose. ``probe`` gives each point's depth behind it (negative outside)
    and the surface's normal there, from the nearest triangle: found among those of the nearest
    sample points, the sign taken from the one the point is most squarely in front of or behind
    (which is right at an edge two faces share). A point nearest a neutral face is clear."""

    def __init__(self, surface: _Surface, tris: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        self.surface, self.tris = surface, tris
        self.normals = normals / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]
        small = len(tris) - surface.big
        self.face = np.concatenate([np.repeat(np.arange(small), len(_BARY)), np.repeat(np.arange(small, len(tris)), len(_DENSE))])
        self.tree = cKDTree(np.concatenate([np.einsum("bk,fkd->fbd", _BARY, tris[:small]).reshape(-1, 3),
                                            np.einsum("bk,fkd->fbd", _DENSE, tris[small:]).reshape(-1, 3)]))

    def probe(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        depth, face = self._nearest(points, self.face[self.tree.query(points, k=min(NEAREST, len(self.face)))[1]])
        inside = np.flatnonzero(depth > 0)
        if len(inside):
            # Every triangle that meets at the nearest point is wanted (they tie, and the wrong one
            # of two at an edge puts a point behind a surface it is in front of): try its neighbours.
            depth[inside], face[inside] = self._nearest(points[inside], self.surface.near[face[inside]])
            inside = inside[depth[inside] > 0]
            if self.surface.closed and len(inside):
                # A rolled hem's lip still fools the nearest triangle: a closed surface can be asked outright.
                from ..clipping import winding

                out = winding(points[inside], self.tris) < 0.5
                depth[inside[out]] *= -1.0
        return np.where(self.surface.neutral[face], -np.inf, depth), self.normals[face]

    def _nearest(self, points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Of ``faces`` (points, k) the nearest to each point: (signed depth, the face)."""
        offset = points[:, None, :] - _closest(points[:, None, :], self.tris[faces])
        distance = np.sqrt(_dot(offset, offset))
        side = _dot(offset, self.normals[faces])
        # Of the triangles as near as the nearest, the one the point is most squarely off.
        tied = distance <= distance.min(axis=1, keepdims=True) + TIE
        pick = np.argmax(np.where(tied, np.abs(side), -1.0), axis=1)
        rows = np.arange(len(points))
        distance = distance[rows, pick]
        return np.where(side[rows, pick] < 0, distance, -distance), faces[rows, pick]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:           # (np.cross is several times slower)
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _closest(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """The point of each triangle (..., 3, 3) nearest ``p`` (broadcast against it)."""
    a, b, c = tris[..., 0, :], tris[..., 1, :], tris[..., 2, :]
    ab, ac, bc = b - a, c - a, c - b
    n = _cross(ab, ac)
    # In the triangle's plane, if that falls inside it; otherwise on the nearest edge.
    plane = p - n * (_dot(p - a, n) / np.maximum(_dot(n, n), 1e-30))[..., None]
    inside = np.ones(plane.shape[:-1], bool)
    for start, edge in ((a, ab), (b, bc), (c, -ac)):
        inside &= _dot(_cross(edge, plane - start), n) >= 0
    best = np.where(inside[..., None], plane, np.inf)
    best_d = np.where(inside, _dot(p - plane, p - plane), np.inf)
    for start, edge in ((a, ab), (b, bc), (a, ac)):
        t = np.clip(_dot(p - start, edge) / np.maximum(_dot(edge, edge), 1e-30), 0.0, 1.0)
        q = start + edge * t[..., None]
        d = _dot(p - q, p - q)
        closer = d < best_d
        best, best_d = np.where(closer[..., None], q, best), np.where(closer, d, best_d)
    return best


def _fans(faces: np.ndarray, vertices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The openings of ``faces`` (with ``vertices`` at rest), each a loop of open edges: (starts,
    ends, the loop's vertices), directed as the faces run."""
    # (vertices doubled along a UV seam are one: the seam is no opening)
    _, first, same = np.unique(np.round(vertices, 6), axis=0, return_index=True, return_inverse=True)
    faces = first[same.ravel()][faces]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    have = {tuple(e) for e in edges.tolist()}
    open_ = [(a, b) for a, b in edges.tolist() if (b, a) not in have]
    group: dict[int, int] = {}

    def find(x: int) -> int:
        while group.setdefault(x, x) != x:
            x = group[x]
        return x

    for a, b in open_:
        group[find(a)] = find(b)
    loops: dict[int, list] = {}
    for a, b in open_:
        loops.setdefault(find(a), []).append((a, b))
    return [(np.array(loop)[:, 0], np.array(loop)[:, 1], np.unique(np.array(loop))) for loop in loops.values()]


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    return matrix_from_quat(quat_axis_angle(axis, angle))


def _swing(points: np.ndarray, pivot: np.ndarray, along: np.ndarray, outward: np.ndarray,
           obstacles: list[_Obstacle]) -> np.ndarray:
    """The least rotation about ``pivot`` (a rotation vector) that takes ``points`` clear, or
    none if they are (near enough) clear already or nothing within the limit clears them. The
    arm (pointing ``along``) swings ``outward``, away from the body's centre plane: the same way
    in every frame, so swings blend from one frame to the next. Failing that, the way the surface
    it is in faces."""

    def probe(at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        found = [o.probe(at) for o in obstacles]
        deepest = np.argmax([d for d, _ in found], axis=0)
        rows = np.arange(len(at))
        return np.array([d for d, _ in found])[deepest, rows], np.array([n for _, n in found])[deepest, rows]

    depth, normal = probe(points)
    if depth.max() <= SKIN:
        return np.zeros(3)
    inside = depth > 0
    push = (normal[inside] * depth[inside, None]).sum(axis=0)
    reach = np.linalg.norm(points - pivot, axis=1)
    for way in (outward, push):
        axis = np.cross(along, way)
        if np.linalg.norm(axis) < 0.2 * np.linalg.norm(way):
            continue
        axis /= np.linalg.norm(axis)
        # Each step is what the deepest vertex still lacks, as an angle at its distance from the shoulder.
        angle, deep = 0.0, depth
        for _ in range(STEPS):
            worst = int(np.argmax(deep))
            angle += max(OVERSHOOT * (deep[worst] + CLEARANCE) / max(reach[worst], 0.05), MIN_STEP)
            if angle > LIMIT:
                break
            deep = probe(pivot + (points - pivot) @ _rotation(axis, angle).T)[0]
            if deep.max() <= -CLEARANCE:
                return axis * angle
    return np.zeros(3)


def _outline(vertices: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Of each group of vertex indices (a rigid part: a forearm, a hand), the ones on its convex
    hull: what is deepest in anything the part is pushed into is one of these."""
    from scipy.spatial import ConvexHull, QhullError

    out = []
    for group in groups:
        try:
            out.append(group[ConvexHull(vertices[group]).vertices] if len(group) > 4 else group)
        except QhullError:
            out.append(group)
    return np.concatenate(out)


def _spread(angles: np.ndarray, loop: bool) -> np.ndarray:
    """A swing that comes on before it is needed and goes off after: each sample takes the most
    its neighbours need, blurred, and never less than it needs itself. The ends of a one-shot
    clip stay as they are (they are the static poses it runs between)."""
    if not angles.any():
        return angles
    pad = (lambda a: np.concatenate([a[-1:], a, a[:1]])) if loop else (lambda a: np.concatenate([a[:1], a, a[-1:]]))
    wide = pad(angles)
    wide = np.maximum(np.maximum(wide[:-2], wide[2:]), wide[1:-1])
    blur = pad(wide)
    out = np.maximum(0.25 * blur[:-2] + 0.5 * blur[1:-1] + 0.25 * blur[2:], angles)
    if not loop:
        out[0], out[-1] = angles[0], angles[-1]
    return out


def _reversed(clip, other) -> bool:
    """Whether ``clip`` is ``other`` played backwards (getting up, after lying down)."""
    for bone in CHECKED:
        a, b = clip.tracks.get(bone), other.tracks.get(bone)
        if a is None or b is None or a.rotations is None or b.rotations is None or a.rotations.shape != b.rotations.shape:
            return False
        if not np.allclose(a.times, b.times) or np.abs(np.einsum("ki,ki->k", a.rotations, b.rotations[::-1])).min() < 1 - 1e-9:
            return False
    return True


def clear_arms(root: SceneNode) -> dict:
    """Swing the arms of a built humanoid ``root`` clear of its body and clothes in every clip
    (see the module docstring). Rewrites the upper arms' rotation keys; returns
    {clip: the largest swing in degrees} for the clips it changed."""
    from .humanoid import dominant_bones, lining_faces

    body = next((n for n in root.children if n.name == "body" and n.skin is not None), None)
    if body is None or not root.clips:
        return {}
    names = [j.name for j in body.skin.joints]
    mesh = body.mesh
    actor = root.instance()
    nodes = {n.name: n for n in actor.iter_nodes()}
    if not all(f"{side}{bone}" in nodes for side in SIDES for bone in ("UpperArm", "Shoulder", "Hand")):
        return {}

    def on(side: str, bones: tuple[str, ...]) -> np.ndarray:
        """Each vertex's weight on ``side``'s bones whose names start with one of ``bones``."""
        hit = np.array([n.startswith(side) and n.removeprefix(side).startswith(bones) for n in names])
        return np.where(hit[mesh.joints], mesh.weights, 0.0).sum(axis=1)

    # What is kept clear: the forearm and hand, each by the vertices on its hull.
    kept = {}
    for side in SIDES:
        forearm, whole = on(side, ("LowerArm",)) >= 0.5, on(side, KEPT_CLEAR) >= 0.5
        if not whole.any():
            return {}
        kept[side] = _outline(mesh.vertices, [g for g in (np.flatnonzero(forearm), np.flatnonzero(whole & ~forearm)) if len(g)])
    # What it is kept clear of: the body without the arms (and the head and feet, out of their
    # way). The faces round the holes that leaves hold nothing back: a forearm is always next to
    # its own elbow.
    dominant = dominant_bones(mesh, names)
    far = np.array([d.removeprefix("Left").removeprefix("Right").startswith(OUT_OF_REACH) for d in dominant])
    away = (sum(on(side, ARM) for side in SIDES) > 0.2) | far
    left_out = away[mesh.faces].any(axis=1)
    if left_out.all():
        return {}
    on_hole = np.zeros(len(mesh.vertices), bool)
    on_hole[mesh.faces[left_out]] = True
    solid = mesh.faces[~left_out]
    surface = _Surface(solid, mesh.vertices, on_hole[solid].any(axis=1))
    wanted = np.unique(np.concatenate([solid.ravel(), *kept.values()]))
    shell = None
    cloth = nodes.get("clothes")
    if cloth is not None and cloth.mesh is not None:
        # A loose shell: its outer wall, with each opening (hem) capped as clipping QA's solid is.
        wall = cloth.mesh.faces[~lining_faces(cloth.mesh)]
        shell = _Surface(wall, cloth.mesh.vertices, fans=_fans(wall, cloth.mesh.vertices))
    to_world = nodes["body"].world_transform()

    def needed(clip, t: float) -> dict[str, np.ndarray]:
        """Per side, the swing this frame needs as a rotation vector in the shoulder's frame."""
        clip.apply(actor, t)
        blended = np.einsum("vk,vkij->vij", mesh.weights[wanted], nodes["body"].skin.matrices(nodes["body"])[mesh.joints[wanted]])
        posed = np.zeros_like(mesh.vertices)
        posed[wanted] = (np.einsum("vij,vj->vi", blended[:, :3, :3], mesh.vertices[wanted]) + blended[:, :3, 3]) \
            @ to_world[:3, :3].T + to_world[:3, 3]
        obstacles = [surface.at(posed)] + ([shell.at(cloth.world_mesh().vertices)] if shell is not None else [])
        centre = nodes["Chest" if "Chest" in nodes else "Hips"].world_transform()[:3, :3]
        out = {}
        for side, sign in SIDES.items():
            pivot = nodes[f"{side}UpperArm"].world_transform()[:3, 3]
            along = nodes[f"{side}Hand"].world_transform()[:3, 3] - pivot
            along /= max(np.linalg.norm(along), 1e-9)
            swing = _swing(posed[kept[side]], pivot, along, sign * centre[:, 0], obstacles)
            parent = nodes[f"{side}Shoulder"].world_transform()[:3, :3]
            out[side] = (parent / np.linalg.norm(parent, axis=0)).T @ swing
        return out

    report = {}
    done: dict[str, tuple] = {}                                # clip name -> (clip, what its samples need)
    for clip in root.clips:
        tracks = {side: clip.tracks.get(f"{side}UpperArm") for side in SIDES}
        if any(t is None or t.rotations is None for t in tracks.values()):
            continue
        static = clip.name.startswith("pose_")
        if static:
            times = np.array([0.0])
        else:
            count = clip.samples or SAMPLES.get(clip.name, DEFAULT_SAMPLES)
            times = np.linspace(0.0, clip.duration, count + 1)[:None if not clip.loop else -1]
        # A clip that is another played backwards needs what that one does; one that runs between
        # two static poses starts and ends as they are.
        back = next((found for other, found in done.values() if not static and _reversed(clip, other)), None)
        ends = [done.get(f"pose_{clip.meta.get(key)}") for key in ("from", "to")]
        if back is not None:
            found = back[::-1]
        else:
            found = [ends[k and 1][1][0] if not static and k in (0, len(times) - 1) and ends[k and 1] is not None
                     else needed(clip, float(t)) for k, t in enumerate(times)]
        done[clip.name] = (clip, found)
        for side, track in tracks.items():
            vectors = np.array([f[side] for f in found])
            angles = np.linalg.norm(vectors, axis=1)
            if not angles.any():
                continue
            spread = angles if static else _spread(angles, clip.loop)
            # A sample that needs nothing itself swings about its nearest neighbour's axis that does.
            has = np.flatnonzero(angles > 0)
            near = [has[np.argmin(np.abs(has - i))] for i in range(len(angles))]
            vectors = vectors[near] / angles[near, None] * spread[:, None]
            if clip.loop and not static:
                at, vectors = np.append(times, clip.duration), np.vstack([vectors, vectors[:1]])
            else:
                at = times
            rotations = track.rotations.copy()
            for k, t in enumerate(track.times):
                v = vectors[0] if static else np.array([np.interp(t, at, vectors[:, d]) for d in range(3)])
                angle = float(np.linalg.norm(v))
                if angle < 1e-6:
                    continue
                q = quat_from_matrix(_rotation(v / angle, angle) @ matrix_from_quat(rotations[k]))
                rotations[k] = q if k == 0 or np.dot(q, rotations[k - 1]) >= 0 else -q
            track.rotations = rotations
            report[clip.name] = max(report.get(clip.name, 0.0), round(float(np.degrees(spread.max())), 1))
    return report
