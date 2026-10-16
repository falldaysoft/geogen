"""Clipping QA for dressed humanoids: what pokes through what, pose by pose (geogen-z2b.16.15).

``measure(character, pose)`` poses a built humanoid (``pose_<name>`` clip, or a clip at a time)
and reports, in millimetres:

- ``shell_poke``: loose garments (the ``clothes`` shell: skirts, dresses). A body vertex is
  *covered* when it lies inside the shell in the pose the garment was built in (stand); in the
  measured pose, covered vertices outside the shell poke through (unless they are nearer its
  hem's opening than its cloth: they left by the opening).
- ``hands``: hand and forearm vertices inside the rest of the body (thighs, hips, belly) or
  inside the loose shell (a palm through the skirt).
- ``feet_skin``: skin (not shoe) vertices on the heels of a body wearing shoes: behind and below
  the ankle joint (an open instep, as in flats, is fine).
- ``seat``: with ``seat=(node, spot)`` from ``layout.affordance_qa``: body and shell vertices
  inside the seat's meshes, posed on it the way the runtime does (``pose_actor``).

Each metric is ``{count, max_mm}`` over offending vertices deeper than ``TOLERANCE``.
``report(...)`` sweeps presets x outfits x poses; ``tests/test_clipping.py`` guards the
recorded baseline (``tests/data/clipping_baseline.json``) so fixes show their improvement.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import roots
from .core.node import SceneNode

TOLERANCE = 0.002            # m: penetration below this doesn't count
HAND_BONES = ("Hand", "LowerArm", "Thumb", "Index", "Middle", "Ring", "Little")
ARM_BONES = HAND_BONES + ("UpperArm", "Shoulder")
FOOT_BONES = ("Foot", "Toes")


# --- geometry -----------------------------------------------------------------------------------


def winding(points: np.ndarray, tris: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Generalised winding number of each point w.r.t. triangles (n, 3, 3): ~1 inside a closed
    mesh, ~0 outside, robust to small holes (Jacobson et al.)."""
    out = np.zeros(len(points))
    if not len(points) or not len(tris):
        return out
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk][:, None, None, :]
        d = tris[None] - p                                   # (c, f, 3, 3)
        a, b, c = d[:, :, 0], d[:, :, 1], d[:, :, 2]
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la \
            + np.einsum("ijk,ijk->ij", c, a) * lb
        out[i:i + chunk] = np.arctan2(det, den).sum(axis=1) / (2 * np.pi)
    return out


def distance(points: np.ndarray, tris: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Distance from each point to the nearest of the triangles (exact, vectorised)."""
    out = np.full(len(points), np.inf)
    if not len(points) or not len(tris):
        return out
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac = b - a, c - a
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk][:, None, :]
        ap = p - a
        d1, d2 = (ap * ab).sum(-1), (ap * ac).sum(-1)
        bp = p - b
        d3, d4 = (bp * ab).sum(-1), (bp * ac).sum(-1)
        cp = p - c
        d5, d6 = (cp * ab).sum(-1), (cp * ac).sum(-1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = np.where(np.abs(va + vb + vc) < 1e-20, 1e-20, va + vb + vc)
        v = vb / denom
        w = vc / denom
        q = a + ab * v[..., None] + ac * w[..., None]         # inside the face
        # Edges and vertices (Ericson, Real-Time Collision Detection 5.1.5).
        t_ab = np.clip(d1 / np.maximum(d1 - d3, 1e-20), 0, 1)
        t_ac = np.clip(d2 / np.maximum(d2 - d6, 1e-20), 0, 1)
        t_bc = np.clip((d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-20), 0, 1)
        e_ab = a + ab * t_ab[..., None]
        e_ac = a + ac * t_ac[..., None]
        e_bc = b + (c - b) * t_bc[..., None]
        inside = (va >= 0) & (vb >= 0) & (vc >= 0)
        cand = np.stack([q, e_ab, e_ac, e_bc], axis=0)
        dist = np.linalg.norm(cand - p[None], axis=-1)
        dist[0] = np.where(inside, dist[0], np.inf)
        out[i:i + chunk] = dist.min(axis=0).min(axis=1)
    return out


def _tris(mesh, faces=None) -> np.ndarray:
    f = mesh.faces if faces is None else faces
    return mesh.vertices[f]


def _solid(mesh, caps: bool = False):
    """A garment shell's triangles as a closed solid. A hemmed (open) shell is its outer wall and
    rim (not the lining) capped across each hem opening with a fan, so the space it wraps is
    inside, as for a closed shell. With ``caps``: (the solid, the cap fans alone)."""
    from .generators.humanoid import lining_faces

    inner = lining_faces(mesh)
    if not inner.any():
        return (mesh.vertices[mesh.faces], np.empty((0, 3, 3))) if caps else mesh.vertices[mesh.faces]
    outer = mesh.faces[~inner]
    # Directed boundary edges of the outer wall: a -> b with no b -> a.
    edges = np.concatenate([outer[:, [0, 1]], outer[:, [1, 2]], outer[:, [2, 0]]])
    have = {tuple(e) for e in edges.tolist()}
    boundary = [(a, b) for a, b in edges.tolist() if (b, a) not in have]
    # Group into loops (connected components) and fan each from its centroid.
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for a, b in boundary:
        parent[find(a)] = find(b)
    loops: dict[int, list] = {}
    for a, b in boundary:
        loops.setdefault(find(a), []).append((a, b))
    tris = [mesh.vertices[outer]]
    for loop in loops.values():
        centre = mesh.vertices[np.unique(np.array(loop))].mean(axis=0)
        fan = np.array([[mesh.vertices[b], mesh.vertices[a], centre] for a, b in loop])
        tris.append(fan)
    if caps:
        return np.concatenate(tris), np.concatenate(tris[1:]) if len(tris) > 1 else np.empty((0, 3, 3))
    return np.concatenate(tris)


# --- posing -------------------------------------------------------------------------------------


def posed(character: SceneNode, pose: str | None = None, clip: str | None = None, t: float = 0.0) -> SceneNode:
    """An instance of ``character`` in skeletal pose ``pose`` (its ``pose_<pose>`` clip) or at time
    ``t`` of ``clip``."""
    from .core.skin import pose_clips

    actor = character.instance()
    if pose is not None and pose != "stand":
        if not pose_clips(actor, f"pose_{pose}", 0.0):
            raise ValueError(f"no pose '{pose}'")
    if clip is not None:
        if not pose_clips(actor, clip, t):
            raise ValueError(f"no clip '{clip}'")
    return actor


def _meshes(actor: SceneNode) -> dict[str, Any]:
    out = {}
    for n in actor.iter_nodes():
        if n.mesh is not None and n.name in ("body", "clothes", "hair"):
            out[n.name] = n.world_mesh()
    return out


def _bone_mask(node: SceneNode, mesh, keys: tuple[str, ...], threshold: float = 0.5) -> np.ndarray:
    """Vertices whose weights on bones whose names contain one of ``keys`` sum to >= threshold."""
    skin_node = next(n for n in node.iter_nodes() if n.name == "body")
    names = [j.name for j in skin_node.skin.joints]
    hit = np.array([any(k in nm for k in keys) for nm in names])
    w = np.where(hit[mesh.joints], mesh.weights, 0.0).sum(axis=1)
    return w >= threshold


def _slot_vertices(mesh, name: str) -> np.ndarray:
    """Vertices used only by faces of material ``name``."""
    if mesh.face_materials is None:
        return np.ones(len(mesh.vertices), bool) if (mesh.material and mesh.material.name == name) \
            else np.zeros(len(mesh.vertices), bool)
    slots = [i for i, m in enumerate(mesh.materials) if m is not None and m.name == name]
    face_in = np.isin(mesh.face_materials, slots)
    used_in = np.zeros(len(mesh.vertices), bool)
    used_out = np.zeros(len(mesh.vertices), bool)
    used_in[mesh.faces[face_in].ravel()] = True
    used_out[mesh.faces[~face_in].ravel()] = True
    return used_in & ~used_out


def _metric(depths: np.ndarray) -> dict[str, float]:
    bad = depths[depths > TOLERANCE]
    return {"count": int(len(bad)), "max_mm": round(float(bad.max() * 1000), 1) if len(bad) else 0.0}


# --- the waist ----------------------------------------------------------------------------------

WAIST_BAND = 0.12            # m either side of the top's hem that the waist checks look at
TRUNK_BONES = ("Hips", "Spine", "Chest", "UpperChest", "LeftUpperLeg", "RightUpperLeg")
HEM_SPIKE = 0.008            # m a hem vertex may be out of line with its neighbours
_BARY = np.array([[i, j, 4 - i - j] for i in range(5) for j in range(5 - i)]) / 4.0     # 15 points a face


def _dominant(actor: SceneNode, mesh) -> np.ndarray:
    """Each body vertex's heaviest bone, by name."""
    from .generators.humanoid import dominant_bones

    skin_node = next(n for n in actor.iter_nodes() if n.name == "body")
    return dominant_bones(mesh, [j.name for j in skin_node.skin.joints])


def worn(character: SceneNode, part: str) -> dict | None:
    """The garment worn as ``part`` (top | bottom | dress | shoes): ``meta.humanoid.garments``."""
    garments = (character.meta.get("humanoid") or {}).get("garments") or []
    return next((g for g in garments if g["part"] == part), None)


def _waist(character: SceneNode, actor: SceneNode, rest: dict, now: dict) -> dict[str, Any]:
    """The waist of an outfit with a separate top and bottom, in the measured pose:

    - ``waist_gap``: skin showing between them: skin vertices of the pelvis, spine and thighs
      within ``WAIST_BAND`` of the top's hem (rest pose) that a loose bottom doesn't cover
      (``max_mm``: how far outside its shell), or any at all when the bottom is tight too
      (``max_mm``: how far from the hem).
    - ``waist_overlap``: ``min_mm``, the least the two overlap anywhere round the hem. Under a
      loose bottom: from the hem to the nearest body surface that shows (negative: how far the
      hem itself is outside the shell). Tight on tight they are one surface, so it is how far
      the bottom runs on under the top (``meta.humanoid.waist_overlap``), whatever the pose.
    - ``hem``: the hem as a smooth loop: vertices more than ``HEM_SPIKE`` out of line with their
      two neighbours (``max_mm`` the worst), and ``ends``, hem vertices without two neighbours.
      Out of line is how far above or below the line between them the vertex is in the rest
      pose (a level hem; the body's girth isn't a fault), plus how much further off it, any way,
      the pose takes it (a hem torn into spikes).
    """
    from scipy.spatial import cKDTree

    from .generators.humanoid import hem_shape, waist_hem

    top, bottom = worn(character, "top"), worn(character, "bottom")
    if top is None or bottom is None or not len(top["body"]):
        return {}
    body_rest, body = rest["body"], now["body"]
    dominant = _dominant(actor, body)
    mine = np.zeros(len(body.vertices), bool)
    mine[top["body"]] = True
    edges = waist_hem(body_rest, mine[body.faces].all(axis=1), dominant)
    if not len(edges):
        return {}
    hem = np.unique(edges)
    out: dict[str, Any] = {}
    # The body round the waist, sampled over its faces (vertices alone miss a face that shows).
    hem_tree = cKDTree(body_rest.vertices[hem])
    band = hem_tree.query(body_rest.vertices)[0] < WAIST_BAND
    trunk = np.isin(dominant, TRUNK_BONES)
    faces = body.faces[band[body.faces].any(axis=1) & trunk[body.faces].all(axis=1)]
    samples = np.einsum("bk,fkd->fbd", _BARY, body.vertices[faces])
    at_rest = np.einsum("bk,fkd->fbd", _BARY, body_rest.vertices[faces])
    skin = np.ones(len(body.vertices), bool)
    for g in character.meta["humanoid"]["garments"]:
        skin[g["body"]] = False
    bare = skin[faces].all(axis=1)
    in_band = hem_tree.query(at_rest[bare].reshape(-1, 3))[0] < WAIST_BAND if bare.any() else np.zeros(0, bool)
    pts = samples[bare].reshape(-1, 3)[in_band]
    samples = samples.reshape(-1, 3)
    if len(bottom["clothes"]) and "clothes" in now:
        solid, shell = _solid(now["clothes"]), _tris(now["clothes"])
        depth = np.zeros(len(pts))
        outside = winding(pts, solid) < 0.5
        if outside.any():
            depth[outside] = distance(pts[outside], shell)
        out["waist_gap"] = _metric(depth)
        showing = samples[winding(samples, solid) < 0.5]
        hem_pts = body.vertices[hem]
        hem_out = winding(hem_pts, solid) < 0.5
        overlap = np.full(len(hem_pts), WAIST_BAND)
        if len(showing):
            overlap = np.minimum(cKDTree(showing).query(hem_pts)[0], WAIST_BAND)
        if hem_out.any():
            overlap[hem_out] = -distance(hem_pts[hem_out], shell)
        out["waist_overlap"] = {"min_mm": round(float(overlap.min() * 1000), 1)}
    else:
        depth = cKDTree(body.vertices[hem]).query(pts)[0] if len(pts) else np.zeros(0)
        out["waist_gap"] = _metric(depth)
        declared = character.meta["humanoid"].get("waist_overlap")
        if declared is not None:
            out["waist_overlap"] = {"min_mm": round(float(declared) * 1000, 1)}
    was, ends = hem_shape(body_rest.vertices, edges)
    shape, _ = hem_shape(body.vertices, edges)
    level = np.array([abs(body_rest.vertices[v][1] - entry[1][1]) for v, entry in was.items()])
    torn = np.array([max(shape[v][0] - entry[0], 0.0) for v, entry in was.items()])
    off = level + torn
    bad = off[off > HEM_SPIKE]
    out["hem"] = {"count": int(len(bad)), "max_mm": round(float(off.max() * 1000), 1) if len(off) else 0.0,
                  "ends": int(ends)}
    return out


# --- measuring ----------------------------------------------------------------------------------


def measure(character: SceneNode, pose: str | None = "stand", clip: str | None = None, t: float = 0.0,
            seat=None) -> dict[str, Any]:
    rest = _meshes(posed(character))
    actor = posed(character, pose, clip, t)
    now = _meshes(actor)
    body = now["body"]
    out: dict[str, Any] = {}
    # Loose shells: covered at rest, outside now.
    if "clothes" in now:
        covered = winding(rest["body"].vertices, _solid(rest["clothes"])) > 0.5
        covered &= ~_bone_mask(actor, body, ARM_BONES)       # a hand resting in a skirt's folds isn't under it
        pts = body.vertices[covered]
        solid, opening = _solid(now["clothes"], caps=True)
        outside = winding(pts, solid) < 0.5
        depth = np.zeros(len(pts))
        if outside.any():
            wall = distance(pts[outside], _tris(now["clothes"]))
            # Out through the hem's opening (a knee below a skirt that rode up) is not through the cloth.
            left = distance(pts[outside], opening) < wall if len(opening) else np.zeros(len(wall), bool)
            depth[outside] = np.where(left, 0.0, wall)
        out["shell_poke"] = _metric(depth)
    # Hands and forearms in the rest of the body, or in the shell.
    hand = _bone_mask(actor, body, HAND_BONES)
    others = ~np.any(hand[body.faces], axis=1)
    pts = body.vertices[hand]
    inside = winding(pts, _tris(body, body.faces[others])) > 0.5
    if "clothes" in now:
        inside |= winding(pts, _solid(now["clothes"])) > 0.5
    depth = np.zeros(len(pts))
    if inside.any():
        tris = _tris(body, body.faces[others])
        if "clothes" in now:
            tris = np.concatenate([tris, _tris(now["clothes"])])
        depth[inside] = distance(pts[inside], tris)
    out["hands"] = _metric(depth)
    # Bare feet inside shoes (only when the outfit has shoes: leather on the body).
    if body.materials and any(m is not None and m.name == "leather" for m in body.materials):
        skin_feet = _slot_vertices(body, "skin") & _bone_mask(actor, body, FOOT_BONES)
        heel = np.zeros(len(body.vertices), bool)
        for side in ("Left", "Right"):
            ankle = next(n for n in actor.iter_nodes() if n.name == f"{side}Foot")
            m = ankle.world_transform()
            local = (np.linalg.inv(m) @ np.c_[body.vertices, np.ones(len(body.vertices))].T).T[:, :3]
            # Foot bones point +Y toward the toes: behind the ankle is -Y, below it is (bone) -Z.
            below = body.vertices[:, 1] < m[1, 3] - 0.02         # under the ankle (not the ankle itself)
            heel |= (local[:, 1] < 0.0) & (np.linalg.norm(local, axis=1) < 0.2) & below
        out["feet_skin"] = {"count": int((skin_feet & heel).sum()), "max_mm": 0.0}
    # Top over bottom round the waist.
    out.update(_waist(character, actor, rest, now))
    # Posed on a seat.
    if seat is not None:
        from .layout.affordance_qa import pose_actor

        seat_node, spot = seat
        sitter = pose_actor(character, spot)
        mine = _meshes(sitter)
        pts = np.concatenate([m.vertices for m in mine.values() if m is not None])
        tris = []
        for n in seat_node.iter_nodes():
            if n.mesh is not None and len(n.mesh.faces):
                tris.append(n.world_mesh().vertices[n.mesh.faces])
        tris = np.concatenate(tris)
        inside = winding(pts, tris) > 0.5
        depth = np.zeros(len(pts))
        if inside.any():
            depth[inside] = distance(pts[inside], tris)
        out["seat"] = _metric(depth)
    return out


SEATS = {"armchair": "armchair.yaml", "chair": "chair.yaml", "sofa": "sofa.yaml"}


def report(presets, outfits, poses=("stand", "sit", "sit_at_table"), seats=tuple(SEATS), loader=None,
           assets_dir=None) -> dict[str, Any]:
    """{"<preset>/<outfit>": {pose: metrics, "seat_<name>": metrics}} for every combination."""
    from pathlib import Path

    from .layout.affordance_qa import affordance_spots
    from .layout.loader import LayoutLoader

    loader = loader or LayoutLoader()
    assets = Path(assets_dir or roots.CORE)
    seat_nodes = {}
    for name in seats:
        root = SceneNode("seat_" + name)
        node = loader.load(assets / SEATS[name])
        root.add_child(node)
        spots = [s for s in affordance_spots(root) if s.affordance["type"] == "sit"]
        seat_nodes[name] = (node, spots[0])
    out = {}
    for preset in presets:
        for outfit in outfits:
            character = loader.load(assets / "characters" / "humanoid.yaml",
                                    params={"preset": list(preset) if not isinstance(preset, str) else preset,
                                            "outfit": outfit})
            key = f"{'+'.join([preset] if isinstance(preset, str) else preset)}/{outfit}"
            entry = {pose: measure(character, pose) for pose in poses}
            for name, (node, spot) in seat_nodes.items():
                entry[f"seat_{name}"] = measure(character, "stand", seat=(node, spot))["seat"]
            out[key] = entry
    return out
